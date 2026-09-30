"""Pause / resume through the real lifecycle (harness/checkpoint.py): the runner, a program, the
server. (1) A cron program on the weather task: a run paused at a
midnight and resumed from its checkpoint produces the same ledger event
sequence, settled outcomes and score as one that never stopped. (2) The
ext: TM-A memory candidate on the real bnpm world with a fake LLM: at the
pause the resident program winds down as at a real end (transcripts end
on the assistant's wait call), the checkpoint carries no results.json;
resumed from an EDITED snapshot (program and memory changed) every
market agent wakes at the cut on the `__resume__` trigger, the changed
memory is in its next system prompt, `code_change` is ledgered at the
boundary, and the final results cover both stages' claims. (3) The TM-B
candidate: jails and authored programs survive the cut, the curator
parks again. (4) A TM-D cron program resumes without a `__resume__`
firing: its restored crontab fires at the cut."""

from __future__ import annotations

import asyncio
import json
import re
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from harness import checkpoint, llm_proxy
from harness.run import run_experiment
from tests.conftest import make_config, write_weather_csv

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixture_agents"
TASK_DIR = REPO_ROOT / "tasks" / "breakout_news_pm"
EXT = TASK_DIR / "agent" / "ext_candidates"
BUILT = TASK_DIR / "data" / "built"
UTC = timezone.utc
START = datetime(2021, 6, 1, tzinfo=UTC)

bnpm_world = pytest.mark.skipif(
    not (BUILT / "breakpoints.jsonl").exists()
    or not (TASK_DIR / "news" / "tantivy_index_v3").exists(),
    reason="built world / tantivy index not present (gitignored)")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _resp(text: str, model: str) -> tuple[dict, float]:
    return ({"choices": [{"message": {"content": text}}],
             "usage": {"prompt_tokens": 20, "completion_tokens": 10},
             "model": model}, 0.0)


# -- (1) the weather poller: chain == continuous -----------------------------------


def _weather_cfg(tmp_path, name, days=3):
    temps = [25.0] * 14 + [35.0] + [25.0] * 9 + [25.0] * 24 * (days - 1)
    csv = write_weather_csv(tmp_path / f"{name}.csv", START, temps)
    return make_config(run_id=name, sim_start=START,
                       sim_end=START + timedelta(days=days),
                       data_cutoff=START, weather_csv=csv,
                       agent={"scaffold": "ext:fixture"})


def _event_view(rows: list[dict]) -> list[tuple]:
    out = []
    for e in rows:
        if e["type"] == "resume":
            continue
        key = (e["type"], e.get("id"), json.dumps(e.get("payload"), sort_keys=True),
               e.get("ref"), e.get("status"))
        out.append(key)
    return out


def test_weather_poller_chain_equals_continuous(tmp_path):
    cont_cfg = _weather_cfg(tmp_path, "cont")
    cont = asyncio.run(run_experiment(
        cont_cfg, repo_root=tmp_path, run_dir=tmp_path / "cont",
        scaffold_src=FIXTURES / "poller"))

    cfg = _weather_cfg(tmp_path, "chain")
    cfg.task["weather_csv"] = cont_cfg.task["weather_csv"]  # same csv file
    s1, s2 = tmp_path / "s1", tmp_path / "s2"
    paused = asyncio.run(run_experiment(
        cfg, repo_root=tmp_path, run_dir=s1, scaffold_src=FIXTURES / "poller",
        pause_at=START + timedelta(days=1)))
    assert paused is None
    assert (s1 / checkpoint.CHECKPOINT).exists() and (s1 / checkpoint.PARTIAL).exists()
    assert not (s1 / "results.json").exists()
    ck = json.loads((s1 / checkpoint.CHECKPOINT).read_text())
    assert ck["paused_at"] == "2021-06-02T00:00:00Z" and ck["flags"] == []
    # the runner saw `done` and stopped; the stage's own triggers ran
    rows1 = read_jsonl(s1 / "ledger.jsonl")
    assert [e["id"] for e in rows1 if e["type"] == "trigger"] == ["__bootstrap__", "daily"]

    # stage 2: the snapshot is the stage-1 workspace as left
    snapshot = tmp_path / "snapshot"
    shutil.copytree(s1 / "workspace", snapshot, ignore=shutil.ignore_patterns("__pycache__"))
    res = asyncio.run(run_experiment(
        cfg, repo_root=tmp_path, run_dir=s2, scaffold_src=snapshot, resume_from=s1))
    assert res is not None and (s2 / "results.json").exists()
    assert json.loads((s2 / checkpoint.RESUME).read_text())["parent"] == str(s1.resolve())
    rows2 = read_jsonl(s2 / "ledger.jsonl")
    assert rows2[:len(rows1)] == rows1  # the parent's rows verbatim
    resume = [e for e in rows2 if e["type"] == "resume"]
    assert len(resume) == 1 and resume[0]["resume_trigger"] is False  # cron cell
    # a cron cell restarts from its crontab, not a resume one-shot
    after = [e for e in rows2[len(rows1):] if e["type"] == "trigger"]
    assert [(e["id"], e["sim_time"]) for e in after] == [
        ("daily", "2021-06-02T23:00:00Z"), ("daily", "2021-06-03T23:00:00Z")]
    assert [e["seq"] for e in rows2] == list(range(1, len(rows2) + 1))
    # same events in the same order, same settled outcomes, same score
    assert _event_view(rows2) == _event_view(read_jsonl(tmp_path / "cont" / "ledger.jsonl"))
    assert res["performance"] == cont["performance"]
    assert res["task"] == cont["task"]
    assert res["resources"]["spent_usd"] == cont["resources"]["spent_usd"]
    assert res["resources"]["counts"]["trigger"] == cont["resources"]["counts"]["trigger"]


# -- (2) the ext: TM-A memory candidate on the bnpm world --------------------------


def _bnpm_cfg(run_id, tm, scaffold, sim_end="2026-03-04T00:00:00Z"):
    from harness.config import RunConfig

    return RunConfig(
        run_id=run_id,
        task={"name": "breakout_news_pm",
              "markets": [{"market_id": "616902", "start": "2026-03-01T00:00:00Z",
                           "end": sim_end},
                          {"market_id": "678777", "start": "2026-03-01T00:00:00Z",
                           "end": sim_end}]},
        sim_start="2026-03-01T00:00:00Z", sim_end=sim_end, budget_usd=5.0,
        cell={"tm": tm, "tlrn": "none", "sig": "none", "alg": "none"},
        agent={"scaffold": scaffold, "model": "mock-luna"},
        watchdog_seconds=60.0)


def _now_of(body: dict) -> datetime:
    text = " ".join(m["content"] for m in body["messages"])
    times = re.findall(r'(?:woke at |"now": ?")([0-9TZ:\-\.\+]+)', text)
    return datetime.fromisoformat(times[-1].replace("Z", "+00:00"))


def _tma_fake(seen: dict):
    """Market agents: wake -> search -> notify the first hit -> sleep 9 h;
    the curator gets lessons. Records every system prompt it sees."""
    async def fake_upstream(path, body):
        sys_msg = body["messages"][0]["content"]
        if sys_msg.startswith("You curate compact operating memory"):
            return _resp(json.dumps({"lessons": ["Prefer novel, concrete developments."],
                                     "diagnosis": "fine"}), body["model"])
        seen.setdefault("prompts", []).append(sys_msg)
        last = body["messages"][-1]["content"]
        mid = re.search(r"market_id: (\d+)", sys_msg).group(1)
        now = _now_of(body)
        if '"results"' in last:
            hits = re.findall(r'"news_id": ?"([0-9a-f]{64})"', last)
            if hits:
                return _resp(json.dumps({"tool": "notify", "args": {
                    "market_id": mid, "news_id": hits[0], "direction": "up"},
                    "thought": "claim"}), body["model"])
        if last.startswith("(woke at") or '"woke_for"' in last:
            return _resp(json.dumps({"tool": "search_news", "args": {
                "q": "Fed OR strikes OR inflation",
                "date_from": (now - timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ")},
                "thought": "look"}), body["model"])
        until = (now + timedelta(hours=9)).strftime("%Y-%m-%dT%H:%M:%SZ")
        return _resp(json.dumps({"tool": "sleep", "args": {"until": until},
                                 "thought": "wait"}), body["model"])
    return fake_upstream


def _snapshot(stage_dir: Path, dst: Path) -> Path:
    shutil.copytree(stage_dir / "workspace", dst,
                    ignore=shutil.ignore_patterns("__pycache__"))
    return dst


@bnpm_world
def test_tma_candidate_pause_edit_resume(tmp_path, monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(llm_proxy, "_upstream", _tma_fake(seen))
    cand = EXT / "tmA-reflection-only"
    cfg = _bnpm_cfg("bnpm-online-tma", "A", "ext:tma-reflection-only")
    s1, s2 = tmp_path / "s1", tmp_path / "s2"
    cut = "2026-03-02T06:00:00Z"  # not a midnight: the curator's 24 h park
    #                               (a midnight) must have fired before it
    assert asyncio.run(run_experiment(
        cfg, repo_root=tmp_path, run_dir=s1, scaffold_src=cand, pause_at=cut)) is None

    ws1 = s1 / "workspace"
    assert not list((ws1 / "logs").glob("crash-*.log"))
    assert not (s1 / "results.json").exists() and (s1 / checkpoint.CHECKPOINT).exists()
    rows1 = read_jsonl(s1 / "ledger.jsonl")
    n_notify1 = len([e for e in rows1 if e["type"] == "notify"])
    assert n_notify1 >= 1, "the fake agents never notified; corpus empty?"
    # the program wound down as at a real end: one clean exit, and every
    # transcript ends on the assistant's wait call (experiment_over is
    # never appended)
    exits = [e for e in rows1 if e["type"] == "agent_exit"]
    assert [e["outcome"] for e in exits] == ["exit=0"]
    for name in ("m-616902", "m-678777"):
        tr = read_jsonl(ws1 / "logs" / f"transcript_{name}.jsonl")
        assert tr[-1]["role"] == "assistant"
        assert json.loads(tr[-1]["content"])["tool"] == "sleep"
    partial = json.loads((s1 / checkpoint.PARTIAL).read_text())
    assert all(b["t_move_start"] < cut for b in partial["task"]["breakpoints"])
    pending = [a for a in partial["task"]["alerts"] if a["status"] == "pending"]
    # the curator curated at least once before the cut (its 24 h park)
    obs1 = read_jsonl(ws1 / "logs" / "observability.jsonl")
    assert [e for e in obs1 if e["event"] == "curation"]

    # the improver's edit: program and memory
    snap = _snapshot(s1, tmp_path / "snapshot")
    main = snap / "main.py"
    main.write_text(main.read_text() + "\n# stage-2 edit\n")
    mem = snap / "memory" / "reflective_state.json"
    state = json.loads(mem.read_text())
    state["lessons"] = ["INJECTED LESSON: act on the first credible report."]
    mem.write_text(json.dumps(state))
    seen["prompts"] = []

    res = asyncio.run(run_experiment(
        cfg, repo_root=tmp_path, run_dir=s2, scaffold_src=snap, resume_from=s1))
    assert res is not None and (s2 / "results.json").exists()
    assert "failed-degenerate" not in res["flags"]
    ws2 = s2 / "workspace"
    assert not list((ws2 / "logs").glob("crash-*.log"))
    rows2 = read_jsonl(s2 / "ledger.jsonl")
    assert rows2[:len(rows1)] == rows1
    tail = rows2[len(rows1):]
    assert tail[0]["type"] == "resume" and tail[0]["resume_trigger"] is True
    # the resident program restarted AT the cut on the resume one-shot,
    # and the changed main.py was ledgered at the boundary
    trig = [e for e in tail if e["type"] == "trigger"][0]
    assert (trig["id"], trig["kind"], trig["sim_time"]) == (checkpoint.RESUME_TRIGGER, "at", cut)
    assert [e for e in tail if e["type"] == "code_change"]
    for name in ("m-616902", "m-678777"):
        tr = read_jsonl(ws2 / "logs" / f"transcript_{name}.jsonl")
        marker = [i for i, m in enumerate(tr)
                  if m["role"] == "user" and m["content"].startswith(
                      f"(woke at 2026-03-02T06:00:00+00:00 for trigger {checkpoint.RESUME_TRIGGER}/at")]
        assert marker, f"no resume wake marker in {name}"
        # ...right after the pre-cut wait call, with no tool result between
        assert tr[marker[0] - 1]["role"] == "assistant"
        assert json.loads(tr[marker[0] - 1]["content"])["tool"] == "sleep"
    # the edited memory reached the agents' prompts after the cut
    assert any("INJECTED LESSON" in p for p in seen["prompts"])
    # the world continued: claims from both stages are in the final scorer view
    n_notify2 = len([e for e in tail if e["type"] == "notify"])
    assert n_notify2 >= 1
    assert len(res["task"]["alerts"]) == n_notify1 + n_notify2
    assert res["resources"]["counts"]["notify"] == n_notify1 + n_notify2
    if pending:  # a claim pending at the cut settled afterwards
        settled = {a["news_id"]: a["status"] for a in res["task"]["alerts"]}
        assert all(settled[a["news_id"]] != "pending" for a in pending)
    # the curator curated again after the cut
    obs2 = read_jsonl(ws2 / "logs" / "observability.jsonl")
    assert [e for e in obs2 if e["event"] == "curation" and e["sim_time"] > cut]


# -- (3) the ext: TM-B memory candidate ---------------------------------------------


def _tmb_fake():
    async def fake_upstream(path, body):
        sys_msg = body["messages"][0]["content"]
        if sys_msg.startswith("You curate compact operating memory"):
            return _resp(json.dumps({"lessons": ["Prefer novel, concrete developments."],
                                     "diagnosis": "fine"}), body["model"])
        last = body["messages"][-1]["content"]
        if '"bytes"' in last:  # write_file result -> run what we wrote
            return _resp(json.dumps(
                {"tool": "run_program", "args": {"path": "watch.py"},
                 "thought": "watch in my program"}), body["model"])
        mid = re.search(r"market_id: (\d+)", sys_msg).group(1)
        now = _now_of(body)
        until = (now + timedelta(hours=9)).strftime("%Y-%m-%dT%H:%M:%SZ")
        program = (
            "import envkit\n"
            "page = envkit.search_news(q='Fed OR strikes OR inflation', "
            f"date_from='{(now - timedelta(days=2)).strftime('%Y-%m-%dT%H:%M:%SZ')}')\n"
            "hits = page.get('results') or []\n"
            "if hits:\n"
            "    try:\n"
            f"        envkit.notify(market_id='{mid}', news_id=hits[0]['news_id'], direction='up')\n"
            "    except Exception:\n"
            "        pass\n"
            f"envkit.wait('{until}')\n")
        return _resp(json.dumps(
            {"tool": "write_file", "args": {"path": "watch.py", "content": program},
             "thought": "search and claim from code"}), body["model"])
    return fake_upstream


@bnpm_world
def test_tmb_candidate_pause_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_proxy, "_upstream", _tmb_fake())
    cand = EXT / "tmB-reflection-only"
    cfg = _bnpm_cfg("bnpm-online-tmb", "B", "ext:tmb-reflection-only")
    s1, s2 = tmp_path / "s1", tmp_path / "s2"
    cut = "2026-03-02T00:00:00Z"
    assert asyncio.run(run_experiment(
        cfg, repo_root=tmp_path, run_dir=s1, scaffold_src=cand, pause_at=cut)) is None
    ws1 = s1 / "workspace"
    assert not list((ws1 / "logs").glob("crash-*.log"))
    for jail in ("m-616902", "m-678777"):
        assert (ws1 / "agents" / jail / "watch.py").exists()
    assert (ws1 / "agents" / "curator" / "park.py").exists()
    rows1 = read_jsonl(s1 / "ledger.jsonl")
    assert [e["outcome"] for e in rows1 if e["type"] == "agent_exit"] == ["exit=0"]
    for name in ("m-616902", "m-678777"):
        tr = read_jsonl(ws1 / "logs" / f"transcript_{name}.jsonl")
        assert json.loads(tr[-1]["content"])["tool"] == "run_program"

    snap = _snapshot(s1, tmp_path / "snapshot")
    res = asyncio.run(run_experiment(
        cfg, repo_root=tmp_path, run_dir=s2, scaffold_src=snap, resume_from=s1))
    assert res is not None and "failed-degenerate" not in res["flags"]
    ws2 = s2 / "workspace"
    assert not list((ws2 / "logs").glob("crash-*.log"))
    rows2 = read_jsonl(s2 / "ledger.jsonl")
    tail = rows2[len(rows1):]
    trig = [e for e in tail if e["type"] == "trigger"][0]
    assert (trig["id"], trig["sim_time"]) == (checkpoint.RESUME_TRIGGER, cut)
    # the jails came along and the agents ran their programs again
    for jail in ("m-616902", "m-678777"):
        assert (ws2 / "agents" / jail / "watch.py").exists()
    sleeps_after = {e["waiter"] for e in tail if e["type"] == "sleep"}
    assert sleeps_after >= {"m-616902", "m-678777", "coordinator", "curator"}
    assert not [e for e in rows2 if e["type"] == "run_program"]
    obs2 = read_jsonl(ws2 / "logs" / "observability.jsonl")
    assert [e for e in obs2 if e["event"] == "search" and e["sim_time"] >= cut]
    assert len(res["task"]["alerts"]) == res["resources"]["counts"]["notify"] >= 1


# -- (4) a TM-D cron program: no resume one-shot -------------------------------------


def _tmd_fake():
    async def fake_upstream(path, body):
        last = body["messages"][-1]["content"]
        if '"results"' in last or '"error"' in last:
            return _resp(json.dumps({"tool": "done", "args": {}, "thought": "ok"}),
                         body["model"])
        now = _now_of(body)
        return _resp(json.dumps({"tool": "search_news", "args": {
            "q": "Fed OR strikes OR inflation",
            "date_from": (now - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")},
            "thought": "look"}), body["model"])
    return fake_upstream


@bnpm_world
@pytest.mark.slow
def test_tmd_cron_candidate_resumes_from_crontab(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_proxy, "_upstream", _tmd_fake())
    cand = tmp_path / "tmD-base"
    cand.mkdir()
    shutil.copyfile(TASK_DIR / "agent" / "scaffolds" / "per_market_cron_main.py",
                    cand / "main.py")
    cfg = _bnpm_cfg("bnpm-online-tmd", "D", "ext:tmd-base", sim_end="2026-03-03T00:00:00Z")
    s1, s2 = tmp_path / "s1", tmp_path / "s2"
    cut = "2026-03-02T00:00:00Z"
    assert asyncio.run(run_experiment(
        cfg, repo_root=tmp_path, run_dir=s1, scaffold_src=cand, pause_at=cut)) is None
    rows1 = read_jsonl(s1 / "ledger.jsonl")
    fired1 = [(e["id"], e["sim_time"]) for e in rows1 if e["type"] == "trigger"]
    # the firing due exactly at the cut belongs to the next stage
    assert all(t < cut for _, t in fired1) and len(fired1) > 2
    snap = _snapshot(s1, tmp_path / "snapshot")
    res = asyncio.run(run_experiment(
        cfg, repo_root=tmp_path, run_dir=s2, scaffold_src=snap, resume_from=s1))
    assert res is not None and "failed-degenerate" not in res["flags"]
    rows2 = read_jsonl(s2 / "ledger.jsonl")
    tail = rows2[len(rows1):]
    assert tail[0]["type"] == "resume" and tail[0]["resume_trigger"] is False
    fired2 = [(e["id"], e["kind"], e["sim_time"]) for e in tail if e["type"] == "trigger"]
    # ACT_CRON is "5 */6 * * *": the restored crontab's next occurrence
    assert fired2 and fired2[0][1:] == ("cron", "2026-03-02T00:05:00Z")
    assert not [f for f in fired2 if f[0] == checkpoint.RESUME_TRIGGER]
    assert res["resources"]["counts"]["trigger"] == len(fired1) + len(fired2)
