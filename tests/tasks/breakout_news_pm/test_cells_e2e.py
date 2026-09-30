"""Composed-cell smokes on the REAL built world with a mocked LLM
upstream: the scan cell runs end-to-end against
the tantivy index (searches, decides, notifies, reports); the TM-C
learning cell fires its learn dispatch; a TM-B actor arms a real
news_match condition, gets a poll-tick wake, and pays polling-equivalent
billing. Skipped where the gitignored built data / index are absent."""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timedelta
from pathlib import Path

import pytest

import harness.llm_proxy as llm_proxy
from harness.run import run_experiment

TASK_DIR = Path(__file__).resolve().parents[3] / "tasks" / "breakout_news_pm"
BUILT = TASK_DIR / "data" / "built"

pytestmark = pytest.mark.skipif(
    not (BUILT / "breakpoints.jsonl").exists()
    or not (TASK_DIR / "news" / "tantivy_index_v3").exists(),
    reason="built world / tantivy index not present (gitignored)")


def _resp(text: str, model: str) -> tuple[dict, float]:
    return ({"choices": [{"message": {"content": text}}],
             "usage": {"prompt_tokens": 20, "completion_tokens": 10},
             "model": model}, 0.0)


def read_jsonl(path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def make_cfg(**overrides):
    from harness.config import RunConfig

    base = dict(
        run_id="bnpm-e2e",
        task={
            "name": "breakout_news_pm",
            "markets": [
                {"market_id": "616902", "start": "2026-03-01T00:00:00Z",
                 "end": "2026-03-04T00:00:00Z"},
                {"market_id": "678777", "start": "2026-03-01T00:00:00Z",
                 "end": "2026-03-04T00:00:00Z"},
            ],
        },
        sim_start="2026-03-01T00:00:00Z",
        sim_end="2026-03-04T12:00:00Z",
        budget_usd=5.0,
        agent={"scaffold": "tmc", "model": "mock-luna"},
        watchdog_seconds=60.0,
    )
    base.update(overrides)
    return RunConfig(**base)


@pytest.mark.slow
def test_tmc_scan_cell_end_to_end(tmp_path, monkeypatch):
    """tmC-tlrnnone-signone-algnone: hourly scans against the real index,
    one notify with a direction, scored report written."""
    alerted = {"done": False}

    async def fake_upstream(path, body):
        prompt = body["messages"][-1]["content"]
        if '"alerts"' in prompt:  # decision call
            ids = re.findall(r"news_id=(\S+) ", prompt)
            if ids and not alerted["done"]:
                alerted["done"] = True
                return _resp(json.dumps({"alerts": [
                    {"news_id": ids[0], "direction": "up"}]}), body["model"])
            return _resp(json.dumps({"alerts": []}), body["model"])
        return _resp("{}", body["model"])

    monkeypatch.setattr(llm_proxy, "_upstream", fake_upstream)
    cfg = make_cfg(run_id="bnpm-tmc-scan")
    results = asyncio.run(run_experiment(
        cfg, repo_root=tmp_path, run_dir=tmp_path / "run"))

    assert "failed-degenerate" not in results["flags"]
    ws = tmp_path / "run" / "workspace"
    events = read_jsonl(ws / "logs" / "trace.jsonl")
    assert any(e["kind"] == "decide" for e in events)
    acts = [e for e in events if e["kind"] == "action"]
    assert acts and acts[0]["direction"] == "up"
    # scored: the report carries both metric families
    rep = results["task"]
    assert "price_centric" in rep and "news_centric" in rep
    assert rep["price_centric"]["all"]["breakpoints"] >= 1
    # paid for searches (2 markets x hourly) at the real rate
    assert results["resources"]["spend_by_type"].get("news_search", 0) > 0


@pytest.mark.slow
def test_tmc_learning_cell_learn_dispatch(tmp_path, monkeypatch):
    """tmC-tlrndaily-sigoracle-algskills: the learn cron pulls oracle
    outcomes into records and reflection writes the skill block."""
    async def fake_upstream(path, body):
        prompt = body["messages"][-1]["content"]
        if "WRITE THE NEW SKILL BLOCK" in prompt:
            return _resp("Rule: distilled from oracle outcomes.",
                         body["model"])
        if '"alerts"' in prompt:
            return _resp(json.dumps({"alerts": []}), body["model"])
        return _resp("{}", body["model"])

    monkeypatch.setattr(llm_proxy, "_upstream", fake_upstream)
    cfg = make_cfg(run_id="bnpm-tmc-learn",
                   cell={"tm": "C", "tlrn": "daily", "sig": "oracle",
                         "alg": "skills"})
    results = asyncio.run(run_experiment(
        cfg, repo_root=tmp_path, run_dir=tmp_path / "run"))

    assert "failed-degenerate" not in results["flags"]
    ws = tmp_path / "run" / "workspace"
    records = read_jsonl(ws / "memory" / "records.jsonl")
    assert records and all(r["src"] == "oracle" for r in records)
    # the oracle is the idealized signal: settled breakpoints reveal
    # winnability (test_scoring pins that feedback() hides it)
    bps = [r for r in records if r["outcome"].get("kind") == "breakpoint"]
    assert bps and all("winnable" in r["outcome"] for r in bps)
    assert (ws / "memory" / "skills.md").read_text().startswith("Rule:")
    reflects = [e for e in read_jsonl(ws / "logs" / "trace.jsonl")
                if e["kind"] == "reflect"]
    assert len(reflects) >= 2  # daily firings across the 3.5-day window


def test_react_learning_reflection_inputs(tmp_path, monkeypatch):
    """tmA-tlrndaily-sigoracle-algskills: the actor searches (titles get
    registered), alerts (the notify wrapper registers the claim), and the
    RENDERED reflection prompt carries the redesign contract — resolved
    market questions, the own alert with its content, and the cost
    report. Asserting on the rendered prompt, not just on files, is the
    lesson of the own-actions-always-empty bug that shipped silently."""
    reflect_prompts: list[str] = []
    mock = {"notified": False}

    async def fake_upstream(path, body):
        last = body["messages"][-1]["content"]
        if "WRITE THE NEW SKILL BLOCK" in last:
            reflect_prompts.append(last)
            return _resp("Rule: distilled.", body["model"])
        if not mock["notified"]:
            ids = re.findall(r'"news_id": ?"([0-9a-f]{12,})"', last)
            if ids:
                mock["notified"] = True
                return _resp(json.dumps(
                    {"tool": "notify",
                     "args": {"market_id": "616902", "news_id": ids[0],
                              "direction": "up"}}), body["model"])
            if '"results"' not in last:  # nothing fresh in front of us
                return _resp(json.dumps(
                    {"tool": "search_news", "args": {"q": "fed"}}),
                    body["model"])
        text = " ".join(m["content"] for m in body["messages"])
        times = re.findall(r'(?:woke at |"now": ?")([0-9TZ:\-\.\+]+)', text)
        now = datetime.fromisoformat(times[-1].replace("Z", "+00:00"))
        until = (now + timedelta(hours=12)).strftime("%Y-%m-%dT%H:%M:%SZ")
        return _resp(json.dumps(
            {"tool": "sleep", "args": {"until": until}}), body["model"])

    monkeypatch.setattr(llm_proxy, "_upstream", fake_upstream)
    cfg = make_cfg(run_id="bnpm-react-learn",
                   cell={"tm": "A", "tlrn": "daily", "sig": "oracle",
                         "alg": "skills"},
                   agent={"scaffold": "react", "model": "mock-luna"})
    results = asyncio.run(run_experiment(
        cfg, repo_root=tmp_path, run_dir=tmp_path / "run"))

    assert "failed-degenerate" not in results["flags"]
    assert reflect_prompts  # daily learn firings reached reflection
    # every firing carries the cost report and the resolved market table
    for p in reflect_prompts:
        assert "spend since last reflection" in p
        assert "cumulative spend: $" in p
        assert "- 616902:" in p
    # the agent's own alert reached the own-actions section with content:
    # linked claim (notify wrapper) and resolved title (registry)
    linked = [p for p in reflect_prompts if '"did":"alerted"' in p]
    assert linked, "own alert never rendered in a reflection prompt"
    assert any('"news_title":' in p or '"title":"' in p for p in linked)
    # and the substrate shows the linkage, not just the render
    ws = tmp_path / "run" / "workspace"
    recs = read_jsonl(ws / "memory" / "records.jsonl")
    own = [r for r in recs if r["outcome"].get("kind") == "alert"]
    assert own and any(r.get("action") for r in own)
    assert (ws / "memory" / "skills.md").read_text().startswith("Rule:")


def test_tmb_actor_waits_by_program_only(tmp_path, monkeypatch):
    """tmB actor: no sleep tool in the manifest — the authored-program
    tools are the whole timing mechanism. A run_program wait on the
    seeded sleep.py runs the window to completion, and the rendered
    INSTRUCTION.md carries the skill appendix."""
    async def fake_upstream(path, body):
        last = body["messages"][-1]["content"]
        if '"bytes"' in last:  # write_file result -> run what we wrote
            return _resp(json.dumps(
                {"tool": "run_program", "args": {"path": "wait.py"},
                 "thought": "park in my program"}), body["model"])
        text = " ".join(m["content"] for m in body["messages"])
        times = re.findall(r'(?:woke at |"now": ?")([0-9TZ:\-\.+]+)', text)
        now = datetime.fromisoformat(times[-1].replace("Z", "+00:00"))
        until = (now + timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
        return _resp(json.dumps(
            {"tool": "write_file",
             "args": {"path": "wait.py",
                      "content": f'import envkit\nenvkit.wait("{until}")\n'},
             "thought": "wake time lives in the code"}), body["model"])

    monkeypatch.setattr(llm_proxy, "_upstream", fake_upstream)
    cfg = make_cfg(run_id="bnpm-tmb-wake",
                   sim_end="2026-03-02T00:00:00Z",
                   task={
                       "name": "breakout_news_pm",
                       "markets": [
                           {"market_id": "616902",
                            "start": "2026-03-01T00:00:00Z",
                            "end": "2026-03-02T00:00:00Z"}]},
                   cell={"tm": "B", "tlrn": "none", "sig": "none",
                         "alg": "none"},
                   agent={"scaffold": "react", "model": "mock-luna"})
    results = asyncio.run(run_experiment(
        cfg, repo_root=tmp_path, run_dir=tmp_path / "run"))

    assert "failed-degenerate" not in results["flags"]
    manifest = json.loads(
        (tmp_path / "run" / "workspace_manifest.json").read_text())
    names = {t["name"] for t in manifest["tools"]}
    assert "sleep" not in names and "wait_until" not in names
    assert {"run_program", "write_file", "read_file", "ls"} <= names
    ws = tmp_path / "run" / "workspace"
    # the jail was provisioned with the blind-wait exemplar, and the
    # actor's own program carried the wake time in code
    assert (ws / "agents" / "actor" / "sleep.py").is_file()
    assert (ws / "agents" / "actor" / "wait.py").is_file()
    ledger = read_jsonl(tmp_path / "run" / "ledger.jsonl")
    runs = [e for e in ledger if e["type"] == "run_program"]
    assert runs and all(e["path"] == "wait.py" for e in runs)
    # the rendered instruction carries the authored-program skill
    instruction = (ws / "INSTRUCTION.md").read_text()
    assert "# Waiting by program" in instruction


def test_per_market_party_cell_end_to_end(tmp_path, monkeypatch):
    """task:per_market (impl plan wait_party_per_entity_v1): one agent
    per market + a coordinator, all sleeping concurrently through the env
    wait party; the learn cron is routed to the coordinator; separate
    transcripts prove separate histories."""
    async def fake_upstream(path, body):
        last = body["messages"][-1]["content"]
        if '"bytes"' in last:  # write_file result -> run what we wrote
            return _resp(json.dumps(
                {"tool": "run_program", "args": {"path": "wait.py"},
                 "thought": "park in my program"}), body["model"])
        text = " ".join(m["content"] for m in body["messages"])
        times = re.findall(r'(?:woke at |"now": ?")([0-9TZ:\-\.\+]+)', text)
        now = datetime.fromisoformat(times[-1].replace("Z", "+00:00"))
        until = (now + timedelta(hours=36)).strftime("%Y-%m-%dT%H:%M:%SZ")
        return _resp(json.dumps(
            {"tool": "write_file",
             "args": {"path": "wait.py",
                      "content": f'import envkit\nenvkit.wait("{until}")\n'},
             "thought": "monitor my market"}), body["model"])

    monkeypatch.setattr(llm_proxy, "_upstream", fake_upstream)
    cfg = make_cfg(run_id="bnpm-per-market",
                   sim_end="2026-03-02T00:00:00Z",
                   task={
                       "name": "breakout_news_pm",
                       "markets": [
                           {"market_id": "616902",
                            "start": "2026-03-01T00:00:00Z",
                            "end": "2026-03-02T00:00:00Z"},
                           {"market_id": "678777",
                            "start": "2026-03-01T00:00:00Z",
                            "end": "2026-03-02T00:00:00Z"}]},
                   cell={"tm": "B", "tlrn": "daily", "sig": "oracle",
                         "alg": "memory"},
                   agent={"scaffold": "task:per_market",
                          "model": "mock-luna"})
    results = asyncio.run(run_experiment(
        cfg, repo_root=tmp_path, run_dir=tmp_path / "run"))

    assert "failed-degenerate" not in results["flags"]
    ws = tmp_path / "run" / "workspace"
    # separate conversation histories, one per market
    t1 = read_jsonl(ws / "logs" / "transcript_m-616902.jsonl")
    t2 = read_jsonl(ws / "logs" / "transcript_m-678777.jsonl")
    assert t1 and t2
    ledger = read_jsonl(tmp_path / "run" / "ledger.jsonl")
    # the learn cron fired and was routed through the party's sleep
    learns = [e for e in ledger
              if e["type"] == "trigger" and e.get("id") == "learn"]
    assert learns and all(e["via"] == "sleep" for e in learns)
    assert [e for e in ledger if e["type"] == "set_party"]


def test_per_market_algnone_cell_runs(tmp_path, monkeypatch):
    """task:per_market in the no-learning cell (tmA-tlrnnone-signone-
    algnone): records.py is NOT mounted, so the program must not import
    it; agents sleep, the coordinator parks, the run completes."""
    async def fake_upstream(path, body):
        text = " ".join(m["content"] for m in body["messages"])
        times = re.findall(r'(?:woke at |"now": ?")([0-9TZ:\-\.\+]+)', text)
        now = datetime.fromisoformat(times[-1].replace("Z", "+00:00"))
        until = (now + timedelta(hours=36)).strftime("%Y-%m-%dT%H:%M:%SZ")
        return _resp(json.dumps(
            {"tool": "sleep", "args": {"until": until}}), body["model"])

    monkeypatch.setattr(llm_proxy, "_upstream", fake_upstream)
    cfg = make_cfg(run_id="bnpm-per-market-algnone",
                   sim_end="2026-03-02T00:00:00Z",
                   task={
                       "name": "breakout_news_pm",
                       "markets": [
                           {"market_id": "616902",
                            "start": "2026-03-01T00:00:00Z",
                            "end": "2026-03-02T00:00:00Z"},
                           {"market_id": "678777",
                            "start": "2026-03-01T00:00:00Z",
                            "end": "2026-03-02T00:00:00Z"}]},
                   cell={"tm": "A", "tlrn": "none", "sig": "none",
                         "alg": "none"},
                   agent={"scaffold": "task:per_market",
                          "model": "mock-luna"})
    results = asyncio.run(run_experiment(
        cfg, repo_root=tmp_path, run_dir=tmp_path / "run"))

    assert "failed-degenerate" not in results["flags"]
    ws = tmp_path / "run" / "workspace"
    assert not (ws / "records.py").exists()
    assert read_jsonl(ws / "logs" / "transcript_m-616902.jsonl")
    assert read_jsonl(ws / "logs" / "transcript_m-678777.jsonl")
    ledger = read_jsonl(tmp_path / "run" / "ledger.jsonl")
    assert [e for e in ledger if e["type"] == "set_party"]
    assert not [e for e in ledger if e["type"] == "crontab_put"]


@pytest.mark.slow
def test_per_market_cron_cell_end_to_end(tmp_path, monkeypatch):
    """task:per_market_cron: TM-C timing, TmA acting. The acting cron
    fires each market's agent, which searches and notifies by its own
    choice inside the firing; transcripts persist across firings (one
    wake marker per firing); the learn cron pulls oracle outcomes and
    reflection writes the skill block."""
    def market_of(body) -> str | None:
        m = re.search(r"market_id: (\d+)", body["messages"][0]["content"])
        return m.group(1) if m else None

    notified: set[str] = set()

    async def fake_upstream(path, body):
        prompt = body["messages"][-1]["content"]
        if "WRITE THE NEW SKILL BLOCK" in prompt:
            return _resp("Rule: distilled from oracle outcomes.",
                         body["model"])
        mid = market_of(body)
        if '"results"' in prompt:  # a search result page came back
            ids = re.findall(r'"news_id": ?"([0-9a-f]{12,})"', prompt)
            if ids and mid and mid not in notified:
                notified.add(mid)
                return _resp(json.dumps(
                    {"tool": "notify",
                     "args": {"market_id": mid, "news_id": ids[0],
                              "direction": "up"}}), body["model"])
            return _resp(json.dumps({"tool": "done", "args": {}}),
                         body["model"])
        if prompt.startswith("(woke at"):  # fresh firing: go look
            return _resp(json.dumps(
                {"tool": "search_news", "args": {"q": "fed"}}),
                body["model"])
        return _resp(json.dumps({"tool": "done", "args": {}}),
                     body["model"])

    monkeypatch.setattr(llm_proxy, "_upstream", fake_upstream)
    cfg = make_cfg(run_id="bnpm-per-market-cron",
                   sim_end="2026-03-02T02:00:00Z",
                   task={
                       "name": "breakout_news_pm",
                       "markets": [
                           {"market_id": "616902",
                            "start": "2026-03-01T00:00:00Z",
                            "end": "2026-03-02T02:00:00Z"},
                           {"market_id": "678777",
                            "start": "2026-03-01T00:00:00Z",
                            "end": "2026-03-02T02:00:00Z"}]},
                   cell={"tm": "C", "tlrn": "daily", "sig": "oracle",
                         "alg": "skills"},
                   agent={"scaffold": "task:per_market_cron",
                          "model": "mock-luna"})
    results = asyncio.run(run_experiment(
        cfg, repo_root=tmp_path, run_dir=tmp_path / "run"))

    assert "failed-degenerate" not in results["flags"]
    ws = tmp_path / "run" / "workspace"
    # separate persistent conversation histories, one per market, with
    # one wake marker per cron firing routed to the right agent
    t1 = read_jsonl(ws / "logs" / "transcript_m-616902.jsonl")
    t2 = read_jsonl(ws / "logs" / "transcript_m-678777.jsonl")
    assert t1 and t2
    text1 = " ".join(m["content"] for m in t1)
    text2 = " ".join(m["content"] for m in t2)
    assert text1.count("(woke at") > 1  # persisted across firings
    assert "trigger m-616902/cron" in text1
    assert "trigger m-616902/cron" not in text2
    assert "trigger m-678777/cron" in text2
    # the program owns the schedule: per-market hourly entries + learn
    ledger = read_jsonl(tmp_path / "run" / "ledger.jsonl")
    assert [e for e in ledger if e["type"] == "crontab_put"]
    # the agents alerted by their own choice; the notify wrapper's claim
    # was popped onto the settled record by records.action_for
    recs = read_jsonl(ws / "memory" / "records.jsonl")
    own = [r for r in recs if r["outcome"].get("kind") == "alert"]
    assert {r["outcome"]["market_id"] for r in own} == {"616902", "678777"}
    assert any(r.get("action", {}) and r["action"]["did"] == "alerted"
               for r in own)
    # the learn cron fired: reflection wrote the skill block
    assert (ws / "memory" / "skills.md").read_text().startswith("Rule:")
    assert any(e["kind"] == "reflect"
               for e in read_jsonl(ws / "logs" / "trace.jsonl"))


def test_per_market_cron_tmd_schedules_are_pinned_to_their_market(
        tmp_path, monkeypatch):
    """task:per_market_cron under tm=D: each market agent's schedule CRUD is
    pinned to its own market — the first agent's one-time schedule fires that agent only (note in
    its wake marker), and the second agent's cross-delete is refused by
    the wrapper before it reaches the env."""
    def market_of(body) -> str | None:
        m = re.search(r"market_id: (\d+)", body["messages"][0]["content"])
        return m.group(1) if m else None

    async def fake_upstream(path, body):
        msgs = body["messages"]
        mid = market_of(body)
        woke = sum(1 for m in msgs[1:] if m["role"] == "user"
                   and m["content"].startswith("(woke at "))
        acted = any("_schedule" in m["content"] for m in msgs[1:]
                    if m["role"] == "assistant")
        if woke == 1 and not acted and mid == "616902":
            reply = {"tool": "create_schedule",
                     "args": {"id": "look", "at": "2026-03-01T03:00:00Z",
                              "note": "look at the fed wires"}}
        elif woke == 1 and not acted and mid == "678777":
            reply = {"tool": "delete_schedule", "args": {"id": "look"}}
        else:
            reply = {"tool": "done", "args": {}}
        return _resp(json.dumps(reply), body["model"])

    monkeypatch.setattr(llm_proxy, "_upstream", fake_upstream)
    cfg = make_cfg(run_id="bnpm-per-market-cron-tmd",
                   sim_end="2026-03-02T02:00:00Z",
                   task={
                       "name": "breakout_news_pm",
                       "markets": [
                           {"market_id": "616902",
                            "start": "2026-03-01T00:00:00Z",
                            "end": "2026-03-02T02:00:00Z"},
                           {"market_id": "678777",
                            "start": "2026-03-01T00:00:00Z",
                            "end": "2026-03-02T02:00:00Z"}]},
                   cell={"tm": "D", "tlrn": "none", "sig": "none",
                         "alg": "none"},
                   agent={"scaffold": "task:per_market_cron",
                          "model": "mock-luna"})
    results = asyncio.run(run_experiment(
        cfg, repo_root=tmp_path, run_dir=tmp_path / "run"))

    assert "failed-degenerate" not in results["flags"]
    ws = tmp_path / "run" / "workspace"
    text1 = " ".join(m["content"] for m in
                     read_jsonl(ws / "logs" / "transcript_m-616902.jsonl"))
    text2 = " ".join(m["content"] for m in
                     read_jsonl(ws / "logs" / "transcript_m-678777.jsonl"))
    assert "trigger look/at" in text1
    assert "Your note for this schedule: look at the fed wires" in text1
    assert "look/at" not in text2
    assert "belongs to another agent" in text2
    # the base per-market cron still fires both agents
    assert "trigger m-616902/cron" in text1
    assert "trigger m-678777/cron" in text2
    # the instruction names the schedule API and the read-only base
    assert "list_schedules / create_schedule" in \
        (ws / "instructions" / "m-616902.md").read_text()
    ledger = read_jsonl(tmp_path / "run" / "ledger.jsonl")
    created = [e for e in ledger if e["type"] == "schedule_create"]
    assert [e["target"] for e in created] == ["m-616902"]
    assert [e["type"] for e in ledger if e["type"] == "schedule_delete"] == []
    seeded = {e["id"]: e["target"] for e in ledger
              if e["type"] == "schedule_seed"}  # one default per market
    assert seeded == {"m-616902": "m-616902", "m-678777": "m-678777"}
    fired = [(e["id"], e.get("target")) for e in ledger
             if e["type"] == "trigger" and e.get("owner") == "agent"
             and e["id"] not in seeded]
    assert fired == [("look", "m-616902")]
