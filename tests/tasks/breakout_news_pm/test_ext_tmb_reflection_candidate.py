"""The ext: TM-B memory candidate `tmB-reflection-only`: tmB-base plus
reflective memory. Tool calls made from inside authored programs reach the
reflective state; the curator parks in its own jail and curates at the
first sim midnight."""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from harness import llm_proxy
from harness.run import run_experiment
from tests.tasks.breakout_news_pm.test_cells_e2e import _resp, make_cfg, read_jsonl

REPO_ROOT = Path(__file__).resolve().parents[3]
TASK_DIR = REPO_ROOT / "tasks" / "breakout_news_pm"
CAND = TASK_DIR / "agent" / "ext_candidates" / "tmB-reflection-only"


def test_full_episode_tmb_bases_load():
    from harness.config import load_config

    d = TASK_DIR / "configs" / "ext_bases"
    for ep in ("w10", "w13", "w17"):
        a = load_config(d / f"{ep}full-tmA-ext.yaml")
        b = load_config(d / f"{ep}full-tmB-ext.yaml")
        assert b.agent.scaffold == "ext:candidate" and b.cell.tm == "B"
        assert (b.sim_start, b.sim_end, b.budget_usd, b.domain_budgets, b.task) == \
            (a.sim_start, a.sim_end, a.budget_usd, a.domain_budgets, a.task)


@pytest.mark.skipif(
    not (TASK_DIR / "data" / "built" / "markets.jsonl").exists(),
    reason="built breakout_news_pm world absent (DATA.md)")
def test_ext_tmb_reflection_end_to_end(tmp_path, monkeypatch):
    LESSON = "Prefer novel, concrete developments."

    async def fake_upstream(path, body):
        sys_msg = body["messages"][0]["content"]
        if sys_msg.startswith("You curate compact operating memory"):
            return _resp(json.dumps({"lessons": [LESSON], "diagnosis": "fine"}),
                         body["model"])
        last = body["messages"][-1]["content"]
        if '"bytes"' in last:  # write_file result -> run what we wrote
            return _resp(json.dumps(
                {"tool": "run_program", "args": {"path": "watch.py"},
                 "thought": "watch in my program"}), body["model"])
        text = " ".join(m["content"] for m in body["messages"])
        mid = re.search(r"market_id: (\d+)", sys_msg).group(1)
        times = re.findall(r'(?:woke at |"now": ?")([0-9TZ:\-\.\+]+)', text)
        now = datetime.fromisoformat(times[-1].replace("Z", "+00:00"))
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

    monkeypatch.setattr(llm_proxy, "_upstream", fake_upstream)
    cfg = make_cfg(run_id="bnpm-ext-tmb-reflection",
                   sim_end="2026-03-03T00:00:00Z",
                   task={"name": "breakout_news_pm",
                         "markets": [
                             {"market_id": "616902", "start": "2026-03-01T00:00:00Z",
                              "end": "2026-03-03T00:00:00Z"},
                             {"market_id": "678777", "start": "2026-03-01T00:00:00Z",
                              "end": "2026-03-03T00:00:00Z"}]},
                   cell={"tm": "B", "tlrn": "none", "sig": "none", "alg": "none"},
                   agent={"scaffold": "ext:tmb-reflection-only", "model": "mock-luna"})
    results = asyncio.run(run_experiment(
        cfg, repo_root=tmp_path, run_dir=tmp_path / "run", scaffold_src=CAND))

    assert "failed-degenerate" not in results["flags"]
    ws = tmp_path / "run" / "workspace"
    assert not list((ws / "logs").glob("crash-*.log"))
    # the curator curated at the first midnight and its lessons persist
    obs = read_jsonl(ws / "logs" / "observability.jsonl")
    curations = [e for e in obs if e["event"] == "curation"]
    assert curations and curations[0]["sim_time"].startswith("2026-03-02T00:00")
    assert not {"curator_exception", "curator_env_error", "curation_error"} & \
        {e["event"] for e in obs}
    state = json.loads((ws / "memory" / "reflective_state.json").read_text())
    assert state["lessons"] == [LESSON]
    # searches and notifications made from inside programs reached the state
    ledger = read_jsonl(tmp_path / "run" / "ledger.jsonl")
    searches = [e for e in obs if e["event"] == "search"]
    assert searches and len(searches) == len([e for e in ledger if e["type"] == "news_search"])
    n_notify = len([e for e in ledger if e["type"] == "notify"])
    assert n_notify >= 1 and len(state["actions"]) == n_notify
