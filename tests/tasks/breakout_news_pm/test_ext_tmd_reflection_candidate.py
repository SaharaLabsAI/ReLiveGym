"""The ext: TM-D memory candidate `tmD-reflection-only`: the per-market cron
main plus reflective memory, the curator as a hidden `learn` cron row."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from harness import llm_proxy
from harness.run import run_experiment
from tests.tasks.breakout_news_pm.test_cells_e2e import _resp, make_cfg, read_jsonl

REPO_ROOT = Path(__file__).resolve().parents[3]
BNPM = REPO_ROOT / "tasks" / "breakout_news_pm"
CAND = BNPM / "agent" / "ext_candidates" / "tmD-reflection-only"


def test_full_episode_tmd_bases_load():
    from harness.config import load_config

    cells = BNPM / "configs" / "cells"
    for ep in ("w10", "w13", "w17"):
        a = load_config(cells / ep / "tmD-tlrnnone-signone-algnone-permarketcron.yaml")
        b = load_config(cells.parent / "ext_bases" / f"{ep}full-tmD-ext.yaml")
        assert b.agent.scaffold == "ext:candidate" and b.cell == a.cell
        assert (b.sim_start, b.sim_end, b.budget_usd, b.domain_budgets, b.task) == \
            (a.sim_start, a.sim_end, a.budget_usd, a.domain_budgets, a.task)


@pytest.mark.skipif(
    not (BNPM / "data" / "built" / "markets.jsonl").exists(),
    reason="built breakout_news_pm world absent (DATA.md)")
def test_ext_tmd_reflection_end_to_end(tmp_path, monkeypatch):
    LESSON = "Prefer novel, concrete developments."
    prompts = []  # (wake time, full prompt text) per actor firing

    async def fake_upstream(path, body):
        if body["messages"][0]["content"].startswith("You curate compact operating memory"):
            return _resp(json.dumps({"lessons": [LESSON], "diagnosis": "fine"}),
                         body["model"])
        last = body["messages"][-1]["content"]
        if last.startswith("(woke at"):
            prompts.append((last[9:29], " ".join(m["content"] for m in body["messages"])))
            call = {"tool": "search_news", "args": {"q": "Fed OR strikes"}}
        else:
            call = {"tool": "done", "args": {}}
        return _resp(json.dumps({**call, "thought": "t"}), body["model"])

    monkeypatch.setattr(llm_proxy, "_upstream", fake_upstream)
    cfg = make_cfg(run_id="bnpm-ext-tmd-reflection",
                   sim_end="2026-03-02T12:00:00Z",
                   task={"name": "breakout_news_pm",
                         "markets": [
                             {"market_id": "616902", "start": "2026-03-01T00:00:00Z",
                              "end": "2026-03-02T12:00:00Z"}]},
                   cell={"tm": "D", "tlrn": "none", "sig": "none", "alg": "none"},
                   agent={"scaffold": "ext:tmd-reflection-only", "model": "mock-luna"})
    results = asyncio.run(run_experiment(
        cfg, repo_root=tmp_path, run_dir=tmp_path / "run", scaffold_src=CAND))

    assert "failed-degenerate" not in results["flags"]
    ws = tmp_path / "run" / "workspace"
    assert not list((ws / "logs").glob("crash-*.log"))
    obs = read_jsonl(ws / "logs" / "observability.jsonl")
    # the curator fires once, at the first sim midnight, and its lessons persist
    assert [e["sim_time"][:16] for e in obs if e["event"] == "curation"] == ["2026-03-02T00:00"]
    assert not {"curator_exception", "curation_error"} & {e["event"] for e in obs}
    state = json.loads((ws / "memory" / "reflective_state.json").read_text())
    assert state["lessons"] == [LESSON]
    # actor firings after the curation carry the lessons; earlier ones do not
    assert prompts
    for t, p in prompts:
        assert (LESSON in p) == (t >= "2026-03-02T00:05"), t
