"""End-to-end smoke: the silent fixed baseline over 5 real days.

Needs the built world + the bnpm tantivy index (NewsStore); no LLM —
the baseline is a no-op program, so nothing is mocked. Window
2026-03-01..03-06 with one winnable resolver (891191, resolves 03-02)
and one never-resolving load question (665472).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
BUILT = REPO / "tasks" / "resolution_detect" / "data" / "built"
INDEX = REPO / "tasks" / "breakout_news_pm" / "news" / "tantivy_index_v3"

pytestmark = pytest.mark.skipif(
    not (BUILT / "questions.jsonl").exists() or not INDEX.exists(),
    reason="built world or news index absent")


def test_silent_baseline_end_to_end(tmp_path):
    from harness.config import RunConfig
    from harness.run import run_experiment

    cfg = RunConfig(
        run_id="rd-e2e-silent",
        task={"name": "resolution_detect",
              "questions": ["891191", "665472"]},
        sim_start=datetime(2026, 3, 1, tzinfo=timezone.utc),
        sim_end=datetime(2026, 3, 6, tzinfo=timezone.utc),
        budget_usd=5.0,
        agent={"scaffold": "silent"},
        watchdog_seconds=60.0,
    )
    results = asyncio.run(run_experiment(
        cfg, repo_root=REPO, run_dir=tmp_path / "run"))

    assert "failed-degenerate" not in results["flags"]
    perf = results["performance"]
    assert perf["primary"]["name"] == "tc_f1"
    assert perf["primary"]["value"] == 0.0
    assert perf["n_winnable"] == 1 and perf["n_miss"] == 1
    assert perf["n_claims_settled"] == 0
    assert results["resources"]["spent_usd"] == 0.0
