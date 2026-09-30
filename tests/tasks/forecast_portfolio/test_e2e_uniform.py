"""End-to-end smoke: the uniform fixed baseline over 5 real days.

Needs the built world + the bnpm tantivy index (NewsStore); no LLM —
the baseline is a no-LLM cron program, so nothing is mocked.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
BUILT = REPO / "tasks" / "forecast_portfolio" / "data" / "built"
INDEX = REPO / "tasks" / "breakout_news_pm" / "news" / "tantivy_index_v3"

pytestmark = pytest.mark.skipif(
    not (BUILT / "questions.jsonl").exists() or not INDEX.exists(),
    reason="built world or news index absent")


@pytest.mark.slow
def test_uniform_baseline_end_to_end(tmp_path):
    from harness.config import RunConfig
    from harness.run import run_experiment

    cfg = RunConfig(
        run_id="fp-e2e-uniform",
        task={"name": "forecast_portfolio",
              # 1465968 opens 02-28, resolves 03-03 (scored);
              # 561253 never resolves (attention load)
              "questions": ["1465968", "561253"]},
        sim_start=datetime(2026, 3, 1, tzinfo=timezone.utc),
        sim_end=datetime(2026, 3, 6, tzinfo=timezone.utc),
        budget_usd=5.0,
        agent={"scaffold": "uniform"},
        watchdog_seconds=60.0,
    )
    results = asyncio.run(run_experiment(
        cfg, repo_root=REPO, run_dir=tmp_path / "run"))

    assert "failed-degenerate" not in results["flags"]
    perf = results["performance"]
    assert perf["questions_settled"] == 1
    # uniform from the first hourly sweep; the only loss is the sliver
    # before it (life starts at sim_start here, sweep fires in hour 0)
    assert perf["primary"]["name"] == "ta_bss"
    assert perf["primary"]["value"] == pytest.approx(0.5, abs=0.01)
    assert perf["abstention_share"] < 0.01
    # no news spend, no LLM spend
    assert results["resources"]["spent_usd"] == 0.0
    ledger = [json.loads(line) for line in
              (tmp_path / "run" / "ledger.jsonl").read_text().splitlines()]
    assert any(e["type"] == "submit_forecast" for e in ledger)
    assert any(e["type"] == "outcome" and e["ref"] == "1465968"
               for e in ledger)
