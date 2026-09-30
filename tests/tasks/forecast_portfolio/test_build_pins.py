"""Calibration pins over the real built world (skipped when absent).

Numbers pinned from data/README.md: any drift means the frozen
sample or the build changed — investigate, don't re-pin casually.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

BUILT = Path(__file__).resolve().parents[3] / "tasks" / "forecast_portfolio" \
    / "data" / "built"

pytestmark = pytest.mark.skipif(
    not (BUILT / "questions.jsonl").exists(),
    reason="built world absent (run tasks/forecast_portfolio/data/build.py)")


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    return [json.loads(line)
            for line in (BUILT / "questions.jsonl").open()]


def test_counts(rows):
    assert len(rows) == 300
    scored = [r for r in rows if r["scorer"]["scored"]]
    assert len(scored) == 184
    assert sum(1 for r in scored if r["scorer"]["easy"]) == 31


def test_all_binary_outcomes(rows):
    assert all(r["agent"]["outcomes"] == ["Yes", "No"] for r in rows)


def test_mid_window_opens(rows):
    n = sum(1 for r in rows if r["agent"]["open_date"] >= "2026-03-01")
    assert n == 145


def test_agent_scorer_split(rows):
    for r in rows:
        assert set(r["agent"]) == {"question", "description", "outcomes",
                                   "open_date", "scheduled_end"}
        assert r["scorer"]["scored"] in (True, False)


def test_every_scored_question_has_prices(rows):
    for r in rows:
        if not r["scorer"]["scored"]:
            continue
        path = BUILT / "prices" / f"{r['question_id']}.json"
        assert path.exists(), r["question_id"]
        assert json.loads(path.read_text())["points"], r["question_id"]


def test_market_anchor_pin(rows):
    tas = [r["scorer"]["ta_bss_market"] for r in rows if r["scorer"]["scored"]]
    assert sum(tas) / len(tas) == pytest.approx(0.7808, abs=5e-4)


def test_t_res_never_exceeds_scheduled_end(rows):
    from harness.timeutil import as_utc, parse_iso

    for r in rows:
        s = r["scorer"]
        if not s["scored"]:
            continue
        sched = as_utc(parse_iso(r["agent"]["scheduled_end"])).timestamp()
        assert s["t_res"] <= sched + 1e-6, r["question_id"]
