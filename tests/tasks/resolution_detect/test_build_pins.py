"""Calibration pins from gates 1–2 (post easy-exclusion) + anchor
replication: the scripted freeze-probe actors driven through the REAL
scorer must reproduce the frozen anchor numbers.

Needs the built world (data/built/questions.jsonl); the mark@0.99 anchor
additionally needs the shared bnpm price store (skipped without it).
"""

from __future__ import annotations

import importlib.util
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
BUILT = REPO / "tasks" / "resolution_detect" / "data" / "built"
PRICES = REPO / "tasks" / "breakout_news_pm" / "market" / "raw" / "prices"
PROBE = (REPO / "tasks" / "resolution_detect" / "analysis"
         / "settlement_freeze_probe.py")

pytestmark = pytest.mark.skipif(
    not (BUILT / "questions.jsonl").exists(), reason="built world absent")

W0 = datetime(2026, 3, 1, tzinfo=timezone.utc)
W1 = datetime(2026, 7, 1, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def rows():
    return [json.loads(line)
            for line in (BUILT / "questions.jsonl").open()]


@pytest.fixture(scope="module")
def world(rows):
    """All 609 questions loaded over the full window, plus a fresh scorer
    factory (module-scoped rows, per-test scorer)."""
    from tasks.resolution_detect.task import (
        ResolutionDetectConfig, Scorer, load_questions)

    tcfg = ResolutionDetectConfig(
        questions=[r["question_id"] for r in rows], data_dir=BUILT)

    def make_scorer():
        return Scorer(tcfg, load_questions(BUILT, tcfg, W0, W1), news=None)

    return make_scorer


# -- roster and settlement pins --------------------------------------------------------------------


def test_counts(rows):
    scored = [r for r in rows if r["scorer"]["scored"]]
    winnable = [r for r in scored if r["scorer"]["winnable"]]
    assert len(rows) == 609
    assert len(scored) == 123
    assert len(winnable) == 116


def test_winning_side_split(rows):
    scored = [r for r in rows if r["scorer"]["scored"]]
    first = sum(1 for r in scored
                if r["agent"]["outcomes"].index(
                    r["scorer"]["resolution_answer"]) == 0)
    assert first == 64
    assert 0.35 <= first / len(scored) <= 0.65


def test_gap_pins(rows):
    gaps = [r["scorer"]["gap_s"] / 3600 for r in rows
            if r["scorer"]["winnable"]]
    assert statistics.median(gaps) == pytest.approx(60.3, abs=0.5)
    assert sum(1 for r in rows
               if r["scorer"]["scored"] and not r["scorer"]["winnable"]) == 7
    assert sum(1 for r in rows if r["scorer"]["trap"]) == 2
    assert sum(1 for r in rows if r["scorer"]["news_lead_flag"]) == 4


def test_easy_exclusion_invariant(rows):
    """No winnable question is determinable within 1 h of its
    W0-activation (the sample-level exclusion)."""
    from tasks.resolution_detect.task import _ts  # noqa: F401
    from harness.timeutil import as_utc, parse_iso
    for r in rows:
        s = r["scorer"]
        if not s["winnable"]:
            continue
        open_ts = as_utc(parse_iso(r["agent"]["open_date"])).timestamp()
        act = max(open_ts, W0.timestamp())
        assert s["t_det"] - act > 3600, r["question_id"]


def test_agent_scorer_split(rows):
    for r in rows:
        assert set(r["agent"]) == {"question", "description", "outcomes",
                                   "open_date", "scheduled_end"}
        assert len(r["agent"]["outcomes"]) == 2
        s = r["scorer"]
        assert s["winnable"] == (s["t_det"] is not None)


# -- anchor replication through the real scorer ---------------------------------------


def _t(ts_val: float) -> datetime:
    return datetime.fromtimestamp(ts_val, tz=timezone.utc)


def test_perfect_anchor_exactly_one(world):
    scorer = world()
    for q in scorer.questions.values():
        if q.winnable:
            scorer.record_mark(_t(max(q.t_det, q.activation)), q.qid,
                               q.answer)
    scorer.close_due(W1)
    scorer.close_all()
    m = scorer.metrics()
    assert m["primary"]["value"] == pytest.approx(1.0, abs=1e-3)
    assert m["n_covered"] == m["n_winnable"] == 116


def test_silence_anchor_zero(world):
    scorer = world()
    scorer.close_due(W1)
    scorer.close_all()
    m = scorer.metrics()
    assert m["primary"]["value"] == 0.0
    assert m["n_miss"] == 116


def test_blanket_no_anchor_near_zero(world):
    # soft metric: the early No-claims earn exp-decayed slivers
    # (t_det is days-to-months past activation for nearly all of them);
    # the anchor stays annihilated (0.014 on the FDV-free roster)
    scorer = world()
    from harness.task import NotificationError
    for q in scorer.questions.values():
        if q.outcomes != ["Yes", "No"]:
            continue
        try:
            scorer.record_mark(_t(q.activation), q.qid, "No")
        except NotificationError:
            pass  # resolved-at-activation boundary rows
    scorer.close_due(W1)
    scorer.close_all()
    m = scorer.metrics()
    assert m["primary"]["value"] == pytest.approx(0.0139, abs=0.005)
    assert m["n_covered"] == 0


@pytest.mark.skipif(not PRICES.exists(), reason="bnpm price store absent")
def test_mark99_anchor_replicates_freeze_probe(world):
    spec = importlib.util.spec_from_file_location("freeze_probe", PROBE)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    claims = probe.mark_at_c(probe.load_roster(), 0.99)

    scorer = world()
    from harness.task import NotificationError
    for qid, (tau, outcome, _q) in claims.items():
        try:
            scorer.record_mark(_t(tau), qid, outcome)
        except NotificationError:
            pass
    scorer.close_due(W1)
    scorer.close_all()
    m = scorer.metrics()
    assert m["primary"]["value"] == pytest.approx(0.6157, abs=0.005)
    assert m["n_covered"] == 104
    assert m["n_early"] == 11  # cliff burns: episodes early, credit ~0
