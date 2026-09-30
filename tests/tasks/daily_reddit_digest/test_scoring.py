"""daily_reddit_digest calendar + scoring."""

from __future__ import annotations

from datetime import date

import pytest

from harness.task import NotificationError
from tasks.daily_reddit_digest.task import DailyRedditDigestTask
from tests.tasks.daily_reddit_digest.conftest import (
    DAY5, POP, UNPOP, make_digest_config, t, ts)


# -- calendar ---------------------------------------------------------------------------

def test_scored_days_start_at_first_fully_in_run_window(scorer):
    # Mar 2 12:00's window starts Mar 1 12:00 < sim_start -> not scored
    assert scorer.days == [date(2026, 3, d) for d in range(3, 8)]


def test_windows_are_pinned_to_the_days_hour(scorer):
    d = date(2026, 3, 3)
    assert scorer.window_for(d) == (ts(2, 12), ts(3, 12))
    assert scorer.delivery_for(d) == (ts(3, 12), ts(3, 13))
    assert scorer.settle_at(d) == ts(4, 12)


def test_day_for_uses_half_open_delivery_window(scorer):
    d = date(2026, 3, 3)
    assert scorer.day_for(ts(3, 12)) == d
    assert scorer.day_for(ts(3, 12, 59, 59)) == d
    assert scorer.day_for(ts(3, 13)) is None
    assert scorer.day_for(ts(3, 11, 59, 59)) is None
    assert scorer.day_for(ts(2, 12, 30)) is None  # Mar 2 is not scored


def test_delivery_window_may_cross_midnight(built):
    cfg = make_digest_config(built, task=dict(digest_hour_utc=23,
                                              delivery_window_hours=2))
    sc = DailyRedditDigestTask.from_run_config(cfg, built.parent).scorer
    assert sc.day_for(ts(4, 0, 30)) == date(2026, 3, 3)


def test_config_rejects_lookback_beyond_horizon(built):
    with pytest.raises(ValueError, match="lookback_hours"):
        DailyRedditDigestTask.from_run_config(
            make_digest_config(built, task=dict(lookback_hours=30)),
            built.parent)


# -- recording ---------------------------------------------------------------------------

def test_rejects_outside_window_with_next_window_hint(scorer):
    with pytest.raises(NotificationError, match="next window 2026-03-03T12:00"):
        scorer.record_notification(t(3, 11, 59), POP)
    with pytest.raises(NotificationError, match="not in a delivery window"):
        scorer.record_notification(t(3, 13), POP)
    assert scorer.rejections == {"outside_window": 2}


def test_rejects_second_digest_same_day_and_malformed(scorer):
    scorer.record_notification(t(3, 12, 5), POP[:3])
    with pytest.raises(NotificationError, match="already delivered for 2026-03-03"):
        scorer.record_notification(t(3, 12, 40), POP)
    for bad in ([], "pop1", [""], ["pop1", 3]):
        with pytest.raises(NotificationError, match="root_ids"):
            scorer.record_notification(t(4, 12, 5), bad)
    assert scorer.rejections == {"duplicate_day": 1, "malformed": 4}


def test_unknown_ids_are_accepted_silently(scorer):
    scorer.record_notification(t(3, 12, 5), ["nope", "future_x"])
    assert scorer.digests[date(2026, 3, 3)].picks == ["nope", "future_x"]


# -- settlement ----------------------------------------------------------------------------

def test_full_digest_scores_one_and_settles_at_a_plus_24h(scorer):
    scorer.record_notification(t(3, 12, 30), POP)
    assert scorer.close_due(t(4, 11, 59)) == []
    (ev,) = scorer.close_due(t(4, 12))
    assert ev.ref == "2026-03-03" and ev.status == "ok"
    assert ev.detail["score"] == 1.0 and ev.detail["hits"] == 10
    assert ev.detail["delivered_at"] == "2026-03-03T12:30:00Z"


def test_window_edges(scorer):
    scorer.record_notification(
        t(3, 12, 30), ["edge_before", "edge_lo", "edge_hi", "edge_at", "hist"])
    (ev,) = scorer.close_due(t(4, 12))
    assert ev.detail["hits"] == 2  # edge_lo + edge_hi only
    per = {p["root_id"]: p for p in scorer.digests[date(2026, 3, 3)].per_id}
    assert per["edge_lo"]["hit"] and per["edge_hi"]["hit"]
    assert not per["edge_before"]["in_window"] and not per["edge_at"]["in_window"]
    assert per["edge_at"]["descendants"] is None  # no label leak off-window


def test_only_first_ten_distinct_ids_count(scorer):
    # 3 unpopular first, then 10 popular: picks = 3 unpop + pop1..pop7
    scorer.record_notification(t(3, 12, 1), UNPOP + POP)
    g = scorer.digests[date(2026, 3, 3)]
    assert g.picks == UNPOP + POP[:7]
    (ev,) = scorer.close_due(t(4, 12))
    assert ev.detail["score"] == 0.7


def test_duplicates_collapse_before_the_cut(scorer):
    scorer.record_notification(t(3, 12, 1), [POP[0]] * 5 + POP[1:])
    assert scorer.digests[date(2026, 3, 3)].picks == POP
    (ev,) = scorer.close_due(t(4, 12))
    assert ev.detail["score"] == 1.0


def test_popular_bar_is_inclusive(scorer):
    # pop* have exactly 50 comments; unpop* 8
    scorer.record_notification(t(3, 12, 1), POP[:2] + UNPOP)
    (ev,) = scorer.close_due(t(4, 12))
    assert ev.detail["hits"] == 2


def test_missed_days_settle_as_miss_and_close_all_finishes(scorer):
    scorer.record_notification(t(5, 12, 10), DAY5)  # 12 ids -> 10 picks
    evs = scorer.close_due(t(6, 12))  # Mar 3, 4 (miss), Mar 5 (ok)
    assert [(e.ref, e.status) for e in evs] == [
        ("2026-03-03", "miss"), ("2026-03-04", "miss"), ("2026-03-05", "ok")]
    assert evs[2].detail["score"] == 1.0
    rest = scorer.close_all()
    assert [e.ref for e in rest] == ["2026-03-06", "2026-03-07"]
    assert scorer.close_all() == []


# -- metrics / report ---------------------------------------------------------------------------

def test_metrics_mean_over_all_scored_days(scorer):
    scorer.record_notification(t(3, 12, 30), POP)
    scorer.record_notification(t(5, 12, 0), DAY5[:5])
    scorer.close_all()
    m = scorer.metrics()
    assert m["primary"] == {"name": "digest_score", "value": 0.3,
                            "direction": "max"}  # (1 + 0.5) / 5
    assert m["days"] == 5 and m["days_delivered"] == 2 and m["days_full"] == 1
    assert m["delivery_latency_min_mean"] == 15.0
    rep = scorer.report()
    assert set(rep["daily_outcomes"]) == {f"2026-03-0{d}" for d in range(3, 8)}
    assert rep["pending_days"] == []
    assert [g["day"] for g in rep["digests"]] == ["2026-03-03", "2026-03-05"]


def test_metrics_before_any_settlement(scorer):
    assert scorer.metrics()["primary"]["value"] is None


def test_oracle_outcomes_only_settled_days(scorer):
    scorer.record_notification(t(3, 12, 30), POP)
    scorer.close_due(t(4, 12))
    out = scorer.oracle_outcomes(None, ts(4, 12))
    assert [o["day"] for o in out] == ["2026-03-03"] and out[0]["score"] == 1.0
    assert scorer.oracle_outcomes(ts(4, 12), ts(5)) == []


def test_instruction_context_renders_example_day(task):
    ctx = task.instruction_context()
    assert ctx["first_day"] == "2026-03-03" and ctx["n_days"] == 5
    assert ctx["example_window_lo"] == "2026-03-02T12:00:00Z"
    assert ctx["example_deliver_hi"] == "2026-03-03T13:00:00Z"
    assert ctx["example_settle"] == "2026-03-04T12:00:00Z"
    assert ctx["digest_hour"] == "12"
