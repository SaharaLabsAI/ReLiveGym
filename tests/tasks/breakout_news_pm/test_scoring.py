"""Scorer settlement semantics: gold filters, uniform miss,
direction requirement, boundaries, event ordering — metric era: credits instead of penalties, one
standing claim per market instead of duplicate machinery."""

import pytest

from harness.task import NotificationError
from tests.tasks.breakout_news_pm.conftest import NEWS, make_config, t, ts

TIMING = 0.7  # timing_credit (metric constant)

# world: bp1 up @ Mar-3 12:00, gold {n-alpha (pub Mar-2 18:00),
# n-beta (pub Mar-3 11:00)}; bp2 down @ Mar-10 00:00, unwinnable.


def bp1(scorer):
    return scorer.breakpoints[0]


def bp2(scorer):
    return scorer.breakpoints[1]


# -- gold construction (load_breakpoints filters) -------------------------------------


def test_gold_filters(breakpoints):
    b1, b2 = breakpoints
    assert b1.winnable and set(b1.gold) == {"n-alpha", "n-beta"}
    # n-early (> W before), n-atstart (pub == t_start), n-lowconf (< 0.6)
    # all excluded; the 0.5-confidence group is gone entirely
    assert len(b1.gold_groups) == 1
    assert not b2.winnable and b2.gold == {} and b2.no_attribution


def test_gold_window_boundary_inclusive_at_minus_w(built):
    # an article published exactly t_start - W is gold (>= boundary)
    cfg = make_config()
    import json
    rows = [json.loads(l) for l in open(built / "attributions.jsonl")]
    rows[0]["groups"][0]["articles"].append(
        {"news_id": "n-edge", "pub_ts": ts(3, 12) - 24 * 3600})
    with open(built / "attributions.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    from tasks.breakout_news_pm.task import load_breakpoints

    b1 = load_breakpoints(built, cfg)[0]
    assert "n-edge" in b1.gold


# -- coverage -------------------------------------------------------------------------


def test_news_correct_cover_credit_decays_from_publish(scorer):
    # notify n-alpha (pub Mar-2 18:00) at Mar-3 00:00: 6 h into an 18 h lead
    scorer.record_notification(t(3, 0), "m1", "n-alpha", "up")
    events = scorer.close_all()
    b = bp1(scorer)
    assert b.status == "covered_news"
    assert b.credit == pytest.approx(1 - 6 / 18)
    assert scorer.alerts[0].status == "covering_news"
    assert sorted(e.status for e in events) == ["covered_news", "miss"]


def test_timing_only_cover_flat_credit(scorer):
    # right market/timing/direction, but citing a never-attributed article
    scorer.record_notification(t(3, 0), "m1", "n-other", "up")
    scorer.close_all()
    b = bp1(scorer)
    assert b.status == "covered_timing"
    assert b.credit == pytest.approx(TIMING)
    assert scorer.alerts[0].status == "covering_timing"


def test_notification_at_move_start_earns_nothing(scorer):
    # tau == t_start: not eligible (strict tau < t_start)
    scorer.record_notification(t(3, 12), "m1", "n-beta", "up")
    scorer.close_all()
    assert bp1(scorer).status == "miss"
    assert bp1(scorer).credit == 0.0
    # the alert cites gold news of an already-started breakpoint -> stale
    assert scorer.alerts[0].status == "stale"


def test_claim_horizon_boundary_covers(scorer):
    # tau + W == t_start exactly: covers (inclusive boundary), and the
    # settle loop closes the breakpoint before expiring the alert on the tie
    scorer.record_notification(t(2, 12), "m1", "n-other", "up")
    scorer.close_all()
    assert bp1(scorer).status == "covered_timing"
    assert scorer.alerts[0].status == "covering_timing"


def test_just_beyond_claim_horizon_is_false_alarm(scorer):
    # tau + W < t_start: the claim was false when made
    scorer.record_notification(t(2, 11, 59), "m1", "n-other", "up")
    scorer.close_all()
    a = scorer.alerts[0]
    assert a.status == "false_alarm" and not a.excused
    assert bp1(scorer).status == "miss"


# -- direction (plan D2) --------------------------------------------------------------


def test_wrong_direction_on_missed_breakout_is_excused(scorer):
    # Pair rule: the miss already scores 0 in full; the directional
    # attempt must never rank worse than silence -> excused, out of the
    # precision denominator.
    scorer.record_notification(t(3, 0), "m1", "n-alpha", "down")
    scorer.close_all()
    a = scorer.alerts[0]
    assert a.status == "wrong_direction" and a.excused
    assert bp1(scorer).status == "miss"
    m = scorer.metrics()
    assert m["precision"] is None  # the only resolved alert is excused
    assert m["tc_recall"] == pytest.approx(0.0)


def test_hedge_impossible_second_claim_rejected(scorer):
    # One standing claim per market: the down-leg hedge is rejected free
    # of charge while the up claim is pending.
    scorer.record_notification(t(3, 0), "m1", "n-other", "up")
    with pytest.raises(NotificationError, match="one standing claim"):
        scorer.record_notification(t(3, 0, 1), "m1", "n-alpha", "down")
    scorer.close_all()
    assert [a.status for a in scorer.alerts] == ["covering_timing"]
    assert bp1(scorer).status == "covered_timing"


def test_new_claim_allowed_after_resolution(scorer):
    # a claim resolves when its breakpoint covers it; the market reopens
    scorer.record_notification(t(3, 0), "m1", "n-alpha", "up")
    scorer.close_due(t(4, 0))  # bp1 closed at Mar-3 12:00, claim resolved
    scorer.record_notification(t(9, 20), "m1", "n-other", "down")
    scorer.close_all()
    assert [a.status for a in scorer.alerts] == \
        ["covering_news", "covering_timing"]


def test_direction_accuracy_reported(scorer):
    scorer.record_notification(t(3, 0), "m1", "n-alpha", "up")
    scorer.close_due(t(4, 0))
    scorer.record_notification(t(9, 20), "m1", "n-other", "up")  # bp2 is down
    scorer.close_all()
    m = scorer.metrics()
    assert m["direction_accuracy"] == pytest.approx(0.5)
    r = scorer.report()
    assert r["price_centric"]["direction_accuracy"] == pytest.approx(0.5)


# -- uniform treatment of unwinnable (plan D3) ----------------------------------------


def test_unwinnable_miss_scores_zero_credit(scorer):
    events = scorer.close_all()
    assert bp2(scorer).status == "miss"
    assert bp2(scorer).credit == 0.0
    assert [e.status for e in events] == ["miss", "miss"]
    m = scorer.metrics()
    # TC-recall denominates over winnable only (bp1); bp2 is excluded
    assert m["winnable_breakpoints"] == 1
    assert m["tc_recall"] == pytest.approx(0.0)


def test_unwinnable_covered_timing_and_metrics(scorer):
    scorer.record_notification(t(9, 20), "m1", "n-other", "down")
    scorer.close_all()
    assert bp2(scorer).status == "covered_timing"
    r = scorer.report()
    assert r["price_centric"]["all"]["breakpoints"] == 2
    assert r["price_centric"]["all"]["covered"] == 1
    assert r["price_centric"]["unwinnable"]["covered_timing"] == 1
    # news-centric: no positive label on bp2 -> ordinary false positive
    assert r["news_centric"]["precision"] == 0.0
    assert r["news_centric"]["recall"] == 0.0
    # v1: the covering alert counts toward precision but earns no
    # TC-recall credit (bp2 unwinnable). v2: it earns full recall credit
    # — every closed breakpoint is in the denominator.
    m = scorer.metrics()
    assert m["precision"] == pytest.approx(1.0)
    assert m["tc_recall"] == pytest.approx(0.0)
    assert m["cov_recall"] == pytest.approx(0.5)  # 1 of 2 closed
    assert m["primary"]["value"] == pytest.approx(
        2 * 1.0 * 0.5 / 1.5, abs=1e-4)


# -- one standing claim per market ----------------------------------------------------


def test_pending_claim_blocks_and_expiry_reopens(scorer):
    scorer.record_notification(t(5, 0), "m1", "n-other", "up")  # will FA
    with pytest.raises(NotificationError, match="one standing claim"):
        scorer.record_notification(t(5, 1), "m1", "n-alpha", "up")
    # after tau+W the claim expires as false_alarm; the market reopens
    scorer.close_due(t(6, 1))
    assert scorer.alerts[0].status == "false_alarm"
    scorer.record_notification(t(6, 2), "m1", "n-alpha", "up")
    assert len(scorer.alerts) == 2


# -- validation -----------------------------------------------------------------------


def test_validation_rejections(scorer):
    with pytest.raises(NotificationError):  # bad direction
        scorer.record_notification(t(3, 0), "m1", "n-alpha", "sideways")
    with pytest.raises(NotificationError):  # unknown news
        scorer.record_notification(t(3, 0), "m1", "n-nope", "up")
    with pytest.raises(NotificationError):  # not yet published
        scorer.record_notification(t(2, 17), "m1", "n-alpha", "up")
    with pytest.raises(NotificationError):  # outside monitoring window
        scorer.record_notification(t(1, 0), "m1", "n-early", "up")
    assert scorer.alerts == []


# -- metrics --------------------------------------------------------------------------


def test_v2_ignores_cite_decay_that_undercuts_timing_credit(scorer):
    # late gold cite: 17 h into n-alpha's 18 h lead -> v1 credit 1/18,
    # BELOW the 0.7 a non-gold cite would earn at the same instant. v2
    # counts the cover in full regardless of the article cited.
    scorer.record_notification(t(3, 11), "m1", "n-alpha", "up")
    scorer.close_all()
    m = scorer.metrics()
    assert m["tc_recall"] == pytest.approx(1 / 18, abs=1e-4)
    assert m["tc_recall"] < TIMING
    assert m["cov_recall"] == pytest.approx(0.5)  # 1 of 2 closed, full credit
    assert m["primary"]["name"] == "cov_f1"
    assert m["primary"]["value"] == pytest.approx(
        2 * 1.0 * 0.5 / 1.5, abs=1e-4)


def test_cov_f1_primary_and_tc_f1_secondary_metrics(scorer):
    scorer.record_notification(t(3, 0), "m1", "n-alpha", "up")   # news-correct
    scorer.close_due(t(4, 0))
    scorer.record_notification(t(5, 0), "m1", "n-other", "up")   # false alarm
    scorer.close_all()
    m = scorer.metrics()
    assert m["precision"] == pytest.approx(0.5)      # 1 covering of 2
    # v2 primary: binary coverage over all closed (bp1 covered, bp2 not)
    assert m["cov_recall"] == pytest.approx(0.5)
    assert m["primary"]["name"] == "cov_f1"
    assert m["primary"]["direction"] == "max"
    assert m["primary"]["value"] == pytest.approx(0.5, abs=1e-4)
    assert m["breakpoints_covered"] == 1
    # tc_f1: decay credit over winnable only
    credit = 1 - 6 / 18
    assert m["tc_recall"] == pytest.approx(credit, abs=1e-4)  # 1 winnable bp
    p, r_ = 0.5, credit
    assert m["tc_f1"] == pytest.approx(2 * p * r_ / (p + r_), abs=1e-4)
    assert m["false_alarms"] == 1
    rep = scorer.report()
    nc = rep["news_centric"]
    assert nc["precision"] == pytest.approx(0.5)   # 1 of 2 notifications
    assert nc["recall"] == pytest.approx(1.0)      # 1 of 1 winnable
    assert nc["f1"] == pytest.approx(2 * 0.5 / 1.5, abs=1e-4)
    assert nc["publish_latency_hours"]["median"] == pytest.approx(6.0)
    lat = rep["price_centric"]["all"]["notification_latency_hours"]
    assert lat["median"] == pytest.approx(12.0)


def test_feedback_hides_gold_oracle_reveals_it(scorer):
    scorer.record_notification(t(3, 0), "m1", "n-alpha", "up")
    scorer.close_all()
    fb = scorer.feedback()
    bp_rows = [d for d in fb if d["kind"] == "breakpoint"]
    assert bp_rows and all("gold_groups" not in d and "winnable" not in d
                           for d in bp_rows)
    oc = scorer.oracle_outcomes(None, ts(13))
    b1 = next(d for d in oc if d["kind"] == "breakpoint"
              and d["date"] == "2026-03-03")
    assert b1["winnable"] and b1["gold_groups"][0]["articles"]
    b2 = next(d for d in oc if d["kind"] == "breakpoint"
              and d["date"] == "2026-03-10")
    assert b2["no_attributable_news"] and b2["reason"] == "no_attribution"
