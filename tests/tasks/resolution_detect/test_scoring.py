"""Event-settlement math, free-reject matrix, one-shot rule, metrics.

Soft metric: credit = exp(-early/12h) before t_det,
exp(-delay/48h) after; both clocks fixed, independent of the question's
gap. Precision is credit-weighted. Claims on questions with no in-window
determination stay hard false alarms.
"""

from __future__ import annotations

import math

import pytest

from harness.task import NotificationError

from .conftest import SIM_END, T0, t, ts


def settle_all(scorer):
    scorer.close_due(SIM_END)
    scorer.close_all()
    return {q.qid: q for q in scorer.questions.values()}


# -- settlement categories & credit ----------------------------------------------------


def test_claim_at_t_det_full_credit(scorer):
    scorer.record_mark(t(4), "q_win", "Yes")
    q = settle_all(scorer)["q_win"]
    assert q.category == "covered" and q.credit == pytest.approx(1.0)
    assert q.delay_s == 0.0


def test_credit_decays_exponentially_on_the_fixed_late_clock(scorer):
    scorer.record_mark(t(5), "q_win", "Yes")     # 24h after t_det day 4
    scorer.record_mark(t(8.9), "q_mid", "Yes")   # 45.6h after t_det day 7
    qs = settle_all(scorer)
    assert qs["q_win"].category == "covered"
    assert qs["q_win"].credit == pytest.approx(math.exp(-24 / 48))
    assert qs["q_mid"].category == "covered"
    assert qs["q_mid"].credit == pytest.approx(math.exp(-45.6 / 48))


def test_early_claim_credit_decays_on_the_fixed_early_clock(scorer):
    # the P1 pin, soft form: the speculator claiming the CORRECT outcome
    # at activation (96h before t_det) earns essentially nothing; a claim
    # 6h early earns the honest near-miss remainder
    scorer.record_mark(t(0), "q_win", "Yes")
    scorer.record_mark(t(6.75), "q_mid", "Yes")  # t_det day 7: 6h early
    qs = settle_all(scorer)
    assert qs["q_win"].category == "early"
    assert qs["q_win"].credit == pytest.approx(math.exp(-96 / 12))
    assert qs["q_win"].delay_s == pytest.approx(-96 * 3600)
    assert qs["q_mid"].category == "early"
    assert qs["q_mid"].credit == pytest.approx(math.exp(-6 / 12))
    m = scorer.metrics()
    assert m["n_covered"] == 0 and m["n_early"] == 2
    assert m["n_fa_premature"] == 0


def test_wrong_outcome_is_false_alarm(scorer):
    scorer.record_mark(t(5), "q_win", "No")
    q = settle_all(scorer)["q_win"]
    assert q.category == "fa_wrong" and q.credit == 0.0


def test_unclaimed_winnable_is_miss(scorer):
    q = settle_all(scorer)["q_win"]
    assert q.category == "miss"


def test_claim_on_unwinnable_scored_question_is_premature(scorer):
    scorer.record_mark(t(1), "q_unwin", "No")  # correct side, no t_det
    q = settle_all(scorer)["q_unwin"]
    assert q.category == "fa_premature"


def test_unclaimed_unwinnable_is_silent_not_miss(scorer):
    q = settle_all(scorer)["q_unwin"]
    assert q.category == "silent"
    assert scorer.metrics()["n_miss"] == 3  # the three winnable questions


def test_quiet_question_claim_settles_premature_at_close_all(scorer):
    scorer.record_mark(t(2), "q_open", "Yes")
    scorer.record_mark(t(2), "q_late", "Yes")  # even the eventual answer
    assert scorer.close_due(SIM_END) is not None
    qs = {q.qid: q for q in scorer.questions.values()}
    assert not qs["q_open"].settled  # silent until close_all
    events = scorer.close_all()
    assert {e.ref for e in events if e.status == "unresolved"} == {
        "q_open", "q_late"}
    assert qs["q_open"].category == "fa_premature"
    assert qs["q_late"].category == "fa_premature"


def test_late_credit_runs_on_delay_from_t_det_not_gap(scorer):
    # q_lag: t_det day 3, t_res day 6 — a claim just before t_res is
    # ~72h late; its credit depends only on that delay, not on the gap
    scorer.record_mark(t(5.9999), "q_lag", "No")
    q = settle_all(scorer)["q_lag"]
    assert q.category == "covered"
    assert q.credit == pytest.approx(
        math.exp(-(ts(5.9999) - ts(3)) / (48 * 3600)))


def test_settles_at_public_resolution_not_t_res(scorer):
    scorer.record_mark(t(4), "q_lag", "No")
    early = scorer.close_due(t(6.2))  # q_lag past t_res, not yet public
    # unclaimed questions (q_unwin closed day 5, q_win day 6) no longer
    # settle mid-run — they must stay open to a late claim now that
    # resolutions are invisible; their misses book at close_all
    assert early == []
    events = scorer.close_due(t(6.6))
    assert [e.ref for e in events] == ["q_lag"]
    assert {q.qid for q in scorer.questions.values() if q.settled} == {
        "q_lag"}


# -- one-shot rule & free-reject matrix ------------------------------------------------


def test_second_claim_rejected_first_stands(scorer):
    scorer.record_mark(t(4), "q_win", "Yes")
    with pytest.raises(NotificationError, match="already has your claim"):
        scorer.record_mark(t(5), "q_win", "No")
    q = settle_all(scorer)["q_win"]
    assert q.claim.outcome == "Yes" and q.category == "covered"
    assert q.credit == pytest.approx(1.0)


def test_unknown_and_unactivated_reject_identically(scorer):
    with pytest.raises(NotificationError) as unknown:
        scorer.record_mark(t(0), "ghost", "Yes")
    with pytest.raises(NotificationError) as unactivated:
        scorer.record_mark(t(0), "q_mid", "Yes")  # opens day 2
    assert (str(unknown.value).replace("ghost", "q_mid")
            == str(unactivated.value))


def test_resolution_copier_pays_only_decay(scorer):
    # resolutions are invisible, so the
    # already-resolved reject (a status probe channel) is gone —
    # post-resolution claims are accepted and priced by the same fixed
    # 48 h clock, which is what kills copying
    scorer.close_due(t(6.6))
    scorer.record_mark(t(6.6), "q_win", "Yes")  # q_win closed day 6
    q = settle_all(scorer)["q_win"]
    assert q.category == "covered"
    assert q.credit == pytest.approx(
        math.exp(-(ts(6.6) - ts(4)) / (48 * 3600)))  # t_det day 4


@pytest.mark.parametrize("outcome,match", [
    (None, "outcome must be one of"),
    (1, "outcome must be one of"),
    ("Maybe", "outcome must be one of"),
])
def test_bad_outcomes_reject_free(scorer, outcome, match):
    with pytest.raises(NotificationError, match=match):
        scorer.record_mark(t(1), "q_win", outcome)
    assert scorer.questions["q_win"].claim is None


def test_news_id_validated_against_published(scorer):
    with pytest.raises(NotificationError, match="news_id must be"):
        scorer.record_mark(t(1), "q_win", "Yes", news_id="")
    with pytest.raises(NotificationError, match="not-yet-published"):
        scorer.record_mark(t(1), "q_win", "Yes", news_id="ghost_article")
    with pytest.raises(NotificationError, match="not-yet-published"):
        scorer.record_mark(t(0.2), "q_win", "Yes", news_id="n1")  # pub 0.5
    scorer.record_mark(t(1), "q_win", "Yes", news_id="n1")
    assert scorer.questions["q_win"].claim.news_id == "n1"


def test_rejection_consumes_nothing(scorer):
    for bad in [("ghost", "Yes", None), ("q_win", "Maybe", None),
                ("q_win", "Yes", "ghost_article")]:
        with pytest.raises(NotificationError):
            scorer.record_mark(t(1), *bad[:2], news_id=bad[2])
    scorer.record_mark(t(4), "q_win", "Yes")  # claim still available
    assert settle_all(scorer)["q_win"].category == "covered"


# -- metrics ---------------------------------------------------------------------------


def test_tc_f1_hand_computed(scorer):
    scorer.record_mark(t(4), "q_win", "Yes")   # covered, credit 1
    scorer.record_mark(t(4.5), "q_lag", "No")  # covered, 36h late
    scorer.record_mark(t(2), "q_open", "Yes")  # quiet FA
    settle_all(scorer)                          # q_mid -> miss
    m = scorer.metrics()
    credit = 1.0 + math.exp(-36 / 48)
    recall = credit / 3
    precision = credit / 3
    assert m["recall"] == pytest.approx(recall, abs=1e-4)
    assert m["precision"] == pytest.approx(precision, abs=1e-4)
    assert m["primary"]["name"] == "tc_f1"
    assert m["primary"]["value"] == pytest.approx(
        2 * precision * recall / (precision + recall), abs=1e-4)
    # cov_f1 stays the undecayed binary companion (covered claims only)
    assert m["cov_f1"] == pytest.approx(2 * (2 / 3) * (2 / 3) / (4 / 3),
                                        abs=1e-4)


def test_silence_scores_zero(scorer):
    settle_all(scorer)
    m = scorer.metrics()
    assert m["primary"]["value"] == 0.0
    assert m["n_claims_settled"] == 0 and m["n_miss"] == 3


def test_oracle_returns_only_settled_with_own_claim(scorer):
    scorer.record_mark(t(4), "q_win", "Yes")
    scorer.close_due(t(6.2))  # q_unwin (day 5) and q_win (day 6) public
    rows = scorer.oracle_outcomes(None, ts(6.2))
    # deferred-settlement consequence: unclaimed q_unwin is not settled
    # mid-run, so the oracle feed carries only the claimed q_win here;
    # unclaimed resolutions surface at run end (phase-2 note: a live
    # miss feed for learning cells must read resolved_public, not
    # settled)
    assert [r["question_id"] for r in rows] == ["q_win"]
    win = rows[0]
    assert win["category"] == "covered"
    assert win["your_claim"]["outcome"] == "Yes"
    # nothing for questions still open
    assert all(r["question_id"] not in ("q_mid", "q_open", "q_late")
               for r in rows)


def test_report_fa_decomposition(scorer):
    scorer.record_mark(t(0), "q_win", "Yes")    # early (credited, not FA)
    scorer.record_mark(t(1), "q_unwin", "No")   # unwinnable scored
    scorer.record_mark(t(2), "q_open", "Yes")   # quiet
    settle_all(scorer)
    fa = scorer.report()["fa_decomposition"]
    assert fa == {"quiet_question": 1, "unwinnable_question": 1}
    assert scorer.metrics()["n_early"] == 1


# -- load_questions validation ---------------------------------------------------------


def test_question_resolved_before_sim_start_rejected(built):
    from tasks.resolution_detect.task import load_questions
    from .conftest import make_tcfg
    with pytest.raises(ValueError, match="resolved before sim_start"):
        load_questions(built, make_tcfg(built), t(5.5), SIM_END)


def test_run_window_demotes_out_of_window_resolutions(built):
    from tasks.resolution_detect.task import load_questions
    from .conftest import make_tcfg
    qs = load_questions(built, make_tcfg(built), T0, t(5.5))
    # q_win resolves day 6 > sim_end: run-unscored, unwinnable in-run
    assert not qs["q_win"].run_scored and not qs["q_win"].winnable


def test_duplicate_questions_rejected(built):
    from .conftest import make_tcfg
    with pytest.raises(ValueError, match="duplicates"):
        make_tcfg(built, questions=["q_win", "q_win"])
