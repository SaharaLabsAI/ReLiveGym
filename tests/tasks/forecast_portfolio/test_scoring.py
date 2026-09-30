"""Scorer math (exact fractions by construction), settlement timing,
free-reject matrix, and config-vs-data validation."""

from __future__ import annotations

import pytest

from harness.task import NotificationError
from tasks.forecast_portfolio.task import (
    Scorer, _bss, integrate_ta, load_questions,
)

from .conftest import SIM_END, T0, make_tcfg, t, ts


# -- TA math --------------------------------------------------------------------------


def test_no_forecast_scores_as_the_uniform_one(scorer):
    """Silence is the uninformed forecast, not a zero:
    an unattended binary question earns +0.5 for its whole life."""
    scorer.close_all()
    m = scorer.metrics()
    assert m["primary"]["value"] == 0.5
    assert m["questions_settled"] == 3
    assert m["abstention_share"] == 1.0  # still reported as unattended


def test_silence_and_submitted_uniform_are_indistinguishable(scorer):
    scorer.record_forecast(t(0), "q_res", {"Yes": 0.5, "No": 0.5})
    scorer.close_all()
    assert scorer.questions["q_res"].ta == pytest.approx(
        scorer.questions["q_mid"].ta)  # q_mid never got a forecast


def test_uniform_scores_half(scorer):
    scorer.record_forecast(t(0), "q_res", {"Yes": 0.5, "No": 0.5})
    scorer.close_all()
    q = scorer.questions["q_res"]
    assert q.ta == pytest.approx(0.5)
    assert q.abstain_share == 0.0


def test_camp_scores_camp_fraction(scorer):
    # q_res life = [day 0, day 4]; the uniform default holds for 3.6 days,
    # confident-correct for the last 0.4 -> (3.6*0.5 + 0.4*1.0) / 4
    scorer.record_forecast(t(3.6), "q_res", {"Yes": 1.0})
    scorer.close_all()
    assert scorer.questions["q_res"].ta == pytest.approx(0.55)


def test_confident_wrong_is_minus_one(scorer):
    scorer.record_forecast(t(0), "q_lag", {"Yes": 1.0})  # resolves No
    scorer.close_all()
    assert scorer.questions["q_lag"].ta == pytest.approx(-1.0)


def test_partial_mass_scores_the_omitted_outcome_as_zero(scorer):
    scorer.record_forecast(t(0), "q_res", {"Yes": 0.5})
    scorer.close_all()
    # 1 - ((0.5-1)^2 + (0-0)^2) = 0.75 -- leaning the right way
    assert scorer.questions["q_res"].ta == pytest.approx(0.75)


def test_omitting_the_winning_outcome_is_worse_than_silence(scorer):
    """The hazard INSTRUCTION now spells out: an omitted outcome is a
    stated zero, NOT a fallback to the uniform default, so a half
    submission on the wrong side scores below doing nothing at all."""
    scorer.record_forecast(t(0), "q_lag", {"Yes": 0.5})  # resolves No
    scorer.close_all()
    qs = scorer.questions
    assert qs["q_lag"].ta == pytest.approx(-0.25)
    assert qs["q_lag"].ta < qs["q_mid"].ta  # q_mid: never submitted, 0.5


def test_multi_update_exact(scorer):
    # q_lag (y=No), life [0, 6]: {No: .5} for days 0-3, {No: 1} for 3-6
    scorer.record_forecast(t(0), "q_lag", {"No": 0.5})
    scorer.record_forecast(t(3), "q_lag", {"No": 1.0})
    scorer.close_all()
    assert scorer.questions["q_lag"].ta == pytest.approx(
        (3 * 0.75 + 3 * 1.0) / 6)


def test_exact_equals_minute_grid_brute_force(scorer):
    trace = [(ts(0.25), {"Yes": 0.3}, None),
             (ts(1.5), {"Yes": 0.8, "No": 0.1}, "n1"),
             (ts(3), {}, None), (ts(3.75), {"Yes": 0.55, "No": 0.45}, "n2")]
    exact, _ = integrate_ta(trace, ["Yes", "No"], "Yes", ts(0), ts(4))
    minutes = 4 * 24 * 60
    acc = 0.0
    for i in range(minutes):  # left endpoint of each 1-min cell
        now = ts(0) + i * 60
        cur = None
        for tt, d, _ in trace:
            if tt <= now:
                cur = d
        acc += _bss(cur, ["Yes", "No"], "Yes") * 60
    assert exact == pytest.approx(acc / (minutes * 60), abs=1e-9)


def test_proper_scoring_elicits_true_probability():
    # expected instantaneous score under true p is maximized by reporting p
    p_true = 0.7
    grid = [i / 100 for i in range(101)]

    def expected(q: float) -> float:
        d = {"Yes": q, "No": 1 - q}
        return (p_true * _bss(d, ["Yes", "No"], "Yes")
                + (1 - p_true) * _bss(d, ["Yes", "No"], "No"))

    assert max(grid, key=expected) == pytest.approx(p_true)


# -- settlement timing ----------------------------------------------------------------


def test_settles_at_public_resolution_not_t_res(scorer):
    # q_lag: life ends at sched_end (day 6), public at closedTime (day 6.5)
    assert [e.ref for e in scorer.close_due(t(6.2))] == ["q_res"]
    events = scorer.close_due(t(6.6))
    assert [e.ref for e in events] == ["q_lag"]
    assert events[0].status == "resolved"
    assert events[0].detail["outcome"] == "No"


def test_lag_window_submission_accepted_but_earns_nothing(scorer):
    scorer.record_forecast(t(0), "q_lag", {"No": 1.0})
    scorer.record_forecast(t(6.2), "q_lag", {"Yes": 1.0})  # after t_res
    scorer.close_all()
    assert scorer.questions["q_lag"].ta == pytest.approx(1.0)
    with pytest.raises(NotificationError, match="already resolved"):
        scorer.record_forecast(t(6.6), "q_lag", {"Yes": 1.0})


def test_unscored_questions_never_settle(scorer):
    events = scorer.close_all()
    assert {e.ref for e in events} == {"q_res", "q_lag", "q_mid"}
    assert not scorer.questions["q_open"].settled
    assert not scorer.questions["q_late"].settled


def test_close_events_chronological_and_detailed(scorer):
    events = scorer.close_all()
    assert [e.ref for e in events] == ["q_res", "q_lag", "q_mid"]
    assert set(events[0].detail) == {"outcome", "ta_bss", "ta_bss_market",
                                     "n_updates"}


# -- market anchor + easy flag --------------------------------------------------------


def test_market_anchor_exact(scorer):
    scorer.close_all()
    qs = scorer.questions
    assert qs["q_res"].ta_market == pytest.approx(1 - 2 * 0.1 ** 2)
    assert qs["q_lag"].ta_market == pytest.approx(
        (3 * (1 - 2 * 0.4 ** 2) + 3 * (1 - 2 * 0.2 ** 2)) / 6)
    # q_mid: no quote until day 5 of life [2, 8]. The quoteless lead-in
    # mirrors the agent's no-forecast rule, so it is now the uniform 0.5
    # for 3 of the 6 days rather than a zero
    assert qs["q_mid"].ta_market == pytest.approx(
        (3 * 0.5 + 3 * (1 - 2 * 0.2 ** 2)) / 6)


def test_easy_flag_from_band(built):
    tcfg = make_tcfg(built, easy_band=0.85)
    scorer = Scorer(tcfg, built, load_questions(built, tcfg, T0, SIM_END))
    scorer.close_all()
    assert scorer.questions["q_res"].easy is True  # flat 0.9 >= 0.85
    assert scorer.questions["q_lag"].easy is False
    m = scorer.metrics()
    assert m["questions_easy"] == 1
    # nontrivial mean excludes q_res; q_lag and q_mid went unattended, so
    # both sit at the uniform default
    assert m["ta_bss_nontrivial"] == pytest.approx(0.5)


def test_metrics_aggregation(scorer):
    scorer.record_forecast(t(0), "q_res", {"Yes": 1.0})   # TA  1.0
    scorer.record_forecast(t(0), "q_lag", {"Yes": 1.0})   # TA -1.0
    scorer.close_all()                                    # q_mid: TA 0.5
    m = scorer.metrics()
    assert m["primary"] == {"name": "ta_bss", "value": 0.1667,
                            "direction": "max"}
    assert m["skill_vs_market"] == pytest.approx(
        0.5 / 3 - (0.98 + 0.8 + 0.71) / 3, abs=1e-3)
    assert m["mean_updates"] == pytest.approx(2 / 3, abs=1e-4)


def test_oracle_returns_only_settled_with_own_trace(scorer):
    scorer.record_forecast(t(0), "q_res", {"Yes": 0.8, "No": 0.2})
    scorer.close_due(t(6.6))
    out = scorer.oracle_outcomes(None, ts(6.6))
    assert [o["question_id"] for o in out] == ["q_res", "q_lag"]
    rec = out[0]
    assert rec["outcome"] == "Yes"
    assert rec["ta_bss"] == pytest.approx(1 - 2 * 0.2 ** 2)
    assert rec["your_forecasts"][0]["forecast"] == {"Yes": 0.8, "No": 0.2}
    # cursor semantics: nothing new since
    assert scorer.oracle_outcomes(ts(6.6), ts(8.0)) == []


def test_oracle_record_shape_is_hindsight_only(scorer):
    """The sig-oracle contract: outcome + own trace + own score on
    already-resolved questions, and nothing else. `ta_bss_market` in
    particular is derived from the price series the agent never sees."""
    scorer.record_forecast(t(0), "q_res", {"Yes": 0.8, "No": 0.2})
    scorer.close_due(t(6.6))
    for rec in scorer.oracle_outcomes(None, ts(6.6)):
        assert set(rec) == {"kind", "question_id", "t_settled", "outcome",
                            "ta_bss", "abstention_share", "your_forecasts"}


def test_oracle_silent_on_questions_not_publicly_resolved(scorer):
    """q_late resolves after sim_end and q_open never — neither may show
    up, at any cursor, right up to the end of the run. q_lag settles at
    its public closedTime (6.5), not at its scoring t_res (6.0)."""
    for now in (2.0, 4.0, 6.0, 6.5, 9.99):
        scorer.close_due(t(now))
        seen = {o["question_id"] for o in scorer.oracle_outcomes(None, ts(now))}
        assert not seen & {"q_open", "q_late"}
        assert ("q_lag" in seen) == (now >= 6.5)


# -- free-reject matrix ---------------------------------------------------------------


def test_unknown_and_unactivated_reject_identically(scorer):
    with pytest.raises(NotificationError) as unknown:
        scorer.record_forecast(t(1), "nope", {"Yes": 1.0})
    with pytest.raises(NotificationError) as unactivated:
        scorer.record_forecast(t(1), "q_mid", {"Yes": 1.0})  # opens day 2
    assert str(unknown.value).replace("nope", "q_mid") == str(unactivated.value)


@pytest.mark.parametrize("forecast, match", [
    (None, "non-empty"),
    ({}, "non-empty"),
    ("Yes", "non-empty"),
    ({"Maybe": 1.0}, "unknown outcome"),
    ({"Yes": -0.1}, ">= 0"),
    ({"Yes": True}, ">= 0"),
    ({"Yes": "high"}, ">= 0"),
    ({"Yes": 0.7, "No": 0.7}, "sum to"),
])
def test_bad_forecasts_reject_free(scorer, forecast, match):
    with pytest.raises(NotificationError, match=match):
        scorer.record_forecast(t(0), "q_res", forecast)
    assert scorer.questions["q_res"].trace == []


def test_sum_exactly_one_accepted(scorer):
    scorer.record_forecast(t(0), "q_res", {"Yes": 0.6, "No": 0.4})
    assert len(scorer.questions["q_res"].trace) == 1


# -- citation param (recorded, never scored) ------------------------------------------


def test_citation_recorded_and_counted(scorer):
    scorer.record_forecast(t(0), "q_res", {"Yes": 0.6}, "n1")
    scorer.record_forecast(t(1), "q_res", {"Yes": 0.7})
    assert [n for _, _, n in scorer.questions["q_res"].trace] == ["n1", None]
    scorer.close_all()
    rec = next(r for r in scorer.report()["questions"]
               if r["question_id"] == "q_res")
    assert (rec["n_updates"], rec["n_cited"]) == (2, 1)


@pytest.mark.parametrize("news_id, match", [
    ("", "non-empty string"),
    (7, "non-empty string"),
    ("ghost", "unknown or not-yet-published"),
    ("n2", "unknown or not-yet-published"),  # published day 5, cited day 0
])
def test_bad_citation_rejects_free(scorer, news_id, match):
    with pytest.raises(NotificationError, match=match):
        scorer.record_forecast(t(0), "q_res", {"Yes": 0.6}, news_id)
    assert scorer.questions["q_res"].trace == []


def test_citation_does_not_change_the_score(scorer):
    """Recorded, never scored: identical forecasts score identically
    whether or not they name an article."""
    scorer.record_forecast(t(0), "q_res", {"Yes": 0.6, "No": 0.4}, "n1")
    scorer.record_forecast(t(0), "q_lag", {"No": 0.6, "Yes": 0.4})
    scorer.close_all()
    qs = scorer.questions
    assert qs["q_res"].ta == pytest.approx(qs["q_lag"].ta)


# -- config-vs-data validation --------------------------------------------------------


def test_unknown_question_id_rejected(built):
    tcfg = make_tcfg(built, questions=["q_res", "ghost"])
    with pytest.raises(ValueError, match="ghost"):
        load_questions(built, tcfg, T0, SIM_END)


def test_question_resolved_before_sim_start_rejected(built):
    tcfg = make_tcfg(built, questions=["q_res"])
    with pytest.raises(ValueError, match="resolved before sim_start"):
        load_questions(built, tcfg, t(5), SIM_END)  # q_res closed day 4


def test_question_opening_after_sim_end_rejected(built):
    tcfg = make_tcfg(built, questions=["q_mid"])
    with pytest.raises(ValueError, match="opens at or after"):
        load_questions(built, tcfg, T0, t(1.5))  # q_mid opens day 2


def test_run_window_demotes_out_of_window_resolutions(built):
    tcfg = make_tcfg(built, questions=["q_res", "q_lag"])
    qs = load_questions(built, tcfg, T0, t(5))  # q_lag t_res day 6 > end
    assert qs["q_res"].run_scored is True
    assert qs["q_lag"].run_scored is False


def test_duplicate_questions_rejected(built):
    with pytest.raises(ValueError, match="duplicates"):
        make_tcfg(built, questions=["q_res", "q_res"])
