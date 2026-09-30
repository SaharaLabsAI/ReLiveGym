"""Scoring, reveal, and prefix logic for reddit_ai_popularity."""

from __future__ import annotations

import pytest

from harness.task import NotificationError
from tasks.reddit_ai_popularity.task import RedditPopularityTask, pop_of

from tests.tasks.reddit_ai_popularity.conftest import make_reddit_config, t, ts


# -- labels & pop ---------------------------------------------------------------------

def test_labels_within_horizon(roots):
    # descendants = comments within 24h; big's +25h comment is excluded
    assert roots.get("big")["_label"] == 256
    assert roots.get("dud")["_label"] == 2
    assert roots.get("self")["_label"] == 4
    assert roots.get("mid")["_label"] == 8


def test_pop():
    assert pop_of(256) == 9.0
    assert pop_of(2) == 2.0
    assert pop_of(0) == 1.0  # zero-traction floor


# -- reveal gating --------------------------------------------------------------------

def test_root_reveal_hides_label_and_never_leaks_score(roots):
    r = roots.get("big")
    pre = roots.visible_view(r, ts(2, 12))  # +6h, before reveal
    assert pre["revealed"] is False and pre["descendants"] is None
    post = roots.visible_view(r, ts(3, 7))  # after reveal (day3 06:00)
    assert post["revealed"] is True and post["descendants"] == 256
    for v in (pre, post):
        assert "score" not in v
    # num_comments is the live prefix count: comments arrive every 5s from
    # +60s, so at +5min 49 are in; at +6h all 256 within-horizon ones; after
    # reveal the +25h latecomer makes it 257 while the label stays frozen at 256
    early = roots.visible_view(r, ts(2, 6) + 300)
    assert early["num_comments"] == 49 and early["descendants"] is None
    assert pre["num_comments"] == 256 and post["num_comments"] == 257


def test_query_clips_future_and_time_sorted(roots):
    # at day2 12:00, only posts made by then are visible (hist, big, dud, self)
    res = roots.query(None, None, None, "asc", 0, 50, t(2, 12))
    ids = [p["id"] for p in res["posts"]]
    assert "mid" not in ids  # posted day2 15:00, still future
    assert "late" not in ids
    assert res["clipped_until"].startswith("2026-03-02T12:00")


def test_query_sort_comments_ranks_current_prefix_not_label(roots):
    now = t(2, 12)
    res = roots.query(None, None, None, "desc", 0, 50, now, sort="comments")
    assert res["sort"] == "comments"
    counts = [p["num_comments"] for p in res["posts"]]
    assert counts == sorted(counts, reverse=True)
    # ranking uses the snapshot at now, which matches each post's own prefix
    for p in res["posts"]:
        assert p["num_comments"] == roots._cascades.descendants_upto(
            p["id"], now.timestamp())


# -- cascade prefix -------------------------------------------------------------------

def test_prefix_grows_hides_score_and_future(cascades):
    post = ts(2, 6)
    # first 3 comments at post+60,+65,+70; a prefix at post+67s sees 2
    pre = cascades.prefix("big", post + 67, 0, 500)
    assert pre["n_comments"] == 2
    assert all(n["created_utc"] <= post + 67 for n in pre["nodes"])
    assert all("score" not in n for n in pre["nodes"])  # never leak score
    # structure is present: parent_id on comments
    comments = [n for n in pre["nodes"] if n["kind"] == "comment"]
    assert all("parent_id" in n for n in comments)
    # far later, the full within-horizon tree (256 comments) is revealed
    full = cascades.prefix("big", post + 12 * 3600, 0, 5000)
    assert full["n_comments"] == 256


def test_prefix_paginates(cascades):
    post = ts(2, 6)
    p0 = cascades.prefix("big", post + 12 * 3600, 0, 10)
    assert len(p0["nodes"]) == 10 and p0["has_more"] is True
    p1 = cascades.prefix("big", post + 12 * 3600, 10, 10)
    assert p0["nodes"][0]["id"] != p1["nodes"][0]["id"]


# -- settlement math ------------------------------------------------------------------

def test_timely_tail_exact_math(scorer):
    scorer.record_notification(t(2, 6), "big")  # at post, weight 1
    scorer.close_all()
    r = scorer._settled[0]
    assert r.descendants == 256 and r.pop == 9.0
    assert r.weight == 1.0
    assert r.status == "tail"  # 256 >= tail_min_desc 50


def test_weight_halfway(scorer):
    scorer.record_notification(t(2, 18), "big")  # +12h = half horizon
    scorer.close_all()
    r = scorer._settled[0]
    assert r.weight == pytest.approx(0.5)
    assert r.status == "tail"


def test_nontail_wastes_a_slot(scorer):
    scorer.record_notification(t(2, 9), "dud")  # final desc 2 < 50
    scorer.close_all()
    r = scorer._settled[0]
    assert r.status == "nontail"
    assert r.weight == 1.0  # weight is timing; class membership is the label


# -- decay_hours: steeper decay window, decoupled from the reveal horizon --------------

@pytest.fixture
def steep_scorer(built):
    cfg = make_reddit_config(built, task=dict(decay_hours=6.0))
    return RedditPopularityTask.from_run_config(cfg, built.parent).scorer


def test_steep_decay_math(steep_scorer):
    steep_scorer.record_notification(t(2, 9), "big")  # +3h = half the window
    steep_scorer.close_all()
    r = steep_scorer._settled[0]
    assert r.weight == pytest.approx(0.5)
    assert r.status == "tail"


def test_weight_clamps_to_zero_after_window(steep_scorer):
    # +8h: past the 6h decay window but well before the 24h reveal, so the
    # recommendation is accepted yet earns no credit — a wasted cap slot
    steep_scorer.record_notification(t(2, 14), "big")
    steep_scorer.close_all()
    r = steep_scorer._settled[0]
    assert r.weight == 0.0
    assert r.status == "tail"  # class is the label; the credit is 0


def test_decay_hours_validation(built):
    for bad in (0, -1, 25):  # horizon_hours is 24
        with pytest.raises(ValueError, match="decay_hours"):
            RedditPopularityTask.from_run_config(
                make_reddit_config(built, task=dict(decay_hours=bad)),
                built.parent)


# -- rejection rules (all free) -------------------------------------------------------

def test_reject_unknown_and_future_same_message(scorer):
    with pytest.raises(NotificationError) as e1:
        scorer.record_notification(t(2, 12), "nope")
    with pytest.raises(NotificationError) as e2:
        scorer.record_notification(t(2, 12), "mid")  # posted day2 15:00, future
    assert str(e1.value) == str(e2.value).replace("mid", "nope") or \
        "unknown or not yet posted" in str(e2.value)


def test_reject_history(scorer):
    with pytest.raises(NotificationError, match="predates"):
        scorer.record_notification(t(2, 0, 30), "hist")  # posted day1


def test_reject_already_revealed(scorer):
    with pytest.raises(NotificationError, match="already revealed"):
        scorer.record_notification(t(3, 7), "big")  # big revealed day3 06:00


def test_reject_duplicate(scorer):
    scorer.record_notification(t(2, 10), "dud")
    with pytest.raises(NotificationError, match="already recommended"):
        scorer.record_notification(t(2, 11), "dud")


def test_reject_cap(scorer):
    for rid in ("big", "dud", "self"):
        scorer.record_notification(t(2, 16), rid)
    with pytest.raises(NotificationError, match="cap reached"):
        scorer.record_notification(t(2, 16), "mid")


# -- close_due vs close_all -----------------------------------------------------------

def test_late_reveal_only_settles_at_close_all(scorer):
    scorer.record_notification(t(13, 12), "late")  # reveals day14 12:00 > sim_end
    assert scorer.close_due(t(14, 0)) == []        # not due at sim_end
    events = scorer.close_all()
    assert len(events) == 1 and scorer._settled[0].root_id == "late"


# -- payload validation ---------------------------------------------------------------

def test_payload_validation(task):
    for bad in ({}, {"root_id": 123}, {"root_id": ""}, {"root_id": None}):
        with pytest.raises(NotificationError, match="root_id"):
            task.record_notification(t(2, 12), bad)


# -- oracle & report ------------------------------------------------------------------

def test_oracle_feed_own_recs_plus_unrecommended_tails(task):
    # own recs stream regardless of class; of the rest, only tail posts
    # (the recall failures) appear — nontail non-recommended stay silent
    task.scorer.record_notification(t(2, 9), "dud")
    task.close_all()
    out = task.oracle_outcomes(None, t(14, 0))
    recs = [o for o in out if o["kind"] == "recommendation"]
    posts = {o["root_id"]: o for o in out if o["kind"] == "post"}
    assert [o["root_id"] for o in recs] == ["dud"]  # own nontail: streamed
    assert recs[0]["growth"]["24h"] == 2
    assert "big" in posts  # the missed tail is visible evidence
    assert posts["big"]["status"] == "tail"
    # trajectory revealed at settlement: final age equals the label
    assert posts["big"]["growth"]["24h"] == 256
    assert not {"self", "mid", "hist"} & set(posts)  # nontail: silent


def test_metrics_and_report_shape(task):
    task.scorer.record_notification(t(2, 6), "big")   # tail, weight 1
    task.scorer.record_notification(t(2, 9), "dud")   # nontail
    task.close_all()
    m = task.metrics()
    # tail universe = {big, late} (both >= 50 within the window)
    assert m["tail_posts"] == 2
    assert m["primary"]["name"] == "twr_at_cap"
    assert m["primary"]["direction"] == "max"
    assert m["primary"]["value"] == pytest.approx(0.5)  # 1.0 weight / 2
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["tail_hits"] == 1
    # ceiling: 2 tail posts across 2 days, cap 3 -> all catchable
    assert m["twr_ceiling_at_cap"] == pytest.approx(1.0)
    rep = task.report()
    assert rep["daily_outcomes"]["2026-03-02"]["tail"] == 1
    assert rep["daily_outcomes"]["2026-03-02"]["nontail"] == 1
    assert len(rep["recommendations"]) == 2
