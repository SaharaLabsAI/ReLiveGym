"""daily_reddit_digest env surface: the sibling's read tools + digest /
digest_status, and a no-LLM end-to-end through the Sim."""

from __future__ import annotations

import asyncio
import pytest

from harness.api import build_tool_registry
from harness.task import NotificationError
from harness.runtime import Sim
from tasks.daily_reddit_digest.task import DailyRedditDigestTask
from tests.tasks.daily_reddit_digest.conftest import POP, make_digest_config, t


def _sim(built, tmp_path, tm="A"):
    cfg = make_digest_config(
        built, run_id=f"digest-{tm}",
        cell=dict(tm=tm, tlrn="none", sig="none", alg="none"),
        agent=dict(scaffold="react" if tm in "AB" else "task:cron_react"))
    run_dir = tmp_path / f"run-{tm}"
    workspace = run_dir / "workspace"
    workspace.mkdir(parents=True)
    task = DailyRedditDigestTask.from_run_config(cfg, built.parent)
    return Sim(cfg, run_dir, workspace, task)


def call(sim, name, **args):
    entry = build_tool_registry(sim).get(name)
    assert entry is not None, f"tool {name!r} not provisioned"
    return asyncio.run(entry[1](args))


def test_tool_surface(built, tmp_path):
    reg = build_tool_registry(_sim(built, tmp_path))
    assert {"list_posts", "get_post", "get_cascade", "digest",
            "digest_status"} <= set(reg)
    assert "recommend" not in reg and "quota" not in reg
    assert "action" in reg["digest"][0].tags
    assert reg["digest_status"][0].price == "free"


def test_read_tools_serve_live_counts_and_comment_sort(built, tmp_path):
    sim = _sim(built, tmp_path)
    sim.clock.advance_to(t(3, 12, 30))
    page = call(sim, "list_posts", since="2026-03-02T12:00:00Z",
                until="2026-03-03T12:00:00Z", sort="comments", order="desc")
    counts = [p["num_comments"] for p in page["posts"]]
    assert counts == sorted(counts, reverse=True)
    # `until` is inclusive in the sibling's listing: edge_at (= a:00) is
    # listed even though it falls outside day Mar 3's half-open window
    assert page["total_hits"] == 2 + 10 + 3 + 1
    assert all(p["descendants"] is None for p in page["posts"]
               if p["created_utc"] + 86400 > t(3, 12, 30).timestamp())
    assert sim.ledger.total_cost() == pytest.approx(0.00024)


def test_digest_status_is_clock_only(built, tmp_path):
    sim = _sim(built, tmp_path)
    sim.clock.advance_to(t(3, 9))
    s = call(sim, "digest_status")
    assert s["day"] is None and s["delivered"] is False
    assert s["next_window"] == ["2026-03-03T12:00:00Z", "2026-03-03T13:00:00Z"]
    sim.clock.advance_to(t(3, 12, 15))
    s = call(sim, "digest_status")
    assert s["day"] == "2026-03-03" and s["delivered"] is False
    assert s["candidate_window"] == ["2026-03-02T12:00:00Z",
                                     "2026-03-03T12:00:00Z"]
    call(sim, "digest", root_ids=POP)
    s = call(sim, "digest_status")
    assert s["delivered"] is True
    assert not any(k for k in s if "post" in k or "label" in k)


def test_digest_records_and_logs_notify(built, tmp_path):
    sim = _sim(built, tmp_path)
    sim.clock.advance_to(t(3, 12, 15))
    res = call(sim, "digest", root_ids=POP)
    assert res["status"] == "accepted" and res["day"] == "2026-03-03"
    notifies = [e for e in sim.ledger.events if e["type"] == "notify"]
    assert len(notifies) == 1 and notifies[0]["payload"]["root_ids"] == POP
    assert sim.ledger.total_cost() == 0.0
    with pytest.raises(NotificationError, match="already delivered"):
        call(sim, "digest", root_ids=POP)


def test_e2e_scripted_daily_delivery_scores_one(built, tmp_path):
    """A scripted actor that wakes at 12:30 each scored day, lists the
    window sorted by current comments, and digests the first 10 ids at or
    above the bar, scores 1.0 with no LLM."""
    sim = _sim(built, tmp_path)
    sc = sim.task.scorer
    for d in sc.days:
        sim.clock.advance_to(t(d.day, 12, 30))
        sim.book_due_outcomes()
        lo, hi = sc.window_for(d)
        page = call(sim, "list_posts",
                    since=t(d.day - 1, 12).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    until=t(d.day, 12).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    sort="comments", order="desc")
        ids = [p["id"] for p in page["posts"]
               if p["num_comments"] >= 50 and lo <= p["created_utc"] < hi][:10]
        if ids:  # Mar 6, 7 have no qualifying posts in this world
            call(sim, "digest", root_ids=ids)
    sim.clock.advance_to(sim.cfg.sim_end)
    sim.book_all_outcomes()
    m = sim.task.metrics()
    # Mar 3 and Mar 5 have >= 10 popular posts (score 1); Mar 4's window
    # holds only edge_at (score 0.1); Mar 6/7 are empty (miss)
    assert m["days_delivered"] == 3 and m["days_full"] == 2
    assert m["primary"]["value"] == pytest.approx(2.1 / 5)
    outcomes = [e for e in sim.ledger.events if e["type"] == "outcome"]
    assert {e["ref"]: e["status"] for e in outcomes} == {
        "2026-03-03": "ok", "2026-03-04": "ok", "2026-03-05": "ok",
        "2026-03-06": "miss", "2026-03-07": "miss"}


def test_e2e_late_delivery_is_rejected_and_scores_zero(built, tmp_path):
    sim = _sim(built, tmp_path)
    sim.clock.advance_to(t(3, 13, 30))
    with pytest.raises(NotificationError, match="not in a delivery window"):
        call(sim, "digest", root_ids=POP)
    sim.clock.advance_to(sim.cfg.sim_end)
    sim.book_all_outcomes()
    assert sim.task.metrics()["primary"]["value"] == 0.0
