"""Calibration pins against the real built world (skipped where the
gitignored artifacts are absent). Rebuild: python3 data/build.py."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

TASK_DIR = Path(__file__).resolve().parents[3] / "tasks" / "breakout_news_pm"
BUILT = TASK_DIR / "data" / "built"

pytestmark = pytest.mark.skipif(
    not (BUILT / "breakpoints.jsonl").exists(),
    reason="built world not present (data/build.py output is gitignored)")


def test_build_stats_pins():
    s = json.loads((BUILT / "build_stats.json").read_text())
    assert s["markets"] == 350
    assert s["breakpoints"] == 1525
    assert s["hourly_fallbacks"] == 109
    assert s["attributed_episodes"] == 953
    assert s["groups"] == 1073
    assert s["unresolved_prefixes"] == 0
    g = s["gold_preview_defaults"]
    assert (g["attr_threshold"], g["w_hours"]) == (0.6, 24)
    assert g["gold_groups"] == 598
    assert g["gold_articles"] == 1410
    assert g["winnable_breakpoints"] == 580


def test_load_breakpoints_matches_pins():
    from tasks.breakout_news_pm.task import BreakoutNewsPMConfig, load_breakpoints

    cfg = BreakoutNewsPMConfig(
        markets=[{"market_id": "616902", "start": "2026-03-01T00:00:00Z",
                  "end": "2026-03-28T00:00:00Z"},
                 {"market_id": "678777", "start": "2026-03-01T00:00:00Z",
                  "end": "2026-03-28T00:00:00Z"}],
        cost={"news_search_call": 0.01, "miss_cap": 50.0, "false_alarm": 10.0})
    bps = load_breakpoints(BUILT, cfg)
    # the w10 roster: 11 breakpoints, 9 winnable
    assert len(bps) == 11
    assert sum(b.winnable for b in bps) == 9
    for b in bps:
        assert b.direction in ("up", "down")
        for pub in b.gold.values():
            assert b.t_start - 24 * 3600 <= pub < b.t_start


@pytest.mark.skipif(
    not (TASK_DIR / "news" / "tantivy_index_v3").exists(),
    reason="tantivy index not present")
def test_news_visibility_clamp_real_index():
    from tasks.breakout_news_pm.task import NewsStore

    news = NewsStore(TASK_DIR / "news" / "tantivy_index_v3")
    now = datetime(2026, 3, 5, 12, 0, tzinfo=timezone.utc)
    hits = news.search("Federal Reserve", "2026-03-01", "2026-12-31",
                       now=now, top_k=10, offset=0)
    assert hits
    assert all(h["published"] <= "2026-03-05T12:00:00" for h in hits)
    # a hit is fetchable now, hidden before its publish time
    h0 = hits[0]
    assert news.get_article(h0["news_id"], now) is not None
    before = datetime.fromisoformat(
        h0["published"].replace("Z", "+00:00")).replace(tzinfo=timezone.utc)
    from datetime import timedelta

    assert news.get_article(h0["news_id"], before - timedelta(seconds=1)) is None
