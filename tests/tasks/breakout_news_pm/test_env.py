"""Store visibility semantics (prices, news index)."""

import json
from datetime import timedelta
from pathlib import Path

import pytest

from tasks.breakout_news_pm.task import PriceStore, _iso_ts as _iso
from tests.tasks.breakout_news_pm.conftest import t, ts



# -- prices ---------------------------------------------------------------------------


def test_price_visibility_clamp(built, cfg):
    prices = PriceStore(built, ["m1"], cfg.price_delay_minutes)
    # the Mar-3 12:00:30 jump is not visible at 12:05 (delay 10 min) ...
    q = prices.query("m1", t(3, 11), t(3, 13), now=t(3, 12, 5))
    assert q["changes"]["p"] == []
    assert q["level_at_start"] == 0.50
    # ... and visible at 12:11
    q = prices.query("m1", t(3, 11), t(3, 13), now=t(3, 12, 11))
    assert q["changes"]["p"] == [0.70]
    assert q["sparse"] == "changes"


def test_price_level_at_start_forward_fill(built, cfg):
    prices = PriceStore(built, ["m1"], cfg.price_delay_minutes)
    # window starting mid-flat-stretch carries the entering level
    q = prices.query("m1", t(5, 0), t(6, 0), now=t(7, 0))
    assert q["level_at_start"] == 0.70
    assert q["changes"]["p"] == []


def _dense_store(root, cfg) -> PriceStore:
    d = root / "gm"
    (d / "prices").mkdir(parents=True)
    (d / "prices" / "m1.json").write_text(json.dumps({
        "grid_hours": 1 / 60, "sparse": "changes",
        "points": [[ts(2, 0, 10), 0.52], [ts(2, 0, 40), 0.55],
                   [ts(2, 1, 30), 0.55], [ts(2, 2, 15), 0.60]]}))
    return PriceStore(d, ["m1"], cfg.price_delay_minutes)


def test_price_grid_minutes_last_change_per_bucket(tmp_path, cfg):
    prices = _dense_store(tmp_path, cfg)
    q = prices.query("m1", t(2, 0), t(2, 3), now=t(3, 0), grid_minutes=60)
    assert q["grid_minutes"] == 60
    # bucket 0 keeps its LAST change (0.55 at 00:40, actual timestamp);
    # bucket 1's 0.55 duplicates the running level and collapses
    assert q["changes"]["time"] == [_iso(ts(2, 0, 40)), _iso(ts(2, 2, 15))]
    assert q["changes"]["p"] == [0.55, 0.60]


def test_price_grid_minutes_dedupes_against_level_at_start(tmp_path, cfg):
    prices = _dense_store(tmp_path, cfg)
    q = prices.query("m1", t(2, 1), t(2, 3), now=t(3, 0), grid_minutes=60)
    # entering level is 0.55; the first bucket's last change (0.55) is
    # no change under forward-fill
    assert q["level_at_start"] == 0.55
    assert q["changes"]["p"] == [0.60]


def test_price_grid_minutes_default_path_identical(built, cfg):
    prices = PriceStore(built, ["m1"], cfg.price_delay_minutes)
    assert prices.query("m1", t(3, 11), t(3, 13), now=t(3, 12, 11),
                        grid_minutes=1) == \
        prices.query("m1", t(3, 11), t(3, 13), now=t(3, 12, 11))


def test_price_grid_minutes_validation(built, cfg):
    prices = PriceStore(built, ["m1"], cfg.price_delay_minutes)
    for bad in (0, 0.5, -1, True, "60", None):
        with pytest.raises(ValueError, match="grid_minutes"):
            prices.query("m1", t(2), t(3), now=t(4), grid_minutes=bad)


# There is no condition grammar: event-driven waiting is authored
# gatekeeper code whose fetches bill like any other call — covered by
# test_authored_programs-style tests and the harness suites.


# -- no rate limits ------------------------------------------


def test_news_and_price_apis_are_unlimited_by_default():
    from datetime import datetime, timezone

    from harness.limits import RateLimiter
    from tasks.breakout_news_pm.task import BreakoutNewsPMConfig

    tcfg = BreakoutNewsPMConfig(markets=[{
        "market_id": "m1", "start": "2026-03-01T00:00:00Z",
        "end": "2026-03-02T00:00:00Z"}])
    assert tcfg.news_rate_limit == {"window": "none"}
    assert tcfg.price_rate_limit == {"window": "none"}
    lim = RateLimiter("news api", tcfg.news_rate_limit)
    now = datetime(2026, 3, 1, tzinfo=timezone.utc)
    for _ in range(10_000):  # far past any former per-minute budget
        lim.consume(now)
    assert lim.n_rejected == 0 and lim.n_allowed == 10_000
    assert lim.doc() == "none (unlimited)"
    assert lim.stats()["limit"] == "none (unlimited)"
    # a run can still impose one from its task section
    strict = BreakoutNewsPMConfig(markets=tcfg.markets, news_rate_limit={
        "window": "fixed_window", "window_seconds": 60, "budget": 2})
    lim = RateLimiter("news api", strict.news_rate_limit)
    lim.consume(now); lim.consume(now)
    import pytest as _pt
    from harness.limits import RateLimited
    with _pt.raises(RateLimited):
        lim.consume(now)


# -- configurable oracle source ------------------------------------------


def test_attributions_path_selects_oracle_source(built, cfg, tmp_path):
    """`task.attributions_path` swaps the hindsight labeler feeding gold;
    default stays data_dir/attributions.jsonl."""
    from tasks.breakout_news_pm.task import BreakoutNewsPMConfig, load_breakpoints

    default = load_breakpoints(built, cfg)
    dcfg = BreakoutNewsPMConfig(**{**cfg.model_dump(), "data_dir": built})
    assert dcfg.resolve_attributions_path() == built / "attributions.jsonl"
    # alternate labeler: every breakpoint unattributed
    alt = tmp_path / "attributions_alt.jsonl"
    with open(built / "attributions.jsonl") as f, open(alt, "w") as g:
        for line in f:
            e = json.loads(line)
            g.write(json.dumps({**e, "no_attribution": True, "groups": []}) + "\n")
    acfg = BreakoutNewsPMConfig(**{**cfg.model_dump(), "attributions_path": alt})
    assert acfg.resolve_attributions_path() == alt
    swapped = load_breakpoints(built, acfg, acfg.resolve_attributions_path())
    assert [b.market_id for b in swapped] == [b.market_id for b in default]
    assert any(b.winnable for b in default)
    assert not any(b.winnable for b in swapped)
    # relative paths resolve against the repo root passed at load
    rel = BreakoutNewsPMConfig(**{**cfg.model_dump(), "data_dir": built,
                                  "attributions_path": Path("x/attr.jsonl")})
    assert rel.resolve_attributions_path(Path("/repo")) == Path("/repo/x/attr.jsonl")
