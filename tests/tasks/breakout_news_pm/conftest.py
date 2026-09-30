"""Synthetic mini-world for breakout_news_pm tests.

One market ("m1", minute grid, March 2026), two ground-truth breakpoints
(bp1 winnable/up at Mar-3 12:00, bp2 no-attribution/down at Mar-10 00:00),
and a stub news store (the scorer only needs published_ts; search/article
visibility against the real tantivy index is covered by test_build_real).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tasks.breakout_news_pm.task import (
    BreakoutNewsPMConfig,
    BreakoutNewsPMTask,
    NewsStore,
    PriceStore,
    Scorer,
    load_breakpoints,
)

UTC = timezone.utc


def t(day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(2026, 3, day, hour, minute, tzinfo=UTC)


def ts(day: int, hour: int = 0, minute: int = 0) -> float:
    return t(day, hour, minute).timestamp()


# news_id -> pub_ts. bp1 (t_start Mar-3 12:00, W=24h) gold window:
# [Mar-2 12:00, Mar-3 12:00).
NEWS = {
    "n-alpha": ts(2, 18),      # gold for bp1 (lead 18 h)
    "n-beta": ts(3, 11),       # gold for bp1 (lead 1 h)
    "n-early": ts(1, 0),       # cited but > W before bp1 -> dropped early
    "n-atstart": ts(3, 12),    # cited but pub == t_start -> dropped late
    "n-lowconf": ts(3, 0),     # cited by a 0.5-confidence group -> filtered
    "n-other": ts(2, 0),       # never cited: timing-only material
}


class StubNews:
    def published_ts(self, news_id: str) -> float | None:
        return NEWS.get(news_id)


def write_world(root: Path) -> Path:
    built = root / "built"
    (built / "prices").mkdir(parents=True)
    (built / "markets.jsonl").write_text(json.dumps({
        "market_id": "m1", "question": "Will Beta win?", "category": "Politics",
        "event_title": None, "volume": 1e6, "description": "Resolves Yes.",
        "grid_minutes": 1, "start_date": None, "end_date": None,
        "closed_time": None}) + "\n")
    # sparse change-series: flat 0.50, jump to 0.70 at Mar-3 12:00:30,
    # drift down 0.60 at Mar-10 00:00:30
    (built / "prices" / "m1.json").write_text(json.dumps({
        "grid_hours": 1 / 60, "sparse": "changes",
        "points": [[ts(1, 0), 0.50], [ts(3, 12, 0) + 30, 0.70],
                   [ts(10, 0, 0) + 30, 0.60]]}))
    with open(built / "breakpoints.jsonl", "w") as f:
        f.write(json.dumps({
            "market_id": "m1", "date": "2026-03-03", "dp": 0.2, "z": 3.0,
            "p_prev": 0.5, "p": 0.7, "t_move_start": ts(3, 12),
            "t_move_end": ts(3, 12, 1), "step_frac": 1.0,
            "localization": "minute"}) + "\n")
        f.write(json.dumps({
            "market_id": "m1", "date": "2026-03-10", "dp": -0.1, "z": 2.5,
            "p_prev": 0.7, "p": 0.6, "t_move_start": ts(10, 0),
            "t_move_end": ts(10, 0, 1), "step_frac": 1.0,
            "localization": "minute"}) + "\n")
    with open(built / "attributions.jsonl", "w") as f:
        f.write(json.dumps({
            "market_id": "m1", "date": "2026-03-03", "no_attribution": False,
            "groups": [
                {"story": "Alpha drops out", "confidence": 0.9,
                 "likely_reports_move": False,
                 "articles": [{"news_id": "n-alpha", "pub_ts": NEWS["n-alpha"]},
                              {"news_id": "n-beta", "pub_ts": NEWS["n-beta"]},
                              {"news_id": "n-early", "pub_ts": NEWS["n-early"]},
                              {"news_id": "n-atstart",
                               "pub_ts": NEWS["n-atstart"]}]},
                {"story": "Speculative driver", "confidence": 0.5,
                 "likely_reports_move": False,
                 "articles": [{"news_id": "n-lowconf",
                               "pub_ts": NEWS["n-lowconf"]}]},
            ]}) + "\n")
        f.write(json.dumps({
            "market_id": "m1", "date": "2026-03-10", "no_attribution": True,
            "groups": []}) + "\n")
    return built


def make_config(**overrides) -> BreakoutNewsPMConfig:
    kw = dict(
        markets=[{"market_id": "m1", "start": t(2), "end": t(13)}],
        cost=dict(price_call=0.0, news_search_call=0.01),
    )
    kw.update(overrides)
    return BreakoutNewsPMConfig(**kw)


@pytest.fixture
def built(tmp_path) -> Path:
    return write_world(tmp_path)


@pytest.fixture
def cfg() -> BreakoutNewsPMConfig:
    return make_config()


@pytest.fixture
def breakpoints(built, cfg):
    return load_breakpoints(built, cfg)


@pytest.fixture
def scorer(cfg, breakpoints) -> Scorer:
    return Scorer(cfg, StubNews(), breakpoints)


# mini-corpus articles for wake tests: (id, pub_ts, title, description) —
# indexed with the production schema so news_match evaluates the canonical
# search engine, exactly as in a real run
CORPUS = [
    ("w-1", ts(2, 0, 7), "Fed signals patience on rates", "no cuts expected"),
    ("w-2", ts(2, 3, 2), "Alpha drops out of race", "campaign ends"),
    ("w-3", ts(2, 3, 4), "Fed rate decision looms", "markets await the Fed"),
]


def build_mini_index(root: Path) -> Path:
    import tantivy

    from tasks.breakout_news_pm.news.build_index import build_schema

    idx_dir = root / "tantivy_index"
    idx_dir.mkdir()
    index = tantivy.Index(build_schema(), path=str(idx_dir))
    writer = index.writer(heap_size=15_000_000)
    for nid, pub, title, desc in CORPUS:
        iso = datetime.fromtimestamp(pub, tz=UTC).strftime(
            "%Y-%m-%dT%H:%M:%SZ")
        writer.add_document(tantivy.Document(
            id=nid, url=f"https://x.com/{nid}", domain="x.com", title=title,
            description=desc, text=desc, date_publish="", published_at=iso,
            pub_ts=int(pub), pub_date=iso, warc_date=iso))
    writer.commit()
    index.reload()
    return idx_dir


@pytest.fixture
def task(built, cfg, scorer) -> BreakoutNewsPMTask:
    news = NewsStore(build_mini_index(built))
    prices = PriceStore(built, ["m1"], cfg.price_delay_minutes)
    meta = {"m1": {"question": "Will Beta win?", "category": "Politics",
                   "event_title": None, "description": "", "grid_minutes": 1}}
    return BreakoutNewsPMTask(cfg, meta, prices, news, scorer)
