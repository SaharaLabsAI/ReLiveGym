"""Ground-truth publish clocks (build_index.py derive_pub_ts).

v2 (--ts-v2): self-reported publish time counts only when the crawl
corroborates it within 2 h (V2_CORROBORATION_S); otherwise the
crawl time is the clock. v3 (--ts-v3): same corroboration rule, but the
uncorroborated fallback is crawl − 2 h (one crawl cycle before
discovery), and self-reports inside the pinned crawler outage are
trusted as-is (no corroboration was possible there). v1 (min rule) must
stay untouched — every bnpm run uses it.

Also an end-to-end check: a mini index built with the v3 derivation
serves re-dated articles through the UNMODIFIED NewsSearch engine —
range filters, earliest_match, and get_article all follow the v3 clock
natively, so pollers see backdated articles as new at crawl − 2 h.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tasks.breakout_news_pm.news.build_index import (
    OUTAGE_END, OUTAGE_START, V2_CORROBORATION_S, build_schema,
    derive_pub_ts,
)
from tasks.breakout_news_pm.news.search import NewsSearch

T0 = datetime(2026, 3, 1, tzinfo=timezone.utc)


def h(hours: float) -> int:
    return int((T0 + timedelta(hours=hours)).timestamp())


def iso(hours: float) -> str:
    return (T0 + timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")


# -- derivation unit tests ------------------------------------------------------------


def test_frozen_constants():
    assert V2_CORROBORATION_S == 7200
    assert OUTAGE_START == int(
        datetime(2026, 4, 1, 4, tzinfo=timezone.utc).timestamp())
    assert OUTAGE_END == int(
        datetime(2026, 4, 5, 20, tzinfo=timezone.utc).timestamp())


@pytest.mark.parametrize("pub_h,crawl_h,v1_h,v2_h,v3_h", [
    (0, 1, 0, 0, 0),           # corroborated within 2h: self-reported stands
    (0, 2, 0, 0, 0),           # exactly at tolerance: stands (v3 fallback
                               #   crawl-2h coincides anyway)
    (0, 2.001, 0, 2.001, 0.001),  # just beyond: v2 crawl, v3 crawl-2h
    (2, 80, 2, 80, 78),        # the backdated "Paramount Skydance" shape
    (5, 1, 1, 1, 1),           # self-reported AFTER crawl: min rule caps
])
def test_derive_pub_ts(pub_h, crawl_h, v1_h, v2_h, v3_h):
    assert derive_pub_ts(h(crawl_h), iso(pub_h), "v1") == h(v1_h)
    assert derive_pub_ts(h(crawl_h), iso(pub_h), "v2") == h(v2_h)
    assert derive_pub_ts(h(crawl_h), iso(pub_h), "v3") == h(v3_h)


OUT0_H = (OUTAGE_START - h(0)) / 3600  # outage start in hours after T0


@pytest.mark.parametrize("pub_off,crawl_off,expect", [
    (1, 130, "pub"),        # pub inside outage, crawled in the backlog: trusted
    (111.9, 114, "pub"),    # pub just before resume, lag > 2h: still trusted
    (-0.001, 130, "lag"),   # pub just BEFORE outage start: normal v3 fallback
    (112.1, 118, "lag"),    # pub after resume, uncorroborated: normal fallback
    (5, 6, "pub"),          # pub inside outage but corroborated (rare
                            #   trickle crawl): stands via the 2h rule
])
def test_v3_outage_window(pub_off, crawl_off, expect):
    pub_h, crawl_h = OUT0_H + pub_off, OUT0_H + crawl_off
    got = derive_pub_ts(h(crawl_h), iso(pub_h), "v3")
    want = h(pub_h) if expect == "pub" else h(crawl_h) - V2_CORROBORATION_S
    assert got == want


def test_no_self_report_is_crawl_time_all_versions():
    assert derive_pub_ts(h(3), None, "v1") == h(3)
    assert derive_pub_ts(h(3), "", "v2") == h(3)
    assert derive_pub_ts(h(3), None, "v3") == h(3)


# -- mini v3 index through the unmodified engine --------------------------------------

CORPUS = [  # (id, pub_h, crawl_h, title)
    ("n_ok", 0, 1, "verified alpha story"),
    ("n_back", 2, 80, "backdated alpha story"),
    ("n_mid", 30, 30.5, "midweek alpha story"),
]


@pytest.fixture(scope="module")
def v3_engine(tmp_path_factory) -> NewsSearch:
    import tantivy

    root = tmp_path_factory.mktemp("news") / "tantivy_index_v3"
    root.mkdir()
    index = tantivy.Index(build_schema(), path=str(root))
    writer = index.writer(heap_size=15_000_000)
    for nid, pub, crawl, title in CORPUS:
        pub_ts = derive_pub_ts(h(crawl), iso(pub), "v3")
        writer.add_document(tantivy.Document(
            id=nid, url=f"https://x.com/{nid}", domain="x.com", title=title,
            description=title, text=title, date_publish="",
            published_at=iso(pub), pub_ts=pub_ts,
            pub_date=datetime.fromtimestamp(
                pub_ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            warc_date=iso(crawl)))
    writer.commit()
    index.reload()
    return NewsSearch(str(root))


def test_backdated_hidden_until_near_crawl(v3_engine):
    hits = v3_engine.search("alpha", date_from=iso(0), date_to=iso(10))
    assert {r["news_id"] for r in hits} == {"n_ok"}


def test_backdated_appears_as_new_at_crawl_minus_2h(v3_engine):
    # the polling window covering crawl − 2h catches it — the
    # missed-discovery approximation of a query-time mode is
    # structurally impossible here
    hits = v3_engine.search("alpha", date_from=iso(77), date_to=iso(79))
    assert {r["news_id"] for r in hits} == {"n_back"}
    assert hits[0]["published"] == iso(78)


def test_earliest_match_on_v3_clock(v3_engine):
    assert v3_engine.earliest_match("alpha", h(1.5), h(100)) == h(30)
    assert v3_engine.earliest_match("backdated", h(0), h(100)) == h(78)
    assert v3_engine.earliest_match("alpha", h(1.5), h(10)) is None


def test_get_article_serves_v3_clock(v3_engine):
    art = v3_engine.get_article("n_back")
    assert art["pub_ts"] == h(78)
    assert art["pub_date"] == iso(78)
    assert v3_engine.get_article("n_ok")["pub_ts"] == h(0)
