"""BM25 news retrieval over the CC-NEWS tantivy index.

Shared by the hindsight-labeling agent (now) and the prediction env (later).
The env wraps `search()` with its visibility clamp (date_to = min(date_to,
now)); nothing labeling-specific lives here.

Semantics:
- query: tantivy query syntax over title/description/text (free terms are
  OR'd across those fields; quoted phrases and AND/OR work).
- date_from/date_to: inclusive ISO dates or datetimes, filtered on the
  article's `pub_ts` — the canonical publish clock, derived at index build
  as the earlier of the page's own metadata publish time and crawler
  discovery (the derivation is internal; results expose it as `published`).
- Deterministic ranking: BM25 score desc, tie-break id asc (stable across
  runs on the same index artifact).

CLI smoke test:  python3 search.py <index_dir> "iran enrichment" 2026-04-10 2026-04-14
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone

import tantivy

_FIELDS = ["title", "description", "text"]


def _epoch(s: str, end_of_day: bool) -> int:
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    ts = int(dt.timestamp())
    if end_of_day and len(s) <= 10:
        ts += 86399
    return ts


class NewsSearch:
    def __init__(self, index_dir: str):
        self.index = tantivy.Index.open(index_dir)
        self.searcher = self.index.searcher()

    def search(self, query: str, date_from: str | None = None,
               date_to: str | None = None, top_k: int = 20,
               offset: int = 0) -> list[dict]:
        q = self.index.parse_query(query, _FIELDS)
        if date_from or date_to:
            lo = _epoch(date_from, False) if date_from else 0
            hi = _epoch(date_to, True) if date_to else 2**62
            rq = self.index.parse_query(f"pub_ts:[{lo} TO {hi}]")
            q = tantivy.Query.boolean_query(
                [(tantivy.Occur.Must, q), (tantivy.Occur.Must, rq)])
        # over-fetch so the deterministic (score, id) re-sort is stable at
        # the offset boundary
        hits = self.searcher.search(q, limit=offset + top_k + 20).hits
        docs = []
        for score, addr in hits:
            d = self.searcher.doc(addr).to_dict()
            docs.append((score, d["id"][0], d))
        docs.sort(key=lambda x: (-x[0], x[1]))
        out = []
        for score, _id, d in docs[offset:offset + top_k]:
            out.append({
                "news_id": _id,
                "title": d["title"][0],
                "domain": d["domain"][0],
                "published": d["pub_date"][0],
                "snippet": (d["description"][0] or d["text"][0][:300])[:300],
                "score": round(score, 4),
            })
        return out

    def _any_match(self, q, lo_ts: int, hi_ts: int) -> bool:
        """Does any article with pub_ts in (lo_ts, hi_ts] match q?"""
        rq = self.index.parse_query(f"pub_ts:[{lo_ts + 1} TO {hi_ts}]")
        bq = tantivy.Query.boolean_query(
            [(tantivy.Occur.Must, q), (tantivy.Occur.Must, rq)])
        return bool(self.searcher.search(bq, limit=1).hits)

    def earliest_match(self, query: str, lo_ts: float,
                       hi_ts: float) -> int | None:
        """Earliest pub_ts in (lo_ts, hi_ts] of an article matching `query`
        (same syntax as search()) — or None. Bisection over existence
        probes (~log2(span) index queries), exactly equivalent to scanning
        forward in time."""
        q = self.index.parse_query(query, _FIELDS)
        lo_i, hi_i = int(lo_ts), int(hi_ts)
        if hi_i <= lo_i or not self._any_match(q, lo_i, hi_i):
            return None
        lo_b, hi_b = lo_i + 1, hi_i
        while lo_b < hi_b:
            mid = (lo_b + hi_b) // 2
            if self._any_match(q, lo_i, mid):
                hi_b = mid
            else:
                lo_b = mid + 1
        return lo_b

    def get_article(self, news_id: str) -> dict | None:
        """Full stored doc — includes internal provenance fields (warc_date,
        published_at, date_publish); agent-facing callers must expose only
        `pub_date`/`pub_ts`."""
        q = self.index.parse_query(f'id:"{news_id}"')
        hits = self.searcher.search(q, limit=1).hits
        if not hits:
            return None
        d = self.searcher.doc(hits[0][1]).to_dict()
        return {k: v[0] for k, v in d.items() if k != "pub_ts"} | \
               {"pub_ts": d["pub_ts"][0]}


if __name__ == "__main__":
    idx, query = sys.argv[1], sys.argv[2]
    date_from = sys.argv[3] if len(sys.argv) > 3 else None
    date_to = sys.argv[4] if len(sys.argv) > 4 else None
    s = NewsSearch(idx)
    for h in s.search(query, date_from, date_to, top_k=10):
        print(f"{h['score']:7.3f}  {h['published'][:16]}  {h['domain']:24s} {h['title'][:70]}")
