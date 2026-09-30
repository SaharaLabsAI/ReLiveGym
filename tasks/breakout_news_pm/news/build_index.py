#!/usr/bin/env python3
"""Build the tantivy BM25 index over the CC-NEWS corpus.

Index-time policy (on top of the extraction-time corpus policy in
ccnews/census_findings.md):
- drop stale recrawls: date_publish more than --stale-days before warc_date
  (evergreen pages recrawled in-window; ~10% of rows at 30 days)
- drop rows with neither title nor text

Everything is stored in the index (including full text), so search + article
fetch need no other artifact at serving time. ~9.5M docs -> ~25-35GB index.

`--pubtimes` left-joins the pass-3 sidecar (ccnews/timestamps.py) on id.
Each doc gets ONE canonical publish clock, `pub_ts`. Two ground-truth
timestamp versions exist (same schema, separate index artifacts):

  v1 (default; tantivy_index — no task reads it): pub_ts = the
     earlier of the page's own metadata publish time and the crawler
     discovery time (fallback when metadata is absent ~24% or claims a
     post-crawl time ~2%).
  v2 (--ts-v2; tantivy_index_v2): the self-reported time counts ONLY
     when the crawl corroborates it within V2_CORROBORATION_S (2 h,
     the census put the crawler's normal discovery latency at ~1-2 h and
     30.5% of the corpus beyond it); otherwise
     pub_ts = the crawl time. Closes the visible-before-crawl leak
     (backdated CMS timestamps let articles into the sim before the
     crawler proved they existed).
  v3 (--ts-v3; tantivy_index_v3 — the index every task reads):
     v2 with two refinements to the uncorroborated branch. (a) The
     fallback is crawl − 2 h, not crawl: the crawler's normal cycle is
     the tolerance itself, so "existed by one cycle before discovery" is
     the tightest bound the crawl supports (still never earlier than the
     self-report, since lag > 2 h there). (b) Self-reported times inside
     the crawler outage (OUTAGE_START..OUTAGE_END — the single crawl
     silence in the window, pinned by a full-window hour scan) are
     trusted as-is: no corroboration was possible, the lag is
     crawler-caused. Accepted residual: a CMS backdating INTO the outage
     window is trusted too.

Search date filtering runs on `pub_ts`; agent-facing surfaces expose only
its ISO form `pub_date` — the derivation is internal (provenance fields
warc_date / published_at / date_publish stay stored for analysis, never
shown to the labeling agent).

Usage:
  python3 build_index.py \
      --corpus 'tasks/breakout_news_pm/news/corpus/corpus_2026-*.parquet' \
      --pubtimes tasks/breakout_news_pm/news/ccnews/pubtime/published_at.parquet \
      --out tasks/breakout_news_pm/news/tantivy_index [--ts-v2 | --ts-v3]
"""

from __future__ import annotations

import argparse
import glob
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import tantivy


def build_schema() -> tantivy.Schema:
    b = tantivy.SchemaBuilder()
    b.add_text_field("id", stored=True, tokenizer_name="raw")
    b.add_text_field("url", stored=True, tokenizer_name="raw")
    b.add_text_field("domain", stored=True, tokenizer_name="raw")
    b.add_text_field("title", stored=True)
    b.add_text_field("description", stored=True)
    b.add_text_field("text", stored=True)
    b.add_text_field("date_publish", stored=True, tokenizer_name="raw")
    b.add_text_field("published_at", stored=True, tokenizer_name="raw")
    b.add_integer_field("pub_ts", stored=True, indexed=True, fast=True)
    b.add_text_field("pub_date", stored=True, tokenizer_name="raw")
    b.add_text_field("warc_date", stored=True, tokenizer_name="raw")
    return b.build()


def warc_epoch(iso: str) -> int:
    return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp())


V2_CORROBORATION_S = 2 * 3600  # v2/v3 clock tolerance

# The single CC-NEWS crawl silence in the 2026-03..07 window (112 h; the
# full-window hour scan found no other run of >=3 quiet
# hours). Self-reports inside it cannot be crawl-corroborated.
OUTAGE_START = int(datetime(2026, 4, 1, 4, tzinfo=timezone.utc).timestamp())
OUTAGE_END = int(datetime(2026, 4, 5, 20, tzinfo=timezone.utc).timestamp())


def derive_pub_ts(wts: int, published_at: str | None,
                  version: str = "v1") -> int:
    """The canonical publish clock for one doc. v1: min(crawl,
    self-reported). v2: the self-reported time only when the crawl
    corroborates it within V2_CORROBORATION_S, else the crawl time.
    v3: like v2, but the uncorroborated fallback is crawl − tolerance,
    and outage-window self-reports are trusted as-is."""
    if not published_at:
        return wts
    pub = min(wts, warc_epoch(published_at))
    if version == "v1" or wts - pub <= V2_CORROBORATION_S:
        return pub
    if version == "v2":
        return wts
    if OUTAGE_START <= pub < OUTAGE_END:
        return pub
    return wts - V2_CORROBORATION_S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="glob of corpus parquet files")
    ap.add_argument("--pubtimes", default=None,
                    help="published_at.parquet sidecar (pass-3) to join on id")
    ap.add_argument("--out", required=True, help="index directory (created fresh)")
    ap.add_argument("--stale-days", type=int, default=30)
    ap.add_argument("--heap-mb", type=int, default=2000)
    ap.add_argument("--limit-rows", type=int, help="cap rows (smoke tests)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--ts-v2", action="store_true",
                   help="v2 ground-truth clock: crawl-corroborated "
                        "self-reported times only (see docstring)")
    g.add_argument("--ts-v3", action="store_true",
                   help="v3 ground-truth clock: v2 with crawl-2h fallback "
                        "and outage-window trust (see docstring)")
    a = ap.parse_args()
    ts_version = "v3" if a.ts_v3 else "v2" if a.ts_v2 else "v1"

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    schema = build_schema()
    index = tantivy.Index(schema, path=str(out))
    writer = index.writer(heap_size=a.heap_mb * 1024 * 1024)

    files = sorted(glob.glob(a.corpus))
    assert files, f"no files match {a.corpus}"
    join = (f"LEFT JOIN read_parquet('{a.pubtimes}') s USING (id)"
            if a.pubtimes else "")
    pub_col = "s.published_at" if a.pubtimes else "NULL AS published_at"
    con = duckdb.connect()
    reader = con.execute(
        f"SELECT c.*, {pub_col} FROM read_parquet({files}) c {join}"
    ).fetch_record_batch(8192)

    t0 = time.time()
    kept = stale = empty = with_pub = 0
    max_stale_s = a.stale_days * 86400
    done = False
    for batch in reader:
        for r in batch.to_pylist():
            if not (r["title"] or r["text"]):
                empty += 1
                continue
            wts = warc_epoch(r["warc_date"])
            dp = r["date_publish"]
            if dp:
                try:
                    pub = datetime.fromisoformat(dp[:10]).replace(
                        tzinfo=timezone.utc).timestamp()
                    if wts - pub > max_stale_s:
                        stale += 1
                        continue
                except ValueError:
                    pass
            if r["published_at"]:
                with_pub += 1
            pub_ts = derive_pub_ts(wts, r["published_at"], ts_version)
            writer.add_document(tantivy.Document(
                id=r["id"], url=r["url"], domain=r["domain"],
                title=r["title"] or "", description=r["description"] or "",
                text=r["text"] or "", date_publish=dp or "",
                published_at=r["published_at"] or "",
                pub_ts=pub_ts,
                pub_date=datetime.fromtimestamp(
                    pub_ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                warc_date=r["warc_date"]))
            kept += 1
            if a.limit_rows and kept >= a.limit_rows:
                done = True
                break
        if done:
            break
        if kept and kept % 500_000 < 8192:
            print(f"  {kept:,} docs, {time.time()-t0:.0f}s", flush=True)
    writer.commit()
    writer.wait_merging_threads()
    print(f"indexed {kept:,} docs ({with_pub:,} with published_at; "
          f"stale dropped {stale:,}, empty {empty:,}) "
          f"in {(time.time()-t0)/60:.1f}m -> {out}")


if __name__ == "__main__":
    main()
