#!/usr/bin/env python3
"""CC-NEWS extraction pass — corpus build stage 2 (census_findings.md policy).

Rescans the WARCs (sequential streaming beats 9M random reads) and, for each
HTTP response record that passes the language gate, extracts article text +
metadata with trafilatura/htmldate. One parquet shard per WARC; `--merge`
dedups exact URLs (keep earliest warc_date) into monthly corpus files.

Policy v1 (census_findings.md): English only, all domains, drop records with
no extractable text. Visibility clock = warc_date; parsed date_publish is
secondary. Every row keeps (warc_file, offset, length) for exact re-fetch.

Language gate: html `lang` attribute prefix match against --langs; records
with no lang attribute are extracted, then checked with py3langid if
installed (kept on match or when the detector is absent).

Usage (EC2, us-east-1 — see README.md):
  python3.12 extract.py --start 20260301 --end 20260707 --workers 60 --out /data/corpus
  python3.12 extract.py --out /data/corpus --merge

Resumable like census.py: existing shards skipped, failures logged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

from census import init_worker, list_warcs, open_warc

_LANG_RE = re.compile(rb'<html[^>]{0,300}?lang=["\']?([a-zA-Z]{2})', re.I)


def process_warc(args: tuple[str, str, tuple[str, ...]]) -> tuple[str, int, str | None, dict]:
    key, out_dir, langs = args
    stats = {"kept": 0, "lang_skip": 0, "no_text": 0, "langid_skip": 0}
    shard = Path(out_dir) / "shards" / (Path(key).name.removesuffix(".warc.gz") + ".parquet")
    if shard.exists():
        return key, -1, None, stats
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        import trafilatura
        from warcio.archiveiterator import ArchiveIterator
        try:
            import py3langid
        except ImportError:
            py3langid = None

        stream, size = open_warc(key)
        rows = []
        it = ArchiveIterator(stream)
        offsets = []
        for rec in it:
            if rec.rec_type != "response":
                continue
            url = rec.rec_headers.get_header("WARC-Target-URI") or ""
            warc_date = rec.rec_headers.get_header("WARC-Date") or ""
            html = rec.content_stream().read()
            # AFTER the content read: warcio's get_record_offset() consumes
            # the record to find the gzip member boundary, so calling it first
            # would leave content_stream() empty (the original 0-row bug).
            offset = it.get_record_offset()
            m = _LANG_RE.search(html[:4096])
            html_lang = m.group(1).decode().lower() if m else None
            if html_lang is not None and html_lang not in langs:
                stats["lang_skip"] += 1
                continue
            doc = trafilatura.bare_extraction(
                html, url=url, with_metadata=True,
                date_extraction_params={"extensive_search": True, "original_date": True})
            if doc is None or not (doc.text or "").strip():
                stats["no_text"] += 1
                continue
            lang = html_lang
            if lang is None:
                if py3langid is not None:
                    lang, _ = py3langid.classify(doc.text[:2000])
                    if lang not in langs:
                        stats["langid_skip"] += 1
                        continue
                else:
                    lang = "und"
            stats["kept"] += 1
            offsets.append(offset)
            rows.append({
                "id": hashlib.sha256(url.encode()).hexdigest(),
                "url": url,
                "domain": urlparse(url).netloc.lower().removeprefix("www."),
                "title": doc.title or "",
                "description": doc.description or "",
                "text": doc.text,
                "author": doc.author or "",
                "date_publish": doc.date or "",
                "warc_date": warc_date,
                "lang": lang,
                "warc_file": key,
                "offset": offset,
            })
        # length = gap to next kept record's offset is wrong (skipped records in
        # between); recompute against ALL record offsets is census's job — here
        # store a generous cap: gap to next kept offset or EOF (re-fetch reads
        # the first response record in the chunk, extra bytes are harmless).
        lengths = [b - a for a, b in zip(offsets, offsets[1:])] + \
                  ([size - offsets[-1]] if offsets else [])
        for r, ln in zip(rows, lengths):
            r["length"] = ln
        shard.parent.mkdir(parents=True, exist_ok=True)
        cols = ["id", "url", "domain", "title", "description", "text", "author",
                "date_publish", "warc_date", "lang", "warc_file", "offset", "length"]
        table = pa.table({c: [r[c] for r in rows] for c in cols})
        tmp = shard.with_suffix(".tmp")
        pq.write_table(table, tmp, compression="zstd")
        tmp.rename(shard)
        return key, len(rows), None, stats
    except Exception as e:  # noqa: BLE001 — log and continue, rerun retries
        return key, 0, f"{type(e).__name__}: {e}", stats


def merge(out_dir: Path):
    """Shards → corpus_<YYYY-MM>.parquet, exact-URL dedup keeping earliest
    warc_date (~9M url→date entries in memory, ~2GB — fine on the instance)."""
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    shards = ds.dataset(out_dir / "shards", format="parquet")
    best: dict[str, str] = {}
    for batch in shards.to_batches(columns=["url", "warc_date"]):
        for url, wd in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
            if url not in best or wd < best[url]:
                best[url] = wd
    writers: dict[str, pq.ParquetWriter] = {}
    n = 0
    for batch in shards.to_batches():
        urls = batch.column(batch.schema.get_field_index("url")).to_pylist()
        dates = batch.column(batch.schema.get_field_index("warc_date")).to_pylist()
        keep = []
        for i, (u, wd) in enumerate(zip(urls, dates)):
            if best.get(u) == wd:
                keep.append(i)
                best[u] = None  # claimed — identical-timestamp dups lose too
        by_month: dict[str, list[int]] = {}
        for i in keep:
            by_month.setdefault(dates[i][:7], []).append(i)
        for month, idxs in by_month.items():
            sub = batch.take(idxs)
            if month not in writers:
                writers[month] = pq.ParquetWriter(
                    out_dir / f"corpus_{month}.parquet", sub.schema, compression="zstd")
            writers[month].write_batch(sub)
            n += len(idxs)
    for w in writers.values():
        w.close()
    print(f"merged {n:,} articles -> {len(writers)} monthly files in {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--start", default="20260301")
    ap.add_argument("--end", default="20260707")
    ap.add_argument("--out", default="./corpus")
    ap.add_argument("--langs", default="en", help="comma-separated 2-letter codes")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    ap.add_argument("--limit", type=int, help="first N WARCs only (validation)")
    ap.add_argument("--source", choices=("s3", "https"), default="s3")
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.merge:
        merge(out)
        return

    import trafilatura
    langs = tuple(a.langs.split(","))
    keys = list_warcs(a.start, a.end)
    if a.limit:
        keys = keys[:a.limit]
    print(f"{len(keys)} WARCs in [{a.start}, {a.end}], langs={langs}, "
          f"trafilatura={trafilatura.__version__}")

    t0 = time.time()
    done = failed = skipped = 0
    totals = {"kept": 0, "lang_skip": 0, "no_text": 0, "langid_skip": 0}
    with mp.Pool(a.workers, initializer=init_worker, initargs=(a.source,)) as pool, \
         open(out / "failures.log", "a") as flog:
        for key, n, err, stats in pool.imap_unordered(
                process_warc, [(k, str(out), langs) for k in keys], chunksize=1):
            done += 1
            if err:
                failed += 1
                flog.write(json.dumps({"key": key, "error": err}) + "\n")
                flog.flush()
            elif n < 0:
                skipped += 1
            for k2 in totals:
                totals[k2] += stats[k2]
            if done % 25 == 0 or done == len(keys):
                dt = time.time() - t0
                print(f"[{done}/{len(keys)}] kept={totals['kept']:,} "
                      f"lang_skip={totals['lang_skip']:,} no_text={totals['no_text']:,} "
                      f"langid_skip={totals['langid_skip']:,} skipped={skipped} "
                      f"failed={failed} elapsed={dt/60:.1f}m "
                      f"eta={dt / done * (len(keys) - done) / 60:.0f}m", flush=True)
    print("done; merge with --merge" + (f"; {failed} failures logged" if failed else ""))


if __name__ == "__main__":
    sys.exit(main())
