#!/usr/bin/env python3
"""CC-NEWS publish-timestamp pass — corpus build stage 3 (optional sidecar).

The corpus's `date_publish` (htmldate) is day-granularity only, but many
pages carry a full publish timestamp in <head> metadata. This pass rescans
the WARCs and, for every response record where one is found, emits
(id, published_at, source, warc_date) — no trafilatura, just regex over the
document head, so it runs at near-census speed (gunzip-bound).

Extraction preference order (first hit with a time-of-day component wins):
  1. <meta property/name="article:published_time" content="...">
  2. JSON-LD / itemprop "datePublished"
  3. <time datetime="...">
  4. <meta name="pubdate|publishdate|publish-date|publication_date|date">

Output joins onto the corpus on `id` (sha256 of url). Coverage is
publisher-dependent — expect a substantial minority of articles without one;
`warc_date` remains the canonical 100%-coverage visibility clock.

Usage (EC2, us-east-1 — see README.md):
  python3.12 timestamps.py --start 20260301 --end 20260707 --workers 60 --out /data/pubtimes
  python3.12 timestamps.py --out /data/pubtimes --merge   # -> published_at.parquet

Resumable like census.py/extract.py: existing shards skipped, failures logged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import re
import sys
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

from census import init_worker, list_warcs, open_warc

HEAD_BYTES = 262_144  # metadata lives in <head>; JSON-LD occasionally later

_PATTERNS = [
    ("meta_published_time", re.compile(
        rb'<meta[^>]{0,200}?(?:property|name)=["\']article:published_time["\']'
        rb'[^>]{0,200}?content=["\']([^"\']{8,40})["\']', re.I)),
    ("meta_published_time", re.compile(  # content= before property=
        rb'<meta[^>]{0,200}?content=["\']([^"\']{8,40})["\']'
        rb'[^>]{0,200}?(?:property|name)=["\']article:published_time["\']', re.I)),
    ("date_published", re.compile(
        rb'"datePublished"\s*:\s*"([^"]{8,40})"')),
    ("date_published", re.compile(
        rb'<[^>]{0,200}?itemprop=["\']datePublished["\']'
        rb'[^>]{0,200}?(?:content|datetime)=["\']([^"\']{8,40})["\']', re.I)),
    ("time_tag", re.compile(
        rb'<time[^>]{0,200}?datetime=["\']([^"\']{8,40})["\']', re.I)),
    ("meta_date", re.compile(
        rb'<meta[^>]{0,200}?name=["\'](?:pubdate|publishdate|publish-date|'
        rb'publication_date|date)["\'][^>]{0,200}?content=["\']([^"\']{8,40})["\']',
        re.I)),
]


def parse_ts(raw: str) -> str | None:
    """Raw metadata value -> UTC ISO string, only if it has a time-of-day."""
    s = raw.strip().replace("Z", "+00:00")
    dt = None
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        try:
            dt = parsedate_to_datetime(raw.strip())
        except (ValueError, TypeError):
            return None
    if (dt.hour, dt.minute, dt.second) == (0, 0, 0):
        return None  # date-only disguised as midnight — no better than htmldate
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)  # assume UTC when unstated
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def extract_ts(html: bytes) -> tuple[str, str] | None:
    head = html[:HEAD_BYTES]
    for source, pat in _PATTERNS:
        m = pat.search(head)
        if m:
            ts = parse_ts(m.group(1).decode("ascii", "ignore"))
            if ts:
                return ts, source
    return None


def process_warc(args: tuple[str, str]) -> tuple[str, int, str | None, dict]:
    key, out_dir = args
    stats = {"records": 0, "with_ts": 0}
    shard = Path(out_dir) / "shards" / (Path(key).name.removesuffix(".warc.gz") + ".parquet")
    if shard.exists():
        return key, -1, None, stats
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        from warcio.archiveiterator import ArchiveIterator

        stream, _size = open_warc(key)
        rows = []
        for rec in ArchiveIterator(stream):
            if rec.rec_type != "response":
                continue
            stats["records"] += 1
            url = rec.rec_headers.get_header("WARC-Target-URI") or ""
            warc_date = rec.rec_headers.get_header("WARC-Date") or ""
            hit = extract_ts(rec.content_stream().read())
            if hit is None:
                continue
            stats["with_ts"] += 1
            rows.append({
                "id": hashlib.sha256(url.encode()).hexdigest(),
                "published_at": hit[0],
                "source": hit[1],
                "warc_date": warc_date,
            })
        shard.parent.mkdir(parents=True, exist_ok=True)
        cols = ["id", "published_at", "source", "warc_date"]
        table = pa.table({c: [r[c] for r in rows] for c in cols})
        tmp = shard.with_suffix(".tmp")
        pq.write_table(table, tmp, compression="zstd")
        tmp.rename(shard)
        return key, len(rows), None, stats
    except Exception as e:  # noqa: BLE001 — log and continue, rerun retries
        return key, 0, f"{type(e).__name__}: {e}", stats


def merge(out_dir: Path):
    """Shards -> published_at.parquet, one row per id (recrawls carry the
    same page metadata; keep the earliest-warc_date occurrence)."""
    import pyarrow as pa
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    shards = ds.dataset(out_dir / "shards", format="parquet")
    best: dict[str, tuple[str, str, str]] = {}
    for batch in shards.to_batches():
        for i in range(len(batch)):
            _id = batch.column(0)[i].as_py()
            wd = batch.column(3)[i].as_py()
            if _id not in best or wd < best[_id][2]:
                best[_id] = (batch.column(1)[i].as_py(),
                             batch.column(2)[i].as_py(), wd)
    table = pa.table({
        "id": list(best.keys()),
        "published_at": [v[0] for v in best.values()],
        "source": [v[1] for v in best.values()],
    })
    pq.write_table(table, out_dir / "published_at.parquet", compression="zstd")
    print(f"merged {len(best):,} ids -> {out_dir / 'published_at.parquet'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--start", default="20260301")
    ap.add_argument("--end", default="20260707")
    ap.add_argument("--out", default="./pubtimes")
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

    keys = list_warcs(a.start, a.end)
    if a.limit:
        keys = keys[:a.limit]
    print(f"{len(keys)} WARCs in [{a.start}, {a.end}]")

    t0 = time.time()
    done = failed = skipped = 0
    totals = {"records": 0, "with_ts": 0}
    with mp.Pool(a.workers, initializer=init_worker, initargs=(a.source,)) as pool, \
         open(out / "failures.log", "a") as flog:
        for key, n, err, stats in pool.imap_unordered(
                process_warc, [(k, str(out)) for k in keys], chunksize=1):
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
                pct = totals["with_ts"] / totals["records"] if totals["records"] else 0
                print(f"[{done}/{len(keys)}] records={totals['records']:,} "
                      f"with_ts={totals['with_ts']:,} ({pct:.0%}) skipped={skipped} "
                      f"failed={failed} elapsed={dt/60:.1f}m "
                      f"eta={dt / done * (len(keys) - done) / 60:.0f}m", flush=True)
    print("done; merge with --merge" + (f"; {failed} failures logged" if failed else ""))


if __name__ == "__main__":
    sys.exit(main())
