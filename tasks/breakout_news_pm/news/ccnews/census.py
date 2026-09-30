#!/usr/bin/env python3
"""CC-NEWS census pass — corpus build stage 1.

Streams every CC-NEWS WARC in a crawl-date range and records, for each HTTP
response record: url, domain, warc_date (crawler discovery time), byte offset
of the record in the .warc.gz, and record length. No HTML parsing — this pass
answers "what does CC-NEWS contain?" so the corpus inclusion policy can be
decided from data before the (more expensive) extraction pass.

Offsets + lengths enable later random access to any single article via an S3
Range request — no rescan needed to fetch record bodies.

Run on EC2 in us-east-1 (the commoncrawl bucket's region; in-region reads are
free). See README.md for the instance runbook.

Usage:
  python3 census.py --start 20260301 --end 20260707 --workers 60 --out /data/census
  python3 census.py --out /data/census --merge   # after (or during) the run

Resumable: one parquet shard per WARC; existing shards are skipped. Failed
WARCs are listed in <out>/failures.log; rerunning retries them.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import multiprocessing as mp
import sys
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

BUCKET = "commoncrawl"
PATHS_URL = "https://data.commoncrawl.org/crawl-data/CC-NEWS/{year}/{month}/warc.paths.gz"
UA = "program-engineering-dataset-build/0.1"

_s3 = None  # per-worker client


def list_warcs(start: str, end: str) -> list[str]:
    """WARC keys whose crawl timestamp (from the filename) falls in [start, end].

    Filenames look like CC-NEWS-20260301003313-06997.warc.gz; the range is
    compared on the YYYYMMDD prefix.
    """
    months = set()
    y, m = int(start[:4]), int(start[4:6])
    while (y, m) <= (int(end[:4]), int(end[4:6])):
        months.add((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    keys = []
    for y, m in sorted(months):
        req = urllib.request.Request(
            PATHS_URL.format(year=y, month=f"{m:02d}"), headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=60) as r:
            listing = gzip.decompress(r.read()).decode()
        keys.extend(listing.split())
    def stamp(key: str) -> str:
        return Path(key).name.split("-")[2][:8]
    return sorted(k for k in keys if start <= stamp(k) <= end)


def init_worker(source: str):
    """The commoncrawl bucket requires AUTHENTICATED requests (no anonymous
    reads since 2024): s3 mode needs any AWS credentials with s3:GetObject on
    arn:aws:s3:::commoncrawl/* (fast, free in us-east-1). https mode uses
    data.commoncrawl.org — no credentials, but rate-limited: fine for local
    validation, not for the full run."""
    global _s3
    if source == "s3":
        import boto3
        from botocore.config import Config
        _s3 = boto3.client("s3", config=Config(
            retries={"max_attempts": 5, "mode": "adaptive"}))


def open_warc(key: str):
    """-> (file-like stream, total size in bytes)"""
    if _s3 is not None:
        obj = _s3.get_object(Bucket=BUCKET, Key=key)
        return obj["Body"], obj["ContentLength"]
    req = urllib.request.Request(
        f"https://data.commoncrawl.org/{key}", headers={"User-Agent": UA})
    resp = urllib.request.urlopen(req, timeout=300)
    return resp, int(resp.headers["Content-Length"])


def process_warc(args: tuple[str, str]) -> tuple[str, int, str | None]:
    """Census one WARC → parquet shard. Returns (key, n_rows, error)."""
    key, out_dir = args
    shard = Path(out_dir) / "shards" / (Path(key).name.removesuffix(".warc.gz") + ".parquet")
    if shard.exists():
        return key, -1, None
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        from warcio.archiveiterator import ArchiveIterator

        stream, size = open_warc(key)
        urls, domains, dates, offsets = [], [], [], []
        it = ArchiveIterator(stream)
        for rec in it:
            if rec.rec_type != "response":
                continue
            url = rec.rec_headers.get_header("WARC-Target-URI") or ""
            netloc = urlparse(url).netloc.lower()
            domains.append(netloc.removeprefix("www."))
            urls.append(url)
            dates.append(rec.rec_headers.get_header("WARC-Date") or "")
            offsets.append(it.get_record_offset())
        # record length = gap to the next record (last one runs to EOF); response
        # records interleave with request records, so this over-counts by the
        # request record's size — fine, Range re-fetches just read a bit extra
        # and warcio takes the first response record in the chunk.
        lengths = [b - a for a, b in zip(offsets, offsets[1:])] + \
                  ([size - offsets[-1]] if offsets else [])
        table = pa.table({
            "url": urls, "domain": domains, "warc_date": dates,
            "warc_file": pa.array([key] * len(urls), pa.string()),
            "offset": pa.array(offsets, pa.int64()),
            "length": pa.array(lengths, pa.int64()),
        })
        shard.parent.mkdir(parents=True, exist_ok=True)
        tmp = shard.with_suffix(".tmp")
        pq.write_table(table, tmp, compression="zstd")
        tmp.rename(shard)
        return key, len(urls), None
    except Exception as e:  # noqa: BLE001 — log and continue, rerun retries
        return key, 0, f"{type(e).__name__}: {e}"


def merge(out_dir: Path):
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq
    shards = ds.dataset(out_dir / "shards", format="parquet")
    with pq.ParquetWriter(out_dir / "census.parquet", shards.schema,
                          compression="zstd") as w:
        for batch in shards.to_batches():
            w.write_batch(batch)
    meta = pq.read_metadata(out_dir / "census.parquet")
    print(f"merged: {meta.num_rows:,} rows -> {out_dir / 'census.parquet'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--start", default="20260301", help="first crawl day YYYYMMDD")
    ap.add_argument("--end", default="20260707", help="last crawl day YYYYMMDD")
    ap.add_argument("--out", default="./census")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    ap.add_argument("--limit", type=int, help="process only the first N WARCs (validation)")
    ap.add_argument("--source", choices=("s3", "https"), default="s3",
                    help="s3 = authenticated boto3 (full runs, in-region); "
                         "https = data.commoncrawl.org, no credentials (validation)")
    ap.add_argument("--merge", action="store_true", help="merge shards and exit")
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
    done = rows = failed = skipped = 0
    with mp.Pool(a.workers, initializer=init_worker, initargs=(a.source,)) as pool, \
         open(out / "failures.log", "a") as flog:
        for key, n, err in pool.imap_unordered(
                process_warc, [(k, str(out)) for k in keys], chunksize=1):
            done += 1
            if err:
                failed += 1
                flog.write(json.dumps({"key": key, "error": err}) + "\n")
                flog.flush()
            elif n < 0:
                skipped += 1
            else:
                rows += n
            if done % 25 == 0 or done == len(keys):
                dt = time.time() - t0
                print(f"[{done}/{len(keys)}] rows={rows:,} skipped={skipped} "
                      f"failed={failed} elapsed={dt/60:.1f}m "
                      f"eta={dt / done * (len(keys) - done) / 60:.0f}m", flush=True)
    print("done; merge with --merge" + (f"; {failed} failures logged" if failed else ""))


if __name__ == "__main__":
    sys.exit(main())
