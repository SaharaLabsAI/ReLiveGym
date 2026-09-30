#!/usr/bin/env python
"""Build the reddit_ai_popularity world data from the Arctic Shift API.

Provenance + rebuild recipe: README.md (this dir).

What it does (raw collection — NO quantization, NO windows, NO splits, NO labels
baked; the continuously-evolving world is the future task's visibility rule):

  1. Harvest root submissions per subreddit over [--after, --before) via
     /api/posts/search (paginated ascending by created_utc, resumable cache).
     Each subreddit is a base-level topic — its name is the label (no company
     grouping, no alias matching); r/OpenAI and r/ChatGPT are distinct topics.
  2. Keep roots with num_comments >= --min-comments.
  3. For each kept root, fetch its FULL comment tree via /api/comments/search
     (link_id=<root>, all comments, paginated), preserving every created_utc.
  4. Assemble one raw cascade per root (nodes + parent edges) and write
     built/{roots.jsonl, cascades.jsonl, build_stats.json}.

Everything is cached under cache/ and resumable. Be polite to the free API
(no uptime/rate guarantees): single-flight, backoff, ~2 req/s.

Runtime deps: requests, pyyaml (build-only; the eventual task runtime won't need them).
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
import yaml

BASE = "https://arctic-shift.photon-reddit.com"
DATA_DIR = Path(__file__).resolve().parent
CACHE = DATA_DIR / "cache"
BUILT = DATA_DIR / "built"
PAGE = 100          # API max limit per page
HORIZON_DAYS = 7    # collect full tree to +7d; label horizon chosen downstream


class RateLimiter:
    """Global aggregate rate cap shared across worker threads. Threads reserve
    staggered slots under a lock, then sleep to their slot outside it — so N
    workers overlap on network latency but never exceed 1/interval req/s total."""

    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._lock = threading.Lock()
        self._next = 0.0

    def acquire(self) -> None:
        with self._lock:
            slot = max(time.monotonic(), self._next)
            self._next = slot + self.min_interval
        delay = slot - time.monotonic()
        if delay > 0:
            time.sleep(delay)


_LIMITER = RateLimiter(0.2)   # replaced in main() from --rate


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def load_config(path: Path) -> dict:
    """Config is a flat list of subreddits (`subreddits: [...]`). Each subreddit
    is the base unit — its own topic label. No company grouping, no aliases."""
    with open(path) as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------- #
# API + resumable pagination
# --------------------------------------------------------------------------- #
def make_session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = "reddit_ai_popularity-build/0.1 (research; polite)"
    return s


def api_get(sess: requests.Session, path: str, params: dict) -> list[dict]:
    """GET with global rate limit + backoff on 422/429/5xx; returns `data`."""
    for attempt in range(6):
        _LIMITER.acquire()
        try:
            r = sess.get(BASE + path, params=params, timeout=90)
        except requests.RequestException as e:
            wait = 2 ** attempt
            print(f"    net error {e}; retry in {wait}s")
            time.sleep(wait)
            continue
        if r.status_code == 200:
            return r.json().get("data", [])
        # 422 here carries "Timeout. Maybe slow down a bit" — transient throttle.
        if r.status_code in (422, 429, 500, 502, 503, 504):
            wait = 2 ** attempt + 2
            print(f"    HTTP {r.status_code} ({r.text[:60]}); retry in {wait}s")
            time.sleep(wait)
            continue
        raise RuntimeError(f"HTTP {r.status_code} for {path} {params}: {r.text[:200]}")
    raise RuntimeError(f"gave up after retries: {path} {params}")


def paginate(sess, path: str, base_params: dict, cache_file: Path,
             after_ts: int | None = None, before_ts: int | None = None) -> list[dict]:
    """Fetch all items ascending by created_utc, resumable via cache_file.

    A sidecar `<cache_file>.done` marks a completed fetch. On resume we reload
    cached rows and continue from the max created_utc seen (dedup by id)."""
    done = cache_file.with_suffix(cache_file.suffix + ".done")
    seen: dict[str, dict] = {}
    if cache_file.exists():
        for line in cache_file.read_text().splitlines():
            if line.strip():
                o = json.loads(line)
                seen[o["id"]] = o
    if done.exists():
        return list(seen.values())

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cursor = max((o["created_utc"] for o in seen.values()), default=after_ts)
    fh = cache_file.open("a")
    try:
        while True:
            params = dict(base_params, sort="asc", limit=PAGE)
            if cursor is not None:
                params["after"] = int(cursor)
            if before_ts is not None:
                params["before"] = int(before_ts)
            batch = api_get(sess, path, params)
            new = 0
            for o in batch:
                if o["id"] not in seen:
                    seen[o["id"]] = o
                    fh.write(json.dumps(o) + "\n")
                    new += 1
                cursor = max(cursor or 0, o["created_utc"])
            if new == 0 or len(batch) < PAGE:   # exhausted (or all-dupes tie-page)
                break
    finally:
        fh.close()
    done.touch()
    return list(seen.values())


# --------------------------------------------------------------------------- #
# cascade assembly
# --------------------------------------------------------------------------- #
def strip_id(fullname: str | None) -> str | None:
    """t3_abc / t1_abc -> abc."""
    return fullname.split("_", 1)[1] if fullname and "_" in fullname else fullname


def assemble_cascade(root: dict, comments: list[dict]) -> dict:
    """Raw tree: root node + comment nodes, edges by parent_id. Timestamps kept
    exactly (no rounding). Downstream picks a horizon and computes popularity."""
    rid = root["id"]
    nodes = [{
        "id": rid, "kind": "root", "parent_id": None,
        "author": root.get("author"),
        "created_utc": root["created_utc"],
        "body": (root.get("title") or "") + (
            ("\n\n" + root["selftext"]) if root.get("selftext") else ""),
        "score": root.get("score"),
    }]
    for c in comments:
        parent = strip_id(c.get("parent_id"))            # t3_<root> or t1_<comment>
        nodes.append({
            "id": c["id"], "kind": "comment",
            "parent_id": rid if parent == rid else parent,
            "author": c.get("author"),
            "created_utc": c["created_utc"],
            "body": c.get("body"),
            "score": c.get("score"),
        })
    return {"root_id": rid, "nodes": nodes}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path,
                    default=DATA_DIR / "subreddits.yaml")
    ap.add_argument("--after", required=True, help="root window start, unix or YYYY-MM-DD")
    ap.add_argument("--before", required=True, help="root window end (exclusive)")
    ap.add_argument("--min-comments", type=int, default=5,
                    help="only fetch cascades for roots with >= this many comments")
    ap.add_argument("--max-roots", type=int, default=None,
                    help="cap kept roots (smoke)")
    ap.add_argument("--out", type=Path, default=BUILT)
    ap.add_argument("--workers", type=int, default=5,
                    help="concurrent request workers")
    ap.add_argument("--rate", type=float, default=0.2,
                    help="min seconds between ANY two requests (global cap ~1/rate req/s)")
    args = ap.parse_args()

    global _LIMITER
    _LIMITER = RateLimiter(args.rate)

    def to_ts(s: str) -> int:
        if s.isdigit():
            return int(s)
        import datetime as dt
        return int(dt.datetime.strptime(s, "%Y-%m-%d")
                   .replace(tzinfo=dt.timezone.utc).timestamp())

    after_ts, before_ts = to_ts(args.after), to_ts(args.before)
    cfg = load_config(args.config)
    # subreddit IS the base unit — no company grouping. The tag is the subreddit.
    subs = sorted({s.lower() for s in cfg["subreddits"]})
    sess = make_session()
    args.out.mkdir(parents=True, exist_ok=True)

    # 1-2. harvest roots (subs in parallel; each sub paginated inside) --------
    stats_roots = Counter()

    def harvest(sub):
        cache = CACHE / "roots" / f"{sub}__{after_ts}_{before_ts}.jsonl"
        posts = paginate(sess, "/api/posts/search", {"subreddit": sub},
                         cache, after_ts=after_ts, before_ts=before_ts)
        print(f"  r/{sub}: {len(posts)} roots")
        return sub, posts

    kept: list[dict] = []
    per_sub = Counter()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for sub, posts in ex.map(harvest, subs):
            stats_roots[sub] = len(posts)
            for p in posts:
                if (p.get("num_comments") or 0) < args.min_comments:
                    continue
                p["_sub"] = sub                            # crawl key == the topic label
                kept.append(p)
                per_sub[sub] += 1

    kept.sort(key=lambda p: p["created_utc"])
    if args.max_roots:
        kept = kept[: args.max_roots]
    print(f"\nkept {len(kept)} roots (>= {args.min_comments} comments, tagged)")

    # 3-4. fetch comments + assemble cascades (roots in parallel) -------------
    def fetch_one(p):
        rid = p["id"]
        ccache = CACHE / "comments" / f"{rid}.jsonl"
        comments = paginate(sess, "/api/comments/search", {"link_id": rid}, ccache)
        casc = assemble_cascade(p, comments)
        root_row = {
            "id": rid, "name": p.get("name"),
            "subreddit": p.get("subreddit") or p["_sub"],   # the topic label / base unit
            "author": p.get("author"), "created_utc": p["created_utc"],
            "title": p.get("title"), "selftext": p.get("selftext"),
            "url": p.get("url"), "is_self": p.get("is_self"),
            "over_18": p.get("over_18"), "link_flair_text": p.get("link_flair_text"),
            "domain": p.get("domain"),
            "score": p.get("score"), "num_comments": p.get("num_comments"),
            "retrieved_on": p.get("retrieved_on"),    # ingestion time (score caveat)
        }
        return root_row, casc

    roots_out = (args.out / "roots.jsonl").open("w")
    casc_out = (args.out / "cascades.jsonl").open("w")
    size_hist = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(fetch_one, p) for p in kept]
        for fut in as_completed(futs):
            root_row, casc = fut.result()
            size_hist.append(len(casc["nodes"]))
            roots_out.write(json.dumps(root_row) + "\n")
            casc_out.write(json.dumps(casc) + "\n")
            done += 1
            if done % 100 == 0 or done == len(kept):
                print(f"  cascades {done}/{len(kept)}", flush=True)
    roots_out.close()
    casc_out.close()

    # 5. stats ----------------------------------------------------------------
    size_hist.sort()
    def pct(p):
        return size_hist[min(len(size_hist) - 1, int(p / 100 * len(size_hist)))] if size_hist else 0
    stats = {
        "window": {"after": after_ts, "before": before_ts,
                   "after_h": args.after, "before_h": args.before},
        "min_comments": args.min_comments,
        "horizon_days_collected": HORIZON_DAYS,
        "subreddits_scanned": dict(stats_roots),
        "n_kept_roots": len(kept),
        "cascades_per_subreddit": dict(per_sub),
        "cascade_size": {
            "n": len(size_hist),
            "min": size_hist[0] if size_hist else 0,
            "p50": pct(50), "p90": pct(90), "p99": pct(99),
            "max": size_hist[-1] if size_hist else 0,
        },
        "note": "raw cascades; no label baked. score is ingestion-time snapshot.",
    }
    (args.out / "build_stats.json").write_text(json.dumps(stats, indent=2))
    print("\nbuild_stats.json:")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
