"""Fetch price history for the 2026-03..07 Polymarket window.

Uses POST clob.polymarket.com/batch-prices-history: up to 20 tokens per call,
start_ts..end_ts span capped at 15 days (1,296,000 s) server-side, so the
window is fetched in 15-day chunks. A market is only queried in chunks that
overlap its own life (startDate .. min(endDate, closedTime)).

Resume-safe: each (chunk, batch) response lands in raw/price_shards/ and is
skipped on rerun; --merge folds shards into prices/<market_id>.json in the
same {grid_hours, points: [[t, p], ...]} shape as the existing built world.

--fidelity sets the grid in minutes (default 60 = hourly). Sub-hourly runs
use raw/price_shards_<N>m/ and raw/prices_<N>m/ so they never collide with
the hourly world. The API returns a dense forward-filled grid, so sub-hourly
shards keep only price CHANGES (plus each token's first point per chunk) —
lossless under forward-fill, ~20x smaller.

Usage:
    python fetch_prices.py --dry-run              # print plan, no requests
    python fetch_prices.py                        # fetch all markets in input
    python fetch_prices.py --min-volume 10000 --min-days 7
    python fetch_prices.py --fidelity 1 ...       # minute grid
    python fetch_prices.py --merge                # shards -> prices/<mid>.json
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

CLOB_BATCH = "https://clob.polymarket.com/batch-prices-history"
UA = {"User-Agent": "program-engineering-dataset-build/0.1"}
WINDOW_START = datetime(2026, 3, 1, tzinfo=timezone.utc)
WINDOW_END = datetime(2026, 7, 1, tzinfo=timezone.utc)
CHUNK_S = 15 * 86400  # measured server cap: 1,296,000 s
BATCH_TOKENS = 20  # documented cap
SLEEP_S = 0.2  # ~5 req/s

HERE = Path(__file__).parent
SPORTS_TAGS = {
    "Sports", "NBA", "NFL", "MLB", "NHL", "Soccer", "EPL", "Champions League",
    "La Liga", "Serie A", "Bundesliga", "Tennis", "UFC", "MMA", "Boxing",
    "Golf", "F1", "NASCAR", "Esports", "College Football", "College Basketball",
    "Cricket", "Baseball", "Basketball", "Hockey", "Olympics",
}


def parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    s = s.replace("Z", "+00:00")
    if s.endswith("+00"):
        s += ":00"
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def load_roster(path: Path, min_volume: float, min_days: float,
                exclude_sports: bool,
                fetch_start: datetime = WINDOW_START,
                fetch_end: datetime = WINDOW_END,
                only_ids: set[str] | None = None) -> list[dict]:
    """One entry per market: yes-token + the chunk-clipped life inside the
    fetch window (defaults to the main window)."""
    roster = []
    for line in path.open():
        m = json.loads(line)
        if only_ids is not None and str(m.get("id")) not in only_ids:
            continue
        try:
            token = json.loads(m["clobTokenIds"])[0]  # first outcome = Yes
        except (KeyError, TypeError, ValueError, IndexError):
            continue
        if (m.get("volumeNum") or 0) < min_volume:
            continue
        if exclude_sports and set(m.get("tags") or []) & SPORTS_TAGS:
            continue
        start = parse_dt(m.get("startDate")) or WINDOW_START
        end = parse_dt(m.get("endDate")) or WINDOW_END
        closed_t = parse_dt(m.get("closedTime"))
        if closed_t and closed_t < end:
            end = closed_t
        # eligibility (min_days) is always judged on the MAIN window so a
        # warmup fetch (--fetch-start/--fetch-end) uses the same roster
        lo = max(start, WINDOW_START)
        hi = min(end, WINDOW_END)
        if (hi - lo).total_seconds() < min_days * 86400:
            continue
        f_lo = max(start, fetch_start)
        f_hi = min(end, fetch_end)
        if f_hi <= f_lo:
            continue  # not alive inside the fetch window
        roster.append({"market_id": m["id"], "token": token,
                       "lo": f_lo.timestamp(), "hi": f_hi.timestamp()})
    return roster


def plan_batches(roster: list[dict],
                 fetch_start: datetime = WINDOW_START,
                 fetch_end: datetime = WINDOW_END) -> list[tuple[int, int, list[dict]]]:
    """(chunk_start_ts, chunk_end_ts, markets alive in chunk) split into <=20-token groups."""
    batches = []
    t0 = int(fetch_start.timestamp())
    t_end = int(fetch_end.timestamp())
    while t0 < t_end:
        t1 = min(t0 + CHUNK_S, t_end)
        alive = [r for r in roster if r["lo"] < t1 and r["hi"] > t0]
        for i in range(0, len(alive), BATCH_TOKENS):
            batches.append((t0, t1, alive[i:i + BATCH_TOKENS]))
        t0 = t1
    return batches


def compress_changes(pts: list[dict]) -> list[dict]:
    """Keep the first point and every price change; the API grid is dense and
    forward-filled, so this is lossless under forward-fill."""
    out = []
    prev = None
    for p in pts:
        if prev is None or p["p"] != prev:
            out.append(p)
            prev = p["p"]
    return out


def fetch_batch(tokens: list[str], t0: int, t1: int, fidelity: int,
                tries: int = 4) -> dict:
    body = {"markets": tokens, "start_ts": t0, "end_ts": t1,
            "fidelity": fidelity}
    for i in range(tries):
        try:
            r = requests.post(CLOB_BATCH, json=body, headers=UA, timeout=120)
            d = r.json()
            if r.status_code == 200 and "history" in d:
                return d["history"]
            wait = d.get("retry_after_seconds") or 2 * (i + 1)
        except (requests.RequestException, ValueError):
            wait = 2 * (i + 1)
        time.sleep(wait)
    raise RuntimeError(f"batch fetch failed after {tries} tries "
                       f"(chunk {t0}..{t1}, {len(tokens)} tokens)")


def merge_shards(shard_dir: Path, out_dir: Path, fidelity: int) -> None:
    series: dict[str, dict[int, float]] = {}
    token_to_mid = {}
    for shard in sorted(shard_dir.glob("chunk_*.json")):
        d = json.loads(shard.read_text())
        token_to_mid.update(d["tokens"])
        for token, pts in d["history"].items():
            mid = d["tokens"][token]
            series.setdefault(mid, {}).update(
                {int(p["t"]): float(p["p"]) for p in pts})
    out_dir.mkdir(parents=True, exist_ok=True)
    for mid, pts in series.items():
        points = sorted(pts.items())
        doc = {"grid_hours": fidelity / 60, "points": points}
        if fidelity < 60:
            doc["sparse"] = "changes"  # forward-fill between points
        (out_dir / f"{mid}.json").write_text(json.dumps(doc))
    n_pts = sum(len(v) for v in series.values())
    print(f"merged {len(series)} markets, {n_pts:,} points -> {out_dir}")
    empty = set(token_to_mid.values()) - set(series)
    if empty:
        print(f"note: {len(empty)} markets returned zero points everywhere")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", default=str(HERE / "raw" / "markets.jsonl"))
    ap.add_argument("--min-volume", type=float, default=0.0,
                    help="skip markets below this lifetime USD volume")
    ap.add_argument("--min-days", type=float, default=0.0,
                    help="skip markets with fewer in-window days")
    ap.add_argument("--exclude-sports", action="store_true")
    ap.add_argument("--fetch-start", default=None, metavar="YYYY-MM-DD",
                    help="fetch window start override (e.g. 2026-02-15 for "
                         "detector warmup); roster eligibility still uses "
                         "the main window")
    ap.add_argument("--fetch-end", default=None, metavar="YYYY-MM-DD")
    ap.add_argument("--fidelity", type=int, default=60,
                    help="price grid in minutes (60=hourly, 1=minute)")
    ap.add_argument("--only-ids", default=None, metavar="FILE.jsonl",
                    help="restrict roster to market_ids listed in this jsonl "
                         "(e.g. the labeling sample_v1.jsonl)")
    ap.add_argument("--dry-run", action="store_true", help="print plan only")
    ap.add_argument("--merge", action="store_true",
                    help="merge fetched shards instead of fetching")
    args = ap.parse_args()

    suffix = "" if args.fidelity == 60 else f"_{args.fidelity}m"
    shard_dir = HERE / "raw" / f"price_shards{suffix}"
    if args.merge:
        merge_shards(shard_dir, HERE / "raw" / f"prices{suffix}", args.fidelity)
        return

    fetch_start = (datetime.fromisoformat(args.fetch_start).replace(
        tzinfo=timezone.utc) if args.fetch_start else WINDOW_START)
    fetch_end = (datetime.fromisoformat(args.fetch_end).replace(
        tzinfo=timezone.utc) if args.fetch_end else WINDOW_END)
    only_ids = None
    if args.only_ids:
        only_ids = {str(json.loads(line)["market_id"])
                    for line in Path(args.only_ids).open()}
    roster = load_roster(Path(args.markets), args.min_volume, args.min_days,
                         args.exclude_sports, fetch_start, fetch_end, only_ids)
    if only_ids is not None and len(roster) < len(only_ids):
        print(f"warning: {len(only_ids) - len(roster)} of {len(only_ids)} "
              f"requested ids not in roster (missing/filtered)")
    batches = plan_batches(roster, fetch_start, fetch_end)
    market_days = sum(r["hi"] - r["lo"] for r in roster) / 86400
    pts_per_day = 1440 / args.fidelity
    print(f"roster: {len(roster)} markets, {market_days:,.0f} market-days")
    print(f"plan: {len(batches)} batch calls (~{len(batches) * SLEEP_S / 60:.0f} min "
          f"at {1 / SLEEP_S:.0f} req/s), "
          f"est ~{market_days * pts_per_day * 27 / 1e6:,.0f} MB transfer"
          + (" (shards keep changes only)" if args.fidelity < 60 else ""))
    if args.dry_run:
        return

    shard_dir.mkdir(parents=True, exist_ok=True)
    done = skipped = 0
    for t0, t1, group in batches:
        shard = shard_dir / f"chunk_{t0}_{group[0]['market_id']}.json"
        if shard.exists():
            skipped += 1
            continue
        history = fetch_batch([r["token"] for r in group], t0, t1,
                              args.fidelity)
        if args.fidelity < 60:
            history = {tok: compress_changes(pts)
                       for tok, pts in history.items()}
        shard.write_text(json.dumps({
            "start_ts": t0, "end_ts": t1,
            "tokens": {r["token"]: r["market_id"] for r in group},
            "history": history,
        }))
        done += 1
        if done % 100 == 0:
            print(f"  {done}/{len(batches)} batches", flush=True)
        time.sleep(SLEEP_S)
    print(f"fetched {done} batches ({skipped} already present); "
          f"now run with --merge")


if __name__ == "__main__":
    main()
