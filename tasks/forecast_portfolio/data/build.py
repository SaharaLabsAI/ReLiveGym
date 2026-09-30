"""Build the forecast_portfolio world from the frozen sample_v1.jsonl.

Outputs (data/built/, gitignored):
  questions.jsonl   one row per sampled market: an `agent` block (the ONLY
                    fields that may ever cross a tool boundary) and a
                    `scorer` block (ground truth + provenance slices)
  prices/<mid>.json {grid_hours: 1, points: [[t, p], ...]} hourly Yes-price,
                    forward-fill between points — scorer-side only (the
                    ta_bss_market anchor + the `easy` flag); the agent has
                    no price surface anywhere

Price sources: hard-copy from the bnpm hourly roster
(tasks/breakout_news_pm/market/raw/prices/) where present; the rest fetched
via clob batch-prices-history by reusing the bnpm fetch machinery
(fetch_prices.load_roster/plan_batches/fetch_batch). The build FAILS if any
scored question (resolved in-window, clean answer) ends up priceless.

Scoring facts pinned here:
  activation = max(startDate, 2026-03-01)         (staggered visibility)
  t_res      = min(closedTime, scheduled endDate) (UMA lag earns nothing)
  scored     = closedTime in [2026-03-01, 2026-07-01) and answer_status ok
  easy       = market price stayed >= 0.95 (or <= 0.05) for the question's
               entire scored life — reporting flag, settlement never
               branches on it

Usage:
    python build.py            # full build (network only for missing prices)
    python build.py --no-fetch # fail instead of fetching missing prices
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from tasks.breakout_news_pm.market import fetch_prices as fp  # noqa: E402

SAMPLE = HERE / "sample_v1.jsonl"
BUILT = HERE / "built"
BNPM_RAW = REPO / "tasks" / "breakout_news_pm" / "market" / "raw"

W0 = dt.datetime(2026, 3, 1, tzinfo=dt.timezone.utc)
W1 = dt.datetime(2026, 7, 1, tzinfo=dt.timezone.utc)
EASY_BAND = 0.95


def parse_ts(s: str | None) -> float | None:
    d = fp.parse_dt(s)
    return None if d is None else d.timestamp()


def fetch_missing_prices(missing: list[str]) -> dict[str, list[list[float]]]:
    """Hourly Yes-price series for `missing` market ids via the bnpm clob
    batch machinery (20 tokens/call, 15-day chunks), merged in memory."""
    roster = fp.load_roster(BNPM_RAW / "markets.jsonl", 0.0, 0.0,
                            exclude_sports=False, only_ids=set(missing))
    found = {r["market_id"] for r in roster}
    if found != set(missing):
        raise SystemExit(f"no roster entry (clobTokenIds?) for "
                         f"{sorted(set(missing) - found)}")
    series: dict[str, dict[int, float]] = {}
    batches = fp.plan_batches(roster)
    print(f"fetching {len(missing)} price series in {len(batches)} batch calls")
    for t0, t1, group in batches:
        history = fp.fetch_batch([r["token"] for r in group], t0, t1,
                                 fidelity=60)
        tok_mid = {r["token"]: r["market_id"] for r in group}
        for token, pts in history.items():
            series.setdefault(tok_mid[token], {}).update(
                {int(p["t"]): float(p["p"]) for p in pts})
    return {mid: sorted(pts.items()) for mid, pts in series.items()}


def life_levels(points: list[list[float]], lo: float,
                hi: float) -> list[float] | None:
    """Every price level in force during [lo, hi] under forward-fill; None
    if the market has no quote until after `hi`."""
    entering = None
    for t, p in points:
        if t > lo:
            break
        entering = p
    levels = [] if entering is None else [entering]
    levels += [p for t, p in points if lo < t <= hi]
    return levels or None


def market_ta(points: list[list[float]], lo: float, hi: float,
              y_yes: bool) -> float:
    """Time-averaged BSS of the market price read as a (p, 1-p) forecast
    over [lo, hi]; quoteless stretches before the first point score as
    abstention (0), mirroring the agent's no-forecast rule."""
    segs = []  # (t_from, level or None)
    entering = None
    for t, p in points:
        if t <= lo:
            entering = p
        elif t <= hi:
            segs.append((t, p))
    segs.insert(0, (lo, entering))
    total = 0.0
    for (t, p), t_next in zip(segs, [t for t, _ in segs[1:]] + [hi]):
        if p is not None:
            total += (1 - 2 * (p - (1.0 if y_yes else 0.0)) ** 2) * (t_next - t)
    return total / (hi - lo)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true",
                    help="fail instead of fetching missing prices")
    args = ap.parse_args()

    rows = [json.loads(line) for line in SAMPLE.open()]
    bad = [r for r in rows if r.get("fetch_error")]
    if bad:
        raise SystemExit(f"sample has fetch_error rows: {bad}")

    # -- prices: copy from bnpm roster, fetch the rest --------------------------------
    prices_dir = BUILT / "prices"
    prices_dir.mkdir(parents=True, exist_ok=True)
    missing = []
    for r in rows:
        mid = r["market_id"]
        src = BNPM_RAW / "prices" / f"{mid}.json"
        dst = prices_dir / f"{mid}.json"
        if dst.exists():
            continue
        if src.exists():
            shutil.copyfile(src, dst)
        else:
            missing.append(mid)
    if missing and args.no_fetch:
        raise SystemExit(f"{len(missing)} price series missing and --no-fetch "
                         f"set: {missing}")
    if missing:
        for mid, points in fetch_missing_prices(missing).items():
            (prices_dir / f"{mid}.json").write_text(
                json.dumps({"grid_hours": 1, "points": points}))
        still = [m for m in missing if not (prices_dir / f"{m}.json").exists()]
        if still:  # zero points everywhere — dead market, keep an empty file
            print(f"note: {len(still)} markets returned zero price points: "
                  f"{still}")
            for mid in still:
                (prices_dir / f"{mid}.json").write_text(
                    json.dumps({"grid_hours": 1, "points": []}))

    # -- questions.jsonl --------------------------------------------------------------
    out = []
    n_easy = 0
    ta_market = []
    for r in rows:
        mid = r["market_id"]
        open_ts = parse_ts(r["start_date"])
        sched_end_ts = parse_ts(r["scheduled_end"])
        closed_ts = parse_ts(r["resolution_date"])
        scored = bool(r["resolved_in_window"] and r["answer_status"] == "ok")
        act = max(open_ts or W0.timestamp(), W0.timestamp())
        t_res = (min(closed_ts, sched_end_ts) if scored else None)
        easy = None
        ta_m = None
        if scored:
            if t_res <= act:
                raise SystemExit(f"{mid}: t_res <= activation "
                                 f"({t_res} <= {act})")
            points = json.loads(
                (prices_dir / f"{mid}.json").read_text())["points"]
            levels = life_levels(points, act, t_res)
            if levels is None:
                raise SystemExit(f"{mid}: scored question has no price "
                                 f"quote inside its life")
            easy = (min(levels) >= EASY_BAND
                    or max(levels) <= 1 - EASY_BAND)
            n_easy += easy
            ta_m = market_ta(points, act, t_res,
                             y_yes=r["resolution_answer"] == "Yes")
            ta_market.append(ta_m)
        out.append({
            "question_id": mid,
            "agent": {
                "question": r["question"],
                "description": r["description"],  # verbatim
                "outcomes": r["outcomes"],
                "open_date": r["start_date"],
                "scheduled_end": r["scheduled_end"],
            },
            "scorer": {
                "scored": scored,
                "t_res": t_res,
                "closed_time": r["resolution_date"],
                "resolution_answer": r["resolution_answer"],
                "answer_status": r["answer_status"],
                "resolved_in_window": r["resolved_in_window"],
                "easy": easy,
                "ta_bss_market": (round(ta_m, 6) if ta_m is not None
                                  else None),
                "family": r["family"],
                "sched_end_bucket": r["sched_end_bucket"],
                "volume_usd": r["volume_usd"],
                "event_id": r["event_id"],
                "event_title": r["event_title"],
                "tags": r["tags"],
            },
        })
    with (BUILT / "questions.jsonl").open("w") as f:
        for q in out:
            f.write(json.dumps(q) + "\n")

    # -- audit block (pin in data/README.md + tests) ----------------------------------
    scored_rows = [q for q in out if q["scorer"]["scored"]]
    opens = sorted(parse_ts(q["agent"]["open_date"]) for q in out
                   if parse_ts(q["agent"]["open_date"]) >= W0.timestamp())
    print(f"questions: {len(out)}  scored: {len(scored_rows)}  "
          f"easy: {n_easy}  mid-window opens: {len(opens)}")
    print(f"ta_bss_market over scored: "
          f"mean {sum(ta_market) / len(ta_market):.4f}  "
          f"min {min(ta_market):.4f}  max {max(ta_market):.4f}")
    lives = sorted((q["scorer"]["t_res"]
                    - max(parse_ts(q["agent"]["open_date"]) or 0,
                          W0.timestamp())) / 86400 for q in scored_rows)
    print(f"scored life days: median {lives[len(lives) // 2]:.1f}  "
          f"p90 {lives[int(0.9 * len(lives))]:.1f}  under 7d "
          f"{sum(1 for d in lives if d < 7)}")


if __name__ == "__main__":
    main()
