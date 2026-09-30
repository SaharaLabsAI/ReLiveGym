"""Hindsight breakpoint detection over raw/prices/*.json (daily grid).

Resample to UTC daily closes
(forward-filled across no-trade days), then flag day d when

    |close(d) - close(d-1)| >= min_dp   AND
    |dp| / stdev(trailing `window` daily changes, ddof=1) >= z_min

with at least `min_history` prior changes required. stdev == 0 with a
qualifying |dp| counts as a breakpoint (z reported as inf-capped 999).

Each breakpoint is then localized on the hourly series: the move provably
lies in (t_prev_trade, t_last_trade] (the trades anchoring the two closes);
[t_move_start, t_move_end] is the minimal contiguous hourly sub-interval
covering >= 80% of the daily move, and step_frac is the share of the move
carried by the single largest hourly step inside it. For news labeling,
"before the breakpoint" should mean published_at < t_move_start.

Output: raw/breakpoints.jsonl (market_id, date, t, p_prev, p, dp, z,
t_prev_trade, t_last_trade, t_move_start, t_move_end, step_frac) and
raw/breakpoints_stats.json (roster-level summary).

Usage:
    python detect_breakpoints.py [--min-dp 0.02] [--z-min 2.0]
                                 [--window 14] [--min-history 10]
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).parent
DAY = 86400
WINDOW_START = int(datetime(2026, 3, 1, tzinfo=timezone.utc).timestamp())
WINDOW_END = int(datetime(2026, 7, 1, tzinfo=timezone.utc).timestamp())


def daily_closes(points: list[list[float]]) -> list[tuple[int, float, int]]:
    """(day_start_ts, close, close_trade_ts) per UTC day, forward-filled."""
    closes: dict[int, tuple[float, int]] = {}
    for t, p in points:
        closes[int(t) // DAY * DAY] = (p, int(t))  # time-sorted; last wins
    days = sorted(closes)
    out, prev = [], None
    for d in range(days[0], days[-1] + DAY, DAY):
        prev = closes.get(d, prev)
        out.append((d, prev[0], prev[1]))
    return out


def localize(points: list[list[float]], t0: int, t1: int,
             dp: float) -> tuple[int, int, float]:
    """Minimal hourly sub-interval of (t0, t1] covering >=80% of the move dp.

    Returns (t_move_start, t_move_end, step_frac) where t_move_start is the
    timestamp of the last point BEFORE the interval (the pre-move anchor)
    and step_frac is the largest single hourly step's share of dp.
    """
    seg = [(int(t), p) for t, p in points if t0 <= t <= t1]
    if len(seg) < 2:
        return t0, t1, 1.0
    target = 0.8 * abs(dp)
    best = (seg[0][0], seg[-1][0])
    for i in range(len(seg) - 1):
        for j in range(i + 1, len(seg)):
            if (abs(seg[j][1] - seg[i][1]) >= target
                    and seg[j][0] - seg[i][0] < best[1] - best[0]):
                best = (seg[i][0], seg[j][0])
                break  # longer j only widens this i's window
    step = max(abs(seg[k][1] - seg[k - 1][1]) for k in range(1, len(seg)))
    return best[0], best[1], round(min(step / abs(dp), 1.0), 3) if dp else 1.0


def detect(points: list[list[float]], min_dp: float, z_min: float,
           window: int, min_history: int) -> list[dict]:
    closes = daily_closes(points)
    changes = [(closes[i][0], closes[i][1] - closes[i - 1][1], closes[i - 1][1],
                closes[i][1], closes[i - 1][2], closes[i][2])
               for i in range(1, len(closes))]
    bps = []
    for i, (d, dp, p_prev, p, t_prev, t_last) in enumerate(changes):
        if not (WINDOW_START <= d < WINDOW_END) or abs(dp) < min_dp:
            continue
        hist = [c[1] for c in changes[max(0, i - window):i]]
        if len(hist) < min_history:
            continue
        sd = statistics.stdev(hist) if len(hist) >= 2 else 0.0
        z = min(abs(dp) / sd, 999.0) if sd > 0 else 999.0
        if z >= z_min:
            m0, m1, step_frac = localize(points, t_prev, t_last, dp)
            bps.append({"t": d, "date": datetime.fromtimestamp(d, timezone.utc)
                        .strftime("%Y-%m-%d"), "p_prev": round(p_prev, 4),
                        "p": round(p, 4), "dp": round(dp, 4), "z": round(z, 2),
                        "t_prev_trade": t_prev, "t_last_trade": t_last,
                        "t_move_start": m0, "t_move_end": m1,
                        "step_frac": step_frac})
    return bps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default=str(HERE / "raw" / "prices"))
    ap.add_argument("--min-dp", type=float, default=0.02)
    ap.add_argument("--z-min", type=float, default=2.0)
    ap.add_argument("--window", type=int, default=14)
    ap.add_argument("--min-history", type=int, default=10)
    args = ap.parse_args()

    out_path = HERE / "raw" / "breakpoints.jsonl"
    n_markets = n_with_bp = n_bp = 0
    by_month: dict[str, int] = {}
    per_market: dict[str, int] = {}
    with out_path.open("w") as f:
        for pf in sorted(Path(args.prices).glob("*.json")):
            mid = pf.stem
            points = json.loads(pf.read_text())["points"]
            n_markets += 1
            bps = detect(points, args.min_dp, args.z_min,
                         args.window, args.min_history)
            if not bps:
                continue
            n_with_bp += 1
            per_market[mid] = len(bps)
            for bp in bps:
                f.write(json.dumps({"market_id": mid, **bp}) + "\n")
                by_month[bp["date"][:7]] = by_month.get(bp["date"][:7], 0) + 1
                n_bp += 1

    counts = sorted(per_market.values(), reverse=True)
    stats = {
        "params": {k: getattr(args, k) for k in
                   ("min_dp", "z_min", "window", "min_history")},
        "markets_scanned": n_markets,
        "markets_with_breakpoint": n_with_bp,
        "total_breakpoints": n_bp,
        "breakpoints_by_month": dict(sorted(by_month.items())),
        "per_market_max": counts[0] if counts else 0,
        "per_market_median": counts[len(counts) // 2] if counts else 0,
    }
    (HERE / "raw" / "breakpoints_stats.json").write_text(
        json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"wrote {n_bp} breakpoints -> {out_path}")


if __name__ == "__main__":
    main()
