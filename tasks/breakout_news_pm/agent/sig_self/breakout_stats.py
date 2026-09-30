"""The one robust-z breakpoint detector.

Median/MAD z on daily closes, after the dataset paper's Eq. 6. The env serves a
sparse minute change-series ({level_at_start, changes:{time,p}}, constant
between points), so `daily_closes` densifies it into one close per
calendar day before the z pass. Constants are fixed for the experiment
(parameters appendix) — do not tune per run. Pure functions of the price
payload: no env access, no spend.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

Z_THRESHOLD = 3.0   # robust z that counts as a breakout
TRAILING_DAYS = 30  # volatility window for the z denominator
MIN_DIFFS = 11      # minimum day-over-day moves before z is defined
MIN_SCALE = 0.005   # z-denominator floor: a dead-flat market (MAD 0)
#                     that steps is a breakout, not a divide-by-zero


def _parse(t: str) -> datetime:
    return datetime.fromisoformat(t.replace("Z", "+00:00"))


def daily_closes(payload: dict, start: datetime,
                 exclude_day: str | None = None) -> list[tuple[str, float, str]]:
    """(day, close, close_time) per calendar day of the payload's visible
    span, sorted. Sparse series are forward-filled: a day without changes
    closes at the carried level; its close_time is the day's last covered
    instant. Pass the current day as `exclude_day` when it is incomplete."""
    if payload.get("visible_until") is None:
        return []
    times = [_parse(t) for t in payload["changes"]["time"]]
    ps = payload["changes"]["p"]
    end = _parse(payload["visible_until"])
    level, close_time = payload["level_at_start"], None
    out: list[tuple[str, float, str]] = []
    i = 0
    day = start.astimezone(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0)
    while day <= end:
        day_end = min(day + timedelta(days=1) - timedelta(seconds=1), end)
        close_time = None
        while i < len(times) and times[i] <= day_end:
            level, close_time = ps[i], times[i]
            i += 1
        d = day.strftime("%Y-%m-%d")
        if level is not None and d != exclude_day:
            t_close = (close_time or day_end).strftime("%Y-%m-%dT%H:%M:%SZ")
            out.append((d, level, t_close))
        day += timedelta(days=1)
    return out


def robust_z(diffs: list[float]) -> float | None:
    """z of the last move vs the trailing window (median/MAD, Eq. 6 style)."""
    if len(diffs) < MIN_DIFFS:
        return None
    latest = diffs[-1]
    trailing = sorted(diffs[-(TRAILING_DAYS + 1):-1])
    n = len(trailing)
    med = trailing[n // 2] if n % 2 else (trailing[n // 2 - 1] + trailing[n // 2]) / 2
    devs = sorted(abs(d - med) for d in trailing)
    mad = devs[n // 2] if n % 2 else (devs[n // 2 - 1] + devs[n // 2]) / 2
    return (latest - med) / max(1.4826 * mad, MIN_SCALE)


def detect_breakpoints(closes: list[tuple[str, float, str]], since: datetime,
                       until: datetime) -> list[dict]:
    """Pure detector pass over daily closes: breakout days in [since, until]
    as {"t_move", "day", "z", "p_before", "p_after", "direction"}."""
    out = []
    lo, hi = since.strftime("%Y-%m-%d"), until.strftime("%Y-%m-%d")
    for k in range(1, len(closes)):
        day, p_after, close_time = closes[k]
        if not (lo <= day <= hi):
            continue
        diffs = [abs(closes[j + 1][1] - closes[j][1]) for j in range(k)]
        z = robust_z(diffs)
        if z is not None and z >= Z_THRESHOLD:
            p_before = closes[k - 1][1]
            out.append({"t_move": close_time, "day": day, "z": round(z, 2),
                        "p_before": p_before, "p_after": p_after,
                        "direction": "up" if p_after >= p_before else "down"})
    return out
