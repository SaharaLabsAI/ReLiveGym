"""Sig-self feedback compilation for breakout_news_pm.

There is no feedback tool in this run. This module compiles a hindsight
learning signal from the ordinary paid tools: it buys realized price
history, runs the hard-coded breakpoint detector (robust median/MAD z on
daily closes, after the dataset paper's Eq. 6), and — only where a
breakpoint was detected — spends one metered LLM call attributing
candidate news items to the move. Candidates near no detected breakpoint
cost no LLM money.

The detector is self-derived and will NOT reproduce the gold breakpoint
set exactly; that noise is part of what a deployable signal costs.
Verdicts carry the detected move's direction so reflection can learn
direction errors, not just relevance. Constants below are fixed for the
experiment (do not tune them per run).

This file is provisioned only in `sig: self` cells. It holds no secrets
and cannot change what is measured (metering is server-side).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from breakout_stats import TRAILING_DAYS, daily_closes, detect_breakpoints  # noqa: F401
from runtime import llm_client
from runtime.env_client import iso

GRACE_HOURS = 24        # a move's period is closed this long after it
ATTR_WINDOW_HOURS = 96  # candidates published within this window before the
#                         move (+ the move day itself) are attributable
ATTR_THRESHOLD = 0.5    # LLM score that counts as attributed

_ATTR_PROMPT = """\
You are auditing a prediction market in hindsight.

Market question: {question}

On {day} the market's price moved abruptly: daily close went from
{p_before:.3f} to {p_after:.3f} (robust z = {z:.1f} vs the trailing month).

Below are candidate news items published shortly before the move. For each,
score from 0.0 to 1.0 how likely it is that this news item DROVE the move
(1.0 = clearly the driver, 0.0 = unrelated/noise). Score each item
independently; unrelated items must get low scores even if nothing fits.

{items}

Reply with ONLY a JSON object mapping news_id to score, e.g.
{{"abc123": 0.9, "def456": 0.1}}"""


def _parse(t: str) -> datetime:
    return datetime.fromisoformat(t.replace("Z", "+00:00"))


def hindsight_verdicts(env, market_id: str, since: datetime, until: datetime,
                       candidates: list[dict], question: str = "") -> dict:
    """Compile hindsight verdicts for `candidates` (dicts with news_id,
    title, published) over the closed period [since, until].

    Raises ValueError before spending anything if the period is not closed
    yet (until + grace must be in the past)."""
    now = env.now()
    if until + timedelta(hours=GRACE_HOURS) > now:
        raise ValueError(
            f"period not closed: until={until.isoformat()} + {GRACE_HOURS}h "
            f"grace is after now={now.isoformat()}")

    query_start = since - timedelta(days=TRAILING_DAYS + 2)
    prices = env.call("get_prices", market_id=market_id,
                      start=iso(query_start), end=iso(until))
    closes = daily_closes(prices, query_start)
    breakpoints = detect_breakpoints(closes, since, until)

    # news_id -> (best score, t_move, direction)
    scores: dict[str, tuple[float, str, str]] = {}
    for bp in breakpoints:
        t_move = _parse(bp["t_move"])
        day_end = datetime.strptime(bp["day"], "%Y-%m-%d").replace(
            tzinfo=timezone.utc) + timedelta(days=1)
        window_lo = t_move - timedelta(hours=ATTR_WINDOW_HOURS)
        eligible = [c for c in candidates
                    if window_lo <= _parse(c["published"]) <= day_end]
        if not eligible:
            continue
        items = "\n".join(
            f"- news_id={c['news_id']} published={c['published']}: "
            f"{c['title']}" for c in eligible)
        reply = llm_client.chat_json(env, [{
            "role": "user",
            "content": _ATTR_PROMPT.format(
                question=question or f"market {market_id}",
                day=bp["day"], p_before=bp["p_before"],
                p_after=bp["p_after"], z=bp["z"], items=items),
        }])
        if not isinstance(reply, dict):
            continue  # unparseable attribution: degrade to no attribution
        for c in eligible:
            try:
                s = float(reply.get(c["news_id"], 0.0))
            except (TypeError, ValueError):
                s = 0.0
            if s >= ATTR_THRESHOLD and s >= scores.get(
                    c["news_id"], (0.0, "", ""))[0]:
                scores[c["news_id"]] = (s, bp["t_move"], bp["direction"])

    verdicts = []
    for c in candidates:
        if c["news_id"] in scores:
            s, t_move, direction = scores[c["news_id"]]
            verdicts.append({"news_id": c["news_id"], "verdict": "attributed",
                             "score": round(s, 3), "t_move": t_move,
                             "direction": direction})
        else:
            verdicts.append({"news_id": c["news_id"], "verdict": "no_breakout"})
    return {"market_id": market_id, "since": since.isoformat(),
            "until": until.isoformat(), "breakpoints": breakpoints,
            "verdicts": verdicts}
