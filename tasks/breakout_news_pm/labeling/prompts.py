"""Prompts + episode-context builders for the hindsight attribution agent
. All labeling text lives here; run_labeling.py owns the tool loop.

One episode = one breakpoint. The agent sees the market, the breakpoint's
price context, and two retrieval tools over the CC-NEWS corpus; it must
search the full market question verbatim at least once, then submit
attribution groups with calibrated confidences via the `submit` tool.
"""

from __future__ import annotations

from datetime import datetime, timezone

MAX_DESCRIPTION_CHARS = 2_000
DAILY_BEFORE_D, DAILY_AFTER_D = 14, 7
HOURLY_BEFORE_S, HOURLY_AFTER_S = 72 * 3600, 24 * 3600

SYSTEM_PROMPT = """\
You are a hindsight news-attribution labeler for prediction-market price
moves. For one Polymarket market and one detected price breakpoint, your job
is to find the news story (if any) that caused the move, using a search tool
over a large news corpus, and to report it with a calibrated probability.

## The news corpus

~9.5M English articles from the Common Crawl news crawl, 2026-03-01 to
2026-07-07, ~8,700 domains. Important properties:

- Major fast wires are ABSENT (Reuters, AP, BBC, Bloomberg, NYT, WaPo,
  Guardian, CNBC, Politico block the crawler). The same stories still appear,
  with minutes-to-hours delay, via outlets that ARE present: Anadolu (aa.com.tr,
  anews.com.tr), Al Jazeera, TASS, CBS News, Fox News, press-release wires
  (PR Newswire, GlobeNewswire), finance aggregators (nasdaq.com, benzinga,
  marketbeat), cointelegraph, and thousands of regional papers. Do not
  conclude "no news" just because you don't see a wire-service byline.
- Every article carries a single `published` timestamp (UTC): when the
  article entered the news stream. It is reliable as an upper bound — the
  article existed by that time.
- The same story is usually syndicated across many domains. That is expected;
  you will group such duplicates into one attribution group.

## Search tools

- `search_news(query, date_from?, date_to?, top_k?)` — BM25 over
  title/description/body. Free terms are OR'd; use "quoted phrases" and AND
  for precision. Date filters apply to `published` (inclusive, ISO dates or
  datetimes).
- `get_article(news_id)` — full text of one result.
- `submit(...)` — deliver your final answer; this ends the episode.

## Required procedure

1. FIRST, run one `search_news` whose query is the FULL market question,
   verbatim, unquoted. This is mandatory (submissions are rejected until you
   have done it).
2. Then run several more searches with keyword combinations: named entities,
   event nouns, synonyms, tickers, place names. Search the candidate window
   given in the episode context first; widen or drop the date filter if it
   comes up empty. 3–8 searches is typical.
3. Open the most promising hits with `get_article` to verify that content and
   timing actually support causation, not just topical relevance.
4. Group articles reporting the SAME underlying story/event into one group
   (any number of publishers), and submit.

## Causation standard

A story can be the cause only if the underlying event became known before or
during the move window (compare `published` and the event time described in
the text against `t_move_start`). Direction
must make sense: news that should push the price the way it actually moved.
Set `likely_reports_move: true` on a group whose articles report the market
move itself ("Polymarket odds surge after...") rather than the cause — such
post-hoc reportage may still be useful evidence but is not the cause; if it
names the trigger event, prefer finding and attributing the trigger story
directly.

## Confidence calibration

`confidence` = P(this story is what actually drove this breakpoint). Across
all groups you ever emit at confidence c, a fraction ≈ c should truly be the
cause. Anchor on these bands:

- 0.80–0.95 — clear same-topic story, correct direction, and timing aligns
  with the hourly move window (event breaks hours before or during the move).
  Reserve ≥0.90 for airtight timing + explicit mechanism.
- 0.60–0.80 — strong candidate but timing is loose (day-level only) or the
  causal chain has one inferential step.
- 0.40–0.60 — plausible indirect driver (related development that could move
  the market, but no tight link).
- 0.20–0.40 — speculative; emit at most one such group, only if nothing
  stronger exists.
- Below 0.20 — do NOT emit. Background/topical relevance alone is not
  attribution.

Expected empirical distribution (a weak prior from comparable labeling
efforts on prediction-market moves; will be recalibrated after a pilot):
roughly 40–60% of breakpoints have NO attributable news at all — thin-market
noise, mechanical drift toward expiry, on-chain/whale flows, or information
that never hit the news. Submitting empty `groups` with
`no_attribution: true` is likely the single most common correct outcome; do
not force a low-quality attribution. Among episodes that
DO have attributions, a rough target histogram of group confidences:

    0.8–0.95  ████████████  ~35%
    0.6–0.8   █████████     ~25%
    0.4–0.6   ██████████    ~30%
    0.2–0.4   ███           ~10%

Be decisive: when timing and direction clearly line up, use the high band —
do not hedge a clear hit down to 0.6.
"""

# Tool definitions in OpenAI Responses-API (flat) format.
SUBMIT_TOOL = {
    "type": "function",
    "name": "submit",
    "description": "Deliver the final attribution for this breakpoint. "
                   "Ends the episode.",
    "parameters": {
        "type": "object",
        "properties": {
            "groups": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "story": {"type": "string",
                                  "description": "one-line story summary"},
                        "news_ids": {"type": "array", "minItems": 1,
                                     "items": {"type": "string"}},
                        "confidence": {"type": "number",
                                       "minimum": 0.2, "maximum": 0.95},
                        "likely_reports_move": {"type": "boolean"},
                        "evidence": {"type": "string",
                                     "description": "timing + content "
                                                    "rationale"},
                    },
                    "required": ["story", "news_ids", "confidence",
                                 "likely_reports_move", "evidence"],
                },
            },
            "no_attribution": {
                "type": "boolean",
                "description": "true iff groups is empty",
            },
            "notes": {"type": "string"},
        },
        "required": ["groups", "no_attribution"],
    },
}

SEARCH_TOOLS = [
    {
        "type": "function",
        "name": "search_news",
        "description": "BM25 search over the news corpus. Returns "
                       "headlines + snippets with news_ids.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "date_from": {"type": "string",
                              "description": "inclusive ISO date/datetime "
                                             "on the publish time"},
                "date_to": {"type": "string"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 25},
            },
            "required": ["query"],
        },
    },
    {
        "type": "function",
        "name": "get_article",
        "description": "Fetch one article's full text by news_id.",
        "parameters": {
            "type": "object",
            "properties": {"news_id": {"type": "string"}},
            "required": ["news_id"],
        },
    },
]


def _utc(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC")


def daily_closes(points: list[list[float]], bp_date: str) -> str:
    """Forward-filled UTC daily closes around the breakpoint day."""
    day0 = datetime.fromisoformat(bp_date).replace(
        tzinfo=timezone.utc).timestamp() // 86400
    closes: dict[int, float] = {}
    for t, p in points:
        closes[int(t // 86400)] = p  # points are time-sorted: last wins
    lines, last = [], None
    for d in range(int(day0) - DAILY_BEFORE_D, int(day0) + DAILY_AFTER_D + 1):
        if d in closes:
            last = closes[d]
        if last is None:
            continue
        date = datetime.fromtimestamp(d * 86400, tz=timezone.utc).date()
        mark = "   <-- breakpoint day" if str(date) == bp_date else ""
        lines.append(f"  {date}  {last:.3f}{mark}")
    return "\n".join(lines)


def hourly_moves(points: list[list[float]], bp: dict) -> str:
    """Hourly series ±(72h, 24h) around the move, compressed to price
    changes so quiet stretches don't burn tokens."""
    lo = bp["t_move_start"] - HOURLY_BEFORE_S
    hi = bp["t_move_end"] + HOURLY_AFTER_S
    window = [(t, p) for t, p in points if lo <= t <= hi]
    lines, prev = [], None
    for t, p in window:
        if prev is not None and abs(p - prev) < 1e-9:
            continue
        lines.append(f"  {_utc(int(t))}  {p:.3f}")
        prev = p
    return "\n".join(lines) or "  (no trades in window)"


def build_episode_prompt(market: dict, bp: dict,
                         points: list[list[float]]) -> str:
    desc = (market.get("description") or "").strip()
    if len(desc) > MAX_DESCRIPTION_CHARS:
        desc = desc[:MAX_DESCRIPTION_CHARS] + " [...truncated]"
    d = datetime.fromisoformat(bp["date"])
    win_from = (d.replace(tzinfo=timezone.utc).timestamp() - 3 * 86400)
    win_from_s = datetime.fromtimestamp(win_from, tz=timezone.utc).date()
    direction = "UP" if bp["dp"] > 0 else "DOWN"
    return f"""\
## Market

Question: {market['question']}
Category: {market['category']}   Volume: ${market['volume']:,.0f}
Event: {market.get('event_title') or '-'}
Market window: {market.get('start_date')} .. {market.get('end_date')}\
{f" (closed {market['closed_time']})" if market.get('closed_time') else ''}

Resolution criteria:
{desc or '(none available)'}

## Breakpoint to attribute

Detector definition: a flagged day is a UTC daily-close move |Δp| ≥ 0.02
whose magnitude is ≥ 2 standard deviations of the trailing 14 daily changes.

Day: {bp['date']}   close moved {bp['p_prev']:.3f} -> {bp['p']:.3f} \
(Δp = {bp['dp']:+.3f}, {direction}, z = {bp['z']:.1f})
Move localized hourly: starts {_utc(bp['t_move_start'])}, \
ends {_utc(bp['t_move_end'])}\
{f" (one dominant step)" if bp.get('step_frac', 0) >= 0.9 else ''}
Last trade before the move: {_utc(bp['t_prev_trade'])}

## Price context (Yes-price, UTC)

Daily closes:
{daily_closes(points, bp['date'])}

Hourly around the move (rows shown only where the price changed):
{hourly_moves(points, bp)}

## Task

Find the news story, if any, that caused this move. Default candidate news
window: {win_from_s} .. {bp['date']} (you may search outside it). Remember:
your FIRST search must be the full market question verbatim. Then keyword
searches, open promising articles, and submit calibrated groups.
"""
