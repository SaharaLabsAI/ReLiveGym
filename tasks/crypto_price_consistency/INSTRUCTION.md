# Task: hourly crypto price reporting, robust to API failures

You are a long-running automation agent. Every hour, for each of
${n_symbols} symbol(s) — ${symbols} — you must either **report** the
current spot price in USDT or explicitly **abstain**. The run's horizon is
unknown to you; hours keep coming until the experiment ends.

## Objective

Minimize your reporting error, measured at the end of the run:

- Each reported symbol-hour scores
  `excess_bps = min(${cap_bps}, max(0, error_bps − ${free_bps}))` against
  the reference price (the cross-venue median at the moment you submit —
  report promptly once you trust your number). Within ${free_bps} bps is
  a perfect report.
- Your score is the **mean excess_bps over reported hours** — lower is
  better — subject to two hard rules:
  - **Answer every hour.** A symbol-hour with neither report nor abstain
    is a miss — a protocol violation that invalidates the run's ranking.
  - **Abstentions are budgeted.** You may abstain on at most
    ${abstain_budget_pct}% of scored hours. Abstaining is the honest move
    when no source can be trusted; it books no error but spends budget.

The hierarchy still matters: a verified report beats abstaining, an
abstention beats a wildly wrong number (which costs the full ${cap_bps}
bps cap), and silence is never acceptable.

## Budget and rate limits

You have **${budget_usd} total for the entire run**.${domain_caps} LLM calls bill real
token cost against it (${llm_price_table}); when the budget is spent, LLM
calls fail — the run continues and your reporting degrades. The venue
APIs are free public endpoints, but each venue enforces its documented
rate limits (429 on excess; Binance escalates ignored 429s into long IP
bans). Track your own usage; nobody reports it to you.

## Data sources

Three exchanges serve public 5-minute candles over plain HTTP via
`http_fetch(venue, path, params, timeout_s?)`. Call `get_exchange_docs()`
(free) for the endpoint contract:

| symbol | venue spellings |
|---|---|
${symbols_table}

These are real public APIs with real behavior. Expect everything you would
expect operating against the open internet for months: outages, slow or
flaky responses, rate limits, and occasionally responses that are
well-formed but not what they should be. No source is authoritative; you
have three for a reason.

## Scoring detail

Each full hour, per symbol, the **last** action you took inside that hour
is scored (earlier actions in the hour are superseded — correcting
yourself is free):

| outcome | effect |
|---|---|
| report within ${free_bps} bps | excess_bps 0 (perfect) |
| report off by e bps | excess_bps = min(${cap_bps}, e − ${free_bps}) |
| abstain | no error booked; consumes abstention budget |
| no action that hour | protocol violation (miss) |

## APIs

- `get_exchange_docs()` — free: venues, endpoints, reply shapes, rate
  limits.
- `http_fetch(venue, path, params, timeout_s?)` — free, rate-limited by
  each venue (timeout_s is capped at ${timeout_max_s}s; timeouts still
  cost you the wait in freshness). Returns the raw wire result:
  `{status, headers, body, elapsed_ms}` with `body` an unparsed string,
  or `{error, elapsed_ms}` when no response arrived.
- `report(symbol, price)` / `abstain(symbol, reason?)` — free; scored as
  above.
- `get_costs()` — free: cumulative spend by category.
- `get_feedback(since?)` — free: settled hours with the reference price,
  your submission, and its excess_bps — available once each hour closes
  (may be disabled in some runs).
- `get_time()` and the scheduling and waiting tools in your tool
  manifest — free: simulated time and scheduling.

An hour costs you at minimum the work it takes to know a price; spend
that work where it changes what you report.
