# Task: breakout prediction for prediction markets, from news

You are a long-running automation agent. You monitor ${n_markets} prediction
market(s), each over a fixed window:

| market_id | question | monitor from | monitor until |
|---|---|---|---|
${markets_table}

Real-world news moves these markets. Your job: **predict abrupt price moves
("breakouts") before they start, from the news that drives them.** A
notification is the claim: *"this market will start a breakout in this
direction within the next ${breakout_window_hours} hours — this article is
my evidence."* Payload to `/notify`:
`{"market_id": ..., "news_id": ..., "direction": "up"|"down"}`.

A notification at or after the start of the move earns nothing — there is
no partial credit for reacting to a move already underway. The run's
horizon is unknown to you; data keeps coming until the experiment ends.

## Objective

After the fact, every *abrupt* price move (a day-over-day change
statistically large against that market's recent volatility, localized to
the minute it starts) is a breakout. Your notification is judged against
the first breakout of the named market that starts after it and within
${breakout_window_hours} hours, with the direction you named:

- **covered** — such a breakout exists: the breakout counts as covered and
  your notification as a covering one. The article you cite is recorded
  as your evidence but does not change the score; what counts is that the
  claim preceded the move in the right direction.
- **miss** — a breakout no notification of yours preceded.
- **false alarm** — no matching breakout within ${breakout_window_hours}
  hours. Wrong-direction calls count as false alarms, except when the
  opposite breakout was missed anyway — naming a direction never ranks
  worse than silence.

Your score is **cov-F1**: the harmonic mean of
**coverage** (covered breakouts / all breakouts) and
**precision** (covering notifications / all resolved notifications).
Higher is better. Quiet stretches with news but no moves, and moves with
no findable news, are both part of the task.

**One standing claim per market**: while a notification of yours on a
market is unresolved (it resolves when a breakout starts, or expires
${breakout_window_hours} h after you made it), further notifications on
that market are rejected free of charge. A wasted claim locks you out of
that market for up to ${breakout_window_hours} h — commit carefully.

## Budget and rate limits

You have **${budget_usd} total for the entire run**.${domain_caps} LLM calls bill real
token cost against it (${llm_price_table}); news calls bill the real
commercial rate below. When the budget is spent, LLM calls fail and paid
calls are refused — the run continues and your score suffers. Rate limits
(HTTP 429 on excess; track your own usage, nobody reports it):

| API | rate limit |
|---|---|
| news (search + articles) | ${news_rate_limit} |
| market data | ${price_rate_limit} |

## Data visibility

- News: searchable from the moment it is published.
- Prices: minute-grid change-points; a point at time t becomes visible at
  t + ${price_delay_minutes} minutes. Between two points the price is the
  earlier point's value (quiet stretches are flat, not missing).

## APIs

- `get_markets()` — the table above plus resolution criteria. Free.
- `get_costs()` — your cumulative spend so far, by category. Free.
- `get_prices(market_id, start, end, grid_minutes?)` — free (rate-limited).
  Minute change-series, clipped to visibility; grid_minutes >= 1
  coarsens the reply to the last change per N-minute bucket.
- `search_news(q, date_from?, date_to?, offset?)` — ${news_search_call} per
  page of ${search_top_k}. BM25 over published news; quoted phrases and
  AND/OR work; date filters apply to publish time.
- `get_article(news_id)` — ${article_call}. Full article text.
- `notify(market_id, news_id, direction)` — free; scored as above; one
  standing claim per market.

Time, scheduling, notify, and feedback are free.

The clock starts when the story is published and runs out when the move
starts: you must judge news on content, not wait for prices to confirm.
