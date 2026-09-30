# Task: high-temperature alerting

You are a long-running automation agent. Watch the hourly temperature and, for
each UTC calendar day whose temperature reaches **${threshold_c} °C at any
hour**, send **exactly one notification for that day** — as promptly as
possible after the first crossing hour becomes observable. Days that never
reach ${threshold_c} °C must get no notification.

## Objective

Maximize your detection score, computed at the end of the run:

- Each crossing day you notify validly earns a **timeliness credit**
  `max(0, 1 − delay/${credit_hours})`, where delay = hours between the first
  crossing hour and your notification. Missed crossing days earn 0.
- **TC-recall** = mean credit over all crossing days. **Precision** =
  properly notified crossing days / (those + days with a false alarm).
- Your score is **TC-F1**, the harmonic mean of precision and TC-recall.
  Higher is better.

The run's horizon is unknown to you; data keeps coming until the experiment
ends.

## Budget and rate limit

You have **${budget_usd} total for the entire run**.${domain_caps} LLM calls bill real
token cost against it (${llm_price_table}); when the budget is spent, LLM
calls fail and paid calls are refused — the run continues and your score
suffers. The weather query itself is free but rate-limited to
**${weather_rate_limit}** — exceeding it gets HTTP 429. Track your own
usage; the environment does not report it.

## Data visibility

The temperature reading for hour H covers the interval H..H+1 and becomes
visible at time H+1 — you can never observe the current, incomplete hour.
Queries are clipped to what exists; the response reports the effective range
(`clipped_start` / `clipped_end`).

## Notifications

- A notification names one UTC calendar day: payload `{"date": "YYYY-MM-DD"}`.
- Send it only after you have seen data proving the crossing. A notification
  sent before the crossing data was visible counts as a false alarm, even if
  the day later crosses (the day can still be properly notified afterwards).
- One per day: extra notifications for the same day count as duplicates
  (reported, never rewarded).
- A day closes for scoring ${grace_hours} hours after it ends; closed days can
  no longer be notified, and a crossing day with no valid notification by then
  counts as missed.

Because data for hour H appears at H+1, the best achievable delay is 1 hour.

## Feedback

Scored outcomes for closed days are available for free from the environment's
feedback tool (may be disabled in some runs).
