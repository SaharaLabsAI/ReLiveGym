# Task: standing forecasts on real-world questions

You are a long-running automation agent. You maintain **standing
probability forecasts** on a portfolio of real-world questions (elections,
conflicts, economic events, ...). Questions arrive over time and resolve
to exactly one of their listed outcomes; new questions may appear at any
moment. The run's horizon is unknown to you; data keeps coming until the
experiment ends.

## Objective

For each question, your **standing forecast** is a probability assignment
over its outcomes, submitted via `submit_forecast`. It stays in force
until you replace it or the question resolves. At every moment of a
question's life, your current forecast earns the score

```
1 - sum over outcomes o of (p_o - result_o)^2
```

where `result_o` is 1 for the outcome the question resolves to and 0
otherwise. Your score on the question is the **time average** of this over
the question's entire life — from the moment it is added until it
resolves. Anchors: a confidently correct forecast earns +1, a confidently
wrong one −1, and probability spread evenly over the outcomes earns
+0.5. Your overall score is the average across questions. Higher is
better.

**Until you first submit on a question, the even spread stands in for
you**: an unattended question scores as if you had said 1/N on each of
its N outcomes (+0.5 on a two-outcome question). So silence is not a
zero — it is the uninformed forecast, and what you are paid for is
being better than it, for as much of each question's life as you can.

**Submit a complete distribution.** `forecast` must give a probability
for **every** outcome of the question, summing to 1 — `{"Yes": 0.7,
"No": 0.3}`, not `{"Yes": 0.7}`. Any outcome you leave out is scored as
if you had said its probability is **zero**; it does not fall back to
the even spread. Omitting the outcome that actually happens is the
single most expensive mistake available: on a two-outcome question,
`{"Yes": 0.5}` scores −0.25 when the answer is No, which is worse than
never having submitted at all.

Properties that follow directly from the rule:

- Score accrues continuously from the moment a question is added; time
  spent below +0.5 is averaged in and cannot be earned back later.
- A question can resolve **before** its scheduled close, at any moment.
- Re-submitting identical numbers changes nothing.

## Budget and rate limits

You have **${budget_usd} total for the entire run**.${domain_caps} LLM calls bill real
token cost against it (${llm_price_table}); news calls bill the real
commercial rate below. When the budget is spent, LLM calls fail and paid
calls are refused — the run continues and your score suffers. Rate limits
(HTTP 429 on excess; track your own usage, nobody reports it):

| API | rate limit |
|---|---|
| news (search + articles) | ${news_rate_limit} |

## Data visibility

- News: searchable from the moment it is published.
- Questions: visible from the moment they are added. When a question
  resolves, its outcome appears in `list_questions` / `get_question`.

## APIs

- `list_questions(status?, added_after?)` — free. Index of the questions
  added so far: id, question, added_at, scheduled_close, status; resolved
  ones include their outcome. `added_after` returns only questions added
  strictly after that time.
- `get_question(question_id)` — free. Full detail: outcomes and the
  question's precise resolution criteria.
- `get_forecasts(question_id?)` — free. Your own submission history and
  current standing forecasts.
- `get_costs()` — your cumulative spend so far, by category. Free.
- `search_news(q, date_from?, date_to?, offset?)` — ${news_search_call} per
  page of ${search_top_k}. BM25 over published news; quoted phrases and
  AND/OR work; date filters apply to publish time.
- `get_article(news_id)` — ${article_call}. Full article text.
- `submit_forecast(question_id, forecast, news_id?)` — free; the scored
  action. `forecast` is `{outcome: probability, ...}` covering **every**
  outcome of the question, probabilities ≥ 0 summing to 1 (a sum below 1
  is accepted, but the missing mass is scored as zero on the outcomes
  you omitted — see the objective). You may cite the article the
  forecast rests on via `news_id`. Invalid submissions (unknown
  question, resolved question, bad probabilities, a `news_id` that is
  unknown or not yet published) are rejected free of charge.

Time and scheduling are free.
