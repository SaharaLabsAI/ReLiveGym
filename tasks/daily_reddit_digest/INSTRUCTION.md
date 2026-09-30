# Task: deliver a daily digest of the AI-subreddit posts that became popular

You are a long-running automation agent watching a set of AI subreddits (OpenAI,
Anthropic, LocalLLaMA, r/singularity, …). New posts arrive continuously. Your
job: **once a day, deliver a digest of ${digest_size} posts from the previous
24 hours whose discussions turn out popular.** A digest is one call to the
environment's `/notify` with payload `{"root_ids": [...]}`.

## The daily calendar (all times UTC)

Each scored day `d` is pinned to **${digest_hour}:00 UTC**:

- **Candidate window**: posts created in
  `[${digest_hour}:00 − ${lookback_hours} h, ${digest_hour}:00)` of day `d`.
  Only these posts can earn credit for day `d`. The window is fixed to the
  day's ${digest_hour}:00, not to the moment you deliver.
- **Delivery window**: your digest for day `d` is accepted only while the
  simulated clock is in
  `[${digest_hour}:00, ${digest_hour}:00 + ${delivery_window_hours} h)` of
  day `d`. The first accepted digest in that window is day `d`'s digest.
- **Settlement**: day `d` is scored at `${digest_hour}:00 + ${horizon_hours} h`,
  once every candidate's final comment count is frozen.

Worked example — the first scored day, ${first_day}: candidate window
`${example_window_lo}` to `${example_window_hi}`; digest accepted from
`${example_deliver_lo}` until `${example_deliver_hi}`; scored at
`${example_settle}`. There are ${n_days} scored days in this run
(${first_day} through ${last_day}); a day without an accepted digest scores 0.

## Objective

A post's popularity is the size of its comment cascade `${horizon_hours}`
hours after posting: `descendants` = the number of comments in that window.
It is hidden until the reveal, then frozen. A post is **popular** when its
final `descendants ≥ ${popular_min_desc}`.

Day score:

    picks(d)  = the first ${digest_size} distinct ids of your root_ids, in list order
    hits(d)   = picks that are in day d's candidate window AND are popular
    score(d)  = hits(d) / ${digest_size}            (0 if no digest was accepted for d)

Run score = the average of `score(d)` over all ${n_days} scored days. Higher
is better. Duplicate ids count once; ids beyond the first ${digest_size}
distinct ones are ignored; ids outside the window, unknown ids, and
non-popular posts earn nothing.

## Budget and rate limit

You have **${budget_usd} total for the entire run**.${domain_caps} LLM calls bill real token
cost against it (${llm_price_table}); the API calls below bill ${api_call}
each. When the budget is spent, LLM calls fail and paid calls are refused —
the run continues and your score suffers. The API additionally enforces
**${rate_limit}** (HTTP 429 on excess). Track your own usage; the environment
does not report it.

## Rules (violating requests are rejected free of charge, HTTP 400)

- A digest outside every delivery window is rejected (the message names the
  next window).
- A second digest inside the same delivery window is rejected.
- `root_ids` must be a non-empty list of post ids. Ids are not checked when
  you deliver; they are scored at settlement.

## Data visibility

- A post's static metadata (subreddit, author, title, selftext, url, flair) is
  visible from the moment it is posted.
- Every post carries `num_comments` — the comments posted so far, i.e. the
  live cascade size (it equals `descendants` once revealed). Listing sorts
  by time (`sort=new`) or by current `num_comments` (`sort=comments`); the
  ranking is a snapshot of now, never of the future.
- `get_cascade` returns the reply tree *as it has grown so far* — every
  comment with `created_utc ≤ now`, its `parent_id` (the tree structure),
  `author`, and `body`.
- What you **cannot** see: any comment's vote score (never shown), future
  comments (only those posted by `now`), and the final `descendants` label,
  which is null until `reveal_at = posted_at + ${horizon_hours} h`.
- Data reaches back to ${history_start}; everything older than
  `${horizon_hours}` h is fully revealed (final `descendants` and complete
  trees).

## APIs

- `list_posts(since?, until?, subreddit?, sort?=new|comments, order?=asc|desc,
  offset?)` — one page of up to ${page_size} root posts plus `total_hits`;
  each post carries `num_comments` (so far). `descendants` is null for posts
  not yet revealed.
- `get_post(root_id)` — one post's metadata including `selftext`.
- `get_cascade(root_id, offset?)` — the revealed reply-tree prefix (up to
  ${page_size} nodes/page) with running `n_nodes` / `n_comments` counts.
- `digest_status()` — free: whether now is inside a delivery window, whether
  today's digest has been delivered, and the next window. Clock-only.
- `/notify` (`digest(root_ids)`) — free to call; the accepted digest settles
  at the day's settlement time.

All three data APIs bill ${api_call} per call and share the
${rate_limit} limit. Time, scheduling, notify, and digest_status are free.

The run's budget covers everything you spend — LLM tokens and API fees.
Spend your attention where it changes decisions.
