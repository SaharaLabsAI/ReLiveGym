# Task: recommend the AI-subreddit threads that will become popular

You are a long-running automation agent watching a set of AI subreddits (OpenAI,
Anthropic, LocalLLaMA, r/singularity, …). New posts arrive continuously. Your
job: **recommend the posts whose discussion will turn out large — as early as
possible**. A recommendation names one post: payload `{"root_id": ...}` to the
environment's `/notify`.

## Objective

A post's popularity is the size of its comment cascade `${horizon_hours}` hours
after posting: `descendants` = the number of comments in that window. It is
hidden until the reveal, then frozen. A post is a **tail post** when its final
`descendants ≥ ${tail_min_desc}` — those are the posts you are meant to catch.

Your score, computed at the end of the run, is **time-weighted tail recall**:

    TWR = Σ over tail posts of weight(your first recommendation) / (number of tail posts)
    weight = max(0, 1 − (hours between posting and your recommendation) / ${decay_hours})

- Recommend at the moment of posting: weight 1 — full credit.
- The weight falls linearly with delay. Concretely: ${decay_examples}.
- Tail posts you never recommend earn 0. Higher TWR is better.

You get at most **${daily_cap} accepted recommendations per rolling 24 h** —
far fewer than the tail posts that exist — so every slot spent on a post that
fizzles is a tail post forgone. Roughly 3–4% of posts end tail-sized; the
median post ends with only a handful of comments.

## Budget and rate limit

You have **${budget_usd} total for the entire run**.${domain_caps} LLM calls bill real token
cost against it (${llm_price_table}); the API calls below bill ${api_call}
each. When the budget is spent, LLM calls fail and paid calls are refused —
the run continues and your score suffers. The API additionally enforces
**${rate_limit}** (HTTP 429 on excess). Track your own usage; the environment
does not report it.

## Rules (violating requests are rejected free of charge, HTTP 400)

- At most ${daily_cap} accepted recommendations per rolling 24 h window.
- Each post can be recommended at most once.
- Only posts made after the run started and not yet revealed are recommendable:
  pre-run history and already-revealed posts are rejected.

## Data visibility

- A post's static metadata (subreddit, author, title, selftext, url, flair) is
  visible from the moment it is posted.
- **The comment cascade is your early signal.** `get_cascade` returns the reply
  tree *as it has grown so far* — every comment with `created_utc ≤ now`, its
  `parent_id` (the tree structure), `author`, and `body`. You watch reply
  velocity and branching build up and judge whether a post is taking off.
- What you **cannot** see: any comment's vote score (never shown), future
  comments (only those posted by `now`), and the final `descendants` label,
  which is null until `reveal_at = posted_at + ${horizon_hours} h`.
- Data reaches back to ${history_start}; everything older than
  `${horizon_hours}` h is fully revealed (final `descendants` and complete
  trees) — free study material for what early cascade shapes lead to large
  final discussions.
- Every post carries `num_comments` — the comments posted so far, i.e. the
  live cascade size (it equals `descendants` once revealed). Listing sorts
  by time (`sort=new`) or by current `num_comments` (`sort=comments`); the
  ranking is a snapshot of now, never of the future.

## APIs

- `list_posts(since?, until?, subreddit?, sort?=new|comments, order?=asc|desc,
  offset?)` — one page of up to ${page_size} root posts plus `total_hits`;
  each post carries `num_comments` (so far). `descendants` is null for posts
  not yet revealed.
- `get_post(root_id)` — one post's metadata including `selftext`.
- `get_cascade(root_id, offset?)` — the revealed reply-tree prefix (up to
  ${page_size} nodes/page) with running `n_nodes` / `n_comments` counts.
- `quota()` — free: your rolling-24h recommendation budget.
- `/notify` (`recommend(root_id)`) — free to call; accepted recommendations
  settle at the post's reveal as tail (with your timeliness weight) or nontail.

All three data APIs bill ${api_call} per call and share the
${rate_limit} limit. Time, scheduling, notify, quota, and feedback are free.

Every hour you wait before recommending shrinks the credit linearly — but the
longer you watch a cascade, the better you can tell a trender from a dud that
wastes one of your ${daily_cap} daily slots. Spend your attention where it
changes decisions.
