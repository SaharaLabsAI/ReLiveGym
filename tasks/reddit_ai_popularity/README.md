# reddit_ai_popularity

Recommend the AI-subreddit discussion threads whose comment cascade will grow
large — as early as possible. The distinguishing feature is the
**timestamped comment cascade**: the agent watches the reply tree grow in
simulated time and calls a trend before the outcome is revealed.

## Task structure

- **World**: `data/built_min0/` — 229,810 root posts across 10 AI subreddits
  (Mar–Jul 2026), each with its full timestamped reply tree. Built by
  `data/build.py`; see [data/README.md](data/README.md).
- **Root = a submission**; cascade = its comment tree. Metadata (subreddit,
  author, title, selftext, url, flair) is visible from post time.
- **Observation** (`env/apps.py`): `list_posts` (page sorted by time or by
  current `num_comments`; every post carries `num_comments`-so-far), `get_post`,
  `get_cascade` (the growing reply-tree prefix), `quota` (free).
- **Action**: `recommend(root_id)` — flag a post as going-to-trend.

## Ground truth & scoring

- **Label** `descendants` = number of comments within `horizon_hours` (24 h) of
  the root, computed from the cascade timestamps. Hidden until
  `reveal_at = post + 24 h`.
- **Popularity** `pop = log2(max(descendants, 1)) + 1`.
- Each accepted recommendation settles at the reveal as **tail**
  (`descendants ≥ tail_min_desc`, default 50) or **nontail**, with a
  timeliness weight `max(0, 1 − delay/decay_hours)`. `results.json` carries
  `twr_at_cap` — time-weighted tail recall under the rolling `daily_cap` —
  as `performance.primary`, with precision, recall and latency as
  secondaries. The paper's metric, **time-weighted F1** (precision =
  popular / settled recommendations; recall = the first-recommendation
  decay weights of popular posts / Σ_days min(cap, popular posts that
  day)), is derived from `results.json` by `scripts/primary_metric.py`.
  Resources are constraints, not score: the `budget_usd` wallet (LLM
  tokens at real rates + `api_call` $0.00024 fees) and the documented
  1000-per-10-min rate limit.
- **No `score`, ever.** Reddit vote scores (root and comment) are ingestion-time
  snapshots and are never loaded or served; nor is the root's ingested final
  `num_comments` — the served `num_comments` is recomputed from the cascade
  prefix at `now`. The only exposed label is `descendants` at the reveal.

## Leak-safety (the reveal)

Visibility gating lives in `RootStore`/`CascadeStore` (`task.py`):
future posts/nodes are never returned (time filters clip to `now`); the
`descendants` label is `null` until `reveal_at`; the cascade prefix is
score-stripped. Reply trees are causal, so a revealed prefix is always a
connected subtree.

## Evaluation

`metrics()` gives `twr_at_cap`, precision, recall, the tail universe size,
median delay, and a weight-1 `twr_ceiling_at_cap`; `report()` adds per-day
outcome tables and the full recommendation list. Reference ceiling:
`agent/baselines/cascade_forecaster` (no-LLM blanket prober).

## Learning cells

`sig: oracle` adds `get_feedback`, the settled-outcome feed of the agent's
own recommendations; `alg: memory` renders those outcomes into the prompt
(`agent/records.py`, `agent/prompts/reflect.md`) on the `tlrn: daily`
trigger. Under TM-D the main is `agent/scaffolds/cron_react_main.py`, an
hourly base cron the agent may edit.

## Config reference

`configs/cells/`: `tm{A,B,D}-tlrnnone-signone-algnone.yaml` and
`tm{A,B,D}-tlrndaily-sigoracle-algmemory.yaml` (April 2026). Task fields: `history_days`
(revealed lookback), `horizon_hours` (=24), `decay_hours`, `daily_cap`,
`tail_min_desc`, `page_size`, `data_dir`, `cost.api_call`, and
`rate_limit`. `from_run_config` validates the run window against
`build_stats["window"]` and that `horizon_hours ≤
horizon_days_collected·24`.
