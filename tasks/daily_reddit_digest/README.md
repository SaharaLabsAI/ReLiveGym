# daily_reddit_digest

Once a day, deliver a digest of 10 AI-subreddit posts from the previous
24 h that end up popular (final 24 h cascade ≥ 50 comments). A rung *below*
`reddit_ai_popularity` on the same world: nothing to predict — every
qualifying post is observable at delivery time (`num_comments` so far is
monotone) — so the task isolates **trigger discipline and cost**: act once in
a one-hour window each day, never miss a day, don't spend on monitoring.

## Task structure

- **World**: the sibling's `tasks/reddit_ai_popularity/data/built_min0`
  (no copy); `RootStore`/`CascadeStore`, visibility rule, fees
  (`api_call` $0.00024) and rate limit (1000 / 10 min) imported from
  `tasks/reddit_ai_popularity/task.py`.
- **Observation** (`tasks/reddit_ai_popularity/env/apps.py::RedditReadApp`):
  `list_posts` (sort by time or current `num_comments`), `get_post`,
  `get_cascade`.
- **Action** (`env/apps.py::DigestApp`): `digest(root_ids)` — one per day,
  accepted only inside the delivery window; `digest_status()` free,
  clock-only.

## Calendar & scoring (`task.py`)

`a = digest_hour_utc` (12). A scored day `d` needs its whole candidate
window inside the run (`a:00(d) − lookback ≥ sim_start`, `a:00(d) <
sim_end`) — 29 days in the April cell.

- candidate window `[a:00 − 24 h, a:00)`; delivery window `[a:00, a:00 + 1 h)`;
  settlement at `a:00 + 24 h`.
- `picks` = first `digest_size` distinct ids in list order; `hits` = picks in
  the window with final `descendants ≥ popular_min_desc`;
  `score(d) = hits / digest_size`, 0 for a day without a digest.
- **Primary metric** `digest_score` = mean over scored days (max). Secondaries:
  `days_delivered`, `days_full`, `mean_hits`, `delivery_latency_min_mean`,
  `rejections`.
- Free rejections (clock/shape only): outside every delivery window, second
  digest in a window, malformed payload. Ids are never validated at call
  time.

Feasibility: every April day has ≥ 35 posts already past 50
comments at 12:00 inside the window (≥ 29 for any hour), all on page 0 of
`list_posts(since=a−24h, sort="comments")`.

## Scaffolds

- TM-A / TM-B: generic `react` (`scaffolds/compose.py::_render_react`);
  TM-B's authored example is `agent/example_gatekeeper.py` (mechanism only).
- TM-C / TM-D: `task:cron_react` → `agent/scaffolds/cron_react_main.py`
  (the sibling's program; base cron once a day at `digest_hour_utc`:05 UTC, derived from `cell_config.TASK_PARAMS`); `agent/compose_spec.py` pins
  `alg: ("none",)` — no learning cells.

## Config reference

`configs/cells/tm{A,B,D}-tlrnnone-signone-algnone.yaml`. Task fields:
`history_days`, `horizon_hours` (=24), `digest_hour_utc`,
`delivery_window_hours`, `lookback_hours` (≤ horizon), `digest_size`,
`popular_min_desc`, `page_size`, `data_dir`, `cost.api_call`, `rate_limit`.

## Tests

`tests/tasks/daily_reddit_digest/` — calendar + scoring, env tool surface,
scaffold mounts, and a no-LLM end-to-end (scripted deliveries at 12:30 score
1.0; at 13:30 score 0).
