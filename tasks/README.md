# Tasks

One directory per task. The harness (`harness/`) is task-agnostic; everything
task-specific lives here behind the `Task` interface in
[harness/task.py](../harness/task.py).

## Anatomy of a task

```
tasks/<name>/
  README.md        experimenter-facing: task structure, ground truth, metrics, config reference
  INSTRUCTION.md   agent-facing task spec — ONLY what the agent may know. A string.Template
                   rendered into the workspace at run init from the run's actual config, so
                   the spec never disagrees with the prices being charged
  task.py          TASK, a harness.task.Task subclass: config model, data store + visibility
                   rule, env-app provisioning, notification validation, scorer
  env/apps.py      the task's server-side tools (observation + action), served through the
                   tool manifest (GET /tools)
  agent/           agent-side material the constructor mounts into workspaces: compose_spec.py
                   (what to mount), scaffolds/ (checked-in program mains), records.py +
                   prompts/reflect.md (the learning stack's task semantics), example_gatekeeper.py
                   (the TM-B teaching example), baselines/ (no-LLM anchors)
  configs/         run YAMLs of the paper's arms
  data/            build scripts + provenance (built data is not shipped: see ../DATA.md)
```

Run YAML shape: harness fields (`run_id`, `cell`, `sim_start`, `sim_end`,
`budget_usd`, `domain_budgets`, `agent`, `seed`) at top level; task fields under
`task:` with a `name:` that selects `tasks/<name>/task.py`. The model is a
launch-time argument (`--model api_provider:model`), never a YAML field.

## The eight benchmark tasks

| directory | paper name | world | commitment | primary metric |
|---|---|---|---|---|
| `breakout_news_pm` | Breakout News Detection | Polymarket prices + CC-NEWS search | alert ⟨market, article, direction⟩ before a price breakout | coverage F1 (`cov_f1`) |
| `forecast_portfolio` | Forecast Portfolio | Polymarket questions + CC-NEWS search (no prices) | standing probability forecast per question | time-averaged Brier skill score |
| `resolution_detect` | Resolution Detection | Polymarket questions + CC-NEWS search (no prices) | one claim per question that its outcome is decided | time-weighted F1 (`tc_f1`) |
| `reddit_ai_popularity` | Popularity Prediction | ten AI subreddits, replayed comment cascades | recommend a post that will reach 50 comments | time-weighted F1 |
| `daily_reddit_digest` | Daily Digest | the same subreddit replay | one digest of 10 popular posts per day | on-time accuracy (`digest_score`) |
| `crypto_price_consistency` | Market Price Check | three exchanges' 5-minute candles behind an incident-injecting proxy | hourly spot price report per symbol | on-time accuracy (`price_acc`) |
| `edgar_portfolio` | Web Filing Tracking | replayed SEC EDGAR endpoints + a portfolio web app (browser) | filing records entered through the web form | on-time accuracy (`filing_score`) |
| `broker_ops` | Web Broker Ops | broker web portal + read-only account API + ops inbox (browser) | protective stops and funding instructions in time | on-time accuracy (`routine_score`) |

`weather_fixture/` is not a benchmark task: it is the small reference task the harness
test suite and its fixture agents run against.

## Program shapes (`agent.scaffold`)

- `react` — the constructor-generated persistent ReAct actor (TM-A / TM-B).
- `task:<name>` — a checked-in main declared in the task's `compose_spec.py`
  (`TASK_SCAFFOLDS`), mounted with the same cell-gated learning files as `react`
  plus a generated `cell_config.py`: `cron_react` (TM-C / TM-D, one agent for the
  whole roster), `per_market` / `per_market_cron` (breakout news: one agent per
  market).
- `ext:<name>` — an externally supplied program directory run verbatim
  (breakout news memory arm, `agent/ext_candidates/`).
- `<baseline>` — a task-owned no-LLM anchor under `agent/baselines/`.

## Cell axes (`cell:`)

- `tm` — trigger mechanism: `A` sleep tool, `B` authored watcher programs,
  `C` fixed cron, `D` cron with agent-managed schedules.
- `tlrn` — learning trigger: `none`, or `daily` (a `learn` cron at midnight).
- `sig` — feedback signal: `none`, or `oracle` (the task's settled-outcome feed).
- `alg` — learning algorithm: `none`, `memory` (raw outcome memory rendered
  into the prompt), `skills` (LLM-curated skill block; not used in the paper).
