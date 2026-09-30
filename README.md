# Long-lived agents over replayed reality — preliminary code release

This folder is a self-contained copy of the code behind the paper's main-body
experiments: the simulated environment, the eight benchmark tasks with their
scorers and run configurations, the three trigger mechanisms (sleep, watcher,
cron), and the two test-time learning arms (outcome memory on popularity
prediction, reflection memory on breakout news detection).


## Layout

```
harness/      the environment (server side): simulated clock, schedule store, wait party,
              wallet + rate limits, ledger, metered LLM proxy, tool manifest, task interface;
              entry points run.py (combined), serve.py + launcher.py (detached), mcp.py +
              episode_drive.py (episode mode for the two browser tasks)
scaffolds/    compose.py — the constructor that compiles a run YAML into a workspace program;
              runtime/ — the program library copied into every workspace (actor runner,
              ReAct agent, env client, memory / reflection)
tasks/        one directory per task (tasks/README.md); tasks/weather_fixture is the test fixture
configs/      model_costs.yaml (the sole pricing source), model_limits.yaml (context windows)
scripts/      run_seeds.py, gen_contract.py, llm_smoke.py, primary_metric.py,
              prepare_episode.py / run_web_episodes.py / settle_dead_episode.py (web tasks),
              data_sync/ (installing a released data bundle)
tests/        pytest suite (tests/README.md)
```

## Install

Python 3.12 or newer.

```sh
uv sync                       # or: pip install -e ".[dev]"
cp .env.example .env          # add OPENAI_API_KEY and/or OPENROUTER_API_KEY
```

A model is named at launch as `api_provider:model` (`openai:gpt-5.6-luna`,
`openrouter:qwen/qwen3.7-plus`). Every model must have a row in
`configs/model_costs.yaml`: booked cost comes from that table, the LLM proxy
refuses unlisted models, and the launch entry points fail fast. Browser
episodes additionally need a row in `configs/model_limits.yaml`.

Data: install or rebuild the task worlds as described in [DATA.md](DATA.md).
Each task's `task.py` documents the default paths it reads.

```sh
python scripts/llm_smoke.py openai:gpt-5.6-luna           # one real metered round trip
pytest                                                    # fast suite, ~1 min, no API key needed
```

## How a run works

- **Run config.** A YAML under `tasks/<task>/configs/` names the task section,
  the simulated window, the wallet (`budget_usd`, plus `domain_budgets.llm`
  where the LLM cap differs), the program shape (`agent.scaffold`) and the
  cell axes: `tm` (trigger mechanism), `tlrn` (learning trigger), `sig`
  (feedback signal), `alg` (learning algorithm). The model and the seed are
  launch arguments.
- **Simulated time.** One sim per run, one clock. The clock advances only
  when the agent waits (`sleep`, an authored watcher program, or a cron
  trigger); every query is clamped to what was visible at that instant.
  Priced tool calls and LLM tokens debit the wallet; a run whose LLM cap is
  exhausted continues without LLM calls until `sim_end`.
- **Run directory.** `runs/task_<task>/<model-slug>/<run_id>-s<seed>/` with
  `ledger.jsonl` (every event, priced), `workspace/` (the agent's program and
  logs), `config.json`, and `results.json` once the clock reaches `sim_end`.
- **Run modes.** `python -m harness.run` runs server, scheduler and program
  runner in one process; `harness.serve` + `harness.launcher` run them
  detached (used for the `ext:` memory arm below); `harness.mcp` runs an
  external agent app as the actor (the browser tasks).

Launching one cell:

```sh
python -m harness.run --config tasks/daily_reddit_digest/configs/cells/tmA-tlrnnone-signone-algnone.yaml \
    --model openai:gpt-5.6-luna --seed 0
python scripts/run_seeds.py tasks/daily_reddit_digest/configs/cells/tmA-tlrnnone-signone-algnone.yaml \
    --seeds 0 1 2 --model openrouter:qwen/qwen3.7-plus     # one process per repetition
python scripts/primary_metric.py runs/task_daily_reddit_digest/gpt-5.6-luna/*-s?
```

`--mock-llm` (on `harness.run`, `harness.serve` and `prepare`) selects a
canned zero-cost upstream for plumbing checks. Seeds label repetitions; the
harness does not seed the provider, so repetitions differ through sampling.

## Trigger mechanisms

The mechanism is the cell's `tm` value; the same program shape serves all of
them, and the constructor changes only which wait tools the agent holds.

| `tm` | paper name | what the agent gets |
|---|---|---|
| `A` | sleep harness | a `sleep` tool: name a duration, the loop resumes when it expires |
| `B` | watcher harness | no `sleep`; file tools jailed to its scratch dir plus `run_program`, which runs an authored Python watcher against the same priced tools until it hands over, a schedule fires or its deadline passes (`harness/authored.py`; the skill text appended to the instruction is `harness/authored_skill.md`) |
| `D` | cron harness | a fixed schedule fires a bounded ReAct episode per trigger; `list/create/update/delete_schedule` let the agent manage its own one-time and recurring triggers (`harness/apps.py`) |

(`tm: C`, a fixed cron without agent-managed triggers, exists in the code but
is not a paper arm.)

Configs of the paper's no-learning cells (`tlrnnone-signone-algnone`), one
YAML per mechanism:

| task | configs under `configs/cells/` | program shape (A / B, D) | window |
|---|---|---|---|
| breakout_news_pm | `w10/`, `w13/`, `w17/`: `tmA-…-permarket.yaml`, `tmB-…-permarket.yaml`, `tmD-…-permarketcron.yaml` | `task:per_market` / `task:per_market_cron` (one agent per market plus a coordinator) | three four-week episodes |
| forecast_portfolio | `tm{A,B,D}-….yaml` | `react` / `task:cron_react` | March 2026 |
| resolution_detect | `e1/`, `e2/`, `e3/` | `react` / `task:cron_react` | three episodes |
| reddit_ai_popularity | `tm{A,B,D}-….yaml` | `react` / `task:cron_react` | April 2026 |
| daily_reddit_digest | `tm{A,B,D}-….yaml` | `react` / `task:cron_react` | April 2026 |
| crypto_price_consistency | `tmA-….yaml`, `tmB-….yaml`, `tmD-…-cronreact.yaml` | `react` / `task:cron_react` | March 1–15 2026 |
| edgar_portfolio | `tm{A,B,D}-….yaml` | browser episodes (below) | April 1 – June 1 2026 |
| broker_ops | `tm{A,B,D}-….yaml` | browser episodes (below) | April 1–15 2026 |

Every cell was run with three seeds (six for some breakout-news cells) for
the eight models below. Repetitions of one cell land side by side under the
model's directory; `scripts/primary_metric.py` prints the paper's metric for
each (`cov_f1`, Brier skill score over the base rate, `tc_f1`, time-weighted
F1, digest and price accuracy, `filing_score`, `routine_score`).


## Models

| paper name | launch name |
|---|---|
| GPT-5.6 luna | `openai:gpt-5.6-luna` |
| GPT-5.6 terra | `openai:gpt-5.6-terra` |
| Gemini-3.5-flash-lite | `openrouter:google/gemini-3.5-flash-lite` |
| Gemini-3.5-flash | `openrouter:google/gemini-3.5-flash` |
| Claude-haiku-4.5 | `openrouter:anthropic/claude-haiku-4.5` |
| Qwen3.7-plus | `openrouter:qwen/qwen3.7-plus` |
| MiniMax-M3 | `openrouter:minimax/minimax-m3` |
| DeepSeek-V4-flash | `openrouter:deepseek/deepseek-v4-flash` |

Rates are pinned in `configs/model_costs.yaml`.

## Browser tasks (edgar_portfolio, broker_ops)

The two web tasks serve HTTP hosts (a read-only replay and a writable portal
behind a browser-only rule) and are acted on by OpenCode with a Playwright
browser, both metered through the run's LLM proxy. An episode is prepared,
acted, then settled:

```sh
# tm A / B: one OpenCode session for the whole window
python scripts/prepare_episode.py --task broker_ops --tm A --app opencode \
    --model openai:gpt-5.6-luna --out runs/episodes --run-id bops-luna-tmA
sh runs/episodes/bops-luna-tmA-s0/run_opencode.sh
python -m harness.mcp settle --episode runs/episodes/bops-luna-tmA-s0/episode.json

# tm D: an external driver fires the schedule, one OpenCode turn per trigger
python scripts/prepare_episode.py --task edgar_portfolio --tm D --app opencode \
    --model openai:gpt-5.6-luna --out runs/episodes --run-id edgar-luna-tmD
python -m harness.mcp drive  --episode runs/episodes/edgar-luna-tmD-s0/episode.json
python -m harness.mcp settle --episode runs/episodes/edgar-luna-tmD-s0/episode.json

# the grid: 8 models x {A, B, D} x 3 seeds, six episodes in flight
python scripts/run_web_episodes.py --tasks edgar_portfolio broker_ops --dry-run
nohup python scripts/run_web_episodes.py --tasks edgar_portfolio broker_ops > runs/web_episodes.log 2>&1 &
```

Results land in `<episode>/server/results.json` (`scripts/primary_metric.py`
accepts the episode directory). `scripts/settle_dead_episode.py` rebuilds and
settles an episode whose server died before `settle`.

Prerequisites (macOS): `sandbox-exec` for the two network fences; OpenCode on
`PATH`; the Playwright MCP checkout
(`npm install --prefix .tools/playwright-mcp @playwright/mcp@0.0.81`, or
`--playwright-cli`); Google Chrome or `npx playwright install chromium`; the
EDGAR snapshot (DATA.md). On Linux there is no OS fence: the agent's shell
tool is disabled in `opencode.json` instead and Playwright runs over stdio.
Add `--mock-llm --stretch 2d` to `prepare` for a free plumbing check.

## Memory arms

**Popularity prediction — outcome memory**

```sh
C=tasks/reddit_ai_popularity/configs/cells
python scripts/run_seeds.py $C/tmA-tlrndaily-sigoracle-algmemory.yaml --seeds 0 1 2 --model openai:gpt-5.6-luna
python scripts/run_seeds.py $C/tmB-tlrndaily-sigoracle-algmemory.yaml --seeds 0 1 2 --model openai:gpt-5.6-luna
python scripts/run_seeds.py $C/tmD-tlrndaily-sigoracle-algmemory.yaml --seeds 0 1 2 --model openai:gpt-5.6-luna
```

**Breakout news detection — reflection memory**

```sh
B=tasks/breakout_news_pm/configs/ext_bases
python -m harness.launcher --port 8766 --runs-root runs/task_breakout_news_pm/gpt-5.6-luna \
    --bases w10full=$B/w10full-tmA-ext.yaml w13full=$B/w13full-tmA-ext.yaml w17full=$B/w17full-tmA-ext.yaml
CAND=$PWD/tasks/breakout_news_pm/agent/ext_candidates/tmA-reflection-only
for base in w10full w13full w17full; do
  curl -s -X POST http://127.0.0.1:8766/runs -H 'Content-Type: application/json' \
    -d "{\"base\":\"$base\",\"candidate\":\"$CAND\",\"model\":\"openai:gpt-5.6-luna\",\"seeds\":3}"
done
curl -s http://127.0.0.1:8766/runs            # status; DELETE /runs/<id> stops one
```

## Reading a run

- `results.json`: `performance.primary` (name, value, direction),
  `performance.*` (task detail), `resources.spent_usd` and the per-domain
  spend, `flags` (`budget_exhausted`, `llm_budget_exhausted`, ...), and the
  task's full outcome list.
- `ledger.jsonl`: every priced call, wait, trigger, LLM call (with booked and
  provider-reported cost), settled outcome and daily cost brief, in sim order.
- `workspace/logs/`: the program's own trace (`trace.jsonl`), per-agent
  transcripts, crash logs; `workspace/memory/` the learned block where a
  learning cell wrote one.
