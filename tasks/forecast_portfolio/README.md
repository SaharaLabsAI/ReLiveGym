# forecast_portfolio

Standing-forecast maintenance over Polymarket questions with billed
CC-NEWS evidence: the agent keeps a probability distribution per question,
scored **continuously** against the eventual resolution. Formulation
adapted from FutureSim with four deviations (fixed outcome sets,
time-averaged scoring, billed evidence, no forced cadence) that make it
deterministic, timing-sensitive, and budget-forced.

## Task structure

- **World**: the bnpm 2026-03→07 substrate end to end — CC-NEWS BM25 index
  (`tasks/breakout_news_pm/news/tantivy_index_v3`, publish-time
  visibility on the crawl-corroborated clock of `news/build_index.py`)
  plus the hindsight-free 300-question sample (`data/sample_v1.jsonl` →
  `data/build.py` → `data/built/`). Prices exist scorer-side only.
- **Questions** are visible from their real market-open date (staggered
  arrivals: 145/300 open mid-window); a resolution becomes an observable
  world fact at settlement time, for every arm — that is observability,
  not a sig treatment.
- **Observation tools** (env/apps.py): `list_questions(status?,
  added_after?)` compact index, `get_question(id)` detail (verbatim
  resolution criteria), `get_forecasts(id?)` own-submission history — all
  free; `search_news` / `get_article` billed at bnpm's commercial anchor
  rates on the shared "news" limiter.
- **Action**: `submit_forecast(question_id, {outcome: p}, news_id?)`, free,
  free-rejected on unknown/unactivated id (byte-identical errors — id
  probing must not leak future arrivals), resolved question, bad
  probabilities, or Σp > 1. A complete distribution summing to 1 is what
  INSTRUCTION and the tool doc both ask for; Σp < 1 is accepted but the
  omitted outcomes are scored as zeros, not as abstention. `news_id` is
  the optional citation — **recorded, never scored** (evidence
  provenance for run analysis); an unknown or not-yet-published id is a
  free reject, so it cannot name an article the agent could not read.
  Surfaces as `n_cited` per question in `report()` and verbatim in the
  ledger.

## Ground truth & scoring

Per scored question: `BSS(t) = 1 − Σ_o (p_o(t) − 1[o=y])²` on the standing
(piecewise-constant) forecast, integrated **exactly** (no grid — forecasts
and hourly price series are both step functions) over
`[activation, t_res]` and normalized: `TA_q`. `activation =
max(open_date, sim_start)`; `t_res = min(closedTime, scheduled endDate)`
(UMA settlement lag earns nothing). Anchors: confident-correct +1,
uniform +0.5, confident-wrong −1. **A question with no standing forecast
scores as the uniform one** (1/N per outcome, +0.5 here — all 300 sample
questions are binary `["Yes","No"]`), so the working range is [0.5, 1.0]
and a negative score means actively wrong rather than idle. That default
covers only the *absent* forecast: a submitted distribution is scored
literally, so an omitted outcome is a stated zero — `{Yes: 0.5}` on a
question resolving No earns −0.25, below silence.

- **Primary**: `ta_bss` = mean `TA_q` over the run-scored questions
  (clean resolution with `t_res` in the run window), direction max. The
  paper reports the Brier skill score over the constant base-rate
  forecast, derived from `ta_bss` by `scripts/primary_metric.py`.
- Scored set of the full window: 184/300; the rest stay live as attention
  load the agent cannot distinguish.
- **Anchors**: `ta_bss_market` (the withheld market price read as a
  forecaster, same integral — full-window mean **0.7808**) and the
  uniform baseline (≈0.5 by construction, `agent/baselines/uniform`).
- `easy` (price never left the 0.95 band over the scored life, 31/184) is
  a reporting slice like bnpm's `winnable`: settlement never branches on
  it; `ta_bss_nontrivial` excludes it.

## Leak-safety

Agent-visible fields are exactly the `agent` block of
`data/built/questions.jsonl` (question, verbatim description, outcomes,
open_date, scheduled_end) plus post-settlement outcomes. Tool payloads
never carry `t_res`, answers pre-settlement, the easy flag, prices, or
any `ta_*` for open questions. `sig=oracle` adds `get_feedback` →
settled questions with the agent's own score decomposition and trace —
never anything for open questions.

## Evaluation

`metrics()`: primary `ta_bss`, plus `ta_bss_nontrivial`, `ta_bss_market`,
`skill_vs_market`, `abstention_share` (time-weighted), counts,
`mean_updates`. `report()`: per-question rows and slices by family /
resolution month / life band / easy, plus unsettled-question traces.

## Config reference

`configs/cells/tm{A,B,D}-tlrnnone-signone-algnone.yaml`
(March 2026 window, 60-question roster). Task fields:
`questions` (the portfolio — scope is this list: single / subset / full
300), `search_top_k`, `easy_band`, `cost.{news_search_call, article_call,
submit_call}`, `news_rate_limit`, `data_dir`, `news_index_dir`.
`from_run_config` validates: run window inside 2026-03-01..07-01, known
ids, no question resolved before `sim_start`, none opening after
`sim_end`.
