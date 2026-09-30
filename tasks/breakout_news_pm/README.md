# breakout_news_pm

Breakout news detection on the Polymarket + CC-NEWS world: minute-grid
Polymarket prices, a self-built CC-NEWS corpus (9.48 M articles, BM25,
publish-time visibility clock) and hindsight attribution labels. This file
is for experimenters; the agent-facing spec is [INSTRUCTION.md](INSTRUCTION.md).

## Task structure

- **Requests**: the config lists `(market_id, start, end)` monitoring
  windows over the 350 frozen sample markets.
- **Signals** (`env/apps.py`): `get_markets`; `get_prices` (minute
  change-series prices — a point at t is visible from
  t + `price_delay_minutes`); `search_news` (BM25 over the corpus, an
  article visible from its publish time) and `get_article` (full text).
  Event-driven waiting is authored watcher code under TM-B
  (`harness/authored.py`); there is no declarative condition grammar.
- **Action**: `notify {market_id, news_id, direction}` — the claim "this
  market starts a breakout in this direction within W hours; this article
  is my evidence" (W = `breakout_window_hours`, default 24). One standing
  claim per market: a new alert is rejected (free) while a previous alert
  on that market is pending, so same-breakout spam and up/down hedging are
  structurally impossible.
- **Ground truth** (scorer-only): the 1,525 labeled breakpoints,
  minute-localized. Gold-citable articles per breakpoint: cited articles of
  attribution groups with confidence ≥ `attr_threshold` (default 0.6) whose
  publish time lies in `[t_move_start − W, t_move_start)` — articles on
  which an immediate honest W-hour claim would have been true. At defaults:
  580/1,525 breakpoints winnable, 598 gold groups, 1,410 gold articles
  (pinned in `data/built/build_stats.json`).

## Scoring

A breakpoint closes at its minute-localized `t_move_start`; notifications
at or after the start earn nothing.

| event | rule |
|---|---|
| covered_news | earliest eligible notification (τ < t_start ≤ τ+W, direction matches) citing a gold article; time credit decays linearly from the article's publish (1) to the move start (0) |
| covered_timing | eligible notification citing a non-gold article; flat `timing_credit` |
| miss | no eligible notification (winnable or not — winnable is a reporting annotation only) |
| false alarm | the claim expires at τ+W with no breakout matched; tagged `wrong_direction` when a breakout matched on market and timing but not direction, and excused (excluded from precision) when that paired breakout was missed, so attempting a direction never ranks worse than silence |

Primary metric `cov_f1`: the harmonic mean of precision (covering alerts /
non-excused resolved alerts) and coverage recall (covered breakpoints /
all closed breakpoints, binary credit). `tc_f1` (gold-cite decay credit
over winnable breakpoints only) is a reporting-only secondary, alongside
`direction_accuracy`, `coverage_rate` and the alert counts; `report()`
adds per-market and monthly slices.

Price data is free (`cost.price_call` 0: real market feeds are free, and
the zero-after-start rule already blocks converting price data into
reward); `search_news` and `get_article` bill `cost.news_search_call`
(0.002 USD, the commercial rate) per call on the `news` limiter.

## Data

```
market/        Polymarket fetch + breakpoint detection pipeline (raw/ not shipped)
news/          CC-NEWS corpus build, tantivy index, shared BM25 search module
labeling/      the 350-market sample + hindsight attribution labeling
data/build.py  compiles the world -> data/built/ (only build_stats.json is shipped):
               markets.jsonl, prices/, breakpoints.jsonl (minute-localized),
               attributions.jsonl (unfiltered — the scorer applies attr_threshold
               and W at load, so thresholds stay live without a rebuild)
```

Rebuild: `python data/build.py` (~2 min). Provenance: `market/README.md`,
`news/ccnews/README.md`, `../../DATA.md`.

## Config reference (`task:` params)

| param | default | meaning |
|---|---|---|
| `markets` | — | monitoring windows |
| `breakout_window_hours` | 24 | W: claim horizon ≡ gold article window |
| `attr_threshold` | 0.6 | gold confidence floor |
| `timing_credit` | 0.7 | covered_timing flat credit (reporting metric `tc_f1`) |
| `price_delay_minutes` | 10 | price visibility delay |
| `search_top_k` | 10 | results per search page |
| `cost.*` | price_call 0, news_search_call 0.002, article_call (= search) | real commercial rates |
| `news_rate_limit`, `price_rate_limit` | `{"window": "none"}` | limiter definitions |

## Program shapes and configs

`agent/compose_spec.py` registers the task-owned mains: `per_market`
(TM-A / TM-B: one agent per monitored market plus a coordinator, sharing
the wait party), `per_market_cron` (TM-D: the same topology under cron)
and `cron_react` (one agent for the whole roster under cron). `agent/`
also holds the learning-stack material the constructor mounts into
learning cells (`scan.py`, `records.py`, `prompts/`, `sig_self/`) and the
TM-B teaching example `example_gatekeeper.py`.

- `configs/cells/{w10,w13,w17}/` — the no-learning cells of the three
  four-week episodes; one YAML per mechanism.
- `configs/ext_bases/` — the same cells with `agent.scaffold: ext:candidate`,
  the bases the reflection-memory candidates in `agent/ext_candidates/`
  run on (launch: `../../README.md`, "Memory arms").
