# forecast_portfolio sample v1

Stratified, hindsight-free sample of 300 Polymarket questions for the
standing-forecast task. Built by `build_sample.py` (seed 20260811, deterministic given
the frozen bnpm markets crawl).

## Provenance & criteria

Universe: `tasks/breakout_news_pm/market/raw/markets.jsonl` (all markets
whose *scheduled* life overlaps 2026-03-01..07-01; gamma crawl 2026-07-21).
Eligibility — everything observable at market open / sim_start:
non-sports, non-mechanical tags (Recurring / Crypto Prices / Up or Down /
5M / 15M / 1H / Weather / temperature / Hit Price / Multi Strikes / Tweet
Markets / Rewards Automation* excluded), fetch-time volume >= $100k
(a quality proxy), not already closed at sim_start.
Eligible universe: 3,709.

Stratification uses open-time features ONLY: tag family x scheduled-endDate
month (proportional, largest remainder), per-event cap 3. **In-window
resolution is an outcome we record, never a selection criterion** — the
sample deliberately contains markets that outlive the window (never-resolve
attention load).

Resolution facts fetched per market from gamma `/markets/<id>`
(no batch-by-id endpoint exists). `resolution_answer` = the outcome with
final price ~1.0; anything else is flagged (`answer_status:
no_unit_price`), never guessed.

## sample_v1.jsonl (300 rows)

Key fields: `question`, `description` (resolution criteria), `outcomes`,
`start_date`, `scheduled_end`, `resolution_date` (closedTime; null if
open), `resolved_in_window`, `resolution_answer`, `answer_status`,
`family`, `sched_end_bucket`, `event_id`, `volume_usd`, `tags`.

Pinned facts (gamma fetch of 2026-08-11):
- fetch errors 0; resolved-in-window 184/300 (61.3% — matches the
  universe's 61%, i.e. stratification didn't distort)
- resolution months (diagnostic, post-hoc): Mar 22 / Apr 53 / May 54 / Jun 55
- answer_status: ok 223 (incl. after-window resolutions), open 75,
  no_unit_price 2 (both GTA-VI novelty markets, resolved 2026-08-01)
- families: geopolitics 132, politics 64, econ 38, culture 27, crypto 19,
  tech 18, other 2

## built/ (build.py)

`build.py` turns the frozen sample into the runnable world (built/ is
gitignored; rebuild = `python build.py`, network only for price series
absent from the bnpm hourly roster — fetched via clob
batch-prices-history, the only true batch API):

- `built/questions.jsonl` — 300 rows, split `agent` block (question,
  verbatim description, outcomes, open_date, scheduled_end — the only
  fields that may cross a tool boundary) / `scorer` block (t_res =
  min(closedTime, scheduled_end), answer, easy flag, ta_bss_market,
  provenance slices).
- `built/prices/<mid>.json` — hourly Yes-price, 261 copied from the bnpm
  roster + 39 fetched (8 batch calls, all returned points).

Pinned build facts (also pinned in tests/tasks/forecast_portfolio):
scored (resolved-in-window, clean answer) 184; easy (price never left
the 0.95 band over the scored life) 31; mid-window opens 145;
`ta_bss_market` over scored mean **0.7808** (min −0.7646, max 1.0);
scored life days median 30, p90 94, 33 under 7 d.

## Caveats

- **News clock**: the task reads the CC-NEWS index `tantivy_index_v3`
  (crawl-corroborated publish times, `news/build_index.py`). Substrate
  fact inherited with it: the crawler was down 2026-04-01 04:00 → 04-05
  20:00 UTC (the only crawl silence in 03-01..07-01), and the backlog was
  crawled 04-06. **This is not a news blackout**: self-reported publish
  times inside the outage are trusted as-is, so those articles serve at
  their claimed times; only the ~24% with no self-report surface late,
  at crawl − 2 h. Residual exposure for this task: 8 of
  the 184 scored questions have `t_res` inside the window and 5.2% of
  total scored life falls in it, so the late-surfacing minority can cost
  a little `ta_bss` there. Arm-identical either way.
- `closedTime` is the resolution timestamp proxy (UMA settlement); it can
  lag the question's deadline by hours (e.g. "by March 31" markets close
  Apr 1 early UTC).
- Fetch-time volume is mildly hindsight-tinged (includes post-window
  trading); accepted deliberately as "questions that turned out to matter".
- Novelty markets (the "before GTA VI?" event) pass the tag filters and
  are kept.
- Agent-visible fields must exclude `resolution_date`, `resolution_answer`,
  `answer_status`, `uma_status`, `resolved_in_window` (scorer-only).
