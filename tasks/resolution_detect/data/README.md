# sample_detect_v1 — resolution_detect roster

Roster for the resolution_detect task. Built by `build_sample.py` (seed
20260812) from `tasks/breakout_news_pm/market/raw/markets.jsonl` (full
Polymarket dump, fetched 2026-07-21); answers enriched per market from
gamma `/markets/<id>` on 2026-08-12; hourly prices for in-window
resolvers live in the shared bnpm price store
(`tasks/breakout_news_pm/market/raw/prices/`), topped up via
`fetch_prices.py --only-ids` (29 markets).

## Admission criterion (pre-window metadata only — no realized outcomes)

- `startDate < 2026-07-01` (opens inside the sim window)
- `endDate >= 2026-07-31` (scheduled deadline ≥ sim_end + 30 d) — the
  far-deadline rule: any in-window resolution is event-triggered by
  construction (no deadline clock-watching)
- non-sports, non-mechanical tags; `volumeNum >= $100k`; alive at
  sim_start
- stratified family × open-month (largest remainder), per-event cap 3
  with seeded global top-up, N = 700 out of a 1,359-market universe

Sizing note: the universe's in-window resolve rate is 25.2%; N=700
targets ~150–200 scored questions *in expectation* — selection never
reads resolution fields. `resolved_in_window` / answers are recorded
OUTCOME data.

## Easy-at-activation exclusion (post-selection)

The **14 scored rows with t_det ≤ activation + 1 h** are dropped from
the artifact (`build_sample.py --refilter`; the filter also runs in
every full build). They were claimable at full credit by pure retrieval
at t=0 — 4 startDate-skew rows determined ~14 d before activation
(IPO-mechanics questions), and 10 questions the world settled at or
before sim_start whose official resolution lagged into the window.
**This step reads realized prices** — the roster is hindsight-free up
to exactly this documented exclusion. Dropped ids: 1021156, 1021159,
1278048, 1278347, 1287603, 1422370, 1422372, 1469372, 1654506, 572481,
661700, 789405, 842087, 842089.

## FDV exclusion (post-selection)

All crypto-launch **"FDV above $X" questions (77 rows)** are dropped by
question-text match (`build_sample.py`, `FDV_RE` — metadata-based,
reads no outcomes). Their determinations are feed-settled token-price
threshold crossings that CC-NEWS never reports, so they were guaranteed
misses compressing recall identically for every arm while measuring
nothing about detection. 686 → **609** rows; the drop is winnable-heavy
(44 scored / 40 winnable removed).

## Audit block (`audit_sample.py`, post-exclusions)

| stat | value | gate |
|---|---|---|
| roster rows | 609 (0 fetch errors) | — |
| scored (resolved in window, answer ok) | **123** | — |
| quiet | 486 (391 still open, 92 resolved post-window, 3 no_unit_price) | quiet majority ✓ |
| winning-side split | 64 first / 59 second → majority share **0.520** | [0.35, 0.65] **PASS** |
| Yes/No-labelled scored subset | 64 Yes / 58 No | — |
| winnable (dip-tolerant t_det exists, θ=0.99, exit 0.95) | 116/123 | — |
| no-crossing (unwinnable) | **7** — ids in audit output | pinned |
| gap hours (t_res − t_det) | median **60.3**, ≥12h 70/116, ≥24h 64/116, min 0.4, max 1978.7 | — |
| trap questions (losing side ≥6h inside 0.95 band) | **2** | pinned |
| easy-at-activation (t_det ≤ act+1h) | **0** | by construction (exclusion above) |
| price coverage (scored) | 123/123 | — |
| event cap | max 3 markets/event, 407 events | ✓ |

Family × (scored/quiet): crypto 10/22, culture 7/15, econ 20/61,
geopolitics 46/255, politics 24/112, tech 16/21.

## Boundary cases (annotations for the task build, not filters)

- **Near-zero gaps** (<1h, min 0.4 h): winnable in name, undetectable in
  practice; they depress the perfect-anchor ceiling honestly — leave in.
- **Long-gap tail** (max 1978.7 h): "world decided long before official
  resolution" markets (e.g. the Dublin by-election). Genuine
  event-triggered detections with large earnable windows.
- **startDate discrepancy**: gamma's current `startDate` occasionally
  postdates the dump's. The extreme cases (t_det ~14 d before activation)
  were removed by the easy exclusion; activation is pinned to the
  sample's `start_date`.
- **CC-NEWS crawl outage 2026-04-01 04:00 → 04-05 20:00 UTC** (112 h,
  the only crawl silence in 03-01..07-01; the backlog was crawled
  04-06). 6/116 winnable questions have t_det inside the window, 4 with
  gaps < 24 h (the 04-02 Cabinet-departure cluster). Under the
  crawl-corroborated clock the news blackout is largely repaired:
  self-reported publish times inside the outage are trusted as-is
  (corroboration was impossible), so in-outage articles with
  self-reports are visible at their claimed times; only the ~24% with no
  self-report surface late (crawl − 2 h, i.e. the 04-06 backlog).
  Substrate fact, not a filter — identical for every arm; keep in mind
  when reading per-question behavior around 04-01..06.
- **2 non-Yes/No binary race markets** (["Anthropic","OpenAI"],
  ["Leadership Change","Ceasefire"] — the latter scored): outcome labels
  handled by winning index everywhere; keep.
- **3 no_unit_price rows** (all "…before GTA VI?" long-horizon races,
  uma resolved without a unit price): quiet members, excluded from
  scoring by `answer_status`.

## Files

- `build_sample.py` — selection + gamma enrichment (deterministic under
  seed; `--dry-run` for sampling only)
- `sample_detect_v1.jsonl` — 609 rows, one per market (700 sampled −
  14 easy-excluded − 77 FDV-excluded)
- `audit_sample.py` — recomputes this audit block; also the reference
  implementation of the dip-tolerant t_det predicate
