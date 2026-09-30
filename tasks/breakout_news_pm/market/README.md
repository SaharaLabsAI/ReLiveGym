# Polymarket 2026-03..07 raw data

All data fetched directly from Polymarket public APIs for the window **2026-03-01 .. 2026-07-01**. The
`raw/` directory is gitignored; rebuild with the two scripts here.

```
python fetch_markets.py                # gamma-api metadata crawl (~50 min)
python fetch_prices.py --min-volume 10000 --min-days 7 --exclude-sports
python fetch_prices.py --merge         # shards -> raw/prices/<mid>.json
python detect_breakpoints.py           # hindsight z-score breakpoints
```

| artifact | contents | pinned facts (2026-07-21 fetch) |
|---|---|---|
| `raw/markets.jsonl` | every market overlapping the window (metadata, volume, tags, clob tokens) | 1,109,925 markets (1,096,082 closed); 51% sports-tagged |
| `raw/price_shards/` | raw batch-prices-history responses (resume cache) | 2,560 shards |
| `raw/prices/<mid>.json` | `{grid_hours: 1, points: [[t,p],...]}` hourly Yes-price | 10,921 markets, 13.27M points; median 94% of in-window hours |

Roster filter (vol >= $10k lifetime, >= 7 in-window days, non-sports tags):
10,928 markets / 616k market-days; 7 returned zero price points.

| artifact | contents | pinned facts |
|---|---|---|
| `raw/breakpoints.jsonl` | daily-grid hindsight breakpoints (`\|dp\|>=0.02`, z>=2 vs trailing-14d stdev, >=10 prior changes), hourly-localized | **26,495 bps on 5,882 markets** |
| `raw/breakpoints_stats.json` | detector params + summary | pinned by detect_breakpoints.py |

Warmup: `fetch_prices.py --fetch-start 2026-02-15 --fetch-end 2026-03-01`
pulled Feb baselines (roster eligibility unchanged) so pre-existing markets
can fire from Mar 1. Chronically volatile markets (baseline daily sd ~0.1)
never fire even on large moves - by design (z is relative to own history).

API notes (verified 2026-07-21):
- gamma `/markets/keyset`: pagination via `after_cursor` (offset caps at
  2,100); **closed markets excluded unless `closed=true`**; tags need
  `include_tag=true`; page cap 100.
- clob `POST /batch-prices-history`: max 20 tokens/call, `start_ts..end_ts`
  span capped at 15 days (1,296,000 s) regardless of fidelity; body params
  are snake_case even though error messages say `startTs`.
- clob `GET /prices-history`: `endTs` broken (returns empty) — use
  `startTs`+`fidelity` and clip client-side; resolved markets keep full
  history; python-urllib UAs are 403'd, requests/curl UAs fine.

News side for the same window: the self-built CC-NEWS corpus at
`tasks/breakout_news_pm/news/` (9.5M articles; `news/ccnews/README.md`).
