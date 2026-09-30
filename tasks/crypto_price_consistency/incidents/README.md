# Incident stream tooling

Tooling for the incident stream: archive scraper, generator fitting,
limiter config, trace generation, the freeze record, and the replay proxy
engine.

## Scrape provider incident archives

Keyless, like the price collector. Run from the repository root:

```bash
python -m tasks.crypto_price_consistency.incidents.scrape_archives --deep
```

Sources and what each yields:

| Source | Mechanism | Depth |
|---|---|---|
| `coinbase` | Statuspage JSON API + paginated `/history` (3 months/page, `--history-months`, default 36) | years; NOTE: dominated by per-asset wallet entries — filter by impact/title at fit time |
| `bitstamp` | same Statuspage machinery | mid-2023 → now |
| `coingecko` | `/incidents` listing via cursor-chained "Next" links; `--deep` fetches each incident page for severity + resolution time | 2021 → now |
| `okx` | `GET /api/v5/system/status?state=completed` + server-rendered status page + date-slugged help-center postmortems | 2020 → now (slug events are date-only) |

Outputs under `calibration/`:

- `events/<source>.json` — normalized `IncidentEvent` lists (committed; these
  are the frozen inputs to generator fitting).
- `manifest.json` — retrieval time, per-source counts and coverage (committed).
- `raw/` — every HTTP payload as fetched, plus saved HTML pages (git-ignored,
  rebuildable; kept for auditability of the parse).

Normalization caveats, preserved rather than papered over:

- Unparseable timestamps leave `started_at`/`ended_at` null; the verbatim
  provider string stays in `raw_timestamp`.
- CoinGecko `ended_at` is the last incident-update timestamp — an upper bound
  on resolution. Late postmortem updates inflate the duration tail; fit with
  robust statistics or cap.
- `impact` keeps each provider's own vocabulary (`critical`/`major`/`minor`/
  `none`/`maintenance`/`unscheduled`); harmonization happens at fit time.

## Fit generator parameters

```bash
python -m tasks.crypto_price_consistency.incidents.fit_generator
```

Reads `calibration/events/*.json`, writes `generator_params_v1.json`
(committed; hash-links the event files it was fit from). Harmonizes provider
severities into `outage`/`degradation`/`maintenance`, filters Coinbase wallet
and fiat-rail noise via an explicit regex, merges overlapping intervals (OKX
phase rows), and fits durations as lognormals via robust quantile matching —
archive resolution timestamps are upper bounds, so tail-sensitive MLE is
avoided. Rates divide merged counts by each archive's observed span; OKX spans
differ per retrieval method.

Venue synthesis: `okx`/`coinbase`/`bitstamp`/`coingecko` are fitted directly;
`binance` is anchored to its published H1 2024/2025 API uptime reports
(2 partial incidents/year, ~28 min each) with pooled-exchange duration spread;
`kucoin` and `defillama` borrow pooled parameters and are flagged
high-uncertainty (a grid axis). The `validation` block records
the cross-source spread: exchange-class outage rates 2.1–10.6/yr with
p50 durations 57–84 min, and expected outages per 153-day window ≈ 0.8–1.1
for Binance/OKX/Coinbase — the sparsity sanity check.

## Rate-limiter machines (`limiters.yaml`)

Layer-M config for the replay proxy: per-venue budgets, window semantics,
429 behavior, and Binance's documented 429→418 escalating IP ban. Values
come from each venue's public rate-limit docs (URLs inline); anything the
docs leave vague is marked `basis: approximate`. Capacity-modulation events
in scenario files multiply these budgets — the "unexplained 429" mode.

## Generate the frozen trace set

```bash
python -m tasks.crypto_price_consistency.incidents.generate_traces
```

Reads `generator_params_v1.json` + `calibration/events/*.json`, writes 68
traces to `scenarios/` (committed) plus `scenarios/manifest.json` with
per-trace SHA-256 hashes: `zero`, `real_replay` (46 archived events on true
timestamps), 20 dev + 20 held-out fitted seeds, an 18-point
rate×duration×correlation grid, and 8 authored stress scenarios. Every event
carries its provenance (`replayed`/`fitted`/`authored`); Layer-E and
capacity-modulation constants have no archive by nature, so they are
authored module constants echoed into each trace header and swept by the
grid's rate axis. Generation is fully deterministic (seeded, no wall-clock)
— reruns are byte-identical. `stress_maintenance_in_vol` computes the
window's highest-realized-volatility day (2026-06-05) from the recorded
Binance candles, so the dataset must be present locally.

`FREEZE_v1.md` records the freeze date and top-level hashes. The v1
artifacts are immutable from that point; changes create a `v2` sibling.

## Replay proxy engine (`proxy.py`)

Pure library — no sockets, no wall clock:
`ReplayProxy(store, trace).request(venue, path, params, sim_time)` returns a
`Response` (status/headers/body/elapsed_ms, or `error` for
timeout/dns/connect-refused) exactly as the venue would answer at that
simulated instant. Candle rows are byte-identical from the datasets'
`raw/*.jsonl`, re-wrapped in each venue's native envelope
(binance `/api/v3/klines`, okx `/api/v5/market/[history-]candles`, kucoin
`/api/ua/v1/market/kline`) and sliced by the request's own query params;
only completed candles are visible (the live APIs' forming candle is never
served — a documented fidelity caveat). Incidents compose M → E → P: the
limiter machines from `limiters.yaml` (incl. Binance's 429→418 escalation
and capacity modulation) run first, edge failures mask the provider, and
provider modes (outage/degraded/maintenance/stale_200/wrong_data/
schema_change/slow_bleed) transform the base response. Per-request
randomness is seeded from (trace, venue, path, t) so replays are exactly
reproducible. `Trace.client_clock_skew(t)` is exposed for the env layer's
clock tool. Tests: `tests/tasks/crypto_price_consistency/test_incident_proxy.py`
(self-contained synthetic recordings).

The env layer (`../env/apps.py` + `../task.py`) wraps this engine as the
`http_fetch` tool and owns billing and scoring — see `../README.md`.
Clock-skew events are inert in the env layer (the harness owns `get_time`).
