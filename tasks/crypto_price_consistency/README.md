# crypto_price_consistency

Hourly BTC/ETH spot-price reporting under injected API incidents — the
routine robustness task of the collection. The agent polls three
exchanges' public candle APIs through a replay proxy for the simulated
window and must, every hour per symbol, `report` a price or explicitly
`abstain`.

- **Data**: `data/` — recorded 5-minute candles, Mar 1 → Aug 1 2026,
  binance/okx/kucoin (`data/README.md`).
- **Incident stream**: `incidents/` — archive scraper, fitted generator,
  frozen traces (`incidents/FREEZE_v1.md`), limiter machines, and the pure
  replay engine `incidents/proxy.py`. The generator was fitted to public
  status-page archives and frozen before any agent existed; every
  injected event carries its provenance (replayed / fitted / authored).

## Harness wiring

Standard task layout (`harness/task.py`): `task.py` (TASK, config model,
scoring), `env/apps.py` (ExchangeApp), `INSTRUCTION.md`, `configs/`,
`agent/` (`compose_spec.py` registering the `cron_react` main used under
TM-D, and the TM-B teaching example).

The observation tool is deliberately **HTTP-fidelity**, not semantic:
`http_fetch(venue, path, params, timeout_s?)` returns raw status/headers/
unparsed body (or timeout/dns/connect errors) from the replay proxy at the
current sim time. Status codes, Retry-After headers, HTML challenge pages,
renamed fields, and frozen candles all reach the agent intact — a
`get_price()` tool would absorb exactly the failures this task measures.
Fetches are free (public endpoints) behind each venue's documented rate
limiter; `get_exchange_docs` publishes the endpoint contract. The scored
actions are `report(symbol, price)` and `abstain(symbol, reason?)`.

## Scoring

Each full hour × symbol, the last action in the hour settles at the next
hour boundary (visible via `get_feedback`). A report settles `ok` when its
error against the untouched cross-venue median at submission time lies
inside the free band (`free_bps`, 5) and `priced` otherwise, with
`excess_bps = min(cap_bps, max(0, bps_error − free_bps))`; an hour without
an action is a `miss`. `results.json` carries `mean_excess_bps` over
reported hours plus availability, abstain rate and the status counts. The
paper's metric, **price accuracy** (`price_acc`: the share of symbol-hours
settled `ok`, misses counted as 0, hours after the run's first
budget-refused LLM call unscored), is derived from the ledger by
`scripts/primary_metric.py`. Constraints: answer every hour, abstain on at
most `abstain_budget_frac` (10 %) of scored hours, the `budget_usd` wallet
(LLM at real token rates) and the venue limiters.

## Known limitations

- `clock_skew` trace events are inert (the harness owns `get_time`), as
  are trace events on venues without recorded data
  (coinbase/bitstamp/coingecko/defillama).
- Only the recorded endpoints/symbols/5-minute interval exist behind
  `http_fetch`.
