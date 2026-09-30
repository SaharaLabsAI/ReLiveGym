# Keyless crypto price data

This directory contains the reproducible collector for comparing the same spot
instrument across independent venues. API keys are neither read nor accepted.
Generated datasets are not shipped; this README and the Python build tools
are.

## Configured cohorts

| Cohort | Instrument | Sources |
|---|---|---|
| `btc_usdt_spot` | BTC/USDT spot | Binance, OKX, KuCoin, Bybit |
| `eth_usdt_spot` | ETH/USDT spot | Binance, OKX, KuCoin, Bybit |
| `btc_usd_spot` | BTC/USD | Coinbase Exchange, Bitstamp, DefiLlama reference |
| `eth_usd_spot` | ETH/USD | Coinbase Exchange, Bitstamp, DefiLlama reference |
| `btc_usd_hourly` | BTC/USD hourly | Coinbase, Bitstamp, DefiLlama, CoinGecko |
| `eth_usd_hourly` | ETH/USD hourly | Coinbase, Bitstamp, DefiLlama, CoinGecko |

USD and USDT are deliberately separate cohorts. Spot candles are never mixed
with perpetual, index, or mark-price candles. DefiLlama is included only in the
USD cohorts and is explicitly labeled `market_type=reference`: its Coins API
returns sampled aggregate prices, not an executable exchange market or OHLCV.
The sampled values are snapped to the nearest requested UTC boundary, assigned
as the close of the preceding interval, and stored as zero-range candles; raw
timestamps and confidence remain in `raw/defillama.jsonl`. Its nominal 1-minute
chart can contain gaps, so the collector automatically requests DefiLlama at
the output interval when that interval is supported.
Venue symbols live in
[`markets.py`](markets.py), making more assets straightforward to add.

Direct CoinGecko data is kept in separate hourly cohorts. Its keyless public
API returns arbitrary past ranges hourly; explicit historical `interval=5m` is
an Enterprise-only feature. The collector does not forward-fill or interpolate
those hourly observations into the five-minute cohorts.

Bybit documents that its API blocks US and mainland-China IP addresses, so it
is opt-in rather than part of the default USDT selection. It remains available
with `--source bybit`. A source failure does not discard successful responses
from other venues; it is recorded in the dataset manifest. Pass `--strict`
when every requested source must succeed.

## Fetch a bounded dataset

Run commands from the repository root. Times are UTC, `start` is inclusive,
and `end` is exclusive. Both boundaries must align to the output interval.

```bash
python -m tasks.crypto_price_consistency.data.fetch_prices \
  --cohort btc_usdt_spot \
  --start 2026-08-01T00:00:00Z \
  --end 2026-08-02T00:00:00Z
```

For the USD cohort including DefiLlama, the default produces aligned 5-minute
observations. The explicit equivalent is:

```bash
python -m tasks.crypto_price_consistency.data.fetch_prices \
  --cohort btc_usd_spot \
  --source-interval-minutes 5 \
  --lookback-hours 24
```

Fetch the direct CoinGecko hourly comparison for the same historical window:

```bash
python -m tasks.crypto_price_consistency.data.fetch_prices \
  --cohort btc_usd_hourly \
  --start 2026-08-01T00:00:00Z \
  --end 2026-08-02T00:00:00Z
```

The default is to fetch 1-minute venue candles and construct complete UTC
5-minute candles locally. That gives every venue identical interval boundaries.
To inspect a plan without making requests:

```bash
python -m tasks.crypto_price_consistency.data.fetch_prices \
  --cohort btc_usdt_spot --lookback-hours 24 --dry-run
```

Limit a run to selected venues by repeating `--source`:

```bash
python -m tasks.crypto_price_consistency.data.fetch_prices \
  --cohort btc_usdt_spot \
  --source binance --source okx --source kucoin \
  --lookback-hours 6
```

Exclude one or more default sources without enumerating the rest:

```bash
python -m tasks.crypto_price_consistency.data.fetch_prices \
  --cohort btc_usdt_spot \
  --exclude-source kucoin \
  --lookback-hours 6
```

Include the opt-in Bybit adapter explicitly:

```bash
python -m tasks.crypto_price_consistency.data.fetch_prices \
  --cohort btc_usdt_spot \
  --source bybit \
  --lookback-hours 6
```

List all configured symbols:

```bash
python -m tasks.crypto_price_consistency.data.fetch_prices --list-cohorts
```


The datasets the task reads (`btc_usdt_spot_mar_jul`, `eth_usdt_spot_mar_jul`)
are fetched with the commands in `../../../DATA.md`.

## Dataset layout

Every collection is self-contained beneath `datasets/<dataset-id>/`:

```text
manifest.json
raw/
  binance.jsonl
  okx.jsonl
normalized/
  cohort=btc_usdt_spot/
    source=binance/interval=5m/candles.csv
    source=okx/interval=5m/candles.csv
comparisons/
  cohort=btc_usdt_spot/interval=5m/
    close_matrix.csv
    pairwise_spreads.csv
    summary.json
```

- `raw/*.jsonl` preserves each HTTP URL, retrieval time, and unmodified JSON
  response for auditability.
- `normalized/**/candles.csv` is long-form OHLCV with consistent names,
  decimal-preserving values, UTC timestamps, completion status, and source-bar
  counts. `vwap` is populated when a venue supplies both base and quote volume.
- `close_matrix.csv` has one row per timestamp and close/VWAP/deviation
  columns per source. The reference price is the cross-source median close.
- `pairwise_spreads.csv` contains every complete overlapping source pair.
  `a_minus_b_bps` uses the symmetric pair midpoint as denominator.
- `summary.json` gives coverage and pairwise mean, p50, p95, and maximum absolute
  spread in basis points.
- `manifest.json` records the requested window, symbols, failures, row counts,
  schema version, and artifact paths.

Partial 5-minute buckets remain in normalized output with `complete=false`.
They stay visible in the matrix but are excluded from medians and pairwise
statistics. Missing candles are never forward-filled.

Comparison artifacts can be rebuilt without network access:

```bash
python -m tasks.crypto_price_consistency.data.compare_prices \
  tasks/crypto_price_consistency/data/datasets/<dataset-id>
```

## Public sources and provenance

- Binance market-data-only REST API and [bulk archive documentation](https://github.com/binance/binance-public-data)
- [OKX historical candlesticks](https://www.okx.com/docs-v5/en/#rest-api-market-data-get-candlesticks-history)
- [KuCoin public klines](https://www.kucoin.com/docs-new/3473244e0)
- [Bybit public klines](https://bybit-exchange.github.io/docs/v5/market/kline)
- [Coinbase Exchange public candles](https://docs.cdp.coinbase.com/api-reference/exchange-api/rest-api/products/get-product-candles)
- [Bitstamp public OHLC](https://www.bitstamp.net/api/#tag/Market-info/operation/GetOHLCData)
- [DefiLlama free Coins API and official SDK](https://github.com/DefiLlama/api-sdk#prices)
- [CoinGecko keyless public API](https://docs.coingecko.com/docs/keyless-public-api)

Before redistributing collected data or using it commercially, review each
venue's current terms. Bitstamp explicitly directs commercial users to arrange
a data license.
