"""Keyless exchange adapters for normalized historical spot candles."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import httpx

from tasks.crypto_price_consistency.data.markets import MarketCohort, MarketSource
from tasks.crypto_price_consistency.data.models import Candle, decimal_or_none


USER_AGENT = "program-engineering-crypto-price-consistency/0.1"
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class FetchError(RuntimeError):
    """An exchange response could not be safely normalized."""


class JsonClient(Protocol):
    def get_json(self, url: str, params: Mapping[str, object]) -> Any: ...


class HttpJsonClient:
    """Small retrying HTTP client that records every successful raw payload."""

    def __init__(
        self,
        *,
        raw_path: Path,
        timeout_seconds: float = 30,
        retries: int = 4,
    ) -> None:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        self._raw_file = raw_path.open(mode="w", encoding="utf-8")
        self._client = httpx.Client(
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            timeout=timeout_seconds,
            follow_redirects=True,
        )
        self._retries = retries

    def __enter__(self) -> "HttpJsonClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()
        self._raw_file.close()

    def get_json(self, url: str, params: Mapping[str, object]) -> Any:
        last_error: Exception | None = None
        for attempt in range(self._retries + 1):
            try:
                response = self._client.get(url, params=params)
                if (
                    response.status_code in RETRYABLE_STATUS_CODES
                    and attempt < self._retries
                ):
                    retry_after = response.headers.get("retry-after")
                    delay = float(retry_after) if retry_after else min(2**attempt, 8)
                    time.sleep(delay)
                    continue
                response.raise_for_status()
                payload = response.json()
                self._raw_file.write(
                    json.dumps(
                        {
                            "fetched_at": datetime.now(UTC).isoformat(),
                            "url": str(response.request.url),
                            "payload": payload,
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                self._raw_file.flush()
                return payload
            except httpx.HTTPStatusError as exc:
                # Authentication, symbol, and regional errors will not improve
                # on retry. Preserve enough of the response for diagnosis.
                try:
                    error_payload: object = exc.response.json()
                except ValueError:
                    error_payload = exc.response.text
                self._raw_file.write(
                    json.dumps(
                        {
                            "fetched_at": datetime.now(UTC).isoformat(),
                            "url": str(exc.request.url),
                            "status_code": exc.response.status_code,
                            "error_payload": error_payload,
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                self._raw_file.flush()
                raise FetchError(
                    f"HTTP {exc.response.status_code} for {exc.request.url}"
                ) from exc
            except (httpx.TransportError, ValueError) as exc:
                last_error = exc
                if attempt < self._retries:
                    time.sleep(min(2**attempt, 8))
                    continue
                break
        raise FetchError(f"request failed for {url}: {last_error}") from last_error


def _windows(
    start_ms: int,
    end_ms: int,
    *,
    interval_minutes: int,
    page_size: int,
) -> Iterator[tuple[int, int]]:
    window_ms = interval_minutes * 60_000 * page_size
    cursor = start_ms
    while cursor < end_ms:
        page_end = min(end_ms, cursor + window_ms)
        yield cursor, page_end
        cursor = page_end


def _empty_candle(
    *,
    interval_minutes: int,
    open_time_ms: int,
    open_value: object,
    high: object,
    low: object,
    close: object,
    base_volume: object = None,
    quote_volume: object = None,
    trades: object = None,
    complete: bool,
) -> Candle:
    open_decimal = decimal_or_none(open_value)
    high_decimal = decimal_or_none(high)
    low_decimal = decimal_or_none(low)
    close_decimal = decimal_or_none(close)
    if None in {open_decimal, high_decimal, low_decimal, close_decimal}:
        raise FetchError("candle is missing an OHLC value")
    return Candle(
        cohort="",
        source="",
        symbol="",
        market_type="",
        base="",
        quote="",
        interval_minutes=interval_minutes,
        open_time_ms=open_time_ms,
        open=open_decimal,
        high=high_decimal,
        low=low_decimal,
        close=close_decimal,
        base_volume=decimal_or_none(base_volume),
        quote_volume=decimal_or_none(quote_volume),
        trades=int(trades) if trades not in (None, "") else None,
        complete=complete,
    )


class SourceAdapter:
    name: str
    supported_intervals: frozenset[int]

    def fetch(
        self,
        client: JsonClient,
        *,
        symbol: str,
        interval_minutes: int,
        start_ms: int,
        end_ms: int,
        now_ms: int,
    ) -> list[Candle]:
        raise NotImplementedError

    def validate_interval(self, interval_minutes: int) -> None:
        if interval_minutes not in self.supported_intervals:
            choices = ", ".join(map(str, sorted(self.supported_intervals)))
            raise ValueError(
                f"{self.name} does not support {interval_minutes}m; choose {choices}"
            )

    def collection_interval_minutes(
        self, requested_interval_minutes: int, output_interval_minutes: int
    ) -> int:
        """Choose the source-native interval used for a collection run."""

        return requested_interval_minutes


class BinanceAdapter(SourceAdapter):
    name = "binance"
    supported_intervals = frozenset({1, 3, 5})
    url = "https://data-api.binance.vision/api/v3/klines"

    def fetch(
        self,
        client: JsonClient,
        *,
        symbol: str,
        interval_minutes: int,
        start_ms: int,
        end_ms: int,
        now_ms: int,
    ) -> list[Candle]:
        self.validate_interval(interval_minutes)
        result: list[Candle] = []
        for page_start, page_end in _windows(
            start_ms, end_ms, interval_minutes=interval_minutes, page_size=1000
        ):
            payload = client.get_json(
                self.url,
                {
                    "symbol": symbol,
                    "interval": f"{interval_minutes}m",
                    "startTime": page_start,
                    "endTime": page_end - 1,
                    "limit": 1000,
                },
            )
            if not isinstance(payload, list):
                raise FetchError(f"binance returned unexpected payload: {payload!r}")
            for row in payload:
                result.append(
                    _empty_candle(
                        interval_minutes=interval_minutes,
                        open_time_ms=int(row[0]),
                        open_value=row[1],
                        high=row[2],
                        low=row[3],
                        close=row[4],
                        base_volume=row[5],
                        quote_volume=row[7],
                        trades=row[8],
                        complete=int(row[6]) < now_ms,
                    )
                )
        return result


class BybitAdapter(SourceAdapter):
    name = "bybit"
    supported_intervals = frozenset({1, 3, 5})
    url = "https://api.bybit.com/v5/market/kline"

    def fetch(
        self,
        client: JsonClient,
        *,
        symbol: str,
        interval_minutes: int,
        start_ms: int,
        end_ms: int,
        now_ms: int,
    ) -> list[Candle]:
        self.validate_interval(interval_minutes)
        result: list[Candle] = []
        interval_ms = interval_minutes * 60_000
        for page_start, page_end in _windows(
            start_ms, end_ms, interval_minutes=interval_minutes, page_size=1000
        ):
            payload = client.get_json(
                self.url,
                {
                    "category": "spot",
                    "symbol": symbol,
                    "interval": str(interval_minutes),
                    "start": page_start,
                    "end": page_end - 1,
                    "limit": 1000,
                },
            )
            if payload.get("retCode") != 0:
                raise FetchError(f"bybit error: {payload!r}")
            for row in payload.get("result", {}).get("list", []):
                open_time_ms = int(row[0])
                result.append(
                    _empty_candle(
                        interval_minutes=interval_minutes,
                        open_time_ms=open_time_ms,
                        open_value=row[1],
                        high=row[2],
                        low=row[3],
                        close=row[4],
                        base_volume=row[5],
                        quote_volume=row[6],
                        complete=open_time_ms + interval_ms <= now_ms,
                    )
                )
        return result


class OkxAdapter(SourceAdapter):
    name = "okx"
    supported_intervals = frozenset({1, 3, 5})
    url = "https://www.okx.com/api/v5/market/history-candles"

    def fetch(
        self,
        client: JsonClient,
        *,
        symbol: str,
        interval_minutes: int,
        start_ms: int,
        end_ms: int,
        now_ms: int,
    ) -> list[Candle]:
        self.validate_interval(interval_minutes)
        result: list[Candle] = []
        for page_start, page_end in _windows(
            start_ms, end_ms, interval_minutes=interval_minutes, page_size=300
        ):
            payload = client.get_json(
                self.url,
                {
                    "instId": symbol,
                    "bar": f"{interval_minutes}m",
                    "after": page_end,
                    "limit": 300,
                },
            )
            if payload.get("code") != "0":
                raise FetchError(f"okx error: {payload!r}")
            for row in payload.get("data", []):
                open_time_ms = int(row[0])
                if not page_start <= open_time_ms < page_end:
                    continue
                result.append(
                    _empty_candle(
                        interval_minutes=interval_minutes,
                        open_time_ms=open_time_ms,
                        open_value=row[1],
                        high=row[2],
                        low=row[3],
                        close=row[4],
                        base_volume=row[5],
                        quote_volume=row[7],
                        complete=row[8] == "1",
                    )
                )
        return result


class KucoinAdapter(SourceAdapter):
    name = "kucoin"
    supported_intervals = frozenset({1, 3, 5})
    url = "https://api.kucoin.com/api/ua/v1/market/kline"

    def fetch(
        self,
        client: JsonClient,
        *,
        symbol: str,
        interval_minutes: int,
        start_ms: int,
        end_ms: int,
        now_ms: int,
    ) -> list[Candle]:
        self.validate_interval(interval_minutes)
        result: list[Candle] = []
        interval_ms = interval_minutes * 60_000
        for page_start, page_end in _windows(
            start_ms, end_ms, interval_minutes=interval_minutes, page_size=1500
        ):
            payload = client.get_json(
                self.url,
                {
                    "tradeType": "SPOT",
                    "symbol": symbol,
                    "interval": f"{interval_minutes}min",
                    "startAt": page_start // 1000,
                    "endAt": (page_end - 1) // 1000,
                },
            )
            if payload.get("code") != "200000":
                raise FetchError(f"kucoin error: {payload!r}")
            data = payload.get("data", [])
            rows = data.get("list", []) if isinstance(data, dict) else data
            if not isinstance(rows, list):
                raise FetchError(f"kucoin returned unexpected payload: {payload!r}")
            for row in rows:
                open_time_ms = int(row[0]) * 1000
                result.append(
                    _empty_candle(
                        interval_minutes=interval_minutes,
                        open_time_ms=open_time_ms,
                        open_value=row[1],
                        high=row[2],
                        low=row[3],
                        close=row[4],
                        base_volume=row[5],
                        quote_volume=row[6],
                        complete=open_time_ms + interval_ms <= now_ms,
                    )
                )
        return result


class CoinbaseAdapter(SourceAdapter):
    name = "coinbase"
    supported_intervals = frozenset({1, 5, 60})
    url_template = "https://api.exchange.coinbase.com/products/{symbol}/candles"

    def fetch(
        self,
        client: JsonClient,
        *,
        symbol: str,
        interval_minutes: int,
        start_ms: int,
        end_ms: int,
        now_ms: int,
    ) -> list[Candle]:
        self.validate_interval(interval_minutes)
        result: list[Candle] = []
        interval_ms = interval_minutes * 60_000
        for page_start, page_end in _windows(
            start_ms, end_ms, interval_minutes=interval_minutes, page_size=300
        ):
            payload = client.get_json(
                self.url_template.format(symbol=symbol),
                {
                    "granularity": interval_minutes * 60,
                    "start": datetime.fromtimestamp(page_start / 1000, tz=UTC).isoformat(),
                    "end": datetime.fromtimestamp(page_end / 1000, tz=UTC).isoformat(),
                },
            )
            if not isinstance(payload, list):
                raise FetchError(f"coinbase returned unexpected payload: {payload!r}")
            for row in payload:
                open_time_ms = int(row[0]) * 1000
                if not page_start <= open_time_ms < page_end:
                    continue
                result.append(
                    _empty_candle(
                        interval_minutes=interval_minutes,
                        open_time_ms=open_time_ms,
                        open_value=row[3],
                        high=row[2],
                        low=row[1],
                        close=row[4],
                        base_volume=row[5],
                        complete=open_time_ms + interval_ms <= now_ms,
                    )
                )
        return result


class BitstampAdapter(SourceAdapter):
    name = "bitstamp"
    supported_intervals = frozenset({1, 3, 5, 60})
    url_template = "https://www.bitstamp.net/api/v2/ohlc/{symbol}/"

    def fetch(
        self,
        client: JsonClient,
        *,
        symbol: str,
        interval_minutes: int,
        start_ms: int,
        end_ms: int,
        now_ms: int,
    ) -> list[Candle]:
        self.validate_interval(interval_minutes)
        result: list[Candle] = []
        interval_ms = interval_minutes * 60_000
        for page_start, page_end in _windows(
            start_ms, end_ms, interval_minutes=interval_minutes, page_size=1000
        ):
            payload = client.get_json(
                self.url_template.format(symbol=symbol),
                {
                    "step": interval_minutes * 60,
                    "limit": 1000,
                    "start": page_start // 1000,
                    "end": (page_end - 1) // 1000,
                    "exclude_current_candle": "true",
                },
            )
            rows = payload.get("data", {}).get("ohlc")
            if rows is None:
                raise FetchError(f"bitstamp returned unexpected payload: {payload!r}")
            for row in rows:
                open_time_ms = int(row["timestamp"]) * 1000
                if not page_start <= open_time_ms < page_end:
                    continue
                result.append(
                    _empty_candle(
                        interval_minutes=interval_minutes,
                        open_time_ms=open_time_ms,
                        open_value=row["open"],
                        high=row["high"],
                        low=row["low"],
                        close=row["close"],
                        base_volume=row.get("volume"),
                        complete=open_time_ms + interval_ms <= now_ms,
                    )
                )
        return result


class DefillamaAdapter(SourceAdapter):
    """DefiLlama sampled USD reference prices exposed by its free Coins API.

    Chart observations are normally within a few seconds of the requested UTC
    boundary. They are snapped to the nearest boundary and represented as the
    close of the preceding interval because the endpoint does not expose OHLCV.
    """

    name = "defillama"
    supported_intervals = frozenset({1, 3, 5, 60})
    url_template = "https://coins.llama.fi/chart/{symbol}"

    def collection_interval_minutes(
        self, requested_interval_minutes: int, output_interval_minutes: int
    ) -> int:
        # Its nominal 1-minute chart has intermittent gaps. When possible,
        # request one reference observation per comparison bucket instead.
        if output_interval_minutes in self.supported_intervals:
            return output_interval_minutes
        return requested_interval_minutes

    def fetch(
        self,
        client: JsonClient,
        *,
        symbol: str,
        interval_minutes: int,
        start_ms: int,
        end_ms: int,
        now_ms: int,
    ) -> list[Candle]:
        self.validate_interval(interval_minutes)
        result: list[Candle] = []
        interval_ms = interval_minutes * 60_000
        interval_seconds = interval_minutes * 60
        for page_start, page_end in _windows(
            start_ms, end_ms, interval_minutes=interval_minutes, page_size=500
        ):
            span = (page_end - page_start) // interval_ms
            payload = client.get_json(
                self.url_template.format(symbol=symbol),
                {
                    # Request interval-close observations. A point at 00:05 is
                    # assigned to the candle [00:00, 00:05), matching exchange
                    # close semantics used by the comparison artifacts.
                    "start": (page_start + interval_ms) // 1000,
                    "period": (
                        "1h" if interval_minutes == 60 else f"{interval_minutes}m"
                    ),
                    "span": span,
                },
            )
            coin = payload.get("coins", {}).get(symbol)
            if not isinstance(coin, dict) or not isinstance(coin.get("prices"), list):
                raise FetchError(
                    f"defillama returned unexpected payload for {symbol}: {payload!r}"
                )
            for point in coin["prices"]:
                timestamp_seconds = int(point["timestamp"])
                # DefiLlama samples near, rather than exactly on, each requested
                # boundary (usually by only a few seconds).
                boundary_seconds = (
                    (timestamp_seconds + interval_seconds // 2) // interval_seconds
                ) * interval_seconds
                open_time_ms = boundary_seconds * 1000 - interval_ms
                if not page_start <= open_time_ms < page_end:
                    continue
                price = point.get("price")
                result.append(
                    _empty_candle(
                        interval_minutes=interval_minutes,
                        open_time_ms=open_time_ms,
                        open_value=price,
                        high=price,
                        low=price,
                        close=price,
                        complete=open_time_ms + interval_ms <= now_ms,
                    )
                )
        return result


class CoingeckoAdapter(SourceAdapter):
    """CoinGecko keyless hourly USD aggregate prices.

    Arbitrary historical 5-minute data is intentionally unsupported because
    CoinGecko reserves ``interval=5m`` for Enterprise customers. Hourly samples
    are treated as closes of the preceding hour, consistent with DefiLlama's
    sampled-reference normalization.
    """

    name = "coingecko"
    supported_intervals = frozenset({60})
    url_template = (
        "https://api.coingecko.com/api/v3/coins/{symbol}/market_chart/range"
    )

    def fetch(
        self,
        client: JsonClient,
        *,
        symbol: str,
        interval_minutes: int,
        start_ms: int,
        end_ms: int,
        now_ms: int,
    ) -> list[Candle]:
        self.validate_interval(interval_minutes)
        result: list[Candle] = []
        interval_ms = interval_minutes * 60_000
        # CoinGecko documents up to 100 days per explicit-hourly request.
        for page_start, page_end in _windows(
            start_ms, end_ms, interval_minutes=interval_minutes, page_size=2400
        ):
            payload = client.get_json(
                self.url_template.format(symbol=symbol),
                {
                    "vs_currency": "usd",
                    "from": (page_start + interval_ms) // 1000,
                    "to": page_end // 1000,
                    "interval": "hourly",
                    "precision": "full",
                },
            )
            prices = payload.get("prices")
            if not isinstance(prices, list):
                raise FetchError(
                    f"coingecko returned unexpected payload for {symbol}: {payload!r}"
                )
            for point in prices:
                if not isinstance(point, list) or len(point) < 2:
                    raise FetchError(f"coingecko returned malformed price: {point!r}")
                timestamp_ms = int(point[0])
                boundary_ms = (
                    (timestamp_ms + interval_ms // 2) // interval_ms
                ) * interval_ms
                open_time_ms = boundary_ms - interval_ms
                if not page_start <= open_time_ms < page_end:
                    continue
                price = point[1]
                result.append(
                    _empty_candle(
                        interval_minutes=interval_minutes,
                        open_time_ms=open_time_ms,
                        open_value=price,
                        high=price,
                        low=price,
                        close=price,
                        complete=open_time_ms + interval_ms <= now_ms,
                    )
                )
        return result


ADAPTERS: dict[str, SourceAdapter] = {
    adapter.name: adapter
    for adapter in (
        BinanceAdapter(),
        BybitAdapter(),
        OkxAdapter(),
        KucoinAdapter(),
        CoinbaseAdapter(),
        BitstampAdapter(),
        DefillamaAdapter(),
        CoingeckoAdapter(),
    )
}


def fetch_market_source(
    client: JsonClient,
    *,
    cohort: MarketCohort,
    market_source: MarketSource,
    interval_minutes: int,
    start_ms: int,
    end_ms: int,
    now_ms: int,
) -> list[Candle]:
    adapter = ADAPTERS[market_source.source]
    rows = adapter.fetch(
        client,
        symbol=market_source.symbol,
        interval_minutes=interval_minutes,
        start_ms=start_ms,
        end_ms=end_ms,
        now_ms=now_ms,
    )
    identified = [
        row.with_identity(
            cohort=cohort.name,
            source=market_source.source,
            symbol=market_source.symbol,
            market_type=market_source.market_type or cohort.market_type,
            base=cohort.base,
            quote=cohort.quote,
        )
        for row in rows
        if start_ms <= row.open_time_ms < end_ms
    ]
    deduplicated = {row.open_time_ms: row for row in identified}
    return [deduplicated[timestamp] for timestamp in sorted(deduplicated)]
