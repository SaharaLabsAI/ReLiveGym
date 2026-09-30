from typing import Any

import pytest

from tasks.crypto_price_consistency.data.markets import MarketCohort, MarketSource
from tasks.crypto_price_consistency.data.sources import ADAPTERS, fetch_market_source


class StubClient:
    def __init__(self, payload: Any) -> None:
        self.payload = payload
        self.calls: list[tuple[str, dict[str, object]]] = []

    def get_json(self, url: str, params: dict[str, object]) -> Any:
        self.calls.append((url, dict(params)))
        return self.payload


COHORT = MarketCohort(
    name="test_usdt_spot",
    description="test",
    base="BTC",
    quote="USDT",
    market_type="spot",
    sources=(),
)


@pytest.mark.parametrize(
    ("source", "symbol", "payload"),
    [
        (
            "binance",
            "BTCUSDT",
            [[0, "100", "102", "99", "101", "2", 59_999, "202", 7, "0", "0", "0"]],
        ),
        (
            "bybit",
            "BTCUSDT",
            {
                "retCode": 0,
                "result": {
                    "list": [["0", "100", "102", "99", "101", "2", "202"]]
                },
            },
        ),
        (
            "okx",
            "BTC-USDT",
            {
                "code": "0",
                "data": [["0", "100", "102", "99", "101", "2", "2", "202", "1"]],
            },
        ),
        (
            "kucoin",
            "BTC-USDT",
            {
                "code": "200000",
                "data": {
                    "tradeType": "SPOT",
                    "symbol": "BTC-USDT",
                    "list": [[0, "100", "102", "99", "101", "2", "202"]],
                },
            },
        ),
    ],
)
def test_usdt_adapters_normalize_identity_and_ohlcv(
    source: str, symbol: str, payload: Any
) -> None:
    client = StubClient(payload)

    rows = fetch_market_source(
        client,
        cohort=COHORT,
        market_source=MarketSource(source, symbol),
        interval_minutes=1,
        start_ms=0,
        end_ms=60_000,
        now_ms=120_000,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.source == source
    assert row.symbol == symbol
    assert row.base == "BTC"
    assert row.quote == "USDT"
    assert str(row.open) == "100"
    assert str(row.high) == "102"
    assert str(row.low) == "99"
    assert str(row.close) == "101"
    assert str(row.base_volume) == "2"
    assert str(row.quote_volume) == "202"
    assert row.complete is True
    assert len(client.calls) == 1


@pytest.mark.parametrize(
    ("source", "symbol", "payload"),
    [
        ("coinbase", "BTC-USD", [[0, "99", "102", "100", "101", "2"]]),
        (
            "bitstamp",
            "btcusd",
            {
                "data": {
                    "ohlc": [
                        {
                            "timestamp": "0",
                            "open": "100",
                            "high": "102",
                            "low": "99",
                            "close": "101",
                            "volume": "2",
                        }
                    ]
                }
            },
        ),
    ],
)
def test_usd_adapters_normalize_candles(
    source: str, symbol: str, payload: Any
) -> None:
    client = StubClient(payload)
    cohort = MarketCohort(
        name="test_usd_spot",
        description="test",
        base="BTC",
        quote="USD",
        market_type="spot",
        sources=(),
    )

    rows = fetch_market_source(
        client,
        cohort=cohort,
        market_source=MarketSource(source, symbol),
        interval_minutes=1,
        start_ms=0,
        end_ms=60_000,
        now_ms=120_000,
    )

    assert len(rows) == 1
    assert str(rows[0].open) == "100"
    assert str(rows[0].close) == "101"
    assert rows[0].quote == "USD"


def test_defillama_normalizes_near_boundary_reference_price() -> None:
    client = StubClient(
        {
            "coins": {
                "coingecko:bitcoin": {
                    "symbol": "BTC",
                    "confidence": 0.99,
                    "prices": [
                        {"timestamp": 61, "price": 101.25},
                        {"timestamp": 122, "price": 102.5},
                    ],
                }
            }
        }
    )
    cohort = MarketCohort(
        name="test_usd_spot",
        description="test",
        base="BTC",
        quote="USD",
        market_type="spot",
        sources=(),
    )

    rows = fetch_market_source(
        client,
        cohort=cohort,
        market_source=MarketSource(
            "defillama", "coingecko:bitcoin", "reference"
        ),
        interval_minutes=1,
        start_ms=0,
        end_ms=120_000,
        now_ms=180_000,
    )

    assert [row.open_time_ms for row in rows] == [0, 60_000]
    assert [str(row.close) for row in rows] == ["101.25", "102.5"]
    assert rows[0].open == rows[0].high == rows[0].low == rows[0].close
    assert rows[0].market_type == "reference"
    assert rows[0].base_volume is None
    assert client.calls[0][1] == {"start": 60, "period": "1m", "span": 2}


def test_defillama_prefers_supported_output_interval() -> None:
    adapter = ADAPTERS["defillama"]

    assert adapter.collection_interval_minutes(1, 5) == 5
    assert adapter.collection_interval_minutes(1, 10) == 1


def test_coingecko_normalizes_hourly_reference_close() -> None:
    client = StubClient(
        {
            "prices": [[3_600_000, 101.25], [7_200_000, 102.5]],
            "market_caps": [],
            "total_volumes": [],
        }
    )
    cohort = MarketCohort(
        name="test_usd_hourly",
        description="test",
        base="BTC",
        quote="USD",
        market_type="spot",
        sources=(),
    )

    rows = fetch_market_source(
        client,
        cohort=cohort,
        market_source=MarketSource("coingecko", "bitcoin", "reference"),
        interval_minutes=60,
        start_ms=0,
        end_ms=7_200_000,
        now_ms=10_800_000,
    )

    assert [row.open_time_ms for row in rows] == [0, 3_600_000]
    assert [str(row.close) for row in rows] == ["101.25", "102.5"]
    assert rows[0].market_type == "reference"
    assert client.calls[0][1] == {
        "vs_currency": "usd",
        "from": 3600,
        "to": 7200,
        "interval": "hourly",
        "precision": "full",
    }
