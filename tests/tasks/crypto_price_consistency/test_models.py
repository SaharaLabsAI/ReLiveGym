from decimal import Decimal

from tasks.crypto_price_consistency.data.models import Candle, aggregate_candles


def candle(minute: int, *, complete: bool = True) -> Candle:
    value = Decimal(100 + minute)
    return Candle(
        cohort="btc_usdt_spot",
        source="example",
        symbol="BTCUSDT",
        market_type="spot",
        base="BTC",
        quote="USDT",
        interval_minutes=1,
        open_time_ms=minute * 60_000,
        open=value,
        high=value + 2,
        low=value - 2,
        close=value + 1,
        base_volume=Decimal("2"),
        quote_volume=value * 2,
        trades=3,
        complete=complete,
    )


def test_aggregate_five_one_minute_candles() -> None:
    result = aggregate_candles(
        [candle(minute) for minute in range(5)],
        output_interval_minutes=5,
        start_ms=0,
        end_ms=5 * 60_000,
    )

    assert len(result) == 1
    aggregated = result[0]
    assert aggregated.open == Decimal("100")
    assert aggregated.high == Decimal("106")
    assert aggregated.low == Decimal("98")
    assert aggregated.close == Decimal("105")
    assert aggregated.base_volume == Decimal("10")
    assert aggregated.quote_volume == Decimal("1020")
    assert aggregated.vwap == Decimal("102")
    assert aggregated.trades == 15
    assert aggregated.complete is True
    assert aggregated.observation_count == 5
    assert aggregated.expected_observation_count == 5


def test_aggregate_retains_but_marks_gap_incomplete() -> None:
    result = aggregate_candles(
        [candle(0), candle(1), candle(3), candle(4)],
        output_interval_minutes=5,
        start_ms=0,
        end_ms=5 * 60_000,
    )

    assert len(result) == 1
    assert result[0].complete is False
    assert result[0].observation_count == 4
    assert result[0].expected_observation_count == 5

