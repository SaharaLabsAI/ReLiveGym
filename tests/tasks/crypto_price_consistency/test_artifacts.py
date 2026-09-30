import csv
import json
from decimal import Decimal

from tasks.crypto_price_consistency.data.artifacts import (
    read_normalized,
    write_comparison_artifacts,
    write_normalized,
)
from tasks.crypto_price_consistency.data.models import Candle


def candle(source: str, close: str, *, complete: bool = True) -> Candle:
    price = Decimal(close)
    return Candle(
        cohort="btc_usdt_spot",
        source=source,
        symbol="BTCUSDT",
        market_type="spot",
        base="BTC",
        quote="USDT",
        interval_minutes=5,
        open_time_ms=0,
        open=price,
        high=price,
        low=price,
        close=price,
        base_volume=Decimal("2"),
        quote_volume=price * 2,
        complete=complete,
        observation_count=5,
        expected_observation_count=5,
    )


def test_normalized_round_trip_and_comparison_outputs(tmp_path) -> None:
    normalized = tmp_path / "candles.csv"
    write_normalized(normalized, [candle("alpha", "100")])
    restored = read_normalized(normalized)
    assert restored == [candle("alpha", "100")]

    output = tmp_path / "comparison"
    summary = write_comparison_artifacts(
        output,
        {
            "alpha": [candle("alpha", "100")],
            "beta": [candle("beta", "102")],
            "gamma": [candle("gamma", "500", complete=False)],
        },
        expected_sources=["alpha", "beta", "gamma"],
    )

    with (output / "close_matrix.csv").open(encoding="utf-8") as source:
        matrix = list(csv.DictReader(source))
    assert matrix[0]["median_close"] == "101"
    assert matrix[0]["complete_source_count"] == "2"
    assert matrix[0]["gamma_close"] == "500"
    assert matrix[0]["gamma_deviation_bps"] == ""

    assert summary["pairs"]["alpha__beta"]["overlap_count"] == 1
    disk_summary = json.loads((output / "summary.json").read_text())
    assert disk_summary == summary
