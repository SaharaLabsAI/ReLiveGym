"""Read, write, and compare normalized cross-venue candle artifacts."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from decimal import Decimal
from itertools import combinations
from pathlib import Path

from tasks.crypto_price_consistency.data.models import Candle, decimal_or_none, iso_utc


NORMALIZED_FIELDS = (
    "cohort",
    "source",
    "symbol",
    "market_type",
    "base",
    "quote",
    "interval_minutes",
    "open_time_ms",
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "base_volume",
    "quote_volume",
    "vwap",
    "trades",
    "complete",
    "observation_count",
    "expected_observation_count",
)


def _decimal_text(value: Decimal | None) -> str:
    return "" if value is None else format(value, "f")


def _open_text(path: Path, mode: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open(mode=mode, encoding="utf-8", newline="")


def candle_to_row(candle: Candle) -> dict[str, object]:
    return {
        "cohort": candle.cohort,
        "source": candle.source,
        "symbol": candle.symbol,
        "market_type": candle.market_type,
        "base": candle.base,
        "quote": candle.quote,
        "interval_minutes": candle.interval_minutes,
        "open_time_ms": candle.open_time_ms,
        "open_time": candle.open_time,
        "open": _decimal_text(candle.open),
        "high": _decimal_text(candle.high),
        "low": _decimal_text(candle.low),
        "close": _decimal_text(candle.close),
        "base_volume": _decimal_text(candle.base_volume),
        "quote_volume": _decimal_text(candle.quote_volume),
        "vwap": _decimal_text(candle.vwap),
        "trades": "" if candle.trades is None else candle.trades,
        "complete": "true" if candle.complete else "false",
        "observation_count": candle.observation_count,
        "expected_observation_count": candle.expected_observation_count,
    }


def write_normalized(path: Path, candles: Iterable[Candle]) -> None:
    with _open_text(path, "wt") as output:
        writer = csv.DictWriter(output, fieldnames=NORMALIZED_FIELDS)
        writer.writeheader()
        for candle in sorted(candles, key=lambda row: row.open_time_ms):
            writer.writerow(candle_to_row(candle))


def read_normalized(path: Path) -> list[Candle]:
    rows: list[Candle] = []
    with _open_text(path, "rt") as source:
        for row in csv.DictReader(source):
            rows.append(
                Candle(
                    cohort=row["cohort"],
                    source=row["source"],
                    symbol=row["symbol"],
                    market_type=row["market_type"],
                    base=row["base"],
                    quote=row["quote"],
                    interval_minutes=int(row["interval_minutes"]),
                    open_time_ms=int(row["open_time_ms"]),
                    open=Decimal(row["open"]),
                    high=Decimal(row["high"]),
                    low=Decimal(row["low"]),
                    close=Decimal(row["close"]),
                    base_volume=decimal_or_none(row["base_volume"]),
                    quote_volume=decimal_or_none(row["quote_volume"]),
                    trades=int(row["trades"]) if row["trades"] else None,
                    complete=row["complete"].lower() == "true",
                    observation_count=int(row["observation_count"]),
                    expected_observation_count=int(
                        row["expected_observation_count"]
                    ),
                )
            )
    return rows


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _median(values: Sequence[Decimal]) -> Decimal:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _quantile(values: Sequence[Decimal], quantile: Decimal) -> Decimal | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = Decimal(len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _bps(numerator: Decimal, denominator: Decimal) -> Decimal | None:
    if denominator == 0:
        return None
    return numerator / denominator * Decimal(10_000)


def build_comparison_rows(
    candles_by_source: Mapping[str, Sequence[Candle]],
    *,
    expected_sources: Sequence[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Build a timestamp matrix and pairwise close-price spread rows.

    Comparison statistics only use complete candles. Incomplete observations
    remain visible in the matrix with an explicit ``*_complete`` column.
    Pairwise basis-point spreads use the symmetric mid-price denominator, so
    changing source order changes the sign but not the absolute magnitude.
    """

    indexed = {
        source: {row.open_time_ms: row for row in rows}
        for source, rows in candles_by_source.items()
    }
    timestamps = sorted(
        {timestamp for source_rows in indexed.values() for timestamp in source_rows}
    )

    matrix_rows: list[dict[str, object]] = []
    pairwise_rows: list[dict[str, object]] = []
    for timestamp in timestamps:
        observed = {
            source: indexed.get(source, {}).get(timestamp) for source in expected_sources
        }
        complete = {
            source: candle
            for source, candle in observed.items()
            if candle is not None and candle.complete
        }
        closes = [candle.close for candle in complete.values()]
        median_close = _median(closes) if closes else None
        matrix: dict[str, object] = {
            "open_time_ms": timestamp,
            "open_time": iso_utc(timestamp),
            "observed_source_count": sum(
                candle is not None for candle in observed.values()
            ),
            "complete_source_count": len(complete),
            "expected_source_count": len(expected_sources),
            "median_close": _decimal_text(median_close),
            "min_close": _decimal_text(min(closes) if closes else None),
            "max_close": _decimal_text(max(closes) if closes else None),
            "range_bps": _decimal_text(
                _bps(max(closes) - min(closes), median_close)
                if closes and median_close is not None
                else None
            ),
        }
        for source in expected_sources:
            candle = observed[source]
            matrix[f"{source}_close"] = (
                _decimal_text(candle.close) if candle is not None else ""
            )
            matrix[f"{source}_vwap"] = (
                _decimal_text(candle.vwap) if candle is not None else ""
            )
            matrix[f"{source}_complete"] = (
                "true" if candle is not None and candle.complete else "false"
            )
            matrix[f"{source}_deviation_bps"] = (
                _decimal_text(_bps(candle.close - median_close, median_close))
                if candle is not None
                and candle.complete
                and median_close is not None
                else ""
            )
        matrix_rows.append(matrix)

        for source_a, source_b in combinations(expected_sources, 2):
            candle_a = complete.get(source_a)
            candle_b = complete.get(source_b)
            if candle_a is None or candle_b is None:
                continue
            midpoint = (candle_a.close + candle_b.close) / 2
            difference = candle_a.close - candle_b.close
            pairwise_rows.append(
                {
                    "open_time_ms": timestamp,
                    "open_time": iso_utc(timestamp),
                    "source_a": source_a,
                    "source_b": source_b,
                    "close_a": _decimal_text(candle_a.close),
                    "close_b": _decimal_text(candle_b.close),
                    "a_minus_b": _decimal_text(difference),
                    "a_minus_b_bps": _decimal_text(_bps(difference, midpoint)),
                    "absolute_bps": _decimal_text(
                        abs(_bps(difference, midpoint) or Decimal(0))
                    ),
                }
            )
    return matrix_rows, pairwise_rows


def _write_dict_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        with _open_text(path, "wt"):
            pass
        return
    with _open_text(path, "wt") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_comparison(
    candles_by_source: Mapping[str, Sequence[Candle]],
    pairwise_rows: Sequence[Mapping[str, object]],
    *,
    expected_sources: Sequence[str],
) -> dict[str, object]:
    source_summary: dict[str, object] = {}
    for source in expected_sources:
        rows = list(candles_by_source.get(source, []))
        complete_rows = [row for row in rows if row.complete]
        source_summary[source] = {
            "row_count": len(rows),
            "complete_row_count": len(complete_rows),
            "first_open_time": rows[0].open_time if rows else None,
            "last_open_time": rows[-1].open_time if rows else None,
        }

    grouped_pairs: dict[tuple[str, str], list[Decimal]] = {}
    for row in pairwise_rows:
        key = (str(row["source_a"]), str(row["source_b"]))
        grouped_pairs.setdefault(key, []).append(Decimal(str(row["a_minus_b_bps"])))

    pair_summary: dict[str, object] = {}
    for (source_a, source_b), spreads in sorted(grouped_pairs.items()):
        absolute = [abs(value) for value in spreads]
        pair_summary[f"{source_a}__{source_b}"] = {
            "overlap_count": len(spreads),
            "mean_signed_bps": float(sum(spreads, Decimal(0)) / len(spreads)),
            "mean_absolute_bps": float(sum(absolute, Decimal(0)) / len(absolute)),
            "p50_absolute_bps": float(
                _quantile(absolute, Decimal("0.50")) or Decimal(0)
            ),
            "p95_absolute_bps": float(
                _quantile(absolute, Decimal("0.95")) or Decimal(0)
            ),
            "max_absolute_bps": float(max(absolute)),
        }
    return {"sources": source_summary, "pairs": pair_summary}


def write_comparison_artifacts(
    output_dir: Path,
    candles_by_source: Mapping[str, Sequence[Candle]],
    *,
    expected_sources: Sequence[str],
) -> dict[str, object]:
    matrix_rows, pairwise_rows = build_comparison_rows(
        candles_by_source, expected_sources=expected_sources
    )
    _write_dict_rows(output_dir / "close_matrix.csv", matrix_rows)
    _write_dict_rows(output_dir / "pairwise_spreads.csv", pairwise_rows)
    summary = summarize_comparison(
        candles_by_source, pairwise_rows, expected_sources=expected_sources
    )
    write_json(output_dir / "summary.json", summary)
    return summary
