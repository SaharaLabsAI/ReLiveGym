"""Normalized candle model and deterministic interval aggregation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal
from typing import Iterable


def decimal_or_none(value: object) -> Decimal | None:
    """Convert an API value to Decimal without passing through binary float."""

    if value is None or value == "":
        return None
    return Decimal(str(value))


def iso_utc(timestamp_ms: int) -> str:
    """Render an epoch-millisecond timestamp as a canonical UTC string."""

    return datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC).isoformat().replace(
        "+00:00", "Z"
    )


@dataclass(frozen=True, slots=True)
class Candle:
    """A venue candle normalized across all supported public APIs.

    ``open_time_ms`` is the inclusive UTC bucket boundary. ``complete`` means
    that the venue considers the source candle closed; for aggregated candles,
    it additionally means that every expected source interval was present.
    """

    cohort: str
    source: str
    symbol: str
    market_type: str
    base: str
    quote: str
    interval_minutes: int
    open_time_ms: int
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    base_volume: Decimal | None = None
    quote_volume: Decimal | None = None
    trades: int | None = None
    complete: bool = True
    observation_count: int = 1
    expected_observation_count: int = 1

    @property
    def vwap(self) -> Decimal | None:
        if (
            self.base_volume is None
            or self.quote_volume is None
            or self.base_volume == 0
        ):
            return None
        return self.quote_volume / self.base_volume

    @property
    def open_time(self) -> str:
        return iso_utc(self.open_time_ms)

    def with_identity(
        self,
        *,
        cohort: str,
        source: str,
        symbol: str,
        market_type: str,
        base: str,
        quote: str,
    ) -> "Candle":
        return replace(
            self,
            cohort=cohort,
            source=source,
            symbol=symbol,
            market_type=market_type,
            base=base,
            quote=quote,
        )


def _sum_optional(values: Iterable[Decimal | None]) -> Decimal | None:
    values = list(values)
    if not values or any(value is None for value in values):
        return None
    return sum((value for value in values if value is not None), Decimal(0))


def _sum_optional_int(values: Iterable[int | None]) -> int | None:
    values = list(values)
    if not values or any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


def aggregate_candles(
    candles: Iterable[Candle],
    *,
    output_interval_minutes: int,
    start_ms: int,
    end_ms: int,
) -> list[Candle]:
    """Aggregate one source's candles onto fixed UTC buckets.

    Partial buckets are retained but marked incomplete. Duplicate source
    timestamps are deterministically collapsed, preferring a completed candle.
    The caller can therefore inspect data gaps without silently forward-filling.
    """

    rows = list(candles)
    if not rows:
        return []

    source_intervals = {row.interval_minutes for row in rows}
    identities = {
        (row.cohort, row.source, row.symbol, row.market_type, row.base, row.quote)
        for row in rows
    }
    if len(source_intervals) != 1:
        raise ValueError("cannot aggregate mixed source intervals")
    if len(identities) != 1:
        raise ValueError("cannot aggregate candles from different markets or sources")

    source_interval = source_intervals.pop()
    if output_interval_minutes < source_interval:
        raise ValueError("output interval cannot be finer than the source interval")
    if output_interval_minutes % source_interval:
        raise ValueError("output interval must be a multiple of the source interval")

    deduplicated: dict[int, Candle] = {}
    for row in rows:
        if not (start_ms <= row.open_time_ms < end_ms):
            continue
        existing = deduplicated.get(row.open_time_ms)
        if existing is None or (row.complete and not existing.complete):
            deduplicated[row.open_time_ms] = row

    bucket_ms = output_interval_minutes * 60_000
    source_ms = source_interval * 60_000
    expected_count = output_interval_minutes // source_interval
    grouped: dict[int, list[Candle]] = {}
    for row in deduplicated.values():
        bucket = row.open_time_ms - (row.open_time_ms % bucket_ms)
        grouped.setdefault(bucket, []).append(row)

    aggregated: list[Candle] = []
    for bucket, bucket_rows in sorted(grouped.items()):
        bucket_rows.sort(key=lambda row: row.open_time_ms)
        first = bucket_rows[0]
        expected_times = {bucket + index * source_ms for index in range(expected_count)}
        observed_times = {row.open_time_ms for row in bucket_rows}
        complete = (
            observed_times == expected_times
            and all(row.complete for row in bucket_rows)
            and bucket >= start_ms
            and bucket + bucket_ms <= end_ms
        )
        aggregated.append(
            Candle(
                cohort=first.cohort,
                source=first.source,
                symbol=first.symbol,
                market_type=first.market_type,
                base=first.base,
                quote=first.quote,
                interval_minutes=output_interval_minutes,
                open_time_ms=bucket,
                open=first.open,
                high=max(row.high for row in bucket_rows),
                low=min(row.low for row in bucket_rows),
                close=bucket_rows[-1].close,
                base_volume=_sum_optional(row.base_volume for row in bucket_rows),
                quote_volume=_sum_optional(row.quote_volume for row in bucket_rows),
                trades=_sum_optional_int(row.trades for row in bucket_rows),
                complete=complete,
                observation_count=len(bucket_rows),
                expected_observation_count=expected_count,
            )
        )
    return aggregated

