"""Replayed prices for broker_ops: one venue's 5-minute candles from
the crypto_price_consistency datasets (layer 1, read-only).
`mark(symbol, now)` is the close of the last candle completed at or
before now; `between(symbol, t_from, t_to)` yields the candles whose
close time lies in (t_from, t_to] — what fills and margin checks scan."""

from __future__ import annotations

import bisect
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

CANDLE_SECONDS = 300


@dataclass(frozen=True)
class Candle:
    close_ts: float
    open: float
    high: float
    low: float
    close: float

    @property
    def close_time(self) -> datetime:
        return datetime.fromtimestamp(self.close_ts, tz=timezone.utc)


class CandleStore:
    def __init__(self, datasets_dir: Path, symbols: list[str], suffix: str,
                 venue: str):
        self._series: dict[str, list[Candle]] = {}
        self._keys: dict[str, list[float]] = {}
        for sym in symbols:
            cohort = f"{sym.lower()}_usdt_spot"
            path = (datasets_dir / f"{cohort}_{suffix}" / "normalized"
                    / f"cohort_{cohort}" / f"source_{venue}" / "interval_5m"
                    / "candles.csv")
            if not path.exists():
                raise ValueError(f"no candles for {sym} at {path}")
            rows: list[Candle] = []
            with path.open() as fh:
                for r in csv.DictReader(fh):
                    rows.append(Candle(int(r["open_time_ms"]) / 1000 + CANDLE_SECONDS,
                                       float(r["open"]), float(r["high"]),
                                       float(r["low"]), float(r["close"])))
            rows.sort(key=lambda c: c.close_ts)
            self._series[sym] = rows
            self._keys[sym] = [c.close_ts for c in rows]
        self.symbols = list(symbols)

    def coverage(self, sym: str) -> tuple[datetime, datetime]:
        ks = self._keys[sym]
        return (datetime.fromtimestamp(ks[0], tz=timezone.utc),
                datetime.fromtimestamp(ks[-1], tz=timezone.utc))

    def mark(self, sym: str, now: datetime) -> float | None:
        i = bisect.bisect_right(self._keys[sym], now.timestamp()) - 1
        return self._series[sym][i].close if i >= 0 else None

    def between(self, sym: str, t_from: datetime, t_to: datetime) -> list[Candle]:
        ks = self._keys[sym]
        lo = bisect.bisect_right(ks, t_from.timestamp())
        hi = bisect.bisect_right(ks, t_to.timestamp())
        return self._series[sym][lo:hi]
