"""crypto_price_consistency: hourly BTC/ETH price reporting under incidents.

The system under test polls simulated venue HTTP (the incidents/proxy.py
replay engine behind the `http_fetch` tool) and must, every hour, either
`report` a price per symbol or explicitly `abstain`. Ground truth is the
untouched cross-venue median of the recorded 5-minute candles — incidents
are injected only into what the system sees.

Scoring (constrained optimization, no dollars): each full hour of the run
is a scoring period per symbol. The
last action inside the hour wins:

  report   excess_bps = min(cap_bps, max(0, bps_error - free_bps)), where
           bps_error is measured against the truth median at the moment
           the report was submitted (freshness within the 5-minute candle
           grid is the reporter's problem, not the scorer's)
  abstain  no error booked; counts against the abstention budget
           (abstain_budget_frac of scored hours)
  nothing  a miss — a protocol violation (the contract requires an
           explicit answer every hour)

Primary metric: mean excess_bps over reported hours (lower is better),
valid subject to the constraints (abstain rate within budget, zero
misses, run budget respected). Venue fetches are free at real rates —
public endpoints — behind the documented limiter machines; request
latency still costs freshness.

Deliberate v1 limitations, documented not hidden: clock_skew trace events
are inert (the harness owns get_time and the env cannot skew it without a
harness hook), as are trace events for venues without recorded data.
"""

from __future__ import annotations

import bisect
import csv
import math
import statistics
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import BaseModel, model_validator

from harness.task import NotificationError, OutcomeEvent, Task
from harness.timeutil import iso

TASK_DIR = Path(__file__).resolve().parent
DATA_WINDOW = (datetime(2026, 3, 1, tzinfo=UTC),
               datetime(2026, 8, 1, tzinfo=UTC))
CANDLE_SECONDS = 300
HOUR = timedelta(hours=1)

# venue-native symbol spelling for each cohort symbol (also the contract
# get_exchange_docs publishes)
VENUE_SYMBOLS = {
    "binance": "{sym}USDT",
    "okx": "{sym}-USDT",
    "kucoin": "{sym}-USDT",
}


class CryptoPriceConsistencyConfig(BaseModel):
    trace: str = "fitted_dev_s001"   # scenario name under incidents/scenarios/
    symbols: list[str] = ["BTC", "ETH"]
    dataset_suffix: str = "mar_jul"  # <sym>_usdt_spot_<suffix> dataset dirs
    timeout_max_s: float = 30.0      # clamp on http_fetch timeout_s
    free_bps: float = 5.0            # cross-venue spread noise is not an error
    cap_bps: float = 250.0           # a wildly wrong report caps here
    abstain_budget_frac: float = 0.10  # allowed abstentions / scored hours
    datasets_dir: str | None = None  # test overrides; default repo data
    scenarios_dir: str | None = None
    limiters_path: str | None = None

    @model_validator(mode="after")
    def _symbols(self) -> "CryptoPriceConsistencyConfig":
        if not self.symbols:
            raise ValueError("symbols must be non-empty")
        if len(set(self.symbols)) != len(self.symbols):
            raise ValueError("symbols must be unique")
        if not 0 < self.free_bps < self.cap_bps:
            raise ValueError("need 0 < free_bps < cap_bps")
        if not 0 <= self.abstain_budget_frac <= 1:
            raise ValueError("abstain_budget_frac must be in [0, 1]")
        return self

    def resolve(self, attr: str, default: Path) -> Path:
        value = getattr(self, attr)
        return Path(value) if value else default


class TruthStore:
    """Per-symbol cross-venue median close, indexed by candle close time.
    Built from the normalized candle CSVs, which the incident stream never
    touches."""

    def __init__(self, datasets_dir: Path, symbols: list[str], suffix: str):
        self._series: dict[str, tuple[list[float], list[float]]] = {}
        for symbol in symbols:
            cohort = f"{symbol.lower()}_usdt_spot"
            root = (datasets_dir / f"{cohort}_{suffix}" / "normalized"
                    / f"cohort_{cohort}")
            venue_closes: list[dict[float, float]] = []
            for candles in sorted(root.glob(
                    "source_*/interval_5m/candles.csv")):
                closes: dict[float, float] = {}
                with candles.open() as handle:
                    for row in csv.DictReader(handle):
                        close_ts = (int(row["open_time_ms"]) / 1000
                                    + CANDLE_SECONDS)
                        closes[close_ts] = float(row["close"])
                venue_closes.append(closes)
            if not venue_closes:
                raise ValueError(f"no candle files under {root}")
            keys = sorted(set().union(*venue_closes))
            medians = [statistics.median(
                [v[ts] for v in venue_closes if ts in v]) for ts in keys]
            self._series[symbol] = (keys, medians)

    def truth_at(self, symbol: str, t: datetime) -> float | None:
        """Median close of the last candle completed at or before t."""
        keys, medians = self._series[symbol]
        index = bisect.bisect_right(keys, t.timestamp()) - 1
        return medians[index] if index >= 0 else None


@dataclass
class _Submission:
    at: datetime
    kind: str            # "report" | "abstain"
    price: float | None
    bps_error: float | None  # measured at submission time


class Scorer:
    """Hourly settlement per symbol; last submission in the hour wins."""

    def __init__(self, tcfg: CryptoPriceConsistencyConfig, truth: TruthStore,
                 sim_start: datetime, sim_end: datetime):
        self.tcfg = tcfg
        self.truth = truth
        self.sim_end = sim_end
        self._next_hour = sim_start
        self._latest: dict[tuple[str, float], _Submission] = {}
        self._settled: list[dict] = []

    def record(self, sim_time: datetime, symbol: str, kind: str,
               price: float | None) -> dict:
        if symbol not in self.tcfg.symbols:
            raise NotificationError(
                f"unknown symbol {symbol!r} (configured: {self.tcfg.symbols})")
        bps: float | None = None
        if kind == "report":
            if not isinstance(price, (int, float)) or not math.isfinite(price) \
                    or price <= 0:
                raise NotificationError(
                    f"price must be a finite positive number, got {price!r}")
            truth = self.truth.truth_at(symbol, sim_time)
            if truth is None:
                raise NotificationError("no scoring data at this time")
            bps = abs(price / truth - 1) * 10_000
        hour = sim_time.replace(minute=0, second=0, microsecond=0)
        self._latest[(symbol, hour.timestamp())] = _Submission(
            sim_time, kind, float(price) if price is not None else None, bps)
        return {"hour": iso(hour)}

    def _settle_hour(self, hour: datetime) -> list[OutcomeEvent]:
        tcfg = self.tcfg
        events = []
        for symbol in tcfg.symbols:
            submission = self._latest.pop((symbol, hour.timestamp()), None)
            excess: float | None = None
            if submission is None:
                status = "miss"
            elif submission.kind == "abstain":
                status = "abstain"
            else:
                excess = min(tcfg.cap_bps,
                             max(0.0, submission.bps_error - tcfg.free_bps))
                status = "ok" if excess == 0 else "priced"
            record = {
                "hour": iso(hour), "symbol": symbol, "status": status,
                "t_settled": hour + HOUR,
                "truth": self.truth.truth_at(symbol, hour + HOUR),
            }
            detail = {}
            if submission is not None and submission.kind == "report":
                record["reported"] = submission.price
                record["bps_error"] = round(submission.bps_error, 4)
                record["excess_bps"] = round(excess, 4)
                detail["excess_bps"] = round(excess, 4)
            self._settled.append(record)
            events.append(OutcomeEvent(
                ref=f"{symbol}@{iso(hour)}", status=status, detail=detail))
        return events

    def close_due(self, now: datetime) -> list[OutcomeEvent]:
        events = []
        while self._next_hour + HOUR <= min(now, self.sim_end):
            events.extend(self._settle_hour(self._next_hour))
            self._next_hour += HOUR
        return events

    def close_all(self) -> list[OutcomeEvent]:
        return self.close_due(self.sim_end)

    def oracle_outcomes(self, since: datetime | None,
                        now: datetime) -> list[dict]:
        out = []
        for record in self._settled:
            settled = record["t_settled"]
            if settled <= now and (since is None or settled > since):
                out.append({**record, "t_settled": iso(settled)})
        return out

    def metrics(self) -> dict:
        by_status: dict[str, int] = {}
        excess = []
        for record in self._settled:
            by_status[record["status"]] = by_status.get(record["status"], 0) + 1
            if "excess_bps" in record:
                excess.append(record["excess_bps"])
        scored = len(self._settled)
        reported = by_status.get("ok", 0) + by_status.get("priced", 0)
        abstains = by_status.get("abstain", 0)
        return {
            "primary": {"name": "mean_excess_bps",
                        "value": (round(sum(excess) / len(excess), 4)
                                  if excess else None),
                        "direction": "min"},
            "excess_bps_p90": (round(statistics.quantiles(
                excess, n=10)[-1], 4) if len(excess) >= 10 else None),
            "hours_scored": scored,
            "availability": round(reported / scored, 4) if scored else None,
            "abstain_rate": round(abstains / scored, 4) if scored else None,
            "abstain_budget_frac": self.tcfg.abstain_budget_frac,
            "misses": by_status.get("miss", 0),
            "by_status": by_status,
        }

    def constraint_violations(self) -> list[str]:
        scored = len(self._settled)
        if not scored:
            return []
        out = []
        abstains = sum(1 for r in self._settled if r["status"] == "abstain")
        if abstains / scored > self.tcfg.abstain_budget_frac:
            out.append("abstain_budget_exceeded")
        if any(r["status"] == "miss" for r in self._settled):
            out.append("missed_hours")
        return out

    def report(self) -> dict:
        by_status: dict[str, int] = {}
        bps = []
        for record in self._settled:
            by_status[record["status"]] = by_status.get(record["status"], 0) + 1
            if "bps_error" in record:
                bps.append(record["bps_error"])
        return {
            "by_status": by_status,
            "bps_error_median": (round(statistics.median(bps), 4)
                                 if bps else None),
        }


class CryptoPriceConsistencyTask(Task):
    name = "crypto_price_consistency"

    def __init__(self, tcfg: CryptoPriceConsistencyConfig, scorer: Scorer,
                 proxy, trace, sim_start: datetime):
        self.tcfg = tcfg
        self.scorer = scorer
        self.proxy = proxy      # incidents.proxy.ReplayProxy
        self.trace = trace
        self.sim_start = sim_start

    @classmethod
    def from_run_config(cls, cfg, repo_root: Path) -> "CryptoPriceConsistencyTask":
        from tasks.crypto_price_consistency.incidents.proxy import (
            CandleStore, LimiterBank, ReplayProxy, Trace)

        tcfg = CryptoPriceConsistencyConfig(**cfg.task_params)
        if cfg.sim_start < DATA_WINDOW[0] or cfg.sim_end > DATA_WINDOW[1]:
            raise ValueError(
                f"sim window must lie within the recorded window "
                f"[{iso(DATA_WINDOW[0])}, {iso(DATA_WINDOW[1])}]")
        if cfg.sim_start.minute or cfg.sim_start.second:
            raise ValueError("sim_start must be on an hour boundary")

        datasets_dir = tcfg.resolve(
            "datasets_dir", TASK_DIR / "data" / "datasets")
        scenarios_dir = tcfg.resolve(
            "scenarios_dir", TASK_DIR / "incidents" / "scenarios")
        limiters = tcfg.resolve(
            "limiters_path", TASK_DIR / "incidents" / "limiters.yaml")
        trace_path = scenarios_dir / f"{tcfg.trace}.jsonl"
        if not trace_path.is_file():
            raise ValueError(f"unknown trace {tcfg.trace!r} ({trace_path})")

        truth = TruthStore(datasets_dir, tcfg.symbols, tcfg.dataset_suffix)
        scorer = Scorer(tcfg, truth, cfg.sim_start, cfg.sim_end)
        store = CandleStore(sorted(
            p for p in datasets_dir.iterdir() if (p / "raw").is_dir()))
        proxy = ReplayProxy(store, Trace(trace_path), LimiterBank(limiters))
        return cls(tcfg, scorer, proxy, proxy.trace, cfg.sim_start)

    # -- environment API -------------------------------------------------------------

    def env_apps(self, sim) -> list:
        from tasks.crypto_price_consistency.env.apps import ExchangeApp

        return [ExchangeApp(sim, self)]

    def record_notification(self, sim_time: datetime, payload: dict) -> None:
        kind = payload.get("kind")
        if kind not in ("report", "abstain"):
            raise NotificationError(
                f"payload needs kind 'report'|'abstain', got: {payload!r}")
        self.scorer.record(sim_time, payload.get("symbol"), kind,
                           payload.get("price"))

    # -- authored wait programs (TM-B) ---------------------------------------------------

    def authored_example(self) -> str | None:
        return (Path(__file__).resolve().parent / "agent" /
                "example_gatekeeper.py").read_text(encoding="utf-8")

    # -- scoring ---------------------------------------------------------------------

    def close_due(self, now: datetime) -> list[OutcomeEvent]:
        return self.scorer.close_due(now)

    def close_all(self) -> list[OutcomeEvent]:
        return self.scorer.close_all()

    def oracle_outcomes(self, since: datetime | None,
                        now: datetime) -> list[dict]:
        return self.scorer.oracle_outcomes(since, now)

    def metrics(self) -> dict:
        return self.scorer.metrics()

    def constraint_violations(self) -> list[str]:
        return self.scorer.constraint_violations()

    def report(self) -> dict:
        return {"trace": self.tcfg.trace, "symbols": self.tcfg.symbols,
                **self.scorer.report()}

    # -- instruction -----------------------------------------------------------------

    def instruction_context(self) -> dict[str, object]:
        rows = []
        for symbol in self.tcfg.symbols:
            spellings = ", ".join(
                f"{venue}: `{pattern.format(sym=symbol)}`"
                for venue, pattern in VENUE_SYMBOLS.items())
            rows.append(f"| {symbol} | {spellings} |")
        return {
            "symbols": ", ".join(self.tcfg.symbols),
            "symbols_table": "\n".join(rows),
            "n_symbols": len(self.tcfg.symbols),
            "free_bps": f"{self.tcfg.free_bps:g}",
            "cap_bps": f"{self.tcfg.cap_bps:g}",
            "abstain_budget_pct":
                f"{self.tcfg.abstain_budget_frac * 100:g}",
            "timeout_max_s": self.tcfg.timeout_max_s,
        }


TASK = CryptoPriceConsistencyTask
