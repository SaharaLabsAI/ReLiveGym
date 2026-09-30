"""Weather high-temperature alerting task (the study's reference task).

Everything task-specific for `weather_fixture` lives here: the config model, the
data store with the visibility rule, the paid /weather endpoint, and the
day-based scorer. See tasks/weather_fixture/README.md for the experimenter view and
tasks/weather_fixture/INSTRUCTION.md for the agent-facing spec.

Availability rule: hour H's reading is the measurement of the H..H+1 interval
and becomes visible at sim time H+1 — so at time `now` the last visible hour
is floor_hour(now - 1h). Queries are clipped to [data_cutoff, last_visible].

Scoring (metrics, not dollars), for each
UTC day D in [sim_start, sim_end):

- crossing day (first hour h* with temp >= threshold):
    * notifications with t <  h* + 1h are *premature* (the data proving the
      crossing was not yet visible): they count as a false alarm; the day
      can still be properly notified later.
    * the first notification with t >= h* + 1h scores timeliness credit
      max(0, 1 - d/24) with d = (t - h*) in hours; later ones are duplicates.
    * no valid notification by close time -> miss (credit 0).
- non-crossing day: any notification is a false alarm; extras are duplicates.

Metrics: TC-recall = mean credit over crossing days; precision = properly
notified crossing days / (those + false-alarm days); primary = TC-F1
(their harmonic mean). The API is free at real rates (open-meteo) behind a
documented rate limit; the run's binding constraint is budget_usd.

A day *closes* `grace_hours` after it ends; oracle outcomes (mounted only in
`cell.sig: oracle` runs) reveal closed days only.
Notifications are validated at record time: unknown/out-of-window days, future
days, and already-closed days are rejected free of charge.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, field_validator, model_validator

from harness.task import NotificationError, OutcomeEvent, Task
from harness.timeutil import HOUR, as_utc, day_of, day_start, floor_hour, iso, parse_iso

if TYPE_CHECKING:
    from fastapi import FastAPI

    from harness.config import RunConfig
    from harness.runtime import Sim

DATA_DIR = Path(__file__).resolve().parent / "data"

LOCATION_FILES = {
    "LA": "open-meteo-34.06N118.24W91m.csv",
    "Berlin": "open-meteo-52.55N13.41E38m.csv",
}


# -- config ---------------------------------------------------------------------------


class WeatherConfig(BaseModel):
    location: str = "LA"
    weather_csv: Path | None = None  # explicit override; else DATA_DIR/LOCATION_FILES[location]
    data_cutoff: datetime  # earliest visible weather timestamp
    threshold_c: float
    grace_hours: int = 24  # a day closes for scoring this long after it ends
    credit_hours: float = 24.0  # timeliness credit = max(0, 1 - delay/this)
    # Documented free-tier quota of the real API (open-meteo: 10,000
    # calls/day, basis: documented) — advertised, enforced, never metered
    # for the agent.
    rate_limit: dict = {"window": "fixed_window", "window_seconds": 86400,
                        "budget": 10000}

    @field_validator("data_cutoff")
    @classmethod
    def _utc(cls, v: datetime) -> datetime:
        return as_utc(v)

    def resolve_weather_csv(self, base_dir: Path | None = None) -> Path:
        path = self.weather_csv
        if path is None:
            if self.location not in LOCATION_FILES:
                raise ValueError(f"unknown location {self.location!r} and no weather_csv given")
            return DATA_DIR / LOCATION_FILES[self.location]
        if not path.is_absolute() and base_dir is not None:
            path = base_dir / path
        return path


# -- data store -----------------------------------------------------------------------


class WeatherStore:
    def __init__(self, csv_path: str | Path, data_cutoff: datetime):
        self.data_cutoff = data_cutoff
        self._t0, self._temps = self._load(Path(csv_path))
        self._t_last = self._t0 + (len(self._temps) - 1) * HOUR

    @staticmethod
    def _load(path: Path) -> tuple[datetime, list[float]]:
        times: list[datetime] = []
        temps: list[float] = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            in_data = False
            for row in reader:
                if not in_data:
                    if row and row[0] == "time":  # open-meteo data header row
                        in_data = True
                    continue
                if len(row) < 2 or not row[0]:
                    continue
                times.append(parse_iso(row[0]))
                temps.append(float(row[1]))
        if not times:
            raise ValueError(f"no data rows found in {path}")
        # The hourly grid must be contiguous: indexing below is pure arithmetic.
        for i in range(1, len(times)):
            if times[i] - times[i - 1] != HOUR:
                raise ValueError(f"non-hourly gap in {path} at row {i}: "
                                 f"{times[i - 1]} -> {times[i]}")
        return times[0], temps

    # -- index arithmetic --------------------------------------------------------

    def _index(self, t: datetime) -> int:
        return int((t - self._t0) / HOUR)

    def _temp_at(self, hour: datetime) -> float:
        return self._temps[self._index(hour)]

    @staticmethod
    def last_visible(now: datetime) -> datetime:
        return floor_hour(now - HOUR)

    # -- agent-facing query (clipped) ---------------------------------------------

    def query(self, start: datetime, end: datetime, now: datetime) -> dict:
        """Hourly temps in [start, end], clipped to visibility and cutoff.

        Returns the open-meteo-like shape; `clipped_start` /
        `clipped_end` report the effective bounds so the agent can tell when it
        asked for data that does not exist yet (or is before the cutoff).
        """
        lo = floor_hour(max(start, self.data_cutoff, self._t0))
        hi = min(floor_hour(end), self.last_visible(now), self._t_last)
        out_times: list[str] = []
        out_temps: list[float] = []
        if lo <= hi:
            i, j = self._index(lo), self._index(hi)
            out_times = [iso(lo + k * HOUR) for k in range(j - i + 1)]
            out_temps = self._temps[i : j + 1]
        return {
            "hourly": {"time": out_times, "temperature_2m": out_temps},
            "clipped_start": iso(lo) if lo <= hi else None,
            "clipped_end": iso(hi) if lo <= hi else None,
        }

    # -- scorer-facing ground truth (full data, no clipping) -----------------------

    def first_crossing(self, day: date, threshold: float) -> datetime | None:
        """First hour of UTC day `day` with temp >= threshold, or None."""
        start = day_start(day)
        for k in range(24):
            hour = start + k * HOUR
            if self._t0 <= hour <= self._t_last and self._temp_at(hour) >= threshold:
                return hour
        return None

    def coverage(self) -> tuple[datetime, datetime]:
        return self._t0, self._t_last


# -- scorer ---------------------------------------------------------------------------


@dataclass
class DayOutcome:
    day: date
    crossing_hour: datetime | None
    notified_at: datetime | None = None
    delay_hours: float | None = None
    credit: float = 0.0  # timeliness credit, ok days only
    false_alarms: int = 0
    duplicates: int = 0
    status: str = "quiet"  # ok | miss | false_alarm | quiet

    def to_dict(self) -> dict:
        return {
            "date": self.day.isoformat(),
            "crossing_hour": iso(self.crossing_hour) if self.crossing_hour else None,
            "notified_at": iso(self.notified_at) if self.notified_at else None,
            "delay_hours": self.delay_hours,
            "credit": round(self.credit, 6),
            "false_alarms": self.false_alarms,
            "duplicates": self.duplicates,
            "status": self.status,
        }


@dataclass
class _DayState:
    notifications: list[datetime] = field(default_factory=list)


class Scorer:
    def __init__(self, tcfg: WeatherConfig, sim_start: datetime, sim_end: datetime,
                 weather: WeatherStore):
        self._tcfg = tcfg
        self._weather = weather
        self._first_day = day_of(sim_start)
        self._last_day = day_of(sim_end - timedelta(microseconds=1))
        self._open: dict[date, _DayState] = {}
        self._closed: dict[date, DayOutcome] = {}

    # -- recording ----------------------------------------------------------------

    def record_notification(self, sim_time: datetime, day: date) -> None:
        if not (self._first_day <= day <= self._last_day):
            raise NotificationError(f"day {day} is outside the run window")
        if day > day_of(sim_time):
            raise NotificationError(f"cannot notify a future day {day}")
        if day in self._closed:
            raise NotificationError(f"day {day} is already closed for scoring")
        self._open.setdefault(day, _DayState()).notifications.append(sim_time)

    # -- closing ------------------------------------------------------------------

    def _close_time(self, day: date) -> datetime:
        return day_start(day) + timedelta(days=1, hours=self._tcfg.grace_hours)

    def close_days(self, now: datetime) -> list[DayOutcome]:
        """Close every not-yet-closed day whose close time has passed."""
        out = []
        d = self._first_day
        while d <= self._last_day and self._close_time(d) <= now:
            if d not in self._closed:
                out.append(self._score_day(d))
            d += timedelta(days=1)
        return out

    def close_all(self) -> list[DayOutcome]:
        """End of run: close everything remaining (grace effectively expires)."""
        out = []
        d = self._first_day
        while d <= self._last_day:
            if d not in self._closed:
                out.append(self._score_day(d))
            d += timedelta(days=1)
        return out

    def _score_day(self, day: date) -> DayOutcome:
        h_star = self._weather.first_crossing(day, self._tcfg.threshold_c)
        o = DayOutcome(day=day, crossing_hour=h_star)
        notifications = sorted(self._open.pop(day, _DayState()).notifications)

        if h_star is None:
            if notifications:
                o.status = "false_alarm"
                o.false_alarms = 1
                o.duplicates = len(notifications) - 1
        else:
            visible_from = h_star + HOUR
            premature = [t for t in notifications if t < visible_from]
            valid = [t for t in notifications if t >= visible_from]
            if premature:
                o.false_alarms = 1
                o.duplicates += len(premature) - 1
            if valid:
                o.notified_at = valid[0]
                o.delay_hours = (valid[0] - h_star) / HOUR
                o.credit = max(0.0, 1 - o.delay_hours
                               / self._tcfg.credit_hours)
                o.duplicates += len(valid) - 1
                o.status = "ok"
            else:
                o.status = "miss"

        self._closed[day] = o
        return o

    # -- reporting ----------------------------------------------------------------

    def feedback(self) -> list[dict]:
        """Outcomes for all closed days, oldest first (the /feedback payload)."""
        return [self._closed[d].to_dict() for d in sorted(self._closed)]

    def metrics(self) -> dict:
        closed = list(self._closed.values())
        crossing = [o for o in closed if o.crossing_hour]
        ok = [o for o in crossing if o.status == "ok"]
        fa_days = sum(1 for o in closed if o.false_alarms)
        tc_recall = (sum(o.credit for o in ok) / len(crossing)
                     if crossing else None)
        precision = (len(ok) / (len(ok) + fa_days)
                     if (ok or fa_days) else None)
        if precision is None or tc_recall is None:
            tc_f1 = None
        elif precision + tc_recall == 0:
            tc_f1 = 0.0
        else:
            tc_f1 = 2 * precision * tc_recall / (precision + tc_recall)
        delays = sorted(o.delay_hours for o in ok)
        return {
            "primary": {"name": "tc_f1",
                        "value": round(tc_f1, 4) if tc_f1 is not None else None,
                        "direction": "max"},
            "tc_recall": round(tc_recall, 4) if tc_recall is not None else None,
            "precision": round(precision, 4) if precision is not None else None,
            "recall": (round(len(ok) / len(crossing), 4)
                       if crossing else None),
            "crossing_days": len(crossing),
            "days_closed": len(closed),
            "false_alarm_days": fa_days,
            "duplicates": sum(o.duplicates for o in closed),
            "median_delay_hours": (delays[len(delays) // 2]
                                   if delays else None),
        }

    def monthly_buckets(self) -> dict[str, dict]:
        """Outcome counts bucketed by calendar month (adaptation metrics)."""
        buckets: dict[str, dict] = {}
        for d in sorted(self._closed):
            o = self._closed[d]
            key = f"{d.year:04d}-{d.month:02d}"
            b = buckets.setdefault(key, {
                "days": 0, "crossing_days": 0, "ok": 0, "miss": 0,
                "false_alarm_days": 0, "credit": 0.0,
            })
            b["days"] += 1
            b["crossing_days"] += 1 if o.crossing_hour else 0
            b["ok"] += 1 if o.status == "ok" else 0
            b["miss"] += 1 if o.status == "miss" else 0
            b["false_alarm_days"] += 1 if o.false_alarms else 0
            b["credit"] = round(b["credit"] + o.credit, 6)
        return buckets


# -- task -----------------------------------------------------------------------------


class WeatherTask(Task):
    name = "weather_fixture"

    def __init__(self, tcfg: WeatherConfig, store: WeatherStore, scorer: Scorer):
        self.tcfg = tcfg
        self.store = store
        self.scorer = scorer

    @classmethod
    def from_run_config(cls, cfg: RunConfig, repo_root: Path) -> "WeatherTask":
        tcfg = WeatherConfig(**cfg.task_params)
        if tcfg.data_cutoff > cfg.sim_start:
            raise ValueError("data_cutoff must not be after sim_start")
        store = WeatherStore(tcfg.resolve_weather_csv(repo_root), tcfg.data_cutoff)
        return cls(tcfg, store, Scorer(tcfg, cfg.sim_start, cfg.sim_end, store))

    # -- environment API -------------------------------------------------------------

    def env_apps(self, sim: Sim) -> list:
        from tasks.weather_fixture.env.apps import WeatherApp

        return [WeatherApp(sim, self)]

    def record_notification(self, sim_time: datetime, payload: dict) -> None:
        try:
            day = date.fromisoformat(str(payload.get("date")))
        except (TypeError, ValueError):
            raise NotificationError(
                f"payload needs a valid 'date' (YYYY-MM-DD), got: {payload!r}")
        self.scorer.record_notification(sim_time, day)

    # -- scoring ---------------------------------------------------------------------

    @staticmethod
    def _events(outcomes: list[DayOutcome]) -> list[OutcomeEvent]:
        return [OutcomeEvent(o.day.isoformat(), o.status,
                             {"credit": round(o.credit, 6)})
                for o in outcomes]

    def close_due(self, now: datetime) -> list[OutcomeEvent]:
        return self._events(self.scorer.close_days(now))

    def close_all(self) -> list[OutcomeEvent]:
        return self._events(self.scorer.close_all())

    def oracle_outcomes(self, since: datetime | None, now: datetime) -> list[dict]:
        out = []
        for d in sorted(self.scorer._closed):
            t_settled = self.scorer._close_time(d)
            if (since is None or t_settled > since) and t_settled <= now:
                out.append({"kind": "day", "t_settled": iso(t_settled),
                            **self.scorer._closed[d].to_dict()})
        return out

    def metrics(self) -> dict:
        return self.scorer.metrics()

    def report(self) -> dict:
        return {
            "days": self.scorer.feedback(),
            "monthly_outcomes": self.scorer.monthly_buckets(),
        }

    # -- instruction -----------------------------------------------------------------

    def instruction_context(self) -> dict[str, object]:
        from harness.limits import RateLimiter

        return {
            "threshold_c": f"{self.tcfg.threshold_c:g}",
            "grace_hours": self.tcfg.grace_hours,
            "credit_hours": f"{self.tcfg.credit_hours:g}",
            "weather_rate_limit": RateLimiter("weather",
                                              self.tcfg.rate_limit).doc(),
        }


TASK = WeatherTask
