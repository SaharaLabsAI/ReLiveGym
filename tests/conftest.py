from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from harness.config import RunConfig
from tasks.weather_fixture.task import WeatherStore, WeatherTask


@pytest.fixture(autouse=True)
def _provider_keys_present(monkeypatch):
    """harness.serve refuses to start without the launch provider's API key
    (an unforwardable run must fail fast). Tests run the mock LLM upstream
    and never forward a call, so a placeholder satisfies the preflight
    where no .env is configured."""
    import os
    for k in ("OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        if not os.environ.get(k):
            monkeypatch.setenv(k, "test-placeholder")


@pytest.fixture(autouse=True)
def _mock_models_priced(monkeypatch):
    """Every model must be priced or the proxy refuses the call
    (harness/model_costs). Tests run mock models, so the global table is
    replaced with a zero-rate entry for them (mock calls book $0, as the
    e2e suites assume) — the checked-in configs/model_costs.yaml never
    prices a test run. Tests exercising the table itself monkeypatch
    _cache again on top."""
    import harness.model_costs as model_costs

    monkeypatch.setattr(model_costs, "_cache",
                        {"mock-luna": {"in": 0.0, "out": 0.0}})

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "tasks" / "weather_fixture" / "data"
UTC = timezone.utc

collect_ignore: list[str] = []

# Harness tests also run against the weather task: it is the reference task,
# and the fixture agents under fixture_agents/ speak its /weather endpoint.

CSV_HEADER = (
    "latitude,longitude,elevation,utc_offset_seconds,timezone,timezone_abbreviation\n"
    "34.0,-118.0,91.0,0,GMT,GMT\n"
    "\n"
    "time,temperature_2m (°C)\n"
)

# convenience kwargs of make_config that live in the task section
_TASK_KEYS = ("location", "weather_csv", "data_cutoff", "threshold_c", "grace_hours")


def write_weather_csv(path: Path, start: datetime, temps: list[float]) -> Path:
    """Write a synthetic open-meteo-format CSV of consecutive hourly temps."""
    lines = [CSV_HEADER]
    for k, temp in enumerate(temps):
        t = start + timedelta(hours=k)
        lines.append(f"{t.strftime('%Y-%m-%dT%H:%M')},{temp}\n")
    path.write_text("".join(lines), encoding="utf-8")
    return path


def make_config(**overrides) -> RunConfig:
    task = dict(
        name="weather_fixture",
        location="LA",
        data_cutoff=datetime(2021, 6, 1, tzinfo=UTC),
        threshold_c=33.0,
    )
    for key in _TASK_KEYS:
        if key in overrides:
            task[key] = overrides.pop(key)
    task.update(overrides.pop("task", {}))
    base = dict(
        run_id="test-run",
        task=task,
        sim_start=datetime(2021, 6, 1, tzinfo=UTC),
        sim_end=datetime(2021, 6, 8, tzinfo=UTC),
        agent=dict(scaffold="baseline_poller"),
    )
    base.update(overrides)
    return RunConfig(**base)


def make_weather_task(cfg: RunConfig) -> WeatherTask:
    return WeatherTask.from_run_config(cfg, REPO_ROOT)


@pytest.fixture(scope="session")
def la_store() -> WeatherStore:
    csv_path = DATA_DIR / "open-meteo-34.06N118.24W91m.csv"
    if not csv_path.exists():
        pytest.skip("real LA weather CSV not available")
    return WeatherStore(csv_path, data_cutoff=datetime(2016, 1, 1, tzinfo=UTC))
