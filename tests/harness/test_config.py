from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from harness.config import load_config
from tasks.weather_fixture.task import DATA_DIR, WeatherConfig
from tests.conftest import make_config, make_weather_task

UTC = timezone.utc


def test_weather_config_defaults():
    tcfg = WeatherConfig(**make_config().task_params)
    assert tcfg.credit_hours == 24.0
    assert tcfg.rate_limit["budget"] == 10000


def test_naive_datetimes_become_utc():
    cfg = make_config(sim_start=datetime(2021, 6, 1), sim_end=datetime(2021, 6, 8),
                      data_cutoff=datetime(2021, 6, 1))
    assert cfg.sim_start.tzinfo is not None
    assert cfg.sim_start == datetime(2021, 6, 1, tzinfo=UTC)
    tcfg = WeatherConfig(**cfg.task_params)
    assert tcfg.data_cutoff == datetime(2021, 6, 1, tzinfo=UTC)


def test_window_validation():
    with pytest.raises(ValidationError):
        make_config(sim_end=datetime(2021, 6, 1, tzinfo=UTC))  # end == start
    with pytest.raises(ValueError):
        # cutoff after start: rejected when the task is built from the config
        make_weather_task(make_config(data_cutoff=datetime(2022, 1, 1, tzinfo=UTC)))


def test_task_section_needs_name():
    with pytest.raises(ValidationError):
        make_config(task={"name": ""})


def test_resolve_weather_csv_from_location():
    tcfg = WeatherConfig(**make_config().task_params)
    assert tcfg.resolve_weather_csv(Path("/repo")) == \
        DATA_DIR / "open-meteo-34.06N118.24W91m.csv"
    tcfg = WeatherConfig(**make_config(weather_csv=Path("/abs/custom.csv")).task_params)
    assert tcfg.resolve_weather_csv(Path("/repo")) == Path("/abs/custom.csv")


def test_load_yaml(tmp_path):
    yaml_text = """
run_id: smoke-1
task:
  name: weather_fixture
  location: LA
  data_cutoff: 2023-08-01T00:00:00Z
  threshold_c: 33.0
sim_start: 2023-08-01T00:00:00Z
sim_end: 2023-08-15T00:00:00Z
budget_usd: 10
agent:
  scaffold: baseline_poller
"""
    path = tmp_path / "smoke.yaml"
    path.write_text(yaml_text)
    cfg = load_config(path)
    assert cfg.run_id == "smoke-1"
    assert cfg.task_name == "weather_fixture"
    assert cfg.budget_usd == 10
    assert cfg.sim_start == datetime(2023, 8, 1, tzinfo=UTC)
    assert cfg.agent.scaffold == "baseline_poller"
    # no cell section -> the default no-learning wiring
    assert cfg.cell.tm == "C"
    assert cfg.cell.tlrn == "none"
    assert cfg.cell.sig == "none"
    assert cfg.cell.alg == "none"
    assert cfg.cell.label == "tmC-tlrnnone-signone-algnone"


def test_cell_spec_validation():
    ok = make_config(cell={"tm": "C", "tlrn": "daily", "sig": "self",
                           "alg": "skills"})
    assert ok.cell.label == "tmC-tlrndaily-sigself-algskills"
    assert ok.cell.tlrn_spec == {"source": "sim", "cron": "30 0 * * *"}
    ok = make_config(cell={"tm": "A", "tlrn": "daily", "sig": "oracle",
                           "alg": "memory"})
    assert ok.cell.tm == "A"
    with pytest.raises(ValidationError):  # signal without an alg
        make_config(cell={"tm": "C", "tlrn": "daily", "sig": "oracle",
                          "alg": "none"})
    with pytest.raises(ValidationError):  # alg without a signal
        make_config(cell={"tm": "C", "tlrn": "daily", "sig": "none",
                          "alg": "memory"})
    with pytest.raises(ValidationError):  # learner without a trigger
        make_config(cell={"tm": "C", "sig": "oracle", "alg": "memory"})
    with pytest.raises(ValidationError):  # trigger without a learner
        make_config(cell={"tm": "C", "tlrn": "daily", "sig": "none",
                          "alg": "none"})
    with pytest.raises(ValidationError):  # unregistered tlrn tag
        make_config(cell={"tm": "C", "tlrn": "hourly", "sig": "oracle",
                          "alg": "memory"})
    with pytest.raises(ValidationError):  # ReACT arms cap at Alg-B
        make_config(cell={"tm": "A", "tlrn": "daily", "sig": "self",
                          "alg": "full"})
    with pytest.raises(ValidationError):
        make_config(cell={"tm": "B", "tlrn": "daily", "sig": "oracle",
                          "alg": "config"})
    # TM-D (agent-managed schedules) is a ReACT arm too: alg caps at skills
    ok = make_config(cell={"tm": "D", "tlrn": "daily", "sig": "oracle",
                           "alg": "skills"})
    assert ok.cell.label == "tmD-tlrndaily-sigoracle-algskills"
    with pytest.raises(ValidationError):
        make_config(cell={"tm": "D", "tlrn": "daily", "sig": "oracle",
                          "alg": "config"})
