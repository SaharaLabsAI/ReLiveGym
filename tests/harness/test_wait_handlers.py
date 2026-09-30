"""Wait-handler trigger delivery: a trigger due exactly `now` must be
consumable through the wait tool (regression lineage: the 2026-07-26 tmB
clock wedge, where a same-instant collision left the trigger due and every
subsequent wait short-circuited without consuming it). The only
wait tool is the timed sleep."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from harness.apps import SleepApp
from harness.runtime import Sim
from harness.timeutil import iso
from tests.conftest import make_config, make_weather_task, write_weather_csv

START = datetime(2021, 6, 1, tzinfo=timezone.utc)


def _sim(tmp_path) -> Sim:
    csv = write_weather_csv(tmp_path / "w.csv", START, [20.0] * 24 * 7)
    cfg = make_config(weather_csv=csv, data_cutoff=START,
                      cell={"tm": "B", "tlrn": "daily", "sig": "oracle",
                            "alg": "memory"})
    task = make_weather_task(cfg)
    (tmp_path / "run").mkdir()
    return Sim(cfg, tmp_path / "run", tmp_path / "ws", task)


def test_sleep_delivers_trigger_due_exactly_now(tmp_path):
    sim = _sim(tmp_path)
    now = sim.clock.now
    sim.schedule.run_at("learn", now)  # the wedge state: due right now
    app = SleepApp(sim)
    until = iso(now + timedelta(days=1))

    first = asyncio.run(app.sleep({"until": until}))
    assert first["woke_for"] == "trigger"
    assert first["trigger"]["id"] == "learn"
    assert first["now"] == iso(now)  # zero-time delivery, clock untouched

    # the trigger is consumed: the next sleep advances instead of wedging
    second = asyncio.run(app.sleep({"until": until}))
    assert second["now"] > iso(now)


def test_sleep_with_past_until_returns_immediately(tmp_path):
    sim = _sim(tmp_path)
    now = sim.clock.now
    app = SleepApp(sim)
    result = asyncio.run(app.sleep({"until": iso(now)}))
    assert result == {"now": iso(now), "woke_for": "sleep"}
