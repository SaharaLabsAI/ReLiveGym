"""Anchor test: the baseline
poller run over the full main window must reproduce the pinned reference
metrics. One run end-to-end-validates clock, schedule, weather availability,
rate limiting, notification semantics, and the metric scoring against the
same data the task was calibrated on.

~2 minutes (1097 real agent process spawns); deselect with -m "not anchor".
"""

import asyncio
from datetime import datetime, timezone

import pytest

from harness.run import run_experiment
from tests.conftest import DATA_DIR, REPO_ROOT, make_config

UTC = timezone.utc


@pytest.mark.anchor
def test_baseline_poller_main_window_anchor(tmp_path):
    if not (DATA_DIR / "open-meteo-34.06N118.24W91m.csv").exists():
        pytest.skip("real LA weather CSV not available")
    cfg = make_config(
        run_id="anchor-baseline",
        sim_start=datetime(2021, 6, 1, tzinfo=UTC),
        sim_end=datetime(2024, 6, 1, tzinfo=UTC),
        data_cutoff=datetime(2021, 6, 1, tzinfo=UTC),
    )
    results = asyncio.run(run_experiment(
        cfg, repo_root=REPO_ROOT, run_dir=tmp_path / "run"))

    res = results["resources"]
    assert res["counts"]["weather"] == 1096  # one call per day
    assert res["spent_usd"] == 0.0  # weather API is free; no LLM in baseline
    assert res["rate_limits"]["weather"]["rejected_429"] == 0

    perf = results["performance"]
    assert perf["crossing_days"] == 111
    assert perf["false_alarm_days"] == 0
    # daily poller: 110/111 crossing days caught (one crossing lands too
    # close to the day boundary for the next-day poll), zero false alarms,
    # ~3 h median delay — the pinned metric anchor for this data + policy
    assert perf["recall"] == pytest.approx(0.991, abs=1e-4)
    assert perf["precision"] == pytest.approx(1.0)
    assert perf["tc_recall"] == pytest.approx(0.7207, abs=1e-4)
    assert perf["median_delay_hours"] == pytest.approx(3.0)
    assert perf["primary"]["name"] == "tc_f1"
    assert perf["primary"]["value"] == pytest.approx(0.8377, abs=1e-4)

    days = results["task"]["days"]
    assert len(days) == 1096
    assert results["flags"] == []
    assert results["constraints"]["within_budget"] is True
