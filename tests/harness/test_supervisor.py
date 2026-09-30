"""Integration tests driving the full helper (HTTP server + supervisor)
with tiny fixture agents."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from harness.run import run_experiment
from tests.conftest import make_config, write_weather_csv

UTC = timezone.utc
START = datetime(2021, 6, 1, tzinfo=UTC)
FIXTURES = Path(__file__).resolve().parents[1] / "fixture_agents"


def run_fixture(tmp_path, fixture: str, days: int = 3,
                temps: list[float] | None = None, **cfg_overrides):
    csv = write_weather_csv(tmp_path / "w.csv", START,
                            temps if temps is not None else [20.0] * 24 * days)
    cfg = make_config(
        sim_start=START,
        sim_end=START + timedelta(days=days),
        data_cutoff=START,
        weather_csv=csv,
        **cfg_overrides,
    )
    run_dir = tmp_path / "run"
    results = asyncio.run(run_experiment(
        cfg, repo_root=tmp_path, run_dir=run_dir, scaffold_src=FIXTURES / fixture))
    return results, run_dir


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_cron_installer_respawned_on_schedule(tmp_path):
    results, run_dir = run_fixture(tmp_path, "cron_installer", days=3)
    inv = read_jsonl(run_dir / "workspace" / "invocations.jsonl")
    assert [(i["id"], i["kind"]) for i in inv] == [
        ("__bootstrap__", "at"), ("daily", "cron"), ("daily", "cron"), ("daily", "cron"),
    ]
    assert [i["due_time"] for i in inv[1:]] == [
        "2021-06-01T23:00:00Z", "2021-06-02T23:00:00Z", "2021-06-03T23:00:00Z",
    ]
    assert results["flags"] == []


def test_sleeper_woken_early_by_due_trigger(tmp_path):
    results, run_dir = run_fixture(tmp_path, "sleeper", days=3)
    # one single daemon process for the whole run
    inv = read_jsonl(run_dir / "workspace" / "invocations.jsonl")
    assert len(inv) == 1
    wakes = read_jsonl(run_dir / "workspace" / "wakes.jsonl")
    # asked for 48h naps but the daily 06:00 trigger cuts each one short
    assert [w.get("woke_for") for w in wakes[:3]] == ["trigger"] * 3
    assert [w["trigger"]["id"] for w in wakes[:3]] == ["daily"] * 3
    assert [w["now"] for w in wakes[:3]] == [
        "2021-06-01T06:00:00Z", "2021-06-02T06:00:00Z", "2021-06-03T06:00:00Z",
    ]
    assert wakes[-1].get("experiment_over") is True


def test_hanger_watchdog_then_degenerate(tmp_path):
    results, run_dir = run_fixture(
        tmp_path, "hanger", days=2, temps=[35.0] * 48,
        watchdog_seconds=0.2, max_consecutive_crashes=2)
    assert "failed-degenerate" in results["flags"]
    crash_logs = list((run_dir / "workspace" / "logs").glob("crash-*.log"))
    assert len(crash_logs) == 2
    assert "watchdog kill" in crash_logs[0].read_text()
    # the clock still ran to sim_end: both crossing days scored as misses
    statuses = [d["status"] for d in results["task"]["days"]]
    assert statuses == ["miss", "miss"]
    assert results["performance"]["recall"] == 0.0


def test_wiper_kept_alive_by_fallback(tmp_path):
    results, run_dir = run_fixture(tmp_path, "wiper", days=2)
    inv = read_jsonl(run_dir / "workspace" / "invocations.jsonl")
    assert [(i["id"], i["kind"]) for i in inv] == [
        ("__bootstrap__", "at"), ("__fallback_midnight__", "fallback"),
    ]
    assert inv[1]["due_time"] == "2021-06-02T00:00:00Z"


def test_crash_recovery_kind_on_next_trigger(tmp_path):
    results, run_dir = run_fixture(tmp_path, "crash_once", days=2)
    inv = read_jsonl(run_dir / "workspace" / "invocations.jsonl")
    assert inv[0]["kind"] == "at"
    assert inv[1]["kind"] == "crash_recovery"
    assert "failed-degenerate" not in results["flags"]
    assert len(list((run_dir / "workspace" / "logs").glob("crash-*.log"))) == 1


def test_rerunner_run_at_now_relaunches_same_sim_time(tmp_path):
    results, run_dir = run_fixture(tmp_path, "rerunner", days=1)
    inv = read_jsonl(run_dir / "workspace" / "invocations.jsonl")
    assert [(i["id"], i["kind"]) for i in inv] == [
        ("__bootstrap__", "at"), ("again", "at"),
    ]
    assert inv[0]["due_time"] == inv[1]["due_time"] == "2021-06-01T00:00:00Z"


def test_poller_end_to_end_scoring(tmp_path):
    # day 1 crosses 33C at 14:00, day 2 stays cool
    temps = [25.0] * 14 + [35.0] + [25.0] * 9 + [25.0] * 24
    results, run_dir = run_fixture(tmp_path, "poller", days=2, temps=temps)
    days = {d["date"]: d for d in results["task"]["days"]}
    assert days["2021-06-01"]["status"] == "ok"
    assert days["2021-06-01"]["delay_hours"] == 9.0  # 14:00 -> 23:00
    assert days["2021-06-02"]["status"] == "quiet"
    res = results["resources"]
    assert res["counts"]["weather"] == 2  # 2 daily calls, free at real rates
    assert res["spent_usd"] == 0.0
    perf = results["performance"]
    credit = 1 - 9 / 24
    assert perf["tc_recall"] == pytest.approx(credit, abs=1e-4)
    assert perf["precision"] == pytest.approx(1.0)
    assert perf["primary"]["value"] == pytest.approx(
        2 * credit / (1 + credit), abs=1e-4)


def test_results_artifacts_written(tmp_path):
    results, run_dir = run_fixture(tmp_path, "poller", days=1)
    assert (run_dir / "results.json").exists()
    assert (run_dir / "ledger.jsonl").exists()
    assert (run_dir / "config.json").exists()
    assert (run_dir / "agent_output.log").exists()
    # initial code snapshot taken at first spawn
    snaps = list((run_dir / "code_history").glob("*.py"))
    assert len(snaps) == 1
    ledger = read_jsonl(run_dir / "ledger.jsonl")
    assert ledger[0]["seq"] == 1
    types = {e["type"] for e in ledger}
    assert {"trigger", "agent_exit", "outcome"} <= types


def test_algd_rollback_after_crash_streak(tmp_path):
    """Alg-C/D sandbox: a self-edit that crashes K_ROLLBACK
    consecutive times is reverted to the last good commit and the run
    continues (rollback logged, not failed-degenerate). Since the
    server/actor detach the SCHEDULER decides (ledger + streak reset +
    `rollback_to` in the exit response) and the RUNNER executes the git
    reset — the decision is exercised here, the reset in
    scaffolds/runtime/actor.py."""
    import subprocess

    from harness.runtime import Sim
    from harness.supervisor import Scheduler
    from scaffolds.runtime.actor import Runner
    from tests.conftest import make_weather_task

    csv = write_weather_csv(tmp_path / "w.csv", START, [20.0] * 24 * 3)
    cfg = make_config(sim_start=START, sim_end=START + timedelta(days=3),
                      data_cutoff=START, weather_csv=csv,
                      cell={"tm": "C", "tlrn": "daily", "sig": "oracle",
                            "alg": "full"})
    ws = tmp_path / "workspace"
    ws.mkdir()

    def git(*args):
        subprocess.run(["git", "-C", str(ws), *args],
                       capture_output=True, check=True)

    def head():
        return subprocess.run(["git", "-C", str(ws), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()

    (ws / "main.py").write_text("print('ok')\n")
    git("init", "-q")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")
    git("add", "-A")
    git("commit", "-q", "-m", "good")
    good = head()
    sim = Sim(cfg, tmp_path, ws, make_weather_task(cfg))
    sched = Scheduler(sim)
    sim.schedule.run_at("__bootstrap__", cfg.sim_start)

    async def drive():
        # first `next` seeds last-good from the runner's HEAD
        trig = await sched.next(code_sha="aaaaaaaaaaaa", head_sha=good)
        assert trig["id"] == "__bootstrap__"
        assert sched._last_good_sha == good
        # a reflection commits a broken self-edit during the invocation
        (ws / "main.py").write_text("raise RuntimeError('broken edit')\n")
        git("add", "-A")
        git("commit", "-q", "-m", "reflection @ t")
        bad = head()
        # crash 1: below K_ROLLBACK, nothing to roll back yet
        r1 = await sched.report_exit(trig["id"], 1, False, "boom", bad)
        assert r1["crashed"] and "rollback_to" not in r1
        sim.schedule.run_at("again", sim.clock.now)
        trig2 = await sched.next(code_sha="bbbbbbbbbbbb", head_sha=bad)
        assert trig2["kind"] == "crash_recovery"
        # crash 2: the scheduler decides the rollback and resets the streak
        r2 = await sched.report_exit(trig2["id"], 1, False, "boom", bad)
        assert r2["rollback_to"] == good
        assert r2["failed_degenerate"] is False
        assert sched._consecutive_crashes == 0
        return bad

    bad = asyncio.run(drive())
    (event,) = [e for e in sim.ledger.events if e["type"] == "rollback"]
    assert event["to_sha"] == good[:12] and event["from_sha"] == bad[:12]
    types = [e["type"] for e in sim.ledger.events]
    assert types.count("code_change") == 1  # bbbb after aaaa
    # the runner's half: execute what the server decided
    runner = Runner("http://unused", "t", "", ws, tmp_path, quiet=True)
    runner._rollback(good)
    assert (ws / "main.py").read_text() == "print('ok')\n"


def test_tmd_schedule_crud_fires_notes_and_protects_base(tmp_path):
    """TM-D: agent-created
    schedules fire through the normal trigger loop with their note in
    ENV_TRIGGER, a one-time schedule disappears after firing, deleting a
    recurring one cancels its future fires, and the base crontab is
    refused to update/delete (free 400s)."""
    results, run_dir = run_fixture(
        tmp_path, "schedule_crud", days=3,
        cell={"tm": "D", "tlrn": "none", "sig": "none", "alg": "none"})
    inv = read_jsonl(run_dir / "workspace" / "invocations.jsonl")
    assert [(i["id"], i["kind"]) for i in inv] == [
        ("__bootstrap__", "at"), ("check", "at"), ("daily", "cron"),
        ("daily", "cron"), ("daily", "cron")]
    assert inv[1]["owner"] == "agent"
    assert inv[1]["note"] == "look at the forecast again"
    assert inv[1]["due_time"] == "2021-06-01T06:00:00Z"  # the updated `at`
    assert "owner" not in inv[2] and "note" not in inv[2]
    calls = read_jsonl(run_dir / "workspace" / "calls.jsonl")
    create_twice, create_check, update_check, upd_base, del_base, \
        create_learn, listing, del_twice, listing2 = calls
    assert create_twice["schedule"]["type"] == "recurring"
    assert update_check["schedule"]["at"] == "2021-06-01T06:00:00Z"
    assert upd_base["http_error"] == 400 and "read-only" in str(upd_base)
    assert del_base["http_error"] == 400 and "read-only" in str(del_base)
    assert create_learn["http_error"] == 400 and "reserved" in str(create_learn)
    assert [r["id"] for r in listing["schedules"]] == ["check", "twice", "daily"]
    assert listing["schedules"][2]["owner"] == "system"
    assert del_twice == {"status": "ok", "id": "twice"}
    assert [r["id"] for r in listing2["schedules"]] == ["daily"]
    ledger = read_jsonl(run_dir / "ledger.jsonl")
    kinds = [(e["id"], e.get("owner")) for e in ledger if e["type"] == "trigger"]
    assert ("check", "agent") in kinds and ("daily", None) in kinds
    assert [e["type"] for e in ledger if e["type"].startswith("schedule_")] == [
        "schedule_create", "schedule_create", "schedule_update",
        "schedule_delete"]
    assert all(e["cost"] == 0.0 for e in ledger if e["type"].startswith("schedule_"))
    assert results["flags"] == []
