"""Pause / checkpoint / resume (harness/checkpoint.py): the exactness
claim first — a sim paused at a cut and resumed from its checkpoint ends with the same
scoring state, wallet and settled outcomes as a sim that never stopped;
a claim pending across the cut settles in the next stage; the resume
trigger exists for resident disciplines only; the checkpoint's partial
report never lists an open (future) breakpoint; inconsistent chains are
refused. Drives Sim/Scheduler directly on the bnpm mini-world
(tests/tasks/breakout_news_pm/conftest), with the sleep handler moving
the clock exactly as a program's waits would."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from harness import checkpoint
from harness.apps import SleepApp
from harness.config import RunConfig
from harness.runtime import Sim
from harness.supervisor import Scheduler
from harness.task import NotificationError, load_task_class
from harness.timeutil import iso
from tests.tasks.breakout_news_pm.conftest import build_mini_index, t, write_world

UTC = timezone.utc
PAUSE = t(5)  # Mar-5 00:00

# world (conftest): bp1 up @ Mar-3 12:00 (winnable), bp2 down @ Mar-10 00:00
# (unwinnable); indexed articles w-1 (Mar-2 00:07), w-2 (Mar-2 03:02),
# w-3 (Mar-2 03:04) — none gold, so covers are covered_timing.
SCRIPT = [
    (t(2, 6), "bill", None),
    (t(3, 0), "notify", {"market_id": "m1", "news_id": "w-2", "direction": "up"}),
    # pending across the pause (resolves Mar-5 20:00 as a false alarm)
    (t(4, 20), "notify", {"market_id": "m1", "news_id": "w-1", "direction": "down"}),
    # rejected in both worlds: the standing claim is still pending
    (t(5, 6), "notify_rejected", {"market_id": "m1", "news_id": "w-3", "direction": "up"}),
    (t(6, 0), "bill", None),
    (t(9, 6), "notify", {"market_id": "m1", "news_id": "w-3", "direction": "down"}),
]


@pytest.fixture
def world(tmp_path):
    built = write_world(tmp_path)
    idx = build_mini_index(built)
    return built, idx


def make_cfg(world, *, run_id="ck", seed=0, tm="A", **over) -> RunConfig:
    built, idx = world
    base = dict(
        run_id=run_id, seed=seed,
        task={"name": "breakout_news_pm",
              "markets": [{"market_id": "m1", "start": iso(t(2)), "end": iso(t(13))}],
              "data_dir": str(built), "news_index_dir": str(idx),
              "cost": {"price_call": 0.0, "news_search_call": 0.01}},
        sim_start=t(1), sim_end=t(14), budget_usd=5.0,
        domain_budgets={"llm": 2.0},
        cell={"tm": tm, "tlrn": "none", "sig": "none", "alg": "none"},
        agent={"scaffold": "ext:probe", "model": "mock-luna"})
    base.update(over)
    return RunConfig(**base)


def make_sim(cfg: RunConfig, run_dir: Path, *, pause_at=None,
             resume_from: Path | None = None) -> tuple[Sim, Scheduler]:
    ck = None
    if resume_from is not None:
        ck = checkpoint.prepare_resume(run_dir, resume_from, cfg)
    else:
        run_dir.mkdir(parents=True)
    task = load_task_class(cfg.task_name).from_run_config(cfg, Path("/nonexistent"))
    (run_dir / "config.json").write_text(json.dumps(cfg.model_dump(mode="json"), default=str))
    sim = Sim(cfg, run_dir, run_dir / "workspace", task, pause_at=pause_at)
    sched = Scheduler(sim)
    sim.scheduler = sched
    if ck is not None:
        checkpoint.restore(sim, sched, resume_from, ck)
    else:
        sim.schedule.run_at("__bootstrap__", cfg.sim_start)
    return sim, sched


async def drain_triggers(sched: Scheduler, until: datetime | None = None) -> list[dict]:
    """Play the runner: consume triggers (reporting a clean exit for each)
    while they are due before `until`; return them. Without `until`, run
    to the end (done)."""
    out = []
    while True:
        if until is not None:
            trig = sched.sim.schedule.peek_next(sched.sim.clock.now)
            if trig.due_time > until or sched.sim.clock.finished:
                return out
        r = await sched.next()
        if r.get("done"):
            out.append(r)
            return out
        out.append(r)
        await sched.report_exit(r["id"], 0, False, "")


async def advance(sim: Sim, sched: Scheduler, to: datetime) -> dict:
    """Move the clock to `to` the way a resident program would: sleep,
    re-sleep after each trigger wake, stop at experiment_over."""
    await drain_triggers(sched, until=sim.clock.now)  # anything due now
    while sim.clock.now < to:
        r = await SleepApp(sim).sleep({"until": iso(to)})
        if r.get("experiment_over"):
            return r
    return {"now": iso(sim.clock.now)}


async def play(sim: Sim, sched: Scheduler, script) -> str | None:
    """Run the script; returns 'paused' if the sim stopped early."""
    for when, action, payload in script:
        r = await advance(sim, sched, when)
        if r.get("experiment_over"):
            return "paused"
        assert sim.clock.now == when
        if action == "bill":
            async with sim.lock:
                sim.bill("news_search", 0.01, q="probe")
        elif action == "notify":
            async with sim.lock:
                sim.task.record_notification(sim.clock.now, payload)
                sim.ledger.append("notify", sim.clock.now, payload=payload)
        elif action == "notify_rejected":
            with pytest.raises(NotificationError):
                sim.task.record_notification(sim.clock.now, payload)
    return None


def scoring_view(results: dict) -> dict:
    counts = {k: v for k, v in results["resources"]["counts"].items()
              if k in ("notify", "outcome", "news_search")}
    return {"performance": results["performance"], "task": results["task"],
            "spent": results["resources"]["spent_usd"], "counts": counts,
            "llm": results["resources"]["llm_usd"]}


def ledger_rows(run_dir: Path) -> list[dict]:
    return [json.loads(l) for l in (run_dir / "ledger.jsonl").read_text().splitlines() if l.strip()]


# -- the exactness claim --------------------------------------------------------------


def test_pause_resume_equals_continuous(world, tmp_path):
    async def continuous():
        sim, sched = make_sim(make_cfg(world), tmp_path / "cont")
        assert await play(sim, sched, SCRIPT) is None
        r = (await drain_triggers(sched))[-1]
        assert r == {"done": True, "now": iso(t(14))}
        res = sim.results()
        sim.ledger.close()
        return res

    async def chained():
        cfg = make_cfg(world)
        stage1 = tmp_path / "stage1"
        sim, sched = make_sim(cfg, stage1, pause_at=PAUSE)
        assert await play(sim, sched, SCRIPT) == "paused"
        assert sim.clock.now == PAUSE and sim.clock.paused and sim.clock.finished
        r = (await drain_triggers(sched))[-1]
        assert r == {"done": True, "paused": True, "now": iso(PAUSE)}
        assert sched.paused
        ck = checkpoint.write_checkpoint(sim, sched)
        sim.ledger.close()
        # the partial report: nothing settled by force, nothing from the future
        partial = json.loads((stage1 / checkpoint.PARTIAL).read_text())
        assert [b["t_move_start"] for b in partial["task"]["breakpoints"]] == [iso(t(3, 12))]
        assert [a["status"] for a in partial["task"]["alerts"]] == ["covering_timing", "pending"]
        assert partial["performance"]["breakpoints_closed"] == 1
        assert partial["paused_at"] == iso(PAUSE)
        assert not (stage1 / "results.json").exists()
        assert ck["paused_at"] == iso(PAUSE) and ck["ledger_rows"] == len(ledger_rows(stage1))

        # stage 2: a different seed resumes from the main branch's checkpoint
        cfg2 = make_cfg(world, run_id="ck-s1", seed=1)
        stage2 = tmp_path / "stage2"
        sim2, sched2 = make_sim(cfg2, stage2, resume_from=stage1)
        assert sim2.clock.now == PAUSE and not sim2.clock.finished
        rows = ledger_rows(stage2)
        assert rows[-1]["type"] == "resume" and rows[-1]["seed"] == 1
        assert rows[-1]["notifications_replayed"] == 2
        assert [r["seq"] for r in rows] == list(range(1, len(rows) + 1))  # seq continues
        assert (stage2 / checkpoint.RESUME).exists()
        # the pending claim survived the cut: still pending, still blocking
        pending = [a for a in sim2.task.scorer.alerts if a.status == "pending"]
        assert [a.news_id for a in pending] == ["w-1"]
        assert sim2.wallet_spend == pytest.approx(0.01)
        # resident discipline: the first trigger restarts the program AT the cut
        first = await sched2.next()
        assert (first["id"], first["kind"], first["now"]) == (
            checkpoint.RESUME_TRIGGER, "at", iso(PAUSE))
        assert "costs" in first  # the date's brief travels with the resume wake
        await sched2.report_exit(first["id"], 0, False, "")
        rest = [row for row in SCRIPT if row[0] > PAUSE]
        assert await play(sim2, sched2, rest) is None
        r = (await drain_triggers(sched2))[-1]
        assert r == {"done": True, "now": iso(t(14))}
        res = sim2.results()
        sim2.ledger.close()
        assert (stage1 / "ledger.jsonl").read_text()  # the parent is untouched
        assert json.loads((stage1 / checkpoint.CHECKPOINT).read_text()) == ck
        return res, stage2

    cont = asyncio.run(continuous())
    chain, stage2 = asyncio.run(chained())
    assert scoring_view(chain) == scoring_view(cont)
    assert cont["task"]["alerts"][1]["status"] == "false_alarm"  # settled after the cut
    assert cont["performance"]["breakpoints_closed"] == 2
    assert cont["performance"]["primary"]["value"] == pytest.approx(2 * (2 / 3) * 1 / (2 / 3 + 1))
    # the chain's ledger holds the parent's rows verbatim plus the resume row
    rows = ledger_rows(stage2)
    kinds = [r["type"] for r in rows]
    assert kinds.count("resume") == 1 and kinds.count("notify") == 3
    outcomes = {(r["ref"], r["status"]) for r in rows if r["type"] == "outcome"}
    # (a covering alert settles with its breakpoint: no row of its own)
    assert outcomes == {("m1@2026-03-03T12:00:00Z", "covered_timing"),
                        ("m1/w-1", "false_alarm"),
                        ("m1@2026-03-10T00:00:00Z", "covered_timing")}


def test_cron_discipline_gets_no_resume_trigger(world, tmp_path):
    async def go():
        cfg = make_cfg(world, tm="C")
        sim, sched = make_sim(cfg, tmp_path / "s1", pause_at=PAUSE)
        assert await play(sim, sched, SCRIPT) == "paused"
        await drain_triggers(sched)
        checkpoint.write_checkpoint(sim, sched)
        sim.ledger.close()
        sim2, sched2 = make_sim(cfg, tmp_path / "s2", resume_from=tmp_path / "s1")
        first = await sched2.next()
        # nothing at the cut: the next trigger is whatever the restored
        # schedule holds (here the fallback midnight a day later)
        assert first["id"] == "__fallback_midnight__" and first["now"] == iso(t(6))
        resume_row = [r for r in ledger_rows(tmp_path / "s2") if r["type"] == "resume"][0]
        assert resume_row["resume_trigger"] is False
        sim2.ledger.close()

    asyncio.run(go())


def test_schedule_store_and_code_hash_survive(world, tmp_path):
    async def go():
        cfg = make_cfg(world, tm="D")
        sim, sched = make_sim(cfg, tmp_path / "s1", pause_at=PAUSE)
        await sched.next()  # bootstrap
        await sched.report_exit("__bootstrap__", 0, False, "")
        sched._last_code_hash = "abc123def456"
        sim.schedule.set_crontab([{"id": "act", "cron_expr": "0 */6 * * *"}], sim.clock.now)
        sim.schedule.agent_create("m1-check", now=sim.clock.now, at=t(7, 9), note="look")
        await advance(sim, sched, t(9))
        await drain_triggers(sched)
        assert sched.paused
        ck = checkpoint.write_checkpoint(sim, sched)
        assert ck["code_sha"] == "abc123def456"
        sim.ledger.close()
        sim2, sched2 = make_sim(cfg, tmp_path / "s2", resume_from=tmp_path / "s1")
        rows = sim2.schedule.agent_list(sim2.clock.now)
        assert [r["id"] for r in rows] == ["m1-check"] and rows[0]["at"] == iso(t(7, 9))
        assert sim2.schedule.get_crontab() == [{"id": "act", "cron_expr": "0 */6 * * *"}]
        # the cron entry's last fire came along: the first firing is the
        # one due at the cut, not a re-fire of an earlier occurrence
        first = await sched2.next(code_sha="abc123def456")
        assert (first["id"], first["now"]) == ("act", iso(PAUSE))
        await sched2.report_exit("act", 0, False, "")
        assert not [r for r in sim2.ledger.events if r["type"] == "code_change"]
        nxt = await sched2.next(code_sha="ffffffffffff")  # the program changed at the cut
        assert [r for r in sim2.ledger.events if r["type"] == "code_change"][-1]["sha256_12"] == "ffffffffffff"
        assert (nxt["id"], nxt["now"]) == ("act", iso(t(5, 6)))
        sim2.ledger.close()

    asyncio.run(go())


def test_wallet_and_flags_restored(world, tmp_path):
    async def go():
        cfg = make_cfg(world, budget_usd=0.02)
        sim, sched = make_sim(cfg, tmp_path / "s1", pause_at=PAUSE)
        await advance(sim, sched, t(2))
        async with sim.lock:
            sim.bill("news_search", 0.01)
            sim.bill("news_search", 0.01)
            with pytest.raises(Exception):
                sim.bill("news_search", 0.01)  # the wallet is spent
        assert "budget_exhausted" in sim.flags
        await advance(sim, sched, t(9))
        await drain_triggers(sched)
        checkpoint.write_checkpoint(sim, sched)
        sim.ledger.close()
        sim2, _ = make_sim(cfg, tmp_path / "s2", resume_from=tmp_path / "s1")
        assert sim2.wallet_spend == pytest.approx(0.02)
        assert sim2.domain_spend == {"news_search": pytest.approx(0.02)}
        assert sim2.flags == ["budget_exhausted"]
        async with sim2.lock:
            with pytest.raises(Exception):
                sim2.bill("news_search", 0.01)
        sim2.ledger.close()

    asyncio.run(go())


def test_refusals(world, tmp_path):
    async def stage1(name, **over):
        cfg = make_cfg(world, **over)
        sim, sched = make_sim(cfg, tmp_path / name, pause_at=PAUSE)
        assert await play(sim, sched, SCRIPT) == "paused"
        await drain_triggers(sched)
        checkpoint.write_checkpoint(sim, sched)
        sim.ledger.close()
        return cfg

    cfg = asyncio.run(stage1("a"))
    # a config that differs on the wallet is not the same run
    with pytest.raises(ValueError, match="budget_usd"):
        checkpoint.prepare_resume(tmp_path / "b", tmp_path / "a", make_cfg(world, budget_usd=9.0))
    # a failed-degenerate parent is final
    ck = json.loads((tmp_path / "a" / checkpoint.CHECKPOINT).read_text())
    ck["flags"] = ["failed-degenerate"]
    (tmp_path / "a" / checkpoint.CHECKPOINT).write_text(json.dumps(ck))
    with pytest.raises(ValueError, match="failed-degenerate"):
        checkpoint.prepare_resume(tmp_path / "c", tmp_path / "a", cfg)
    ck["flags"] = []
    (tmp_path / "a" / checkpoint.CHECKPOINT).write_text(json.dumps(ck))
    # a ledger that contradicts the task (a second claim while one is
    # pending) cannot be replayed
    rows = ledger_rows(tmp_path / "a")
    fake = {**rows[-1], "seq": rows[-1]["seq"] + 1, "sim_time": iso(t(4, 22)),
            "type": "notify", "cost": 0.0,
            "payload": {"market_id": "m1", "news_id": "w-3", "direction": "up"}}
    with open(tmp_path / "a" / "ledger.jsonl", "a") as f:
        f.write(json.dumps(fake) + "\n")
    ck["ledger_rows"] += 1
    (tmp_path / "a" / checkpoint.CHECKPOINT).write_text(json.dumps(ck))
    with pytest.raises(ValueError, match="contradicts"):
        make_sim(cfg, tmp_path / "d", resume_from=tmp_path / "a")
    # not a checkpoint at all
    (tmp_path / "e").mkdir()
    with pytest.raises(FileNotFoundError):
        checkpoint.prepare_resume(tmp_path / "f", tmp_path / "e", cfg)
    # a pause outside the window is refused up front
    with pytest.raises(ValueError, match="strictly inside"):
        checkpoint.parse_pause_at(cfg, iso(t(14)))
    assert checkpoint.parse_pause_at(cfg, None) is None
    assert checkpoint.parse_pause_at(cfg, iso(t(5))) == t(5)
