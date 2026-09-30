"""Wait party (harness/waitparty.py).

The only wait shape is the timed sleep (there is no condition grammar —
event-driven waiting is authored
gatekeeper code, billed per fetch). Safety nets: (1) party-of-one books
and returns exactly what the solo sleep handler does; (2) barrier edge
cases: trigger routing to the declared trigger_waiter, simultaneous
resolutions, roster shrink completing the barrier, immediate
past-deadline waits, experiment_over broadcast, and release on abort.

Runs on the weather reference task (tests/conftest)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from harness.apps import PartyApp, SleepApp
from harness.env_tools import ToolError
from harness.runtime import Sim
from harness.timeutil import iso
from tests.conftest import make_config, make_weather_task, write_weather_csv

START = datetime(2021, 6, 1, tzinfo=timezone.utc)


def t(day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(2021, 6, day, hour, minute, tzinfo=timezone.utc)


def make_sim(root, name, sim_end) -> Sim:
    csv = write_weather_csv(root / f"w-{name}.csv", START, [20.0] * 24 * 30)
    cfg = make_config(
        run_id=f"party-{name}", weather_csv=csv, data_cutoff=START,
        sim_start=t(2), sim_end=sim_end,
        cell={"tm": "B", "tlrn": "none", "sig": "none", "alg": "none"},
        agent=dict(scaffold="task:party_fixture"))
    task = make_weather_task(cfg)
    run_dir = root / f"run-{name}"
    run_dir.mkdir()
    sim = Sim(cfg, run_dir, root / f"ws-{name}", task)
    sim.clock.advance_to(t(2))
    return sim


async def declare(sim, ids, trigger_waiter):
    await PartyApp(sim).set_party(
        {"waiter_ids": ids, "trigger_waiter": trigger_waiter})


def sl(app, until, waiter=None):
    args = {"until": iso(until)}
    if waiter is not None:
        args["waiter_id"] = waiter
    return app.sleep(args)


async def eventually_parked(sim, n: int) -> None:
    for _ in range(200):
        await asyncio.sleep(0)
        if sim.party is not None and len(sim.party.parked) >= n:
            return
    raise AssertionError(f"never reached {n} parked waiters")


def ledger_view(sim) -> list[dict]:
    """Ledger events for equivalence checks: drop bookkeeping-only fields
    and the party-only events/fields."""
    out = []
    for e in sim.ledger.events:
        if e["type"] == "set_party":
            continue
        out.append({k: v for k, v in e.items()
                    if k not in ("seq", "real_time", "waiter")})
    return out


# -- safety net 1: party of one == solo --------------------------------------------


def test_party_of_one_matches_solo_sleep(tmp_path):
    solo = make_sim(tmp_path, "solo", t(12))
    party = make_sim(tmp_path, "party", t(12))

    async def script(sim, waiter):
        app = SleepApp(sim)
        got = []
        sim.schedule.run_at("learn", t(2, 1))
        args = {"until": iso(t(2, 2))}
        if waiter:
            args["waiter_id"] = waiter
        got.append(await app.sleep(dict(args)))   # -> trigger at 01:00
        got.append(await app.sleep(dict(args)))   # -> sleep timeout 02:00
        got.append(await app.sleep(dict(args)))   # past deadline -> sleep
        return got

    async def party_script():
        await declare(party, ["w"], "w")
        return await script(party, "w")

    solo_got = asyncio.run(script(solo, None))
    party_got = asyncio.run(party_script())
    assert party_got == solo_got
    assert [g["woke_for"] for g in solo_got] == ["trigger", "sleep", "sleep"]
    assert ledger_view(party) == ledger_view(solo)


# -- barrier edge cases -------------------------------------------------------------


def test_trigger_routes_to_trigger_waiter_and_simultaneous_wake(tmp_path):
    sim = make_sim(tmp_path, "route", t(12))
    sim.schedule.run_at("learn", t(2, 1))
    sim.schedule.run_at("tick", t(5))

    async def scenario():
        await declare(sim, ["c", "m"], "c")
        app = SleepApp(sim)
        # m's deadline lands exactly on the `tick` trigger: both resolve
        # at the same instant — trigger to c, plain wake to m. No wake at
        # the midnights in between: the fallback never ends a wait
        m_task = asyncio.create_task(sl(app, t(5), "m"))
        await eventually_parked(sim, 1)
        r = await sl(app, t(12), "c")
        assert (r["woke_for"], r["trigger"]["id"]) == ("trigger", "learn")
        assert r["now"] == iso(t(2, 1))
        c_task = asyncio.create_task(sl(app, t(12), "c"))
        m = await m_task
        c = await c_task
        assert (m["woke_for"], m["now"]) == ("sleep", iso(t(5)))
        assert (c["woke_for"], c["now"]) == ("trigger", iso(t(5)))
        assert c["trigger"]["id"] == "tick"
        # cleanup: c is the only roster member left waiting
        sim.party.set_roster(["c"], "c")

    asyncio.run(scenario())


def test_set_party_validation_and_double_park(tmp_path):
    sim = make_sim(tmp_path, "valid", t(12))

    async def scenario():
        app = PartyApp(sim)
        with pytest.raises(ToolError, match="duplicate"):
            await app.set_party({"waiter_ids": ["a", "a"],
                                 "trigger_waiter": "a"})
        with pytest.raises(ToolError, match="trigger_waiter"):
            await app.set_party({"waiter_ids": ["a"],
                                 "trigger_waiter": "z"})
        await declare(sim, ["a", "b"], "a")
        sapp = SleepApp(sim)
        with pytest.raises(ToolError, match="unknown waiter_id"):
            await sl(sapp, t(3), "nope")
        b_task = asyncio.create_task(sl(sapp, t(2, 12), "b"))
        await eventually_parked(sim, 1)
        with pytest.raises(ToolError, match="already parked"):
            await sl(sapp, t(2, 12), "b")
        # removing a parked waiter is refused; removing an idle one is how
        # a finished agent leaves — and it completes the barrier
        with pytest.raises(ToolError, match="parked"):
            sim.party.set_roster(["a"], "a")
        sim.party.set_roster(["b"], "b")
        b = await b_task
        assert (b["woke_for"], b["now"]) == ("sleep", iso(t(2, 12)))

    asyncio.run(scenario())


def test_experiment_over_broadcast(tmp_path):
    sim = make_sim(tmp_path, "over", t(2, 12))  # ends before midnight

    async def scenario():
        await declare(sim, ["a", "b"], "a")
        app = SleepApp(sim)
        b_task = asyncio.create_task(sl(app, t(3), "b"))
        await eventually_parked(sim, 1)
        a = await sl(app, t(3), "a")
        b = await b_task
        assert a == {"experiment_over": True, "now": iso(t(2, 12))}
        assert b == a

    asyncio.run(scenario())


def test_abort_releases_parked_waiters(tmp_path):
    sim = make_sim(tmp_path, "abort", t(12))

    async def scenario():
        await declare(sim, ["a", "b"], "a")
        app = SleepApp(sim)
        b_task = asyncio.create_task(sl(app, t(8), "b"))
        await eventually_parked(sim, 1)
        # a crashes without ever parking; the supervisor aborts the party
        async with sim.lock:
            sim.party.abort(sim.clock.now)
        b = await b_task
        assert b == {"aborted": True, "now": iso(t(2))}

    asyncio.run(scenario())


def test_immediate_past_deadline_never_parks(tmp_path):
    sim = make_sim(tmp_path, "imm", t(12))

    async def scenario():
        await declare(sim, ["a", "b"], "a")
        app = SleepApp(sim)
        r = await sl(app, t(2), "b")  # until == now
        assert (r["woke_for"], r["now"]) == ("sleep", iso(t(2)))
        assert not sim.party.parked

    asyncio.run(scenario())


def test_roster_shrink_completes_barrier_after_deadline_pass(tmp_path):
    sim = make_sim(tmp_path, "shrink", t(12))

    async def scenario():
        await declare(sim, ["a", "b"], "a")
        app = SleepApp(sim)
        b_task = asyncio.create_task(sl(app, t(2, 6) + timedelta(0), "b"))
        await eventually_parked(sim, 1)
        # a leaves the party from its own turn; the shrink completes the
        # barrier for b alone
        sim.party.set_roster(["b"], "b")
        b = await b_task
        assert (b["woke_for"], b["now"]) == ("sleep", iso(t(2, 6)))

    asyncio.run(scenario())
