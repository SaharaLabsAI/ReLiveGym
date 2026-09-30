"""Daily cost & budget brief: one brief
per simulated UTC date, carried by the first wake payload on that date
(and by every payload at that same instant); no catch-up for skipped
dates. Covers Sim.daily_brief semantics, the four wake-payload carriers
(supervisor trigger, sleep, wait party, run_program), get_costs, and the
runtime's one-line render."""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from harness.apps import CostsApp, PartyApp, SleepApp
from harness.runtime import Sim
from harness.supervisor import Scheduler
from harness.timeutil import iso
from tests.conftest import make_config, make_weather_task, write_weather_csv
from tests.harness.test_supervisor import read_jsonl, run_fixture

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scaffolds"))

from runtime.agent import brief_line  # noqa: E402

START = datetime(2021, 6, 1, tzinfo=timezone.utc)


def t(day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(2021, 6, day, hour, minute, tzinfo=timezone.utc)


def _sim(tmp_path, **over) -> Sim:
    csv = write_weather_csv(tmp_path / "w.csv", START, [20.0] * 24 * 30)
    cfg = make_config(weather_csv=csv, data_cutoff=START,
                      cell={"tm": "A", "tlrn": "none", "sig": "none",
                            "alg": "none"}, **over)
    task = make_weather_task(cfg)
    (tmp_path / "run").mkdir()
    return Sim(cfg, tmp_path / "run", tmp_path / "ws", task)


# -- Sim.daily_brief ------------------------------------------------------------------


def test_brief_once_per_date_same_instant_repeats_no_catch_up(tmp_path):
    sim = _sim(tmp_path, sim_start=t(1), sim_end=t(11), budget_usd=50.0,
               domain_budgets={"llm": 8.0, "weather": 3.0})
    sim.bill("llm", 1.5)
    sim.bill("weather", 0.25)
    b = sim.daily_brief(t(1))
    assert b["day"] == 1 and b["of"] == 10
    assert b["budget_usd"] == 50.0 and b["spent_usd"] == 1.75
    assert b["remaining_usd"] == 48.25
    assert b["llm"] == {"budget_usd": 8.0, "spent_usd": 1.5,
                        "remaining_usd": 6.5}
    assert b["domains"] == {"weather": {"budget_usd": 3.0, "spent_usd": 0.25,
                                        "remaining_usd": 2.75}}
    assert b["spend_by_type"] == {"llm": 1.5, "weather": 0.25}
    # same instant: every agent waking together sees it
    assert sim.daily_brief(t(1)) == b
    # later the same date: nothing (00:30 learn after 00:00 act)
    assert sim.daily_brief(t(1, 0, 30)) is None
    assert sim.daily_brief(t(1, 23, 59)) is None
    # a skipped date gets no catch-up: the next wake briefs its own date
    b3 = sim.daily_brief(t(3, 9))
    assert b3["day"] == 3
    assert sim.daily_brief(t(3, 9)) == b3
    assert sim.daily_brief(t(3, 10)) is None
    briefs = [e for e in sim.ledger.events if e["type"] == "brief"]
    assert [e["day"] for e in briefs] == [1, 3]
    assert all(e["cost"] == 0.0 for e in briefs)
    assert briefs[0]["llm_spent_usd"] == 1.5


def test_llm_block_always_present_and_partial_last_day(tmp_path):
    sim = _sim(tmp_path, sim_start=t(1), sim_end=t(3, 12),
               domain_budgets={})
    b = sim.daily_brief(t(3, 6))
    assert b["of"] == 3 and b["day"] == 3
    assert b["llm"] == {"budget_usd": 0.0, "spent_usd": 0.0,
                        "remaining_usd": 0.0}
    assert "domains" not in b and b["spend_by_type"] == {}


# -- carriers -----------------------------------------------------------------------


def test_sleep_carries_brief_on_first_wake_of_each_date(tmp_path):
    sim = _sim(tmp_path, sim_start=t(1), sim_end=t(5))
    app = SleepApp(sim)
    first = asyncio.run(app.sleep({"until": iso(t(1, 6))}))
    assert first["woke_for"] == "sleep" and first["costs"]["day"] == 1
    second = asyncio.run(app.sleep({"until": iso(t(1, 12))}))
    assert "costs" not in second
    # a trigger wake across midnight carries it too
    sim.schedule.run_at("learn", t(2, 1))
    third = asyncio.run(app.sleep({"until": iso(t(3))}))
    assert third["woke_for"] == "trigger" and third["costs"]["day"] == 2
    # sleeping past a whole date (no midnight fallback ends a wait): the
    # wake briefs ITS date only
    fourth = asyncio.run(app.sleep({"until": iso(t(4, 8))}))
    assert fourth["now"] == iso(t(4, 8)) and fourth["costs"]["day"] == 4
    assert [e["day"] for e in sim.ledger.events if e["type"] == "brief"] \
        == [1, 2, 4]
    # no-op (past until) and experiment_over never brief
    assert "costs" not in asyncio.run(app.sleep({"until": iso(t(4))}))
    over = asyncio.run(app.sleep({"until": iso(t(9))}))
    assert over.get("experiment_over") and "costs" not in over


def test_party_briefs_every_waiter_at_the_days_first_instant(tmp_path):
    sim = _sim(tmp_path, sim_start=t(1), sim_end=t(5))
    sim.schedule.run_at("learn", t(2))

    async def script():
        await PartyApp(sim).set_party({"waiter_ids": ["a", "b"],
                                       "trigger_waiter": "a"})
        app = SleepApp(sim)
        # both wake at midnight of day 2 (a via trigger, b via deadline)
        ra, rb = await asyncio.gather(
            app.sleep({"until": iso(t(3)), "waiter_id": "a"}),
            app.sleep({"until": iso(t(2)), "waiter_id": "b"}))
        # b alone wakes later the same date: no brief (a stays parked
        # until day 3, so it is a task, released by abort at the end)
        a_task = asyncio.ensure_future(
            app.sleep({"until": iso(t(3)), "waiter_id": "a"}))
        rb2 = await app.sleep({"until": iso(t(2, 6)), "waiter_id": "b"})
        sim.party.abort(sim.clock.now)
        await a_task
        return ra, rb, rb2

    ra, rb, rb2 = asyncio.run(script())
    assert ra["woke_for"] == "trigger" and ra["costs"]["day"] == 2
    assert rb["woke_for"] == "sleep" and rb["costs"] == ra["costs"]
    assert rb2["now"] == iso(t(2, 6)) and "costs" not in rb2


def test_trigger_payload_carries_brief_once_per_date(tmp_path):
    results, run_dir = run_fixture(
        tmp_path, "schedule_crud", days=3,
        cell={"tm": "D", "tlrn": "none", "sig": "none", "alg": "none"})
    inv = read_jsonl(run_dir / "workspace" / "invocations.jsonl")
    # bootstrap 06-01 00:00, check 06-01 06:00, daily 06-01/02/03 23:00
    assert [i.get("costs", {}).get("day") for i in inv] == [1, None, None, 2, 3]
    assert inv[0]["costs"]["of"] == 3
    assert inv[3]["costs"]["llm"]["budget_usd"] == results["resources"][
        "domain_budgets"]["llm"]
    ledger = read_jsonl(run_dir / "ledger.jsonl")
    assert [e["day"] for e in ledger if e["type"] == "brief"] == [1, 2, 3]


def test_learn_trigger_leaves_the_brief_to_the_next_actor_trigger(tmp_path):
    sim = _sim(tmp_path, sim_start=t(1), sim_end=t(4), budget_usd=50.0)
    sched = Scheduler(sim)
    sim.clock.advance_to(t(1, 12))
    sim.schedule.set_crontab([{"id": "learn", "cron_expr": "0 0 * * *"},
                              {"id": "act", "cron_expr": "5 0 * * *"}],
                             sim.clock.now)

    async def drive():
        out = []
        for _ in range(2):
            trig = await sched.next()
            out.append(trig)
            await sched.report_exit(trig["id"], 0, False, "")
        return out

    learn, act = asyncio.run(drive())
    assert learn["id"] == "learn" and "costs" not in learn
    assert act["id"] == "act" and act["costs"]["day"] == 2


def test_get_costs_reports_budgets(tmp_path):
    sim = _sim(tmp_path, budget_usd=40.0, domain_budgets={"llm": 10.0})
    sim.bill("llm", 2.0)
    res = asyncio.run(CostsApp(sim).get_costs({}))
    assert res["spend_by_type"] == {"llm": 2.0} and res["spend_total"] == 2.0
    assert res["budget_usd"] == 40.0 and res["remaining_usd"] == 38.0
    assert res["llm"] == {"budget_usd": 10.0, "spent_usd": 2.0,
                          "remaining_usd": 8.0}


# -- runtime render -------------------------------------------------------------------


def test_brief_line_render():
    costs = {"day": 12, "of": 30, "budget_usd": 200.0, "spent_usd": 41.2,
             "remaining_usd": 158.8,
             "llm": {"budget_usd": 20.0, "spent_usd": 9.7,
                     "remaining_usd": 10.3},
             "domains": {"news_search": {"budget_usd": 100.0,
                                         "spent_usd": 31.5,
                                         "remaining_usd": 68.5}},
             "spend_by_type": {"llm": 9.7, "news_search": 31.5}}
    assert brief_line(costs) == (
        "Day 12 of 30. Budget: $41.20 spent of $200.00 ($158.80 left). "
        "LLM budget: $9.70 spent of $20.00 ($10.30 left). "
        "news_search budget: $31.50 spent of $100.00 ($68.50 left). "
        "By type: llm $9.70, news_search $31.50.")
    assert brief_line({"day": 1, "of": 3, "budget_usd": 5.0,
                       "spent_usd": 0.0, "remaining_usd": 5.0,
                       "llm": {"budget_usd": 2.0, "spent_usd": 0.0,
                               "remaining_usd": 2.0},
                       "spend_by_type": {}}) == (
        "Day 1 of 3. Budget: $0.00 spent of $5.00 ($5.00 left). "
        "LLM budget: $0.00 spent of $2.00 ($2.00 left).")
