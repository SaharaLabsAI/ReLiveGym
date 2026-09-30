"""Cron-fired ReACT topology for crypto_price_consistency (task
scaffold). Mounted as main.py by scaffolds/compose.py for
cfg.agent.scaffold = "task:cron_react".

Port of the reddit / resolution_detect scaffold of the same name: acting
timing is program-owned. A fixed daily cron (ACT_CRON) fires this
program; each firing runs the ONE stream agent — same Agent class, same
persistent transcript and compaction, same tools minus anything that
moves sim time — to completion at one simulated instant (it fetches
candles from the venues it chooses, reports or abstains, calls done).
TM-C discipline: bounded work per firing, save, exit; the supervisor
re-invokes at the next trigger. What to fetch and what to report stays
entirely agent-owned — no task-authored consensus code. This task has
no learning stack: the no-learning cell only. Cell parameters come from
the generated cell_config.py sibling.

TM-D: the same program under
cell tm=D, with exactly one thing changed — the agent keeps the env's
agent-schedule CRUD (list/create/update/delete_schedule) in its
registry and its instruction names it. The base cadence is installed once as a
DEFAULT schedule — an agent-owned row (id `act`) the agent may update
or delete like its own; re-declaring it never brings it back
. Any schedule
firing runs this same program, and a note arrives in the wake marker.
Everything else — episode loop, caps,
transcript — is byte-identical between C and D. The daily base under a
task scored every hour is deliberate: densifying the cadence is the
agent's own call, and the treatment.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import cell_config
from runtime import agent
from runtime.env_client import Env

if cell_config.TM not in ("C", "D"):
    raise RuntimeError("task:cron_react requires TM-C or TM-D")
if cell_config.SIG != "none" or cell_config.ALG != "none":
    raise RuntimeError("task:cron_react runs the no-learning cell only")

ACT_CRON = "0 0 * * *"  # daily cadence: the acting cadence of this arm
MAX_CALLS_PER_FIRING = 30  # per-firing call cap (run_until_done)
AGENT_SCHEDULE_TOOLS = ("list_schedules", "create_schedule",
                        "update_schedule", "delete_schedule")  # TM-D only

env = Env()


# -- the stream agent ----------------------------------------------------------------


def agent_tools() -> dict:
    """The agent's registry: schedule tools are program-owned, wait tools
    are gone (the agent cannot move sim time), and `done` ends the
    firing. Under TM-D the agent-schedule CRUD stays."""
    tools = agent.env_tools(env)
    for n in ("get_crontab", "set_crontab", "run_at", "set_party",
              "sleep", "wait_until", "run_program"):
        tools.pop(n, None)
    if cell_config.TM != "D":  # TM-C: no schedule surface at all
        for n in AGENT_SCHEDULE_TOOLS:
            tools.pop(n, None)
    for n in [n for n, t in tools.items() if "feedback" in t["tags"]]:
        tools.pop(n)
    tools.update(agent.done_tool())
    return tools


def write_instruction(path: Path) -> None:
    base = Path("INSTRUCTION.md").read_text(encoding="utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if cell_config.TM == "D":
        note = (
            "\n\n# Your schedule\n"
            f"A default schedule wakes you once a day (cron "
            f"`{ACT_CRON}`); you may change or remove it like any schedule of your own. Time does not pass "
            "while you work: each waking runs at one simulated instant "
            "and ends when you call done. You can also manage your own "
            "wake-up schedules with the schedule API (list_schedules / "
            "create_schedule / update_schedule / delete_schedule, all "
            "free): a one-time schedule fires once at its `at` and is "
            "then removed; a recurring schedule fires on its cron "
            "expression; when one of your schedules fires, you wake "
            "with its id and note. Schedules never fire while you are "
            "awake, and none fire after the run ends. "
            "The default schedule is already in place: do not create schedules that duplicate it or each other; list_schedules shows what already exists. "
            "Each waking: "
            "observe what you choose to, decide what to report or "
            "abstain on, then call done.\n")
    else:
        note = (
            "\n\n# Your schedule\n"
            "A fixed schedule wakes you once a day; you cannot wait or "
            "schedule anything yourself, and time does not pass while you "
            "work. Each waking: observe what you choose to, decide what "
            "to report or abstain on, then call done.\n")
    path.write_text(base + note, encoding="utf-8")


def run_episode(trigger: dict) -> None:
    """One firing: run the agent to its done call at the current sim
    instant. Persistent transcript across firings — the wake marker is
    the compaction segment boundary."""
    instruction = Path("instructions") / "agent.md"
    write_instruction(instruction)
    a = agent.Agent(env, "agent", agent_tools(),
                    transcript="logs/transcript_agent.jsonl",
                    instruction=instruction,
                    context_tokens=cell_config.CONTEXT_TOKENS)
    a.wake(trigger)
    agent.run_until_done(a, MAX_CALLS_PER_FIRING)


# -- one firing (the program) --------------------------------------------------------

trigger = json.loads(os.environ.get("ENV_TRIGGER", "{}"))

# Idempotent full replacement: an entry whose id and expression are
# unchanged keeps its last-fire state, so re-declaring never re-fires.
# TM-D: the base cadence is a DEFAULT schedule — installed once as an
# agent-owned row the agent may update or delete; re-declaring it never
# brings it back (harness/schedule.py). TM-C: a fixed crontab entry.
entries = [{"id": "act", "cron_expr": ACT_CRON,
            **({"agent_owned": True} if cell_config.TM == "D" else {})}]
env.call("set_crontab", entries=entries)

try:
    # "act", bootstrap, crash recovery, or (TM-D) one of the agent's own
    # schedules: one acting episode
    run_episode(trigger)
except SystemExit:
    pass  # budget/runaway guard: the firing ends
