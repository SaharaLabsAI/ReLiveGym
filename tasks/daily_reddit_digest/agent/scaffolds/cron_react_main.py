"""Cron-fired ReACT topology for daily_reddit_digest (EXPLORATORY task
scaffold). Mounted as main.py by scaffolds/compose.py for
cfg.agent.scaffold = "task:cron_react".

The react (TM-A/B) arm with exactly one thing changed: acting timing is
program-owned. A fixed daily cron (ACT_CRON, a:00 + 5 min) fires this
program; each firing
runs the ONE stream agent — same Agent class, same persistent transcript
and compaction, same tools minus anything that moves sim time — to
completion at one simulated instant (it lists posts, decides whether to
deliver a digest, calls done). TM-C discipline: bounded work per
firing, save, exit; the supervisor re-invokes at the next trigger. What
to observe and when to deliver stays entirely agent-owned — no scan
pipeline, no task-authored forecasting.

The learning stack is the SHARED one (runtime memory/reflect/skills +
this task's records.py hooks), fired by the learn cron (tlrn axis) —
byte-identical learn() across TM arms. Signal intake is program-owned
too: learn() pulls code-side, so feedback tools never reach the agent
registry (they would hand signal timing back to the agent — the very
axis this arm holds fixed). Cell parameters come from the generated
cell_config.py sibling.

TM-D: the same program under
cell tm=D, with exactly one thing changed — the agent keeps the env's
agent-schedule CRUD (list/create/update/delete_schedule) in its
registry and its instruction names it. The base cadence is installed once as a
DEFAULT schedule — an agent-owned row (id `act`) the agent may update
or delete like its own; re-declaring it never brings it back
. Any schedule
firing runs this same program, and a note arrives in the wake marker.
Everything else — episode loop,
caps, transcript, learning — is byte-identical between C and D.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import cell_config
from runtime import agent, memory
from runtime.env_client import Env
from runtime.state import load_state, save_state

if cell_config.TM not in ("C", "D"):
    raise RuntimeError("task:cron_react requires TM-C or TM-D")
if cell_config.ALG not in ("none", "memory", "skills"):
    raise RuntimeError("task:cron_react supports alg none/memory/skills")
if cell_config.SIG not in ("none", "oracle"):
    raise RuntimeError("task:cron_react supports sig none/oracle")

LEARNING = cell_config.ALG != "none"
if LEARNING:
    import records  # mounted iff sig != none (compose axis gating)

    if cell_config.ALG == "skills":
        from runtime import reflect, skills

# Acting cadence of this arm: once a day, 5 minutes into the delivery
# window (a:00 + 5 min; a = the task's digest_hour_utc, read from the
# generated cell_config so the yaml is the single source).
DIGEST_HOUR = int(cell_config.TASK_PARAMS.get("digest_hour_utc", 12))
ACT_CRON = f"5 {DIGEST_HOUR} * * *"
MAX_CALLS_PER_FIRING = 30  # per-firing call cap (run_until_done)
AGENT_SCHEDULE_TOOLS = ("list_schedules", "create_schedule",
                        "update_schedule", "delete_schedule")  # TM-D only

env = Env()
state = load_state()
if cell_config.ALG == "memory":
    memory.bind(records, env, state)  # formatted learned block:
    # task semantics from records.py (memory_formatted_render_v1)


def learn(state):
    """Identical semantics to the generated react learn(): sig pull ->
    record append -> reflection -> save."""
    memory.pull_oracle(env, state, records.action_for)
    if cell_config.ALG == "skills":
        reflect.reflect(env, state, records)
    save_state(state)


# -- the stream agent ----------------------------------------------------------------


def agent_tools() -> dict:
    """The agent's registry: schedule tools are program-owned, wait tools
    are gone , feedback
    tools are gone (learn() pulls code-side), and `done` ends the
    firing."""
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
            f"A default schedule wakes you once a day at "
            f"{DIGEST_HOUR:02d}:05 UTC (cron `{ACT_CRON}`); you may "
            "change or remove it like any schedule of your own. Time does not pass "
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
            f"observe what you choose to, decide what (if anything) to deliver, then call done.\n")
    else:
        note = (
            "\n\n# Your schedule\n"
            f"A fixed schedule wakes you once a day at {DIGEST_HOUR:02d}:05 "
            "UTC; you cannot wait or "
            "schedule anything yourself, and time does not pass while you "
            "work. Each waking: observe what you choose to, "
            f"decide what (if anything) to deliver, then call done.\n")
    path.write_text(base + note, encoding="utf-8")


def run_episode(trigger: dict) -> None:
    """One firing: run the agent to its done call at the current sim
    instant. Persistent transcript across firings — the wake marker is
    the compaction segment boundary."""
    instruction = Path("instructions") / "agent.md"
    write_instruction(instruction)
    block_fn = None
    if cell_config.ALG == "memory":
        block_fn = memory.render_block
    elif cell_config.ALG == "skills":
        block_fn = skills.render_block
    a = agent.Agent(env, "agent", agent_tools(),
                    transcript="logs/transcript_agent.jsonl",
                    block_fn=block_fn,
                    instruction=instruction,
                    context_tokens=cell_config.CONTEXT_TOKENS)
    a.wake(trigger)
    agent.run_until_done(a, MAX_CALLS_PER_FIRING)


# -- one firing (the program) --------------------------------------------------------

trigger = json.loads(os.environ.get("ENV_TRIGGER", "{}"))

# Idempotent full replacement: entries whose id and expression are
# unchanged keep their last-fire state, so re-declaring never re-fires.
# TM-D: the base cadence is a DEFAULT schedule — installed once as an
# agent-owned row the agent may update or delete; re-declaring it never
# brings it back (harness/schedule.py). TM-C: a fixed crontab entry.
entries = [{"id": "act", "cron_expr": ACT_CRON,
            **({"agent_owned": True} if cell_config.TM == "D" else {})}]
if LEARNING:
    entries.append({"id": "learn", "cron_expr": cell_config.LEARN_CRON})
env.call("set_crontab", entries=entries)

try:
    if trigger.get("id") == "learn" and LEARNING:
        learn(state)
    else:  # "act", bootstrap, crash recovery, or (TM-D) one of the
        run_episode(trigger)  # agent's own schedules: one acting episode
except SystemExit:
    pass  # budget/runaway guard: the firing ends; state still saves
save_state(state)
