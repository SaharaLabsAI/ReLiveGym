"""Per-market cron topology for breakout_news_pm (EXPLORATORY task
scaffold). Mounted as main.py by scaffolds/compose.py for
cfg.agent.scaffold = "task:per_market_cron".

The per_market (TM-A/B) topology with exactly one thing changed: acting
timing is program-owned. A fixed cron (ACT_CRON) fires per market; each
firing runs that market's agent — same Agent class, same persistent
per-market transcript and compaction, same tools minus anything that
moves sim time — to completion at one simulated instant (it fetches
news, decides notifications, calls done). TM-C discipline: bounded work
per firing, save, exit; the supervisor re-invokes at the next trigger.

The learning stack is the SHARED one (runtime memory/reflect/skills +
this task's records.py hooks), fired by the learn cron (tlrn axis) —
per-market acting, pooled learning, byte-identical learn() across TM
arms. Signal intake is program-owned too: learn() pulls code-side, so
feedback tools never reach the agent registry (they would hand signal
timing back to the agent — the very axis this arm holds fixed). Cell
parameters come from the generated cell_config.py sibling.

TM-D: the same program
under cell tm=D, with exactly one thing changed — each market agent keeps
the env's agent-schedule CRUD (list/create/update/delete_schedule) in
its registry, pinned to its own market by the same ownership guard as
its action: every schedule it creates carries its market as `target`,
and a fired agent schedule routes to that market's agent only. Each
market's base cadence is installed once as a DEFAULT schedule — an
agent-owned row (id and target `m-<market_id>`) that market's agent may
update or delete like its own; re-declaring it never brings it back
. Everything else is
byte-identical between C and D.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import cell_config
from runtime import agent, memory
from runtime.env_client import Env, iso
from runtime.state import load_state, save_state

if cell_config.TM not in ("C", "D"):
    raise RuntimeError("task:per_market_cron requires TM-C or TM-D")
if cell_config.ALG not in ("none", "memory", "skills", "vskills"):
    raise RuntimeError(
        "task:per_market_cron supports alg none/memory/skills/vskills")

LEARNING = cell_config.ALG != "none"
SKILLS = cell_config.ALG in ("skills", "vskills")  # renders skills.md
if LEARNING:
    import records  # mounted iff sig != none (compose axis gating)

    if SKILLS:
        from runtime import reflect, skills
    if cell_config.ALG == "vskills":  # replay-verified adoption
        import replay

        replay.DIFF_ROWS = cell_config.REFLECT_DIFF_ROWS
        replay.MAX_WORKERS = cell_config.REPLAY_MAX_WORKERS
        reflect.TRANSCRIPT_TOKENS = cell_config.REFLECT_TRANSCRIPT_TOKENS
    if SKILLS:  # reflection render caps + block budget (cell config)
        reflect.OWN_HISTORY_CAP = cell_config.REFLECT_OWN_HISTORY_CAP
        reflect.DIGEST_CAP = cell_config.REFLECT_DIGEST_CAP
        reflect.K_PER_STRATUM = cell_config.REFLECT_EXAMPLES_PER_STRATUM
        skills.BLOCK_TOKENS = cell_config.BLOCK_TOKENS
    if cell_config.ALG == "memory":
        memory.BLOCK_TOKENS = cell_config.BLOCK_TOKENS
    if cell_config.SIG == "self":
        from runtime import trace

        import pull_feedback

ACT_CRON = "5 */6 * * *"  # 6-hourly: the acting cadence of this arm
MAX_CALLS_PER_FIRING = 30  # per-firing call cap (run_until_done)
AGENT_SCHEDULE_TOOLS = ("list_schedules", "create_schedule",
                        "update_schedule", "delete_schedule")  # TM-D only

env = Env()
state = load_state()
if cell_config.ALG == "memory":
    memory.bind(records, env, state)  # formatted learned block:
    # task semantics from records.py (memory_formatted_render_v1)


def scope_schedule_tools(tools: dict, target: str) -> None:
    """TM-D per-entity: pin this agent's schedule CRUD to its
    own entity, the same ownership guard as the action wrapper —
    create/update force `target`, list shows the base schedule plus this
    agent's own rows, update/delete refuse another agent's rows."""
    inner = {n: tools[n]["fn"] for n in AGENT_SCHEDULE_TOOLS}

    def _foreign(sched_id) -> bool:
        rows = {r["id"]: r
                for r in inner["list_schedules"]({})["schedules"]}
        row = rows.get(sched_id)
        # unknown / hidden / base rows fall through to the env's own
        # refusal (no such schedule, read-only)
        return row is not None and row.get("target") != target

    def _list(args):
        res = inner["list_schedules"](args)
        res["schedules"] = [r for r in res["schedules"]
                            if r["owner"] == "system"
                            or r.get("target") == target]
        return res

    def _create(args):
        return inner["create_schedule"]({**args, "target": target})

    def _update(args):
        if _foreign(args.get("id")):
            return {"error": f"schedule {args.get('id')!r} belongs to "
                             "another agent"}
        return inner["update_schedule"]({**args, "target": target})

    def _delete(args):
        if _foreign(args.get("id")):
            return {"error": f"schedule {args.get('id')!r} belongs to "
                             "another agent"}
        return inner["delete_schedule"](args)

    for n, fn in (("list_schedules", _list), ("create_schedule", _create),
                  ("update_schedule", _update), ("delete_schedule", _delete)):
        tools[n] = {**tools[n], "fn": fn}


def drop_stale_schedule(trigger: dict) -> None:
    """A recurring agent schedule whose owner has retired would keep
    firing empty processes for the rest of the run: remove it (a
    one-time schedule is already gone once it fired)."""
    if trigger.get("kind") == "cron":
        try:
            env.call("delete_schedule", id=trigger["id"])
        except Exception:
            pass



def learn(state):
    """Identical semantics to the generated react learn(): sig pull ->
    record append -> reflection -> save. Global across markets."""
    if cell_config.SIG == "oracle":
        memory.pull_oracle(env, state, records.action_for)
    else:  # self
        t = iso(env.now())
        for out in pull_feedback.compile(env, state):
            action = records.action_for(out, state)
            memory.append_record(t=t, src="self", outcome=out, action=action)
            trace.log(t, "feedback", src="self", outcome=out)
    if cell_config.ALG == "skills":
        reflect.reflect(env, state, records)
    elif cell_config.ALG == "vskills":
        reflect.reflect_verified(
            env, state, records, replay, make_episode,
            n_iter=cell_config.REPLAY_MAX_ITER,
            max_items=cell_config.REPLAY_MAX_ITEMS,
            claim_window_hours=cell_config.REPLAY_CLAIM_WINDOW_HOURS,
            price_delay_minutes=cell_config.REPLAY_PRICE_DELAY_MINUTES,
            context_tokens=cell_config.CONTEXT_TOKENS,
            max_calls=cell_config.REPLAY_MAX_CALLS)
    save_state(state)


# -- per-market agents ---------------------------------------------------------------


def market_tools(name: str, env=env, state=state) -> dict:
    """One market agent's registry: schedule/party tools are program-
    owned, wait tools are gone , feedback tools are gone (learn() pulls code-side), and `done`
    ends the firing. search/notify register candidates/actions exactly
    like the per_market wrappers. env /
    state default to the live ones; a replay passes its frozen env and a
    scratch state so the wrappers register nothing live."""
    tools = agent.env_tools(env)
    for n in ("get_crontab", "set_crontab", "run_at", "set_party",
              "sleep", "wait_until"):
        tools.pop(n, None)
    for n in [n for n, t in tools.items() if "feedback" in t["tags"]]:
        tools.pop(n)
    if cell_config.TM == "D":
        scope_schedule_tools(tools, name)
    else:  # TM-C: no schedule surface at all
        for n in AGENT_SCHEDULE_TOOLS:
            tools.pop(n, None)
    tools.update(agent.done_tool())
    if LEARNING:
        search_inner = tools["search_news"]["fn"]

        def _search(args):
            result = search_inner(args)
            records.register_candidates(state, result)
            return result

        tools["search_news"] = {**tools["search_news"], "fn": _search}

        notify_inner = tools["notify"]["fn"]

        def _notify(args):
            result = notify_inner(args)
            nid = str(args.get("news_id"))
            reg = state.get("registered", {}).get(nid, {})
            state.setdefault("actions", {})[nid] = {
                "did": "alerted", "at": result.get("at"),
                "market_id": args.get("market_id"),
                "direction": args.get("direction"),
                "title": reg.get("title"),
                "published": reg.get("published")}
            if state is globals()["state"]:
                save_state(state)
            return result

        tools["notify"] = {**tools["notify"], "fn": _notify}
    return tools


def write_instruction(market: dict, path: Path, n_markets: int) -> None:
    base = Path("INSTRUCTION.md").read_text(encoding="utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        base + "\n\n# Your assignment\n"
        f"You are one of {n_markets} independent agents; each tracks "
        "exactly one market. Yours:\n\n"
        f"- market_id: {market['market_id']}\n"
        f"- question: {market['question']}\n"
        f"- window: {market['start']} to {market['end']}\n\n"
        "Work only this market: search news relevant to it, and only "
        "notify for it. Another agent handles each other market.\n\n"
        + schedule_note(),
        encoding="utf-8")


def schedule_note() -> str:
    if cell_config.TM == "D":
        return (
            f"A default schedule wakes you every 6 hours (cron `{ACT_CRON}`); "
            "you may change or remove it like any schedule of your own. Time does not pass while you work: "
            "each waking runs at one simulated instant and ends when you "
            "call done. You can also manage your own wake-up schedules "
            "with the schedule API (list_schedules / create_schedule / "
            "update_schedule / delete_schedule, all free); they wake "
            "only you: a one-time schedule fires once at its `at` and is "
            "then removed; a recurring schedule fires on its cron "
            "expression; when one of your schedules fires, you wake with "
            "its id and note. Schedules never fire while you are awake, "
            "and none fire after the run ends. "
            "The default schedule is already in place: do not create schedules that duplicate it or each other; list_schedules shows what already exists. "
            "Each waking: search for "
            "news fresh since your previous waking, decide what (if "
            "anything) to notify, then call done.\n")
    return (
        "A fixed schedule wakes you every 6 hours; you cannot wait or "
        "schedule "
        "anything yourself, and time does not pass while you work. Each "
        "waking: search for news fresh since your previous waking, "
        "decide what (if anything) to notify, then call done.\n")


def build_agent(market: dict, n_markets: int, env=env, state=state,
                block_fn=None, transcript=None, label=None) -> agent.Agent:
    """The ONE constructor of a market agent, live and replayed alike:
    same instruction file, same registry (market_tools), same context
    budget. A replay swaps env (frozen clock), state (scratch), block_fn
    (candidate) and the transcript path; the label only names its
    history/trace rows."""
    name = f"m-{market['market_id']}"
    instruction = Path("instructions") / f"{name}.md"
    write_instruction(market, instruction, n_markets)
    return agent.Agent(env, label or name, market_tools(name, env, state),
                       transcript=transcript or f"logs/transcript_{name}.jsonl",
                       block_fn=block_fn,
                       instruction=instruction,
                       context_tokens=cell_config.CONTEXT_TOKENS)


def make_episode(replay_env, agent_name: str, block_fn, transcript,
                 scratch_state: dict, label: str) -> agent.Agent:
    """Episode factory for replay.replay_wake: the live agent rebuilt on
    a frozen env with a candidate block."""
    market = by_trigger[agent_name]
    return build_agent(market, len(markets), env=replay_env,
                       state=scratch_state, block_fn=block_fn,
                       transcript=transcript, label=label)


def run_episode(market: dict, trigger: dict, n_markets: int) -> None:
    """One firing of one market's agent: run to its done call at the
    current sim instant. Persistent transcript across firings — the
    wake marker is the compaction segment boundary."""
    if not (market["start"] <= iso(env.now()) <= market["end"]):
        return  # outside the market's window: nothing to act on
    block_fn = None
    if cell_config.ALG == "memory":
        block_fn = memory.render_block
    elif SKILLS:
        block_fn = skills.render_block
    a = build_agent(market, n_markets, block_fn=block_fn)
    a.wake(trigger)
    agent.run_until_done(a, MAX_CALLS_PER_FIRING)


# -- one firing (the program) --------------------------------------------------------

trigger = json.loads(os.environ.get("ENV_TRIGGER", "{}"))
markets = env.call("get_markets")

# Idempotent full replacement: entries whose id and expression are
# unchanged keep their last-fire state, so re-declaring never re-fires.
# TM-D: each market's base cadence is a DEFAULT schedule — installed once
# as an agent-owned row targeted at that market, which its agent may
# update or delete; re-declaring it never brings it back. TM-C: fixed.
DEFAULT = cell_config.TM == "D"
entries = [{"id": f"m-{m['market_id']}", "cron_expr": ACT_CRON,
            **({"agent_owned": True, "target": f"m-{m['market_id']}"}
               if DEFAULT else {})}
           for m in markets]
if LEARNING:
    entries.append({"id": "learn", "cron_expr": cell_config.LEARN_CRON})
env.call("set_crontab", entries=entries)

by_trigger = {f"m-{m['market_id']}": m for m in markets}
try:
    if trigger.get("id") == "learn" and LEARNING:
        learn(state)
    elif trigger.get("id") in by_trigger:
        run_episode(by_trigger[trigger["id"]], trigger, len(markets))
    elif trigger.get("owner") == "agent":
        # TM-D: one market agent's own schedule — wake that agent only
        m = by_trigger.get(str(trigger.get("target") or ""))
        if m is not None:
            run_episode(m, trigger, len(markets))
        else:
            drop_stale_schedule(trigger)
    else:  # bootstrap / fallback / crash recovery without a market id
        for m in markets:
            try:
                run_episode(m, trigger, len(markets))
            except SystemExit:
                pass  # guard exit: this agent stops, the rest act
except SystemExit:
    pass  # budget/runaway guard: the firing ends; state still saves
save_state(state)
