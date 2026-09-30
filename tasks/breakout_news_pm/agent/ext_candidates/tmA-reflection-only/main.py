"""Per-market topology for breakout_news_pm (EXPLORATORY task scaffold). Mounted as main.py by
scaffolds/compose.py for cfg.agent.scaffold = "task:per_market".

One agent per monitored market — separate conversation histories, so
per-market attention is invariant to the roster size — plus a
coordinator that owns the schedule and the daily learn cycle. All
agents wait concurrently through the env's wait party (set_party):
the clock advances only when every thread is parked.

The learning stack is the SHARED one (runtime memory/reflect/skills +
this task's records.py hooks), fired globally by the coordinator —
per-market acting, pooled daily learning. Cell parameters come from the
generated cell_config.py sibling.

Thread model: k market threads + the coordinator (main thread). Sim
time is frozen while any thread computes, so threads only overlap in
real time. state.json access is serialized by one lock. A market agent
that exits early (runaway/budget guard) leaves the wait party so the
others keep running; the supervisor's next trigger revives it.
"""

from __future__ import annotations

import json
import os
import threading
import traceback
from datetime import timedelta
from pathlib import Path

import cell_config
import reflective
from runtime import agent, memory
from runtime.env_client import Env, EnvError, iso
from runtime.state import load_state, save_state

if cell_config.ALG not in ("none", "memory", "skills", "vskills"):
    raise RuntimeError("task:per_market supports alg none/memory/skills/vskills")

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

env = Env()
state = load_state()
if cell_config.ALG == "memory":
    memory.bind(records, env, state)  # formatted learned block:
    # task semantics from records.py (memory_formatted_render_v1)
_state_lock = threading.Lock()  # state.json single-writer
_roster_lock = threading.Lock()

COORDINATOR = "coordinator"
CURATOR = "curator"
CURATION_HOURS = 24
SIM_END = env.call("get_time")["sim_end"]
SIM_START = min(m["start"] for m in cell_config.TASK_PARAMS["markets"])


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
        # runs in the coordinator thread while the market agents are
        # parked in the wait party; replays use a frozen env, a scratch
        # state and their own transcripts, so the live agents see nothing
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
    """One market agent's registry: schedule tools are coordinator-owned;
    the wait tool carries this agent's waiter_id (under TM-B that is
    run_program, which also scopes the jail per agent); search/notify
    register candidates/actions exactly like the generated react
    wrappers. env / state default to the
    live ones; a replay passes its frozen env and a scratch state."""
    live = state is globals()["state"]
    tools = agent.env_tools(env)
    for n in ("get_crontab", "set_crontab", "run_at", "set_party"):
        tools.pop(n, None)
    for n in ("ls", "read_file", "write_file", "edit_file"):
        # authored-workspace jails are keyed by waiter_id: pin this
        # agent's file tools to ITS jail, matching its run_program
        if n in tools:
            inner = tools[n]["fn"]
            tools[n] = {**tools[n],
                        "fn": (lambda a, _f=inner:
                               _f({**a, "waiter_id": name}))}

    wait_inner = tools[cell_config.WAIT_TOOL]["fn"]

    def _wait(args):
        args = {**args, "waiter_id": name}
        with _state_lock:
            state.setdefault("last_waits", {})[name] = args
            state["last_wait"] = args  # cost-report slot: most recent arm
        return wait_inner(args)

    tools[cell_config.WAIT_TOOL] = {**tools[cell_config.WAIT_TOOL],
                                    "fn": _wait}
    search_inner = tools["search_news"]["fn"]

    def _observed_search(args):
        result = search_inner(args)
        reflective.register_search(name, args, result, iso(env.now()))
        return result

    tools["search_news"] = {**tools["search_news"], "fn": _observed_search}

    article_inner = tools["get_article"]["fn"]

    def _observed_article(args):
        result = article_inner(args)
        reflective.register_article(name, args, result, iso(env.now()))
        return result

    tools["get_article"] = {**tools["get_article"], "fn": _observed_article}

    notify_observed_inner = tools["notify"]["fn"]

    def _observed_notify(args):
        result = notify_observed_inner(args)
        reflective.register_notify(name, args, result, iso(env.now()))
        return result

    tools["notify"] = {**tools["notify"], "fn": _observed_notify}
    if LEARNING:
        search_inner = tools["search_news"]["fn"]

        def _search(args):
            result = search_inner(args)
            with _state_lock:
                records.register_candidates(state, result)
            return result

        tools["search_news"] = {**tools["search_news"], "fn": _search}

        notify_inner = tools["notify"]["fn"]

        def _notify(args):
            result = notify_inner(args)
            nid = str(args.get("news_id"))
            with _state_lock:
                reg = state.get("registered", {}).get(nid, {})
                state.setdefault("actions", {})[nid] = {
                    "did": "alerted", "at": result.get("at"),
                    "market_id": args.get("market_id"),
                    "direction": args.get("direction"),
                    "title": reg.get("title"),
                    "published": reg.get("published")}
                if live:
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
        "notify for it. Another agent handles each other market.\n",
        encoding="utf-8")


def build_agent(market: dict, n_markets: int, env=env, state=state,
                block_fn=None, transcript=None, label=None) -> agent.Agent:
    """The ONE constructor of a market agent, live and replayed alike
    (see per_market_cron_main.build_agent)."""
    name = f"m-{market['market_id']}"
    instruction = Path("instructions") / f"{name}.md"
    write_instruction(market, instruction, n_markets)
    learned_block = lambda: reflective.render_memory(name)
    if block_fn is not None:
        learned_block = lambda _base=block_fn: (
            (_base() or "") + "\n\n" + reflective.render_memory(name))
    return agent.Agent(env, label or name, market_tools(name, env, state),
                       transcript=transcript or f"logs/transcript_{name}.jsonl",
                       block_fn=learned_block,
                       wait_tool=cell_config.WAIT_TOOL,
                       instruction=instruction,
                       context_tokens=cell_config.CONTEXT_TOKENS,
                       event_fn=(lambda event, **fields:
                                 reflective.observe(event, agent=name,
                                                    **fields)))


def make_episode(replay_env, agent_name: str, block_fn, transcript,
                 scratch_state: dict, label: str) -> agent.Agent:
    """Episode factory for replay.replay_wake: a replayed wake runs until
    the agent's first wait call (ReplayEnv ends it there)."""
    market = next(m for m in markets if f"m-{m['market_id']}" == agent_name)
    return build_agent(market, len(markets), env=replay_env,
                       state=scratch_state, block_fn=block_fn,
                       transcript=transcript, label=label)


def make_agent(market: dict, n_markets: int, trigger: dict) -> agent.Agent:
    block_fn = None
    if cell_config.ALG == "memory":
        block_fn = memory.render_block
    elif SKILLS:
        block_fn = skills.render_block
    a = build_agent(market, n_markets, block_fn=block_fn)
    a.wake(trigger)
    return a


def leave_party(name: str) -> None:
    """A finished/stopped agent leaves the roster from its own turn so
    the barrier does not wait for it; the coordinator stays."""
    with _roster_lock:
        if name in ROSTER:
            ROSTER.remove(name)
        try:
            env.call("set_party", waiter_ids=list(ROSTER),
                     trigger_waiter=COORDINATOR)
        except EnvError:
            pass  # end of run: the party is gone with the process


def run_market_agent(a: agent.Agent) -> None:
    try:
        while a.turn():
            pass
    except SystemExit:
        pass  # runaway/budget guard: this agent stops, the rest go on
    except BaseException:
        traceback.print_exc()
        os._exit(1)  # fail the whole process loudly -> crash log
    finally:
        leave_party(a.name)


def run_curator() -> None:
    """Park as a party member and curate hindsight every simulated day."""
    try:
        next_due = env.now() + timedelta(hours=CURATION_HOURS)
        while True:
            wake = env.call("sleep", until=min(next_due, env.now() +
                            timedelta(hours=CURATION_HOURS)).isoformat(),
                            waiter_id=CURATOR)
            if wake.get("experiment_over") or wake.get("aborted"):
                break
            try:
                reflective.curate(env, markets, SIM_START)
            except Exception as exc:
                reflective.observe("curator_exception", sim_time=iso(env.now()),
                                   error=f"{type(exc).__name__}: {exc}")
                traceback.print_exc()
            next_due = env.now() + timedelta(hours=CURATION_HOURS)
    except EnvError as exc:
        reflective.observe("curator_env_error", error=str(exc))
    finally:
        leave_party(CURATOR)


# -- coordinator (main thread) -------------------------------------------------------

trigger = json.loads(os.environ.get("ENV_TRIGGER", "{}"))
markets = env.call("get_markets")
ROSTER = [f"m-{m['market_id']}" for m in markets] + [CURATOR, COORDINATOR]
env.call("set_party", waiter_ids=list(ROSTER), trigger_waiter=COORDINATOR)
if cell_config.WAIT_TOOL == "run_program":
    # TM-B: the coordinator parks via a program in its own jail (market
    # agents never see it); waking is trigger/deadline delivery
    env.call("write_file", waiter_id=COORDINATOR,
             path=cell_config.WAIT_ARGS["path"],
             content="import envkit\nenvkit.wait(envkit.deadline())\n")
if LEARNING:
    env.call("set_crontab", entries=[
        {"id": "learn", "cron_expr": cell_config.LEARN_CRON}])
    if trigger.get("id") == "learn":  # crash recovery: re-invoked at
        learn(state)                  # the learn firing

threads = []
curator_thread = threading.Thread(target=run_curator, name=CURATOR,
                                  daemon=True)
threads.append(curator_thread)
curator_thread.start()
for m in markets:
    th = threading.Thread(target=run_market_agent,
                          args=(make_agent(m, len(markets), trigger),),
                          name=f"m-{m['market_id']}", daemon=True)
    threads.append(th)
    th.start()

while True:
    try:
        wake = env.call(cell_config.WAIT_TOOL, until=SIM_END,
                        waiter_id=COORDINATOR, **cell_config.WAIT_ARGS)
    except EnvError:
        traceback.print_exc()
        break
    if wake.get("experiment_over") or wake.get("aborted"):
        break
    if (wake.get("woke_for") == "trigger"
            and (wake.get("trigger") or {}).get("id") == "learn"
            and LEARNING):
        learn(state)
    # any other trigger (fallback midnight, run_at): nothing to do here

for th in threads:
    th.join(timeout=60)
