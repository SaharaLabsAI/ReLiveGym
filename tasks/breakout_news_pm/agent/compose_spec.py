"""Constructor declarations of breakout_news_pm: what this task asks
scaffolds/compose.py to provision or
splice, declared task-side so the shared constructor never enumerates
tasks. Read by compose only — never copied into a workspace.

CANDIDATE_SEARCH — the search tool every react learning cell wraps with
records.register_candidates (and whose condition-wake payloads register
too), so everything the agent observes gets an id -> title entry in
state["registered"]: sig-self reads it as the hindsight candidate pool,
and the reflection render reads it to resolve news ids into titles.

ACTION_WRAPPER — generated-code lines splicing the action-tool wrapper
into react learning cells: each notify call is registered in
state["actions"] with its content, so records.action_for links it to
the settled outcome (tmC's scan.py registers the same
shape itself).

TASK_SCAFFOLDS — exploratory task-authored programs: per_market runs one agent per monitored
market plus a coordinator, all waiting concurrently through the env's
wait party; learning stays the shared stack, fired globally. Mounted
as main.py with the react learning file set + generated cell_config.py.
per_market_cron is the same topology under TM-C: a fixed cron per
market fires the market's agent to completion at one sim instant —
program-owned timing, agent-owned acting, no retrieval pipeline.
"""

# extra task-agent modules scan.py imports (mounted next to it)
SCAN_DEPS = ["question_keywords.py"]

CANDIDATE_SEARCH = "search_news"

TASK_SCAFFOLDS = {
    "per_market": {"main": "scaffolds/per_market_main.py",
                   "tm": ("A", "B"),
                   "alg": ("none", "memory", "skills")},
    # tm D = the same mains with the agent-schedule CRUD kept in each
    # agent's registry ; under
    # per_market_cron every market agent schedules only itself
    "per_market_cron": {"main": "scaffolds/per_market_cron_main.py",
                        "tm": ("C", "D"),
                        "alg": ("none", "memory", "skills")},
    # single-stream sibling of per_market_cron (the reddit/rd cron_react
    # port, 6-hourly): one agent over all markets, the single-stream
    # comparison cell.
    "cron_react": {"main": "scaffolds/cron_react_main.py",
                   "tm": ("C", "D"),
                   "alg": ("none", "memory", "skills")},
}

ACTION_WRAPPER = [
    "",
    '_action_inner = tools["notify"]["fn"]',
    "",
    "",
    "def _notify(args):  # own actions feed calibration: register",
    "    result = _action_inner(args)  # the claim so records.action_for",
    '    nid = str(args.get("news_id"))  # links it to the outcome',
    '    reg = state.get("registered", {}).get(nid, {})',
    '    state.setdefault("actions", {})[nid] = {',
    '        "did": "alerted", "at": result.get("at"),',
    '        "market_id": args.get("market_id"),',
    '        "direction": args.get("direction"),',
    '        "title": reg.get("title"),',
    '        "published": reg.get("published")}',
    "    save_state(state)",
    "    return result",
    "",
    "",
    'tools["notify"] = {**tools["notify"], "fn": _notify}',
]
