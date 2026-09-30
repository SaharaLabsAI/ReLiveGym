"""Constructor declarations of reddit_ai_popularity: what this task asks
scaffolds/compose.py to provision, declared task-side so the shared
constructor never enumerates tasks. Read by compose only — never copied
into a workspace.

This task needs no candidate-search or action wrappers: the oracle
stream (task.py oracle_outcomes) already carries the agent's own settled
recommendations with their ids, so records.action_for has nothing to
link (records.py).

TASK_SCAFFOLDS — cron_react is the TM-C acting arm (per the bnpm
per_market_cron topology, single-stream): a fixed hourly cron fires
one persistent ReACT agent to completion at one sim instant —
program-owned timing, agent-owned acting, no retrieval pipeline and no
task-authored decision code. Learning stays the shared stack, fired by
the tlrn cron. Mounted as main.py with the react learning file set +
generated cell_config.py.
"""

TASK_SCAFFOLDS = {
    # tm D = the same main with the agent-schedule CRUD kept in the
    # agent's registry
    "cron_react": {"main": "scaffolds/cron_react_main.py",
                   "tm": ("C", "D"),
                   "alg": ("none", "memory", "skills")},
}
