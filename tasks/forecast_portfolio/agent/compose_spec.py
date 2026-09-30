"""Constructor declarations of forecast_portfolio: what this task asks
scaffolds/compose.py to provision, declared task-side so the shared
constructor never enumerates tasks. Read by compose only — never copied
into a workspace.

TASK_SCAFFOLDS — cron_react is the TM-C acting arm (the resolution_detect
port of the reddit cron_react topology, daily cadence): a fixed daily
cron fires one persistent ReACT agent to completion at one sim instant —
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
