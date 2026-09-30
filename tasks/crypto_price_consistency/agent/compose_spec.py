"""Constructor declarations of crypto_price_consistency: what this task
asks scaffolds/compose.py to provision, declared task-side so the shared
constructor never enumerates tasks. Read by compose only — never copied
into a workspace.

This task has no learning stack, so only TASK_SCAFFOLDS is declared.
"""

TASK_SCAFFOLDS = {
    # cron_react: one agent for the whole symbol roster under a fixed
    # cron (tm C) or, under tm D, with the agent-managed schedule tools;
    # no-learning cells only.
    "cron_react": {"main": "scaffolds/cron_react_main.py",
                   "tm": ("C", "D"),
                   "alg": ("none",)},
}
