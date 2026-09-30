"""daily_reddit_digest: how the constructor (scaffolds/compose.py) assembles
agent workspaces for this task.

TM-A/TM-B cells use the generic `react` scaffold — nothing task-side except
the TM-B authored example (example_gatekeeper.py, served through
Task.authored_example). TM-C/TM-D cells use the task scaffold below: the
sibling's cron-fired ReACT program (base cron once a day at a:05) with the digest
action in place of recommend. No learning cells in v1
: alg is pinned to "none" so
a mis-specified learning yaml fails validation instead of running without
records.py / reflect.md.
"""

TASK_SCAFFOLDS = {
    "cron_react": {"main": "scaffolds/cron_react_main.py",
                   "tm": ("C", "D"), "alg": ("none",)},
}
