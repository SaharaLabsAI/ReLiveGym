"""Fixed baseline: uniform standing forecast, submitted once per question.

Polls the question index hourly; any question not yet forecast gets the
even split over its outcomes. No news, no LLM, no updates — the "showed
up" anchor every other condition is read against (time-averaged score of
an even binary split is 0.5 minus the pre-first-poll sliver).
"""

import json
import os

from runtime.env_client import Env, EnvError, iso
from runtime.state import load_state, save_state

env = Env()
trigger = json.loads(os.environ.get("ENV_TRIGGER", "{}"))
env.call("set_crontab", entries=[{"id": "sweep", "cron_expr": "0 * * * *"}])

state = load_state()
done = set(state.get("done", []))

if trigger.get("id") == "sweep" or not done:
    cursor = state.get("cursor")
    kwargs = {"status": "open"}
    if cursor:
        kwargs["added_after"] = cursor
    rows = env.call("list_questions", **kwargs)["questions"]
    for row in rows:
        qid = row["question_id"]
        if qid in done:
            continue
        outcomes = env.call("get_question", question_id=qid)["outcomes"]
        forecast = {o: 1.0 / len(outcomes) for o in outcomes}
        try:
            env.call("submit_forecast", question_id=qid, forecast=forecast)
        except EnvError:
            pass  # rejections are free (e.g. resolved between calls)
        done.add(qid)
    if rows:
        state["cursor"] = max(r["added_at"] for r in rows)
    state["done"] = sorted(done)
    save_state(state)
