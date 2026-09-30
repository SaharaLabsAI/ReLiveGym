"""Temperature monitoring baseline: daily poller (no LLM).

Task: send one notification per day, promptly after the first hour whose
temperature reaches the threshold; the full spec and price table is in
INSTRUCTION.md (same directory).

Strategy: once a day at 23:00 UTC, fetch today's hourly temperatures and
notify if any hour crossed the threshold.
"""

import json
import os
from datetime import timedelta

from runtime.env_client import Env, iso

THRESHOLD_C = 33.0

env = Env()
trigger = json.loads(os.environ.get("ENV_TRIGGER", "{}"))

env.call("set_crontab",
         entries=[{"id": "daily_check", "cron_expr": "0 23 * * *"}])

if trigger.get("id") == "daily_check":
    now = env.now()
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    res = env.call("get_weather", start=iso(day_start),
                   end=iso(day_start + timedelta(hours=23)))
    temps = res["hourly"]["temperature_2m"]
    if temps and max(temps) >= THRESHOLD_C:
        env.call("notify", date=iso(now)[:10])
