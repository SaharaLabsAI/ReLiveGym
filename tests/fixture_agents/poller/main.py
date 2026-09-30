"""Fixture: a hand-rolled daily poller (pre-scaffold version of agent 1).
On cron triggers: fetch today's temps, notify if any hour >= 33 C."""
import json
import os
import urllib.request

BASE = os.environ["ENV_URL"]
TRIG = json.loads(os.environ["ENV_TRIGGER"])
HEADERS = {"Authorization": "Bearer " + os.environ["ENV_TOKEN"],
           "Content-Type": "application/json"}


def call(name, **args):
    req = urllib.request.Request(BASE + "/call/" + name,
                                 data=json.dumps(args).encode(),
                                 method="POST", headers=HEADERS)
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


call("set_crontab", entries=[{"id": "daily", "cron_expr": "0 23 * * *"}])

if TRIG["id"] == "daily":
    today = call("get_time")["now"][:10]
    res = call("get_weather", start=f"{today}T00:00:00Z",
               end=f"{today}T23:00:00Z")
    temps = res["hourly"]["temperature_2m"]
    if temps and max(temps) >= 33.0:
        call("notify", date=today)
