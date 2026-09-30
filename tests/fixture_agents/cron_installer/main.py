"""Fixture: installs a daily crontab on every run (idempotent) and records
each invocation's ENV_TRIGGER."""
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


with open("invocations.jsonl", "a") as f:
    f.write(json.dumps(TRIG) + "\n")

call("set_crontab", entries=[{"id": "daily", "cron_expr": "0 23 * * *"}])
