"""Fixture: daemon-style program. Installs a daily 06:00 crontab, then loops
in the sleep tool asking for 48h naps; due triggers must wake it early
(woke_for=trigger). Records every wake until experiment_over."""
import datetime
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


def record(entry):
    with open("wakes.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")


with open("invocations.jsonl", "a") as f:
    f.write(json.dumps(TRIG) + "\n")

call("set_crontab", entries=[{"id": "daily", "cron_expr": "0 6 * * *"}])

while True:
    now = datetime.datetime.fromisoformat(
        call("get_time")["now"].replace("Z", "+00:00"))
    until = (now + datetime.timedelta(hours=48)).strftime("%Y-%m-%dT%H:%M:%SZ")
    res = call("sleep", until=until)
    record(res)
    if res.get("experiment_over"):
        break
