"""Fixture (TM-D): installs a daily base crontab on every run, and on the
bootstrap firing exercises the agent-owned schedule CRUD: creates a
recurring + a one-time schedule (with a note), moves the one-time one
with update_schedule, tries to update the read-only base entry (must be
refused), and records every error. The one-time firing then deletes the
recurring schedule. Each invocation's ENV_TRIGGER is appended to
invocations.jsonl."""
import json
import os
import urllib.error
import urllib.request

BASE = os.environ["ENV_URL"]
TRIG = json.loads(os.environ["ENV_TRIGGER"])
HEADERS = {"Authorization": "Bearer " + os.environ["ENV_TOKEN"],
           "Content-Type": "application/json"}


def call(name, **args):
    req = urllib.request.Request(BASE + "/call/" + name,
                                 data=json.dumps(args).encode(),
                                 method="POST", headers=HEADERS)
    try:
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"http_error": e.code, "detail": json.loads(e.read())}


def record(path, obj):
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


record("invocations.jsonl", TRIG)
call("set_crontab", entries=[{"id": "daily", "cron_expr": "0 23 * * *"}])

if TRIG["id"] == "__bootstrap__":
    record("calls.jsonl", call("create_schedule", id="twice",
                               cron_expr="0 */12 * * *"))
    record("calls.jsonl", call("create_schedule", id="check",
                               at="2021-06-01T03:00:00Z",
                               note="look at the forecast again"))
    record("calls.jsonl", call("update_schedule", id="check",
                               at="2021-06-01T06:00:00Z",
                               note="look at the forecast again"))
    record("calls.jsonl", call("update_schedule", id="daily",
                               cron_expr="0 1 * * *"))  # read-only base
    record("calls.jsonl", call("delete_schedule", id="daily"))
    record("calls.jsonl", call("create_schedule", id="learn",
                               cron_expr="0 2 * * *"))  # reserved id
    record("calls.jsonl", call("list_schedules"))
elif TRIG["id"] == "check":
    record("calls.jsonl", call("delete_schedule", id="twice"))
    record("calls.jsonl", call("list_schedules"))
