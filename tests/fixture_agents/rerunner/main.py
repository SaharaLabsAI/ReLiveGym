"""Fixture: the self-modification relaunch path — asks to be re-run immediately
at the same sim time via run_at(now), once."""
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

if TRIG["id"] == "__bootstrap__":
    now = call("get_time")["now"]
    call("run_at", id="again", at=now)
