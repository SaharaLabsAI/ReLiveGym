"""Fixture: wipes its own crontab on every run (buggy self-edit simulation);
the env's fallback midnight trigger must keep waking it."""
import json
import os
import urllib.request

BASE = os.environ["ENV_URL"]
TRIG = json.loads(os.environ["ENV_TRIGGER"])
HEADERS = {"Authorization": "Bearer " + os.environ["ENV_TOKEN"],
           "Content-Type": "application/json"}

req = urllib.request.Request(BASE + "/call/set_crontab",
                             data=json.dumps({"entries": []}).encode(),
                             method="POST", headers=HEADERS)
urllib.request.urlopen(req).read()

with open("invocations.jsonl", "a") as f:
    f.write(json.dumps(TRIG) + "\n")
