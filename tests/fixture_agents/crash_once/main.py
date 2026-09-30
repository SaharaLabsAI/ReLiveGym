"""Fixture: crashes on its first invocation, runs clean afterwards; the next
trigger must arrive with kind=crash_recovery."""
import json
import os
import sys

TRIG = json.loads(os.environ["ENV_TRIGGER"])

with open("invocations.jsonl", "a") as f:
    f.write(json.dumps(TRIG) + "\n")

if not os.path.exists("crashed_once"):
    open("crashed_once", "w").close()
    raise RuntimeError("deliberate first-run crash")
