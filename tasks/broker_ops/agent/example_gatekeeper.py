"""example_gatekeeper.py — a minimal, runnable watcher for the broker's
account-events API.

Copy and edit it (with your file tools), dry-run with
run_program(path, validate=true), then run_program(path). The pattern:
sleep to the next poll time, list new events with plain HTTP, hand over
when a fill or a margin call (or any other event you care about)
arrived — you then sign in and act yourself. The API and the inbox are
reachable from programs; the broker portal is not.

NB: `envkit` exists only inside your run_program workspace — this file is
a template, not an importable module of the repo.
"""

import json
import urllib.request
from datetime import timedelta

from envkit import handover, now, wait, workspace

WS = workspace()
params = {"api_url": "http://127.0.0.1:0",      # EDIT ME: ${api_url} from INSTRUCTION.md
          "poll_minutes": 60,
          "kinds": ["fill", "margin_call"],      # what is worth a handover
          "max_polls": 2000}
pfile = WS / "gatekeeper_params.json"
if pfile.exists():
    params.update(json.loads(pfile.read_text()))
state_file = WS / "gatekeeper_state.json"
state = json.loads(state_file.read_text()) if state_file.exists() else {"since": None}

for _ in range(int(params["max_polls"])):
    url = f"{params['api_url']}/api/events"
    if state["since"]:
        url += f"?since={state['since']}"
    with urllib.request.urlopen(url, timeout=30) as r:
        data = json.load(r)
    new = [e for e in data["events"] if e["kind"] in params["kinds"]]
    state["since"] = data["now"]
    state_file.write_text(json.dumps(state))
    if new:
        handover({"woke_at": data["now"], "events": new})  # -> your turn
    wait(now() + timedelta(minutes=float(params["poll_minutes"])))
handover({"woke_at": now().strftime("%Y-%m-%dT%H:%M:%SZ"), "events": []})
