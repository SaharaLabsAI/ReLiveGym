"""example_gatekeeper.py — a minimal, runnable wake-condition program.

Copy and edit it (write_file / edit_file), dry-run with
run_program(path, validate=true), then run_program(path) — wake times
live in code (wait(now() + interval)); `until` is only a hard timeout
backstop. The pattern: poll cheaply, compute features for free, hand
over ONLY the candidates worth an LLM turn. Params live in a file, not in code, so a
future run is retunable by editing data (the file bus works both ways).

NB: `envkit` exists only inside your run_program workspace — this file is
a template, not an importable module of the repo.
"""

import json
from datetime import timedelta

from envkit import get_cascade, handover, list_posts, now, wait, workspace

WS = workspace()
params = {"poll_minutes": 30, "window_hours": 2, "min_comments": 6}
pfile = WS / "gatekeeper_params.json"
if pfile.exists():
    params.update(json.loads(pfile.read_text()))

sfile = WS / "gatekeeper_seen.json"  # posts already escalated
seen = set(json.loads(sfile.read_text())) if sfile.exists() else set()

while True:
    lo = now() - timedelta(hours=params["window_hours"])
    page = list_posts(since=lo.strftime("%Y-%m-%dT%H:%M:%SZ"), order="desc")
    hot = []
    for p in page["posts"]:  # young posts only; velocity = comments so far
        if p["id"] in seen:
            continue
        c = get_cascade(root_id=p["id"])  # paid: the reply tree so far
        if c["n_comments"] >= params["min_comments"]:
            seen.add(p["id"])
            hot.append({"root_id": p["id"], "title": p["title"],
                        "subreddit": p["subreddit"],
                        "posted_at": p["posted_at"],
                        "comments_so_far": c["n_comments"]})
    sfile.write_text(json.dumps(sorted(seen)))
    if hot:
        handover({"candidates": hot})  # -> back to your ReACT turn
    wait(now() + timedelta(minutes=params["poll_minutes"]))
# no deadline handling needed: a trigger or run_program's `until` exits
