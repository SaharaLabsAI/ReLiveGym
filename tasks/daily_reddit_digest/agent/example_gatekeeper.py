"""example_gatekeeper.py — a minimal, runnable wake-condition program.

Copy and edit it (write_file / edit_file), dry-run with
run_program(path, validate=true), then run_program(path) — wake times
live in code (wait(<datetime>)); `until` is only a hard timeout backstop.
The pattern: sleep until a time read from a params file, fetch one page,
hand over ONLY what is worth an LLM turn. Params live in a file, not in
code, so a future run is retunable by editing data (the file bus works
both ways).

NB: `envkit` exists only inside your run_program workspace — this file is
a template, not an importable module of the repo.
"""

import json
from datetime import datetime, timedelta, timezone

from envkit import handover, list_posts, now, wait, workspace

WS = workspace()
params = {"wake_at": None, "window_hours": 24}
pfile = WS / "gatekeeper_params.json"
if pfile.exists():
    params.update(json.loads(pfile.read_text()))

if params["wake_at"]:  # ISO UTC, e.g. "2026-04-02T09:00:00Z"
    target = datetime.fromisoformat(params["wake_at"].replace("Z", "+00:00"))
    if target.tzinfo is None:
        target = target.replace(tzinfo=timezone.utc)
    if target > now():
        wait(target)

lo = now() - timedelta(hours=params["window_hours"])
page = list_posts(since=lo.strftime("%Y-%m-%dT%H:%M:%SZ"), order="desc")
handover({"woke_at": now().strftime("%Y-%m-%dT%H:%M:%SZ"),
          "total_hits": page["total_hits"],
          "posts": [{"root_id": p["id"], "title": p["title"],
                     "subreddit": p["subreddit"], "posted_at": p["posted_at"],
                     "num_comments": p["num_comments"]}
                    for p in page["posts"]]})  # -> back to your ReACT turn
# no deadline handling needed: a trigger or run_program's `until` exits
