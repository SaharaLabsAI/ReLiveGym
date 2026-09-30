"""example_gatekeeper.py — a minimal, runnable wake-condition program:
poll news on a cadence, hand over fresh keyword-matching articles.

Copy and edit it (write_file / edit_file), dry-run with
run_program(path, validate=true), then run_program(path). Wake times
live in code — wait(now() + interval); run_program's `until` is only a
hard timeout backstop. State persists in workspace files, so a trigger
interrupting the run loses nothing: just re-run the same program.

NB: `envkit` exists only inside your run_program workspace — this file is
a template, not an importable module of the repo.
"""

import json
from datetime import timedelta

from envkit import handover, now, search_news, wait, workspace

QUERY = '"rate cut" OR "FOMC"'      # BM25: quoted phrases, AND/OR
KEYWORDS = ("cut", "hike", "fed")   # cheap title filter on top of BM25
POLL_EVERY = timedelta(minutes=30)  # cadence lives in code

seen_file = workspace() / "gatekeeper_seen.json"
seen = set(json.loads(seen_file.read_text())) if seen_file.exists() else set()

while True:
    t = now()
    page = search_news(
        q=QUERY,
        date_from=(t - POLL_EVERY).strftime("%Y-%m-%dT%H:%M:%SZ"))
    hits = [h for h in page["results"]
            if h["news_id"] not in seen
            and any(k in h["title"].lower() for k in KEYWORDS)]
    seen |= {h["news_id"] for h in page["results"]}
    seen_file.write_text(json.dumps(sorted(seen)))
    if hits:
        handover({"matched": [
            {"news_id": h["news_id"], "title": h["title"],
             "published": h["published"]} for h in hits[:5]]})
    wait(t + POLL_EVERY)
