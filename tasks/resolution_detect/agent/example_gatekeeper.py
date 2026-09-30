"""Example gatekeeper: wake the actor when relevant news or a portfolio
change appears.

This is a runnable template, not a prescription — copy it, then change
the query, the cadence, and the wake conditions to fit your intent.
"""

import json

from datetime import timedelta

from envkit import handover, list_questions, now, search_news, wait, workspace

QUERY = '"officially" OR "confirmed"'  # placeholder: pick your own signal
POLL_EVERY = timedelta(minutes=30)

STATE = workspace() / "gatekeeper_seen.json"
seen = set(json.loads(STATE.read_text())) if STATE.exists() else set()

while True:
    t = now()
    matched = []

    page = search_news(
        q=QUERY,
        date_from=(t - POLL_EVERY).strftime("%Y-%m-%dT%H:%M:%SZ"))
    for hit in page["results"]:
        if hit["news_id"] not in seen:
            seen.add(hit["news_id"])
            matched.append({"kind": "news", "news_id": hit["news_id"],
                            "title": hit["title"]})

    idx = list_questions(added_after=(t - POLL_EVERY).strftime(
        "%Y-%m-%dT%H:%M:%SZ"))
    for row in idx["questions"]:
        matched.append({"kind": "new_question",
                        "question_id": row["question_id"],
                        "question": row["question"]})

    STATE.write_text(json.dumps(sorted(seen)))
    if matched:
        handover({"matched": matched})

    wait(t + POLL_EVERY)
