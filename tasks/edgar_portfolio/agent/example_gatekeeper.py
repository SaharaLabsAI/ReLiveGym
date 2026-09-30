"""example_gatekeeper.py — a minimal, runnable watcher for the EDGAR mirror.

Copy and edit it (with your file tools), dry-run with
run_program(path, validate=true), then run_program(path). The pattern:
sleep until the next poll time, fetch each held company's filing index
from the mirror with plain HTTP, hand over ONLY when a new earnings 8-K
or periodic report appeared. Params live in a file, not in code, so a
future run is retunable by editing data.

The mirror is reachable from programs; the portfolio sheet is not — you
record the filing yourself after the handover.

NB: `envkit` exists only inside your run_program workspace — this file is
a template, not an importable module of the repo.
"""

import json
import urllib.request
from datetime import timedelta

from envkit import handover, now, wait, workspace

WS = workspace()
params = {"sec_url": "http://127.0.0.1:0",        # EDIT ME: ${sec_url} from INSTRUCTION.md
          "ciks": ["0000320193"],                   # EDIT ME: held CIKs (10 digits)
          "poll_minutes": 60,                       # cadence between polls
          "max_polls": 500}                         # hard stop
pfile = WS / "gatekeeper_params.json"
if pfile.exists():
    params.update(json.loads(pfile.read_text()))
seen_file = WS / "seen_accessions.json"
seen = set(json.loads(seen_file.read_text())) if seen_file.exists() else set()


def filings(cik: str) -> list[dict]:
    with urllib.request.urlopen(f"{params['sec_url']}/submissions/CIK{cik}.json",
                                timeout=30) as r:
        rec = json.load(r)["filings"]["recent"]
    return [{"cik": cik, "accession": a, "form": f, "items": it, "accepted_at": t}
            for a, f, it, t in zip(rec["accessionNumber"], rec["form"],
                                   rec["items"], rec["acceptanceDateTime"])
            if f in ("10-Q", "10-K") or (f == "8-K" and "2.02" in (it or ""))]


for _ in range(int(params["max_polls"])):
    new = []
    for cik in params["ciks"]:
        for row in filings(cik):
            if row["accession"] not in seen:
                seen.add(row["accession"])
                new.append(row)
    seen_file.write_text(json.dumps(sorted(seen)))
    if new and _ > 0:  # the first poll only seeds `seen` with history
        handover({"woke_at": now().strftime("%Y-%m-%dT%H:%M:%SZ"),
                  "new_filings": new})  # -> back to your turn: record them
    wait(now() + timedelta(minutes=float(params["poll_minutes"])))
handover({"woke_at": now().strftime("%Y-%m-%dT%H:%M:%SZ"), "new_filings": []})
