"""Agent-side semantic trace: logs/trace.jsonl.

One line per prediction, action, feedback arrival, reflection, and edit.
This is the log reflection itself reads — keep it parseable. The harness's
llm_log.jsonl (server-side) carries full LLM bodies; this file carries the
semantics: what was predicted, with which learned-block version, and what
came back.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

TRACE = Path("logs") / "trace.jsonl"
_LOCK = threading.Lock()  # concurrent replay episodes append rows


def log(t: str, kind: str, **fields) -> dict:
    """Append one trace event. `t` is sim time (ISO), `kind` is one of
    predict | action | feedback | reflect | edit | note."""
    event = {"t": t, "kind": kind, **fields}
    TRACE.parent.mkdir(exist_ok=True)
    with _LOCK, open(TRACE, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")
    return event


def read(kind: str | None = None, limit: int | None = None) -> list[dict]:
    if not TRACE.exists():
        return []
    with open(TRACE, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if kind is not None:
        rows = [r for r in rows if r.get("kind") == kind]
    return rows[-limit:] if limit else rows
