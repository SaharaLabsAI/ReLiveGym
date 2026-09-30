"""Append-only event ledger + cost accumulators.

Every API interaction is one JSONL event carrying sim time, real time, type, the
dollar cost booked, and free-form detail. results.json is derivable from this file
alone.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from harness.timeutil import iso


class Ledger:
    def __init__(self, path: str | Path | None = None, *, keep_events: bool = True):
        # keep_events=False: write-through only. Used for llm_log, whose events
        # carry the full request body (the agent's whole transcript) — keeping
        # every call's copy alive made memory O(calls x context), 2+ GB on a
        # long ReACT run. cost_by_type/count_by_type see nothing in that mode.
        self._path = Path(path) if path else None
        self._file = open(self._path, "a", encoding="utf-8") if self._path else None
        self._keep = keep_events
        self.events: list[dict] = []
        self._seq = 0

    def append(self, type: str, sim_time: datetime, cost: float = 0.0, **detail) -> dict:
        self._seq += 1
        event = {
            "seq": self._seq,
            "real_time": datetime.now(timezone.utc).isoformat(),
            "sim_time": iso(sim_time),
            "type": type,
            "cost": round(cost, 8),
            **detail,
        }
        if self._keep:
            self.events.append(event)
        if self._file:
            self._file.write(json.dumps(event) + "\n")
            self._file.flush()
        return event

    def load_existing(self) -> int:
        """Preload the rows already in the file — a resumed run's copied
        history (harness/checkpoint.py) — so aggregation covers the whole
        chain and `seq` continues where the parent stopped. Returns the
        number of rows loaded."""
        if self._path is None or not self._path.exists():
            return 0
        n = 0
        with open(self._path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                event = json.loads(line)
                if self._keep:
                    self.events.append(event)
                self._seq = max(self._seq, int(event.get("seq") or 0))
                n += 1
        return n

    # -- aggregation ----------------------------------------------------------------

    def cost_by_type(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for e in self.events:
            out[e["type"]] = round(out.get(e["type"], 0.0) + e["cost"], 8)
        return out

    def count_by_type(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for e in self.events:
            out[e["type"]] = out.get(e["type"], 0) + 1
        return out

    def total_cost(self) -> float:
        return round(sum(e["cost"] for e in self.events), 8)

    def monthly_cost_buckets(self) -> dict[str, float]:
        """Booked cost bucketed by sim-time calendar month (adaptation metrics)."""
        out: dict[str, float] = {}
        for e in self.events:
            if e["cost"]:
                key = e["sim_time"][:7]  # YYYY-MM
                out[key] = round(out.get(key, 0.0) + e["cost"], 8)
        return out

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None
