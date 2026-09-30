"""Deterministic record-view helpers shared by the reflection render
(runtime.reflect) and the formatted alg=memory learned block
(runtime.memory): own/other classification, per-status counts, the
per-entity outcome ledger, and id->meaning enrichment. Program library.

Task semantics enter only through the records-module hooks documented in
reflect.py; this module knows no task vocabulary.
"""

from __future__ import annotations

import json


def is_own(r: dict, is_own_fn=None) -> bool:
    """A record is the agent's own action when an action linked at intake
    or the task's is_own predicate says so."""
    return bool(r.get("action")) or bool(is_own_fn and is_own_fn(r))


def status_counts(records: list[dict], stratum_of,
                  is_own_fn=None) -> str:
    """Per-status counts on two labeled lines — own actions and other
    outcomes counted separately so neither is silently absent (review
    finding 4: the old single line dropped own alerts entirely)."""
    own: dict[str, int] = {}
    other: dict[str, int] = {}
    for r in records:
        bucket = own if is_own(r, is_own_fn) else other
        s = stratum_of(r)
        bucket[s] = bucket.get(s, 0) + 1
    fmt = lambda d: (", ".join(f"{k}: {n}" for k, n in sorted(d.items()))
                     or "none")  # noqa: E731
    return (f"own actions settled — {fmt(own)}\n"
            f"other outcomes settled — {fmt(other)}")


def entity_ledger(records: list[dict], stratum_of, entity_of,
                  label_for=None, max_rows: int | None = None,
                  noun: str = "entity") -> str:
    """Cumulative per-entity outcome counts (review finding 1): one line
    per ledger key (the task's entity_of hook) seen in any outcome, so an
    outcome settled on a quiet day stays visible in every later render.
    label_for(key) -> display label (default: the key itself). With
    max_rows, the top rows by record count render and the rest roll up
    into one factual line."""
    per: dict[str, dict[str, int]] = {}
    for r in records:
        key = entity_of(r)
        if not key:
            continue
        row = per.setdefault(str(key), {})
        s = stratum_of(r)
        row[s] = row.get(s, 0) + 1
    if not per:
        return "(no outcomes yet)"
    keys = sorted(per, key=lambda k: (-sum(per[k].values()), k))
    shown = keys if max_rows is None else keys[:max_rows]
    lines = [
        f"{label_for(key) if label_for else key}: "
        + ", ".join(f"{k} {n}" for k, n in sorted(per[key].items()))
        for key in shown]
    rest = keys[len(shown):]
    if rest:
        plural = "entities" if noun == "entity" else noun + "s"
        lines.append(f"(the other {len(rest)} {plural} hold "
                     f"{sum(sum(per[k].values()) for k in rest)} records)")
    return "\n".join(lines)


def enrich(records: list[dict], ctx: dict, enrich_fn=None) -> list[dict]:
    """Render copies of the records, bare ids resolved into meaning by
    the task's enrich hook. Records on disk stay
    id-keyed and compact; only the render is enriched — the hook mutates
    a deep copy, never the original. No hook: plain copies."""
    out = []
    for r in records:
        r = json.loads(json.dumps(r))  # deep copy; render-only mutation
        if enrich_fn:
            enrich_fn(r, ctx)
        out.append(r)
    return out
