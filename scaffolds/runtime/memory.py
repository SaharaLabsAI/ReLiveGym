"""Raw outcome memory: append-only records + token-capped render + tools.
Program library.

Files live under memory/ in the workspace:

  memory/records.jsonl    append-only, one outcome record per line

Feedback intake is append-on-arrival and code-side in every learning
cell: tool results that carry outcome records are appended the
moment they arrive — the agent cannot forget to remember. `auto_append` /
`wrap_feedback` install that wrapper on feedback tools; `pull_oracle` is
the deterministic drain used by cron-style programs.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import trace
from .env_client import iso
from .tokens import BUDGET_TOKENS, count_tokens

BLOCK_TOKENS = BUDGET_TOKENS  # set by the program from its cell config

MEMORY_DIR = Path("memory")
RECORDS = MEMORY_DIR / "records.jsonl"

LEGEND = ""  # task-provided factual glossary rendered under the block
#   header — record kinds, field meanings, stream filters. Set by the
#   generated main from the task's records.BLOCK_LEGEND (alg=memory
#   cells); raw records alone can't say what they are or what was
#   filtered out of the stream.


# -- store ----------------------------------------------------------------------------


def append_record(t: str, src: str, outcome: dict, action: dict | None = None,
                  note: str | None = None) -> dict:
    """Append one memory record."""
    record = {"t": t, "src": src, "outcome": outcome,
              "action": action, "note": note}
    MEMORY_DIR.mkdir(exist_ok=True)
    with open(RECORDS, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return record


def read_records(limit: int | None = None) -> list[dict]:
    if not RECORDS.exists():
        return []
    with open(RECORDS, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return rows[-limit:] if limit else rows


# -- render ---------------------------------------------------------------------------


def _render_record(r: dict) -> str:
    """One compact line per record; drop nulls to save budget."""
    slim = {k: v for k, v in r.items() if v is not None}
    return json.dumps(slim, separators=(",", ":"))


def render_block_raw(budget_tokens: int | None = None) -> str:
    """The raw-memory learned block: newest records that fit (whole records
    only), rendered oldest-first. Fallback render when no records module
    is bound (pre-formatted-render arms; tasks without line hooks)."""
    if budget_tokens is None:
        budget_tokens = BLOCK_TOKENS
    lines: list[str] = []
    used = 0
    for r in reversed(read_records()):
        line = _render_record(r)
        cost = count_tokens(line) + 1
        if used + cost > budget_tokens:
            break
        lines.append(line)
        used += cost
    if not lines:
        return ""
    body = "\n".join(reversed(lines))
    head = "## What you have learned so far (raw outcomes)"
    if LEGEND:
        head += f"\n{LEGEND}"
    return f"{head}\n{body}"


# -- formatted render ----------------------
#
# The deterministic formatted learned block: cumulative statistics over
# ALL records ever, then readable per-record lines under budget-driven
# truncation. Same view the skills-tier reflection consumes, rendered to
# the actor with no LLM step. Pure function of records.jsonl (+ the
# enrichment context); no randomness, no wall clock.

RECORDS_MOD = None  # task records module; wired by memory.bind (alg=memory)
ENV = None
STATE = None

ENTITY_ROWS = 12   # per-entity table rows before rollup
OWN_SHARE = 0.45   # record-budget share reserved for own actions
FULL_SHARE = 0.60  # non-own budget share rendered at full resolution
STRATUM_FLOOR = 2  # newest records per stratum guaranteed a line

_NOTE_RESERVE = 40  # tokens held back for an elision note


def bind(records_mod, env, state: dict) -> None:
    """Wire the task semantics (records-module hooks), env, and state into
    the formatted render. Emitted by the constructor / task mains in
    alg=memory cells; unbound, render_block falls back to the raw JSONL
    render."""
    global RECORDS_MOD, ENV, STATE, LEGEND
    RECORDS_MOD = records_mod
    ENV = env
    STATE = state
    LEGEND = getattr(records_mod, "BLOCK_LEGEND", "")


def _cost(line: str) -> int:
    return count_tokens(line) + 1


def _note_line(r: dict) -> str:
    return f"{(r.get('t') or '?')[:16].replace('T', ' ')} | note | " \
        + " ".join(str(r.get("note") or "").split())


def _stat_lines(records: list[dict], is_own_fn) -> list[str]:
    """The task-declared numeric aggregates (records.STAT_FIELDS: tuples
    (outcome field, scope own|other|all, aggs among n/sum/mean)),
    computed over records whose outcome carries the field numerically."""
    from . import view

    out = []
    for field, scope, aggs in getattr(RECORDS_MOD, "STAT_FIELDS", []):
        vals = []
        for r in records:
            if scope == "own" and not view.is_own(r, is_own_fn):
                continue
            if scope == "other" and view.is_own(r, is_own_fn):
                continue
            v = (r.get("outcome") or {}).get(field)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                vals.append(v)
        if not vals:
            continue
        parts = []
        if "n" in aggs:
            parts.append(f"n {len(vals)}")
        if "sum" in aggs:
            parts.append(f"sum {round(sum(vals), 3)}")
        if "mean" in aggs:
            parts.append(f"mean {round(sum(vals) / len(vals), 3)}")
        which = {"own": "own records", "other": "other records",
                 "all": "records"}[scope]
        out.append(f"{field} — " + ", ".join(parts)
                   + f" (over {which} carrying the field)")
    return out


def _pack_newest(rows: list[str], budget: int) -> tuple[list[str], int, int]:
    """Newest-first whole-line packing of chronological rows; returns
    (kept rows oldest-first, tokens used, rows elided). When elision is
    forced, the note reserve is taken off the budget first so the
    elision line itself always fits."""
    total = sum(_cost(x) for x in rows)
    if total <= budget:
        return list(rows), total, 0
    budget = max(0, budget - _NOTE_RESERVE)
    kept: list[str] = []
    used = 0
    for row in reversed(rows):
        c = _cost(row)
        if used + c > budget:
            break
        kept.append(row)
        used += c
    kept.reverse()
    return kept, used + _NOTE_RESERVE, len(rows) - len(kept)


def _others_lines(others: list[dict], full_fn, compact_fn, stratum_of,
                  budget: int) -> tuple[list[str], int, int]:
    """Two-resolution packing of non-own outcomes: newest
    records as full lines until FULL_SHARE of the budget is used, older
    ones as compact lines, with the newest STRATUM_FLOOR records of every
    stratum guaranteed at least a compact line while budget admits.
    Returns (lines oldest-first, tokens used, records elided)."""
    n = len(others)
    fulls = [full_fn(r) for r in others]
    compacts = [compact_fn(r) for r in others]
    total_full = sum(_cost(x) for x in fulls)
    if total_full <= budget:
        return list(fulls), total_full, 0
    budget = max(0, budget - _NOTE_RESERVE)
    floor: set[int] = set()
    seen: dict[str, int] = {}
    for i in range(n - 1, -1, -1):
        s = stratum_of(others[i])
        if seen.get(s, 0) < STRATUM_FLOOR:
            floor.add(i)
            seen[s] = seen.get(s, 0) + 1
    pending = sum(_cost(compacts[i]) for i in floor)  # unmet floor debt
    full_budget = int(budget * FULL_SHARE)
    chosen: dict[int, str] = {}
    used = full_used = 0
    for i in range(n - 1, -1, -1):
        mine = _cost(compacts[i]) if i in floor else 0
        owed = pending - mine  # debt still owed to OTHER floor records
        line = fulls[i] if full_used < full_budget else compacts[i]
        c = _cost(line)
        if used + c + owed > budget:  # degrade to the compact tier
            line, c = compacts[i], _cost(compacts[i])
        pending -= mine  # this record's own debt is settled either way
        if used + c + owed > budget:
            continue  # elided; the counts above cover it
        chosen[i] = line
        used += c
        if line is fulls[i]:
            full_used += c
    lines = [chosen[i] for i in sorted(chosen)]
    return lines, used + _NOTE_RESERVE, n - len(chosen)


def _elision(n: int, total: int) -> str:
    return f"(oldest {n} of {total} omitted; the counts above cover them)"


def render_block(budget_tokens: int | None = None) -> str:
    """The formatted learned block: cumulative counts, numeric
    aggregates and the per-entity ledger over ALL records, then non-own
    outcomes in two resolution tiers and every own action as a readable
    line, all inside the token budget. Falls back to the raw JSONL render
    when no records module with line hooks is bound."""
    from . import view

    if RECORDS_MOD is None or not hasattr(RECORDS_MOD, "digest_line"):
        return render_block_raw(budget_tokens)
    budget = BLOCK_TOKENS if budget_tokens is None else budget_tokens
    raw = read_records()
    if not raw:
        return ""
    order = sorted(range(len(raw)), key=lambda i: (raw[i].get("t") or "", i))
    raw = [raw[i] for i in order]

    ctx: dict = {}
    enrich_fn = getattr(RECORDS_MOD, "enrich", None)
    ctx_fn = getattr(RECORDS_MOD, "render_context", None)
    if enrich_fn and ctx_fn and ENV is not None:
        try:
            ctx = ctx_fn(ENV, STATE if STATE is not None else {})
        except Exception as e:  # the block must never crash acting
            trace.log(raw[-1].get("t") or "", "note",
                      what="render_context_failed", error=str(e))
    records = view.enrich(raw, ctx, enrich_fn)

    notes = [r for r in records if r.get("src") == "note"]
    scored = [r for r in records if r.get("src") != "note"]
    is_own_fn = getattr(RECORDS_MOD, "is_own", None)
    own = [r for r in scored if view.is_own(r, is_own_fn)]
    others = [r for r in scored if not view.is_own(r, is_own_fn)]
    stratum_of = RECORDS_MOD.stratum_of

    # -- head: header + legend + counts (+ notes count + numeric stats)
    head = ["## What you have learned so far (settled outcomes)"]
    if LEGEND:
        head.append(LEGEND)
    head += ["",
             f"### Cumulative counts (all {len(scored)} settled records, "
             "including any not shown below)",
             view.status_counts(scored, stratum_of, is_own_fn)]
    if notes:
        head.append(f"notes you saved — {len(notes)}")
    stat_lines = _stat_lines(scored, is_own_fn)
    label_fn = getattr(RECORDS_MOD, "entity_label", None)
    entity_of = getattr(RECORDS_MOD, "entity_of", None)
    noun = getattr(RECORDS_MOD, "ENTITY_NOUN", "entity")
    ledger = None
    if entity_of:
        ledger = ["", f"### Cumulative outcomes by {noun}",
                  view.entity_ledger(
                      scored, stratum_of, entity_of,
                      label_for=(lambda k: label_fn(k, ctx)) if label_fn
                      else None,
                      max_rows=ENTITY_ROWS, noun=noun)]
    head_cost = sum(_cost(x) for x in head)
    stats_cost = sum(_cost(x) for x in stat_lines)
    ledger_cost = sum(_cost(x) for x in ledger) if ledger else 0
    if head_cost + stats_cost + ledger_cost <= budget:
        head += stat_lines + (ledger or [])
        used = head_cost + stats_cost + ledger_cost
    elif head_cost + stats_cost <= budget:  # drop the ledger first
        head += stat_lines
        used = head_cost + stats_cost
    else:  # then the numeric lines; counts are the last thing standing
        used = head_cost

    # -- record sections under the remaining budget (section scaffolding
    # and the possible "(none yet)" placeholders are budgeted up front so
    # the block stays hard-bounded at any budget)
    scaffold = ["", "### Settled outcomes (history)", "",
                "### Your actions and how they settled",
                "(none yet)", "(none yet)"]
    remainder = budget - used - sum(_cost(x) for x in scaffold)
    if remainder < 0:  # statistics-only block: the counts still cover
        return "\n".join(head)  # every record
    own_rows = [(r, _note_line(r) if r.get("src") == "note"
                 else _own_line(r)) for r in _merge(own, notes)]
    other_full = getattr(RECORDS_MOD, "outcome_line", RECORDS_MOD.digest_line)

    own_budget = int(remainder * OWN_SHARE)
    own_lines, own_used, own_elided = _pack_newest(
        [x for _, x in own_rows], own_budget)
    other_budget = remainder - own_used
    other_lines, other_used, other_elided = _others_lines(
        others, other_full, RECORDS_MOD.digest_line, stratum_of,
        other_budget)
    if own_elided and other_budget - other_used > 0:  # spill back to own
        own_lines, own_used, own_elided = _pack_newest(
            [x for _, x in own_rows],
            own_budget + (other_budget - other_used))

    out = list(head)
    out += ["", "### Settled outcomes (history)"]
    if other_elided:
        out.append(_elision(other_elided, len(others)))
    out += other_lines or ["(none yet)"]
    out += ["", "### Your actions and how they settled"]
    if own_elided:
        out.append(_elision(own_elided, len(own_rows)))
    out += own_lines or ["(none yet)"]
    return "\n".join(out)


def _own_line(r: dict) -> str:
    fn = getattr(RECORDS_MOD, "action_line", RECORDS_MOD.digest_line)
    return fn(r)


def _merge(own: list[dict], notes: list[dict]) -> list[dict]:
    return sorted(own + notes, key=lambda r: r.get("t") or "")


# -- agent tools ----------------------------------------------------------------------


def memory_tools(env) -> dict[str, dict]:
    return {
        "memory_read": {
            "doc": "memory_read() -> your accumulated outcome records "
                   "(token-capped)",
            "fn": lambda args: render_block(),
        },
        "memory_append": {
            "doc": "memory_append(note: str) -> save a free-text note to "
                   "memory",
            "fn": lambda args: append_record(
                t=iso(env.now()), src="note", outcome={},
                note=str(args["note"])),
        },
    }


def read_tools() -> dict[str, dict]:
    """Read-only memory access (e.g. for a reflector agent)."""
    return {
        "memory_read": {
            "doc": "memory_read() -> your accumulated outcome records "
                   "(token-capped)",
            "fn": lambda args: render_block(),
        },
    }


# -- append-on-arrival ----------------------------------------------------------------


def wrap_feedback(tool: dict, env, src: str, extract) -> dict:
    """Wrap one feedback tool so returned outcome records are appended to
    memory the moment they arrive."""
    inner = tool["fn"]

    def fn(args):
        result = inner(args)
        t = iso(env.now())
        for out in extract(result) or []:
            append_record(t=t, src=src, outcome=out)
            trace.log(t, "feedback", src=src, outcome=out)
        return result

    return {**tool, "fn": fn}


def auto_append(tools: dict[str, dict], env, src: str) -> dict[str, dict]:
    """Install the append-on-arrival wrapper on every tool that declares a
    `records` extractor (the shape task feedback_tools use)."""
    out: dict[str, dict] = {}
    for name, tool in tools.items():
        extract = tool.get("records")
        if extract is None:
            out[name] = tool
        else:
            wrapped = wrap_feedback(tool, env, src, extract)
            wrapped.pop("records", None)
            out[name] = wrapped
    return out


# -- deterministic oracle drain (cron-style programs) ---------------------------------


def pull_oracle(env, state: dict, action_for) -> None:
    """Drain new oracle outcomes into memory records; `action_for` links an
    outcome to this agent's own action (task records module)."""
    from .env_client import EnvError

    t = iso(env.now())
    try:
        resp = env.call("get_feedback", **(
            {"since": state["oracle_cursor"]} if state.get("oracle_cursor")
            else {}))
    except EnvError as e:
        trace.log(t, "note", what="oracle_pull_failed", error=str(e))
        return
    for out in resp["outcomes"]:
        action = action_for(out, state)
        append_record(t=t, src="oracle", outcome=out, action=action)
        trace.log(t, "feedback", src="oracle", outcome=out)
    state["oracle_cursor"] = resp["now"]
