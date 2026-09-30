"""The actor's trajectory, rendered for its reviewer. Program library, task-free.

A deterministic read of logs/history_<agent>.jsonl (the append-only twin
of the transcript) split into wake segments: every wake gets one digest
line (tool sequence, each action with what the actor had seen about its
target, the wait it chose), days get a roll-up line, and the first wake
plus the newest VERBATIM_WAKES are rendered as the actor's own messages
with tool results clipped. The whole render is packed under
BUDGET_TOKENS: digest lines are never dropped before verbatim wakes.

What is task-specific comes from the records module's TRAJECTORY spec
(which tool is the scored action, how a listing names items, which
fields hold an item's time / count / label / group) and its optional
`call_line(tool, args, result, seen)` hook that summarizes one tool
result in a line; everything else is generic.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from statistics import median

from .env_client import parse_iso
from .tokens import count_tokens

VERBATIM_WAKES = 6  # newest wakes rendered verbatim (agent.reflect_trajectory_wakes)
BUDGET_TOKENS = 20_000  # the whole render (agent.reflect_trajectory_tokens)
RESULT_CHARS = 300  # tool-result clip in verbatim wakes (agent.reflect_trajectory_result_chars)

WAIT_TOOLS = ("sleep", "wait_until", "run_program")
BRIEF_RE = re.compile(r"LLM budget: \$([0-9.]+) spent")

DEFAULT_SPEC = {"action_tool": None, "id_arg": "id", "list_key": "items",
                "id_key": "id", "time_key": "posted_at",
                "count_key": None, "label_key": "title", "group_key": None,
                "group_prefix": ""}


# -- parse ------------------------------------------------------------------------------


def read_history(agent_name: str) -> list[dict]:
    path = Path("logs") / f"history_{agent_name}.jsonl"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _wake_of(content: str) -> tuple[datetime, str] | None:
    if not content.startswith("(woke at "):
        return None
    head = content.split("\n", 1)[0]
    m = re.match(r"^\(woke at (\S+) (?:from (\w+)|for trigger ([^)]*))\)", head)
    if not m:
        return None
    try:
        t = parse_iso(m.group(1))
    except ValueError:
        return None
    return t, (m.group(2) or f"trigger {m.group(3)}")


def _json(text: str):
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def segments(messages: list[dict]) -> list[dict]:
    """Wake segments: {t, source, marker, msgs, turns:[{tool, args,
    thought, result}]} — a turn is one assistant call and the user
    message that answered it (None when the wake ended first)."""
    segs: list[dict] = []
    cur: dict | None = None
    pending: dict | None = None
    for m in messages:
        role, content = m.get("role"), m.get("content") or ""
        if role == "user":
            w = _wake_of(content)
            if w is not None:
                cur = {"t": w[0], "source": w[1], "marker": content,
                       "msgs": [m], "turns": []}
                segs.append(cur)
                pending = None
                continue
            if cur is None:
                continue
            cur["msgs"].append(m)
            if pending is not None:
                pending["result"] = _json(content)
                pending["result_text"] = content
                pending = None
        elif role == "assistant" and cur is not None:
            cur["msgs"].append(m)
            call = _json(content)
            if isinstance(call, dict) and "tool" in call:
                pending = {"tool": call.get("tool"),
                           "args": call.get("args") or {},
                           "thought": call.get("thought"),
                           "result": None, "result_text": None}
            else:
                pending = {"tool": None, "args": {}, "thought": content,
                           "result": None, "result_text": None}
            cur["turns"].append(pending)
    return segs


# -- digest ----------------------------------------------------------------------------


def _hhmm(t: datetime) -> str:
    return t.strftime("%m-%d %H:%M")


def _hours(a: datetime, b: datetime) -> float:
    return round((b - a).total_seconds() / 3600, 1)


def _seen_items(seen: dict, result, spec: dict) -> None:
    """Register every listed / fetched item of a tool result: what the
    actor had in front of it when it acted."""
    if not isinstance(result, dict):
        return
    items = result.get(spec["list_key"])
    if isinstance(items, list):
        for it in items:
            if isinstance(it, dict) and it.get(spec["id_key"]) is not None:
                seen[str(it[spec["id_key"]])] = it
    if result.get(spec["id_arg"]) is not None and spec["time_key"] in result:
        seen[str(result[spec["id_arg"]])] = result


def _item_note(item_id: str, t: datetime, seen: dict, spec: dict) -> str:
    it = seen.get(item_id)
    if not it:
        return f"{item_id} (not in anything it had listed)"
    parts = []
    if spec["group_key"] and it.get(spec["group_key"]):
        parts.append(f"{spec['group_prefix']}{it[spec['group_key']]}")
    if it.get(spec["time_key"]):
        try:
            parts.append(f"{_hours(parse_iso(it[spec['time_key']]), t)}h old")
        except ValueError:
            pass
    if spec["count_key"] and it.get(spec["count_key"]) is not None:
        parts.append(f"{it[spec['count_key']]}c")
    label = " ".join(str(it.get(spec["label_key"]) or "").split())
    if label:
        parts.append('"' + (label[:40] + "…" if len(label) > 40 else label) + '"')
    return " ".join(parts) or item_id


def _age_hours(item_id: str, t: datetime, seen: dict, spec: dict) -> float | None:
    it = seen.get(item_id)
    if not it or not it.get(spec["time_key"]):
        return None
    try:
        return _hours(parse_iso(it[spec["time_key"]]), t)
    except ValueError:
        return None


def digest_segment(seg: dict, spec: dict) -> dict:
    """One wake -> {line, n_actions, n_accepted, ages, wait_hours,
    data_calls}."""
    t = seg["t"]
    seen: dict = {}
    tokens: list[list] = []  # [name, count]
    actions: list[str] = []
    ages: list[float] = []
    n_actions = n_accepted = data_calls = 0
    wait_hours = None
    wait_desc = None

    def flush_actions():
        if actions and tokens:
            tokens[-1][0] += " [" + "; ".join(actions) + "]"
            actions.clear()

    for turn in seg["turns"]:
        tool = turn["tool"] or "(no tool)"
        res = turn["result"]
        err = isinstance(res, dict) and "error" in res
        _seen_items(seen, res, spec)
        name = tool + ("!" if err else "")
        if tool == spec["action_tool"]:
            n_actions += 1
            item_id = str(turn["args"].get(spec["id_arg"]))
            note = _item_note(item_id, t, seen, spec)
            if err:
                note += f" — refused: {str(res['error'])[:60]}"
            else:
                n_accepted += 1
                a = _age_hours(item_id, t, seen, spec)
                if a is not None:
                    ages.append(a)
            actions.append(note)
        else:
            flush_actions()
            if tool in WAIT_TOOLS:
                until = turn["args"].get("until")
                desc = tool
                if tool == "run_program" and turn["args"].get("path"):
                    desc += f"({turn['args']['path']})"
                if until:
                    try:
                        wait_hours = _hours(t, parse_iso(until))
                        desc += f" {wait_hours}h (until {_hhmm(parse_iso(until))})"
                    except ValueError:
                        desc += f" until {until}"
                wait_desc = desc
                name = desc + ("!" if err else "")
            elif tool not in ("get_time", "quota", "get_costs",
                              "get_feedback", "done") \
                    and not tool.startswith("("):
                data_calls += 1
        if tokens and tokens[-1][0] == name and "[" not in name:
            tokens[-1][1] += 1
        else:
            tokens.append([name, 1])
    flush_actions()
    seq = " → ".join(f"{n}" + (f" ×{k}" if k > 1 else "") for n, k in tokens)
    line = (f"{_hhmm(t)} ({seg['source']}) | {seq or '(no calls)'}"
            + (f" | acted {n_accepted}" if n_actions else ""))
    return {"line": line, "n_actions": n_actions, "n_accepted": n_accepted,
            "ages": ages, "wait_hours": wait_hours, "wait": wait_desc,
            "data_calls": data_calls, "seen": seen}


def _day_rollups(segs: list[dict], digests: list[dict]) -> dict[str, str]:
    """date -> the roll-up line appended after that day's wakes."""
    by_day: dict[str, list[int]] = {}
    for i, s in enumerate(segs):
        by_day.setdefault(s["t"].strftime("%m-%d"), []).append(i)
    spent: dict[str, float] = {}
    for s in segs:
        m = BRIEF_RE.search(s["marker"])
        if m:
            spent.setdefault(s["t"].strftime("%m-%d"), float(m.group(1)))
    out = {}
    for day, idx in by_day.items():
        ds = [digests[i] for i in idx]
        ages = [a for d in ds for a in d["ages"]]
        waits = [d["wait_hours"] for d in ds if d["wait_hours"] is not None]
        parts = [f"{len(ds)} wakes",
                 f"acted {sum(d['n_accepted'] for d in ds)}"
                 f"/{sum(d['n_actions'] for d in ds)} attempts"]
        if ages:
            parts.append(f"median age at action {median(ages)}h")
        if waits:
            parts.append(f"median wait {median(waits)}h")
        parts.append(f"data calls {sum(d['data_calls'] for d in ds)}")
        if day in spent:
            parts.append(f"LLM spent so far ${spent[day]:.2f}")
        out[day] = f"  — {day}: " + ", ".join(parts)
    return out


# -- verbatim --------------------------------------------------------------------------


def _clip(text: str, n: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[:n] + "…"


def render_verbatim(seg: dict, spec: dict, call_line=None,
                    result_chars: int | None = None) -> str:
    n = RESULT_CHARS if result_chars is None else result_chars
    seen: dict = {}
    out = [f"=== wake {_hhmm(seg['t'])} ({seg['source']}) ===", seg["marker"]]
    for turn in seg["turns"]:
        if turn["tool"] is None:
            out.append(f"> (reply without a tool call) {_clip(turn['thought'], n)}")
        else:
            args = json.dumps(turn["args"], separators=(",", ":"))
            thought = f' — "{turn["thought"]}"' if turn.get("thought") else ""
            out.append(f"> {turn['tool']} {_clip(args, n)}{thought}")
        if turn["result_text"] is None:
            continue
        _seen_items(seen, turn["result"], spec)
        summary = None
        if call_line is not None:
            try:
                summary = call_line(turn["tool"], turn["args"], turn["result"],
                                    seen)
            except Exception:  # a hook must never break the render
                summary = None
        out.append("  ← " + (summary if summary else
                             _clip(turn["result_text"], n)))
    return "\n".join(out)


# -- the render --------------------------------------------------------------------------


def render(agent_name: str, records_mod=None, verbatim_wakes: int | None = None,
           budget_tokens: int | None = None,
           result_chars: int | None = None) -> str:
    spec = dict(DEFAULT_SPEC)
    spec.update(getattr(records_mod, "TRAJECTORY", None) or {})
    call_line = getattr(records_mod, "call_line", None)
    k = VERBATIM_WAKES if verbatim_wakes is None else verbatim_wakes
    budget = BUDGET_TOKENS if budget_tokens is None else budget_tokens
    segs = segments(read_history(agent_name))
    if not segs:
        return "(no wakes yet)"
    digests = [digest_segment(s, spec) for s in segs]
    rollups = _day_rollups(segs, digests)
    n_acted = sum(d["n_accepted"] for d in digests)
    head = [f"{len(segs)} wakes from {_hhmm(segs[0]['t'])} to "
            f"{_hhmm(segs[-1]['t'])}; {n_acted} accepted actions.", "",
            "### Every wake, one line each (newest last; ×n = the same "
            "tool n times in a row; [...] = each action and what the "
            "actor had seen about its target; day roll-ups indented)"]
    lines: list[str] = []
    prev_day = None
    for s, d in zip(segs, digests):
        day = s["t"].strftime("%m-%d")
        if prev_day is not None and day != prev_day:
            lines.append(rollups[prev_day])
        lines.append(d["line"])
        prev_day = day
    lines.append(rollups[prev_day])
    used = sum(count_tokens(x) + 1 for x in head + lines)
    if used > budget:  # oldest digest lines go, with a note (rare)
        while lines and used > budget:
            used -= count_tokens(lines.pop(0)) + 1
        note = f"(oldest wake lines omitted to fit the budget)"
        lines.insert(0, note)
        used += count_tokens(note) + 1
    out = head + lines
    # verbatim: the first wake (where the policy was set), then the
    # newest k, newest first — each whole or not at all
    first = render_verbatim(segs[0], spec, call_line, result_chars)
    picks: list[tuple[str, str]] = [("### The first wake, verbatim", first)]
    newest = segs[-k:] if k > 0 else []
    if newest and newest[0] is segs[0]:
        newest = newest[1:]
    chosen: list[str] = []
    for s in reversed(newest):
        text = render_verbatim(s, spec, call_line, result_chars)
        c = count_tokens(text) + 1
        if used + count_tokens(first) + 2 + c > budget:
            break
        chosen.append(text)
        used += c
    if used + count_tokens(first) + 2 <= budget:
        out += ["", picks[0][0], picks[0][1]]
        used += count_tokens(first) + 2
    if chosen:
        out += ["", f"### The newest {len(chosen)} wake(s), verbatim "
                    "(tool results clipped)"]
        out += list(reversed(chosen))
    return "\n".join(out)
