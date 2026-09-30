"""Online observability and honest hindsight curation (tmA-reflection-only,
ported unchanged to TM-B; register_program_call is the one addition).

The candidate has no scorer/oracle feedback tool.  It therefore derives
provisional feedback only from information available to any actor: its own
actions, delayed prices, and previously fetched article metadata.  Generated
memory is intentionally per-run and starts empty in every copied candidate.
"""

from __future__ import annotations

import json
import os
import statistics
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from runtime import llm_client
from runtime.env_client import EnvError, iso, parse_iso


STATE_PATH = Path("memory/reflective_state.json")
OBS_PATH = Path("logs/observability.jsonl")
MAX_ARTICLES = 1200
MAX_FEEDBACK = 160
MAX_LESSONS = 6
_LOCK = threading.RLock()


def _safe_lesson(text: str) -> bool:
    """Reject the most dangerous hindsight leak: requiring price confirmation."""
    low = text.lower()
    price_signal = any(x in low for x in
                       ("price", "repric", "market move", "movement",
                        "observable move", "visible move", "corresponding move"))
    prerequisite = any(x in low for x in
                       ("before claim", "before issuing", "confirm", "require",
                        "withhold", "wait for", "only when"))
    return not (price_signal and prerequisite)


def _blank() -> dict:
    return {
        "articles": {},
        "actions": [],
        "feedback": [],
        "lessons": [],
        "diagnosis": "",
        "last_review": None,
        "curations": 0,
    }


def _load_unlocked() -> dict:
    if not STATE_PATH.exists():
        return _blank()
    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return _blank()
    out = _blank()
    if isinstance(data, dict):
        out.update(data)
    return out


def _save_unlocked(data: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True, default=str),
                   encoding="utf-8")
    os.replace(tmp, STATE_PATH)


def _clip(value, chars: int = 1200):
    if isinstance(value, str):
        return value if len(value) <= chars else value[:chars] + "…"
    if isinstance(value, dict):
        return {str(k): _clip(v, chars) for k, v in value.items()}
    if isinstance(value, list):
        return [_clip(v, chars) for v in value[:30]]
    return value


def observe(event: str, **fields) -> None:
    """Append a compact, machine-readable event; observability must not crash."""
    try:
        row = {
            "real_time": datetime.now(timezone.utc).isoformat(),
            "thread": threading.current_thread().name,
            "event": event,
            **_clip(fields),
        }
        with _LOCK:
            OBS_PATH.parent.mkdir(parents=True, exist_ok=True)
            with OBS_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, separators=(",", ":"),
                                   default=str) + "\n")
    except Exception:
        pass


def register_search(agent_name: str, args: dict, result: dict,
                    sim_time: str) -> None:
    rows = result.get("results", []) if isinstance(result, dict) else []
    with _LOCK:
        state = _load_unlocked()
        articles = state.setdefault("articles", {})
        for row in rows:
            nid = str(row.get("news_id") or "")
            if nid:
                articles[nid] = {
                    "title": row.get("title"),
                    "domain": row.get("domain"),
                    "published": row.get("published"),
                }
        while len(articles) > MAX_ARTICLES:
            articles.pop(next(iter(articles)))
        _save_unlocked(state)
    observe("search", agent=agent_name, sim_time=sim_time,
            query=args.get("q"), date_from=args.get("date_from"),
            date_to=args.get("date_to"), offset=args.get("offset", 0),
            returned=len(rows))


def register_article(agent_name: str, args: dict, result: dict,
                     sim_time: str) -> None:
    if not isinstance(result, dict):
        return
    nid = str(result.get("news_id") or args.get("news_id") or "")
    if nid:
        with _LOCK:
            state = _load_unlocked()
            state.setdefault("articles", {})[nid] = {
                "title": result.get("title"),
                "domain": result.get("domain"),
                "published": result.get("published"),
            }
            _save_unlocked(state)
    observe("article", agent=agent_name, sim_time=sim_time, news_id=nid,
            title=result.get("title"), domain=result.get("domain"))


def register_notify(agent_name: str, args: dict, result: dict,
                    sim_time: str) -> None:
    if not isinstance(result, dict) or result.get("error"):
        return
    nid = str(args.get("news_id") or "")
    with _LOCK:
        state = _load_unlocked()
        meta = state.get("articles", {}).get(nid, {})
        state.setdefault("actions", []).append({
            "agent": agent_name,
            "market_id": str(args.get("market_id") or ""),
            "news_id": nid,
            "title": meta.get("title"),
            "published": meta.get("published"),
            "direction": args.get("direction"),
            "at": result.get("at") or sim_time,
            "reviewed": False,
        })
        _save_unlocked(state)
    observe("notification", agent=agent_name, sim_time=sim_time,
            market_id=args.get("market_id"), news_id=nid,
            direction=args.get("direction"), title=meta.get("title"))


def register_program_call(agent_name: str | None, tool: str, args: dict,
                          result, sim_time: str) -> None:
    """TM-B: envkit calls made inside an authored program bypass the
    agent's tool registry (runtime.program.CALL_OBSERVER routes them here)
    so a program's searches, article reads and notifications enter the
    same state as the agent's own tool calls."""
    name = agent_name or "program"
    if tool == "search_news":
        register_search(name, args, result, sim_time)
    elif tool == "get_article":
        register_article(name, args, result, sim_time)
    elif tool == "notify":
        register_notify(name, args, result, sim_time)


def render_memory(agent_name: str) -> str:
    with _LOCK:
        state = _load_unlocked()
    lessons = [str(x) for x in state.get("lessons", []) if str(x).strip()]
    own = [x for x in state.get("feedback", [])
           if x.get("agent") == agent_name][-5:]
    pending = [x for x in state.get("actions", [])
               if x.get("agent") == agent_name and not x.get("reviewed")][-3:]
    lines = ["# Reflective memory (heuristic, price-derived)"]
    if lessons:
        lines += [f"- {x}" for x in lessons[:MAX_LESSONS]]
    else:
        lines.append("- No curated lessons yet; use the operating policy.")
    if own:
        lines.append("\nRecent inferred outcomes for this market:")
        for x in own:
            lines.append(
                f"- {x.get('kind')}: {x.get('verdict')}; "
                f"evidence={x.get('evidence')}")
    if pending:
        lines.append("\nYour claims still awaiting mature hindsight:")
        for x in pending:
            lines.append(
                f"- {x.get('at')} {x.get('direction')}: "
                f"{x.get('title') or 'article metadata unavailable'}")
    lines.append(
        "Treat this block as fallible feedback, never as permission to "
        "notify without a fresh supporting article. Prices are retrospective "
        "evaluation only: NEVER wait for price confirmation before notifying.")
    return "\n".join(lines)


def _price_points(result: dict) -> list[tuple[datetime, float]]:
    points: list[tuple[datetime, float]] = []
    start = result.get("level_at_start")
    if start is not None:
        # A synthetic point only supplies the baseline for the first change.
        points.append((datetime.min.replace(tzinfo=timezone.utc), float(start)))
    changes = result.get("changes") or {}
    for t, p in zip(changes.get("time", []), changes.get("p", [])):
        points.append((parse_iso(t), float(p)))
    return points


def _jumps(points: list[tuple[datetime, float]]) -> tuple[list[dict], float]:
    if len(points) < 2:
        return [], 0.02
    raw = [abs(points[i][1] - points[i - 1][1])
           for i in range(1, len(points))]
    noise = statistics.median(raw) if raw else 0.0
    # Generic detector: ignore ordinary microstructure while remaining useful
    # on markets with different baseline volatility. This is not the scorer's
    # private breakpoint algorithm.
    threshold = max(0.02, 6.0 * noise)
    rows = []
    for i in range(1, len(points)):
        delta = points[i][1] - points[i - 1][1]
        if abs(delta) < threshold:
            continue
        # Ignore quote flicker that reverses within minutes. A useful
        # hindsight signal should retain most of its displacement for at
        # least 30 minutes. (At the edge of a review window, defer judgment.)
        later = next((p for t, p in points[i + 1:]
                      if t >= points[i][0] + timedelta(minutes=30)), None)
        if later is None:
            continue
        retained = later - points[i - 1][1]
        if retained * delta <= 0 or abs(retained) < abs(delta) * 0.7:
            continue
        rows.append({
            "at": iso(points[i][0]),
            "direction": "up" if delta > 0 else "down",
            "delta": round(delta, 6),
            "retained_30m": round(retained, 6),
        })
    # One durable displacement can arrive in several prints. Keep only the
    # largest within a six-hour neighborhood so the curator does not learn
    # ten “misses” from one repricing episode.
    grouped: list[dict] = []
    for row in rows:
        if grouped and parse_iso(row["at"]) - parse_iso(grouped[-1]["at"]) \
                < timedelta(hours=6):
            if abs(row["delta"]) > abs(grouped[-1]["delta"]):
                grouped[-1] = row
        else:
            grouped.append(row)
    return grouped, threshold


def _review_action(env, action: dict, now: datetime) -> dict | None:
    at = parse_iso(action["at"])
    if now < at + timedelta(hours=24):
        return None
    start = iso(at - timedelta(hours=1))
    end = iso(min(now, at + timedelta(hours=24, minutes=15)))
    result = env.call("get_prices", market_id=action["market_id"],
                      start=start, end=end, grid_minutes=1)
    jumps, threshold = _jumps(_price_points(result))
    after = [j for j in jumps if parse_iso(j["at"]) >= at]
    matching = [j for j in after if j["direction"] == action["direction"]]
    opposite = [j for j in after if j["direction"] != action["direction"]]
    if matching:
        verdict = "likely covered a directional move"
    elif opposite:
        verdict = "likely wrong direction"
    else:
        verdict = "likely false alarm"
    evidence = {
        "threshold": round(threshold, 6),
        "matching_jumps": matching[:3],
        "opposite_jumps": opposite[:3],
    }
    return {
        "kind": "claim_review",
        "agent": action.get("agent"),
        "market_id": action["market_id"],
        "direction": action.get("direction"),
        "title": action.get("title"),
        "verdict": verdict,
        "evidence": evidence,
        "reviewed_at": iso(now),
    }


def _scan_moves(env, market: dict, start: datetime, now: datetime,
                actions: list[dict], prior_keys: set[str]) -> list[dict]:
    result = env.call("get_prices", market_id=str(market["market_id"]),
                      start=iso(start), end=iso(now), grid_minutes=1)
    jumps, threshold = _jumps(_price_points(result))
    out = []
    for jump in jumps:
        key = f"{market['market_id']}@{jump['at']}"
        if key in prior_keys:
            continue
        jt = parse_iso(jump["at"])
        covering = []
        for action in actions:
            if str(action.get("market_id")) != str(market["market_id"]):
                continue
            at = parse_iso(action["at"])
            if at <= jt <= at + timedelta(hours=24) \
                    and action.get("direction") == jump["direction"]:
                covering.append(action)
        out.append({
            "kind": "move_review",
            "key": key,
            "agent": f"m-{market['market_id']}",
            "market_id": str(market["market_id"]),
            "verdict": ("likely anticipated move" if covering
                        else "possible missed move"),
            "evidence": {**jump, "threshold": round(threshold, 6),
                         "prior_matching_claims": len(covering)},
            "reviewed_at": iso(now),
        })
    return out


def _curation_prompt(state: dict, new_feedback: list[dict]) -> list[dict]:
    payload = {
        "existing_lessons": state.get("lessons", []),
        "recent_feedback": new_feedback[-30:],
    }
    return [
        {"role": "system", "content": (
            "You curate compact operating memory for news-driven prediction "
            "market monitoring. Feedback is heuristic, inferred from delayed "
            "prices, and may be wrong. Return JSON with `lessons` (at most 6 "
            "short, general, actionable rules) and `diagnosis` (one short "
            "paragraph). Generalize across markets. Do not include dates, "
            "market ids, article ids, proper names, exact price levels, or "
            "event-specific hints. Prefer lessons about evidence strength, "
            "query design, claim opportunity cost, direction, and cadence. "
            "Do not claim access to scorer truth. The task requires an alert "
            "BEFORE a price breakout. Prices in this feedback are retrospective "
            "evaluation only. Never recommend waiting for, requiring, or using "
            "a price move to confirm a current alert; current alerts must be "
            "decided from fresh news evidence." )},
        {"role": "user", "content": json.dumps(payload, default=str)},
    ]


def curate(env, markets: list[dict], sim_start: str) -> dict:
    """Review matured claims and detected moves, then refresh memory via LLM."""
    now = env.now()
    with _LOCK:
        state = _load_unlocked()
        actions = list(state.get("actions", []))
        existing = list(state.get("feedback", []))
        last_review = parse_iso(state["last_review"]) \
            if state.get("last_review") else parse_iso(sim_start)
    new_feedback = []
    for action in actions:
        if action.get("reviewed"):
            continue
        try:
            review = _review_action(env, action, now)
        except EnvError as exc:
            observe("feedback_error", action=action, error=str(exc))
            continue
        if review:
            new_feedback.append(review)
            action["reviewed"] = True

    prior_keys = {x.get("key") for x in existing if x.get("key")}
    scan_start = max(parse_iso(sim_start), last_review - timedelta(hours=2))
    for market in markets:
        try:
            new_feedback.extend(
                _scan_moves(env, market, scan_start, now, actions, prior_keys))
        except EnvError as exc:
            observe("feedback_error", market_id=market.get("market_id"),
                    error=str(exc))

    with _LOCK:
        state = _load_unlocked()
        # Preserve wrapper updates that may have arrived during price review.
        by_key = {(x.get("market_id"), x.get("news_id"), x.get("at")): x
                  for x in actions}
        for live in state.get("actions", []):
            key = (live.get("market_id"), live.get("news_id"), live.get("at"))
            if key not in by_key:
                actions.append(live)
        state["actions"] = actions
        state["feedback"] = (state.get("feedback", []) + new_feedback)[-MAX_FEEDBACK:]
        state["last_review"] = iso(now)
        state["curations"] = int(state.get("curations", 0)) + 1
        _save_unlocked(state)

    curated = None
    memory_updated = False
    if new_feedback or not state.get("lessons"):
        try:
            curated = llm_client.chat_json(env, _curation_prompt(state, new_feedback))
        except EnvError as exc:
            observe("curation_error", sim_time=iso(now), error=str(exc))
        if isinstance(curated, dict) and isinstance(curated.get("lessons"), list):
            lessons = [str(x).strip() for x in curated["lessons"]
                       if str(x).strip() and _safe_lesson(str(x))][:MAX_LESSONS]
            with _LOCK:
                state = _load_unlocked()
                state["lessons"] = lessons
                state["diagnosis"] = str(curated.get("diagnosis") or "")[:2000]
                _save_unlocked(state)
            memory_updated = True
    summary = {
        "sim_time": iso(now),
        "new_feedback": len(new_feedback),
        "claims_reviewed": sum(x.get("kind") == "claim_review"
                               for x in new_feedback),
        "moves_reviewed": sum(x.get("kind") == "move_review"
                              for x in new_feedback),
        "memory_updated": memory_updated,
    }
    observe("curation", **summary)
    return summary
