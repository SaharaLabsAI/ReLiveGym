"""Record semantics of the breakout_news_pm task: how an outcome links to
this agent's own action, how records stratify for the reflection render,
and how observed news becomes a sig-self candidate."""

from __future__ import annotations

import json

# Factual glossary rendered above the learned block (alg=memory: wired
# by memory.bind). Feed contract only, no strategy — agent-visible
# instrument text.
BLOCK_LEGEND = (
    'Each line is one settled record. kind "alert" = one of your own '
    'notifications and how it settled (its status names the outcome '
    'class the scoring in INSTRUCTION.md defines). kind "breakpoint" = '
    'a settled market move: its status says whether any notification of '
    'yours preceded it, credit = the score credit it carried, and '
    'gold_groups are the judge\'s causal news stories for the move — an '
    'article shows a title only if you observed it yourself; otherwise '
    'only its publish time is known. verdict records, where present, '
    'are hindsight judgments on articles you observed.')

ENTITY_NOUN = "market"

# Numeric aggregates for the cumulative statistics of the formatted
# learned block: (outcome field, scope, aggregates). credit rides on
# breakpoint records (not the alert itself), hence scope "all".
STAT_FIELDS = [("credit", "all", ("n", "sum", "mean"))]


def action_for(outcome: dict, state: dict) -> dict | None:
    """Link an outcome to this agent's own notification, if any; the entry
    is deleted on link (a notification resolves exactly once). Outcomes the
    agent did not act on carry no action field."""
    nid = outcome.get("news_id")
    if nid is None:
        return None
    return state.get("actions", {}).pop(nid, None)


def is_own(record: dict) -> bool:
    """Alert outcomes are the agent's own actions even when no action
    record linked: they always render in the reflection's
    own-actions section, never sampled away."""
    return (record.get("outcome") or {}).get("kind") == "alert"


def render_context(env, state: dict) -> dict:
    """Id -> meaning context for the reflection render:
    the market table with resolution wording (get_markets is free by the
    task contract), and title resolution for every article the agent has
    observed (state["registered"] — free bookkeeping of what it already
    saw; nothing is looked up). The runtime treats this dict as opaque
    except "extra_slots" (template slots this task defines); the id maps
    feed this module's own enrich hook."""
    markets = env.call("get_markets")
    lines = []
    for m in markets:
        desc = " ".join((m.get("description") or "").split())
        if len(desc) > 600:  # cut at a sentence boundary, never mid-clause
            head = desc[:600]
            cut = head.rfind(". ")
            desc = (head[:cut + 1] if cut > 200 else head) + " […]"
        lines.append(f"- {m['market_id']}: {m['question']}"
                     + (f"\n  resolution: {desc}" if desc else ""))
    registered = state.get("registered") or {}
    return {
        "extra_slots": {"markets": "\n".join(lines)},
        "market_questions": {m["market_id"]: m["question"] for m in markets},
        "news_titles": {nid: meta.get("title")
                        for nid, meta in registered.items()
                        if isinstance(meta, dict) and meta.get("title")},
    }


def entity_of(record: dict) -> str | None:
    """Ledger key for the reflection's cumulative per-entity outcome
    table: bnpm outcomes group by market."""
    return (record.get("outcome") or {}).get("market_id")


def entity_label(key: str, ctx: dict) -> str:
    """Ledger row label for the formatted learned block: the market id
    with its question, never the bare id."""
    q = (ctx.get("market_questions") or {}).get(key) or ""
    return f'{key} "{q[:70]}{"…" if len(q) > 70 else ""}"' if q else str(key)


def enrich(record: dict, ctx: dict) -> None:
    """Resolve bare ids into meaning on ONE render copy ,
    in place: market_id -> market_question, news_id -> news_title (own
    outcomes and gold-group articles alike), from the render_context
    maps. A gold article whose title resolves was, by construction of
    state["registered"], observed by the agent -> seen_by_you."""
    questions = ctx.get("market_questions") or {}
    titles = ctx.get("news_titles") or {}
    o = record.get("outcome") or {}
    q = questions.get(o.get("market_id"))
    if q:
        o["market_question"] = q
    title = titles.get(o.get("news_id"))
    if title:
        o["news_title"] = title
    for g in o.get("gold_groups") or []:
        for a in g.get("articles") or []:
            title = titles.get(a.get("news_id"))
            if title:
                a["title"] = title
                a["seen_by_you"] = True


def digest_line(record: dict) -> str:
    """One compact line for the reflection's cumulative outcome digest
    : date | market | direction | status, plus the
    judge's top causal story with the publish->move lead — durable
    catalyst memory, so per-market rules stay auditable after the full
    record has left the interval view. Expects enriched records
    (seen_by_you / news_title resolved)."""
    o = record.get("outcome") or {}
    if o.get("kind") == "breakpoint":
        date = (o.get("t_move_start") or record.get("t") or "")[:10]
        parts = [date, f"{o.get('market_id')}{_q(o)}",
                 o.get("direction") or "?", o.get("status") or "?"]
        golds = o.get("gold_groups") or []
        if golds:
            best = max(golds, key=lambda g: g.get("confidence") or 0)
            pubs = sorted(a.get("published_at") or "~"
                          for a in best.get("articles") or [])
            move = o.get("t_move_start")
            if pubs and pubs[0] != "~" and move:
                lead_h = (_parse(move) - _parse(pubs[0])).total_seconds() / 3600
                parts.append(f"lead {lead_h:.1f}h pub->move")
            story = " ".join((best.get("story") or "").split())
            if len(story) > 220:
                story = story[:220] + "…"
            seen = any(a.get("seen_by_you")
                       for g in golds for a in g.get("articles") or [])
            parts.append(f"gold: {story}" + (" (seen_by_you)" if seen else ""))
        elif o.get("no_attributable_news"):
            parts.append("no attributable news (unwinnable)")
        return " | ".join(parts)
    if o.get("verdict"):  # sig-self hindsight
        title = o.get("news_title") or "(untitled article)"
        return (f"{(record.get('t') or '')[:10]} | "
                f"{o.get('market_id') or '?'}{_q(o)} | "
                f"verdict {o['verdict']} | {title}")
    slim = {k: v for k, v in o.items() if v is not None}
    return f"{(record.get('t') or '')[:10]} | " + \
        json.dumps(slim, separators=(",", ":"))


def _hm(t: str | None) -> str:
    return (t or "?")[:16].replace("T", " ")


def _q(o: dict, width: int = 70) -> str:
    q = o.get("market_question") or ""
    return f' "{q[:width]}{"…" if len(q) > width else ""}"' if q else ""


def action_line(record: dict) -> str:
    """One readable line per own notification for the verified-curation
    prompt (vskills only; alg=skills keeps its JSON view): when | market |
    direction | the cited article (title + publish time) | how it
    resolved. No ids — the curator has no article tool, so ids carry
    nothing it can use. Expects enriched records."""
    o = record.get("outcome") or {}
    title = o.get("news_title") or "(title unknown)"
    pub = (record.get("action") or {}).get("published")
    line = (f"{_hm(o.get('at'))} | {o.get('market_id')}{_q(o)} | "
            f"{o.get('direction') or '?'} | cited \"{title}\"")
    if pub:
        line += f" (published {_hm(pub)})"
    status = o.get("status") or "?"
    settled = o.get("t_settled")
    if status.startswith("covering"):
        line += f" | -> {status}: the move started {_hm(settled)}"
    elif status == "false_alarm":
        line += f" | -> false alarm: no move by {_hm(settled)}"
    else:
        line += f" | -> {status}"
    return line


def outcome_line(record: dict) -> str:
    """One readable line per settled breakout (or other non-own outcome)
    for the verified-curation prompt: when | market | direction | caught
    or missed | the judge's causal stories with their articles (title or
    'untitled', publish time, whether the program had seen it). Expects
    enriched records."""
    o = record.get("outcome") or {}
    if o.get("kind") != "breakpoint":
        return digest_line(record)
    parts = [f"{_hm(o.get('t_move_start'))} | {o.get('market_id')}{_q(o)} | "
             f"{o.get('direction') or '?'}"]
    status = o.get("status") or "?"
    if status.startswith("covered"):
        parts.append(f"CAUGHT ({status}) by the program's notification at "
                     f"{_hm(o.get('notified_at'))}, credit {o.get('credit')}")
    elif status == "miss":
        parts.append("MISSED (no notification preceded it)")
    else:
        parts.append(status)
    golds = o.get("gold_groups") or []
    if golds:
        stories = []
        for g in sorted(golds, key=lambda g: -(g.get("confidence") or 0))[:2]:
            story = " ".join((g.get("story") or "").split())
            if len(story) > 200:
                story = story[:200] + "…"
            arts = []
            for a in sorted(g.get("articles") or [],
                            key=lambda a: a.get("published_at") or "~"):
                t = a.get("title")
                arts.append((f'"{t}"' if t else "untitled article")
                            + f" published {_hm(a.get('published_at'))}"
                            + (" [seen by the program]"
                               if a.get("seen_by_you") else ""))
            stories.append(f"{story} (confidence {g.get('confidence')}): "
                           + "; ".join(arts))
        parts.append("causal news per the judge: " + " || ".join(stories))
    elif o.get("no_attributable_news"):
        parts.append("no attributable news (unwinnable)")
    return " | ".join(parts)


def _parse(t: str):
    from datetime import datetime
    return datetime.fromisoformat(t.replace("Z", "+00:00"))


def stratum_of(record: dict) -> str:
    """Record status for the reflection render: alert statuses
    (covering_news/covering_timing/wrong_direction/false_alarm/stale),
    breakpoint statuses (covered_news/covered_timing/miss),
    or a sig-self verdict (attributed/no_breakout)."""
    o = record.get("outcome", {})
    return o.get("status") or o.get("verdict") or o.get("kind") or "unknown"


def register_candidates(state: dict, search_result: dict) -> None:
    """Register every article observed via search_news as a sig-self
    hindsight candidate. The registry is global (not per market): ReACT
    actors search free-form, so market attribution is the hindsight
    call's job, not the observer's."""
    seen = state.setdefault("registered", {})
    for h in search_result.get("results", []):
        seen.setdefault(h["news_id"], {
            "title": h["title"], "published": h["published"]})
