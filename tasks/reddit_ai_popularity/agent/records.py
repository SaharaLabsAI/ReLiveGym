"""Record semantics of the reddit_ai_popularity task: how oracle outcomes
map to this agent's own actions and how records stratify for the
reflection render.

The oracle stream (task.py oracle_outcomes): kind "recommendation" =
one of this agent's own settled recommendations (either class); kind
"post" = a tail post the agent did NOT recommend, at its reveal — the
recall failures. Non-recommended nontail posts are not streamed. Every
record carries a `growth` trajectory (comment counts at fixed ages)."""

from __future__ import annotations

import re

# Factual glossary rendered above the raw-memory block (alg=memory:
# compose wires memory.LEGEND to this). Feed contract only, no strategy
# — agent-visible instrument text.
BLOCK_LEGEND = (
    'Each line is one settled oracle record. outcome.kind '
    '"recommendation" = a post you recommended, settled tail or nontail; '
    'weight = the credit your timing earned. outcome.kind "post" = a '
    'tail post you did not recommend. growth = comment counts at fixed '
    'ages after posting. The stream is filtered: posts you never '
    'recommended appear only when they settled tail, so these records '
    'do not reflect the tail base rate (the spec above states it).')

ENTITY_NOUN = "subreddit"

# Numeric aggregates for the cumulative statistics of the formatted
# learned block: (outcome field, scope, aggregates). weight rides on
# the agent's own settled recommendations.
STAT_FIELDS = [("weight", "own", ("n", "sum", "mean"))]


def action_for(outcome: dict, state: dict) -> dict | None:
    """Recommendation outcomes already carry what the action was
    (root_id) and when (recommended_at), so there is no separate action
    to link — is_own marks them; post outcomes are not actions."""
    return None


def is_own(record: dict) -> bool:
    """Every settled recommendation is the agent's own action."""
    return (record.get("outcome") or {}).get("kind") == "recommendation"


def stratum_of(record: dict) -> str:
    """Record status for the reflection render: tail or nontail — the
    env judges the class, this module never re-derives it."""
    o = record.get("outcome") or {}
    return o.get("status") or o.get("kind") or "unknown"


def entity_of(record: dict) -> str | None:
    """Ledger key for the cumulative per-entity outcome table: outcomes
    group by subreddit."""
    return (record.get("outcome") or {}).get("subreddit")


def entity_label(key: str, ctx: dict) -> str:
    """Ledger row label for the formatted learned block."""
    return f"r/{key}"


def action_line(record: dict) -> str:
    """One readable line per own settled recommendation for the
    formatted learned block: when recommended, where, how it settled,
    the credit its timing earned, how the post grew, the title. The
    outcome's root_id never prints."""
    o = record.get("outcome") or {}
    title = " ".join((o.get("title") or "").split())
    if len(title) > 60:
        title = title[:60] + "…"
    g = o.get("growth") or {}
    traj = " ".join(f"{k}:{v}" for k, v in g.items())
    w = o.get("weight")
    d = o.get("delay_hours")
    return (f"{(o.get('recommended_at') or record.get('t') or '')[:16]}"
            .replace("T", " ")
            + f" | r/{o.get('subreddit') or '?'} | "
            f"{o.get('status') or '?'} | "
            f"weight {w if w is not None else '?'}"
            + (f" (recommended {d}h after posting)" if d is not None else "")
            + f" | {o.get('descendants', '?')} comments | "
            + (f"growth {traj} | " if traj else "") + title)


def digest_line(record: dict) -> str:
    """One compact line per non-own settlement (a missed tail post) for
    the cumulative outcome digest: when, where, how it grew."""
    o = record.get("outcome") or {}
    title = " ".join((o.get("title") or "").split())
    if len(title) > 60:
        title = title[:60] + "…"
    g = o.get("growth") or {}
    traj = " ".join(f"{k}:{v}" for k, v in g.items())
    return (f"{(o.get('posted_at') or record.get('t') or '')[:13]} | "
            f"r/{o.get('subreddit') or '?'} | "
            f"{o.get('descendants', '?')} comments | "
            + (f"growth {traj} | " if traj else "") + title)


def render_context(env, state: dict) -> dict:
    """Extra template slot ${quota}: the rolling-24h slot state at
    reflection time (the quota tool is free by the task contract),
    stated in words — the skill writer has no tools in hand."""
    q = env.call("quota")
    return {"extra_slots": {"quota": (
        f"used {q['used_last_24h']} of {q['daily_cap']} rolling-24h "
        f"recommendation slots; {q['remaining']} free right now. A used "
        f"slot returns 24 h after the acceptance that consumed it.")}}


# full-tier line of the formatted learned block: a missed tail post's
# digest already carries everything the record holds
outcome_line = digest_line


# -- trajectory render hooks (alg=vskills2; runtime/trajectory.py) ---------------------

# which tool is the scored action and how a listing names its items
TRAJECTORY = {"action_tool": "recommend", "id_arg": "root_id",
              "list_key": "posts", "id_key": "id", "time_key": "posted_at",
              "count_key": "num_comments", "label_key": "title",
              "group_key": "subreddit", "group_prefix": "r/"}

# a reminder that names a post id recites the replay set (reddit ids are
# 7-char base36 starting with 1 and carry a letter — plain numbers pass)
LINT_EXTRA = [("post id", re.compile(r"\b1(?=[0-9a-z]{6}\b)[0-9]*[a-z][0-9a-z]*\b"))]


def _post_note(p: dict, now_iso: str | None = None) -> str:
    title = " ".join((p.get("title") or "").split())
    if len(title) > 40:
        title = title[:40] + "…"
    return (f"r/{p.get('subreddit') or '?'} {p.get('num_comments', '?')}c "
            f"posted {(p.get('posted_at') or '')[5:16].replace('T', ' ')} "
            f"\"{title}\"")


def call_line(tool: str, args: dict, result, seen: dict) -> str | None:
    """One readable line for a tool result in the verbatim trajectory
    (None = fall back to the clipped raw result)."""
    if not isinstance(result, dict) or "error" in result:
        return None
    if tool == "list_posts":
        posts = result.get("posts") or []
        head = (f"{result.get('total_hits', '?')} hits, page of {len(posts)} "
                f"({result.get('sort', 'new')}): ")
        return head + "; ".join(_post_note(p) for p in posts[:5]) + \
            (f"; +{len(posts) - 5} more" if len(posts) > 5 else "")
    if tool == "get_cascade":
        return (f"{result.get('n_comments', '?')} comments so far "
                f"({result.get('n_nodes', '?')} nodes, as of {result.get('now', '?')})")
    if tool == "get_post":
        return _post_note(result) + (f" selftext {len(result.get('selftext') or '')} chars")
    if tool == "recommend":
        return f"{result.get('status', '?')} at {result.get('at', '?')}"
    if tool == "quota":
        return (f"{result.get('remaining', '?')} of {result.get('daily_cap', '?')} "
                f"slots free (used {result.get('used_last_24h', '?')} in 24h)")
    if tool == "get_feedback":
        outs = result.get("outcomes") or []
        return f"{len(outs)} settled outcome(s)"
    return None
