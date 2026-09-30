"""breakout_news_pm scan handler.

Hourly news scan: per active market, one paid keyword search over news
published since the last firing; on fresh hits, one LLM decision call — the
model sees INSTRUCTION.md's economics (the claim: a breakout starts in the
stated direction within the claim window; anything at/after the move start
earns nothing) and decides directly what to notify, with a direction per
alert (no code thresholds). The acting path never reads prices: reward
exists only before the move, so price evidence arrives too late to act on.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from question_keywords import keywords
from runtime import llm_client, skills, trace

SCAN_CRON = "5 * * * *"    # hourly: reward decays from publication to the move

DIRECTIONS = ("up", "down")


def _iso(dt) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse(t: str) -> datetime:
    return datetime.fromisoformat(t.replace("Z", "+00:00"))


def scan(env, state: dict, block: str) -> None:
    now = env.now()
    t = _iso(now)
    cursor = state.get("scan_cursor") or _iso(now - timedelta(hours=1))
    actions = state.setdefault("actions", {})
    registered = state.setdefault("registered", {})  # id -> title for the
    # reflection render; sig-self reads it as its candidate pool
    for m in env.call("get_markets"):
        mid = m["market_id"]
        if not (m["start"] <= t <= m["end"]):
            continue
        hits = env.call("search_news", q=" ".join(keywords(m["question"])),
                        date_from=cursor, date_to=t)["results"]
        fresh = [h for h in hits if h["published"] > cursor]
        if not fresh:
            continue
        for h in fresh:  # register everything seen: sig-self candidates
            registered.setdefault(h["news_id"], {
                "title": h["title"], "published": h["published"]})
        items = "\n".join(f"- news_id={h['news_id']} "
                          f"published={h['published']}: {h['title']} — "
                          f"{h['snippet']}" for h in fresh)
        prompt = (Path("prompts/decide.md").read_text(encoding="utf-8")
                  .replace("${instruction}",
                           Path("INSTRUCTION.md").read_text(encoding="utf-8"))
                  .replace("${block}", block or "(none yet)")
                  .replace("${question}", m["question"])
                  .replace("${items}", items))
        reply = llm_client.chat_json(env, [{"role": "user", "content": prompt}])
        chosen = []
        for a in (reply.get("alerts") or []) if isinstance(reply, dict) else []:
            if (isinstance(a, dict) and a.get("news_id")
                    and a.get("direction") in DIRECTIONS):
                chosen.append((str(a["news_id"]), a["direction"]))
            else:  # malformed entry: dropped and traced, never guessed
                trace.log(t, "note", what="malformed_alert", market_id=mid,
                          entry=a)
        trace.log(t, "decide", block_version=skills.block_version(),
                  market_id=mid, batch=[h["news_id"] for h in fresh],
                  alerts=[n for n, _ in chosen], question=m["question"],
                  items=[{"news_id": h["news_id"], "title": h["title"],
                          "published": h["published"],
                          "snippet": h.get("snippet", "")} for h in fresh])
        already = state.setdefault("alerted", {}).setdefault(mid, [])
        for nid, direction in chosen:
            if nid in already:  # idempotence, not judgment: never re-send
                continue
            try:
                env.call("notify", market_id=mid, news_id=nid,
                         direction=direction)
                already.append(nid)
                reg = registered.get(nid, {})
                actions[nid] = {"did": "alerted", "at": t, "market_id": mid,
                                "direction": direction,
                                "title": reg.get("title"),
                                "published": reg.get("published")}
                trace.log(t, "action", market_id=mid, news_id=nid,
                          direction=direction)
            except Exception as e:  # rejection is a free 400; log and move on
                trace.log(t, "note", what="notify_rejected", market_id=mid,
                          news_id=nid, error=str(e))
    state["scan_cursor"] = t
