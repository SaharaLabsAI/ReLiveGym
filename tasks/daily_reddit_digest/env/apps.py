"""Server-side tools of the daily_reddit_digest task.

Observation tools are the sibling's RedditReadApp (list_posts / get_post /
get_cascade over the same stores, fees, and limiter). DigestApp adds this
task's action surface: `digest` (the scored action) and the free,
clock-only `digest_status`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from harness.env_tools import EnvApp, tool
from harness.timeutil import iso
from tasks.reddit_ai_popularity.task import _iso_ts

if TYPE_CHECKING:
    from tasks.daily_reddit_digest.task import DailyRedditDigestTask


class DigestApp(EnvApp):
    def __init__(self, sim, task: "DailyRedditDigestTask"):
        super().__init__(sim)
        self.task = task

    @tool("digest_status() -> free: today's delivery window (if now is in "
          "one), whether it has been delivered, and the next window. "
          "Clock-only; says nothing about posts",
          schema={"additionalProperties": False, "properties": {}, "type": "object"})
    async def digest_status(self, args: dict) -> dict:
        sc, sim = self.task.scorer, self.sim
        async with sim.lock:
            now = sim.clock.now
            now_ts = now.timestamp()
            d = sc.day_for(now_ts)
            nxt = sc.next_delivery(now_ts)
        out = {"now": iso(now), "day": d.isoformat() if d else None,
               "delivered": bool(d and d in sc.digests),
               "next_window": ([_iso_ts(nxt[0]), _iso_ts(nxt[1])]
                               if nxt else None)}
        if d:
            lo, hi = sc.window_for(d)
            dlo, dhi = sc.delivery_for(d)
            out["candidate_window"] = [_iso_ts(lo), _iso_ts(hi)]
            out["delivery_window"] = [_iso_ts(dlo), _iso_ts(dhi)]
        return out

    @tool("digest(root_ids: [str]) -> the scored action: today's digest. "
          "Accepted only inside the daily delivery window, once per day; "
          "settles when the day's posts reveal (see INSTRUCTION.md)",
          tags=("action",),
          schema={"additionalProperties": False, "properties": {"root_ids": {"items": {"type": "string"}, "type": "array"}}, "required": ["root_ids"], "type": "object"})
    async def digest(self, args: dict) -> dict:
        sim = self.sim
        payload = {"root_ids": args.get("root_ids")}
        async with sim.lock:
            sim.task.record_notification(sim.clock.now, payload)
            sim.ledger.append("notify", sim.clock.now, payload=payload)
            d = self.task.scorer.day_for(sim.clock.now.timestamp())
        return {"status": "accepted", "day": d.isoformat(),
                "n_ids": len(payload["root_ids"]), "at": iso(sim.clock.now)}
