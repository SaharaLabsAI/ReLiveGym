"""Server-side tools of the reddit_ai_popularity task.

RedditReadApp holds the observation tools (list_posts / get_post /
get_cascade) and is reused by sibling tasks on the same world
(daily_reddit_digest); RedditApp adds this task's action surface (quota /
recommend). The oracle per-post lookup exists only in sig=oracle runs —
provisioning, not branching.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from harness.env_tools import EnvApp, ToolError, tool
from harness.limits import RateLimiter
from harness.timeutil import iso, parse_iso

if TYPE_CHECKING:
    from tasks.reddit_ai_popularity.task import RedditPopularityTask


def _time(s, name: str) -> datetime | None:
    if s is None:
        return None
    try:
        return parse_iso(str(s))
    except ValueError:
        raise ToolError(f"invalid datetime for {name!r}: {s!r}")


def _iso_ts(t: float) -> str:
    return iso(datetime.fromtimestamp(t, tz=timezone.utc))


def _view_clock(sim, args: dict) -> datetime:
    """The instant a data view is computed at: the sim clock, or an
    earlier `as_of` (a replayed wake
    re-observes the world as it was at that wake — nothing the caller
    could not have seen then; an `as_of` after the clock is refused).
    Not a documented parameter of any tool: the live actor's manifest is
    unchanged. Call under sim.lock."""
    now = sim.clock.now
    raw = args.get("as_of")
    if raw is None:
        return now
    as_of = _time(raw, "as_of")
    if as_of > now:
        raise ToolError(f"as_of {iso(as_of)} is later than now {iso(now)}")
    return as_of


class RedditReadApp(EnvApp):
    """Observation tools over a task exposing `roots`, `cascades`, and
    `tcfg` (page_size, rate_limit, cost.api_call). Duck-typed so sibling
    tasks on the same data can mount it unchanged."""

    def __init__(self, sim, task):
        super().__init__(sim)
        self.task = task
        self.limiter = sim.limiters.setdefault(
            "reddit", RateLimiter("reddit api", task.tcfg.rate_limit))

    def _visible(self, root_id, now_ts: float) -> dict:
        if not isinstance(root_id, str) or not root_id:
            raise ToolError(f"root_id must be a non-empty string, got {root_id!r}")
        r = self.task.roots.get(root_id)
        if r is None or r["created_utc"] > now_ts:
            # same message for nonexistent and future ids: no existence leak
            raise ToolError(f"unknown or not yet posted root_id {root_id!r}")
        return r

    @tool("list_posts(since?: iso, until?: iso, subreddit?: str, "
          "sort?: new|comments, order?: asc|desc, offset?: int) -> one page "
          "of root posts, each with `num_comments` (comments so far); sorted "
          "by time (new) or by current num_comments (comments); "
          "`descendants` is null until a post is revealed. Billed per call; "
          "rate-limited (see INSTRUCTION.md)",
          price="paid per page",
          schema={"additionalProperties": False, "properties": {"offset": {"type": "integer"}, "order": {"enum": ["asc", "desc"], "type": "string"}, "since": {"format": "date-time", "type": "string"}, "sort": {"enum": ["new", "comments"], "type": "string"}, "subreddit": {"type": "string"}, "until": {"format": "date-time", "type": "string"}}, "required": [], "type": "object"})
    async def list_posts(self, args: dict) -> dict:
        t, sim = self.task, self.sim
        order = args.get("order", "desc")
        sort = args.get("sort", "new")
        offset = args.get("offset", 0)
        subreddit = args.get("subreddit")
        if order not in ("asc", "desc"):
            raise ToolError("order must be 'asc' or 'desc'")
        if sort not in ("new", "comments"):
            raise ToolError("sort must be 'new' or 'comments'")
        if not isinstance(offset, int) or offset < 0:
            raise ToolError("offset must be an int >= 0")
        if subreddit is not None and not isinstance(subreddit, str):
            raise ToolError("subreddit must be a string")
        s = _time(args.get("since"), "since")
        u = _time(args.get("until"), "until")
        async with sim.lock:
            self.limiter.consume(sim.clock.now)
            result = t.roots.query(s, u, subreddit, order, offset,
                                   t.tcfg.page_size, _view_clock(sim, args),
                                   sort=sort)
            sim.bill(
                "reddit_list", t.tcfg.cost.api_call,
                since=args.get("since"), until=args.get("until"),
                subreddit=subreddit, sort=sort, order=order, offset=offset,
                total_hits=result["total_hits"],
                returned=len(result["posts"]))
        return result

    @tool("get_post(root_id: str) -> one root post incl. selftext "
          "(`descendants` null until revealed). Billed per call; "
          "rate-limited", price="paid per call",
          schema={"additionalProperties": False, "properties": {"root_id": {"type": "string"}}, "required": ["root_id"], "type": "object"})
    async def get_post(self, args: dict) -> dict:
        t, sim = self.task, self.sim
        root_id = args.get("root_id")
        async with sim.lock:
            now_ts = _view_clock(sim, args).timestamp()
            self.limiter.consume(sim.clock.now)
            r = self._visible(root_id, now_ts)
            result = t.roots.visible_view(r, now_ts, include_text=True)
            sim.bill("reddit_post", t.tcfg.cost.api_call, root_id=root_id)
        return result

    @tool("get_cascade(root_id: str, offset?: int) -> the reply-tree "
          "prefix visible so far (nodes with created_utc <= now; parent_id "
          "gives the structure; NO score, NO future nodes). Billed per "
          "call; rate-limited", price="paid per page",
          schema={"additionalProperties": False, "properties": {"offset": {"type": "integer"}, "root_id": {"type": "string"}}, "required": ["root_id"], "type": "object"})
    async def get_cascade(self, args: dict) -> dict:
        t, sim = self.task, self.sim
        root_id = args.get("root_id")
        offset = args.get("offset", 0)
        if not isinstance(offset, int) or offset < 0:
            raise ToolError("offset must be an int >= 0")
        async with sim.lock:
            now_ts = _view_clock(sim, args).timestamp()
            self.limiter.consume(sim.clock.now)
            self._visible(root_id, now_ts)
            result = t.cascades.prefix(root_id, now_ts, offset, t.tcfg.page_size)
            sim.bill("reddit_cascade", t.tcfg.cost.api_call,
                     root_id=root_id, offset=offset,
                     n_nodes=result["n_nodes"])
        return result

class RedditApp(RedditReadApp):
    """reddit_ai_popularity's full surface: the read tools + quota/recommend."""

    @tool("quota() -> free: your rolling 24h recommendation budget",
          schema={"additionalProperties": False, "properties": {}, "type": "object"})
    async def quota(self, args: dict) -> dict:
        t, sim = self.task, self.sim
        async with sim.lock:
            used = t.scorer.used_last_24h(sim.clock.now.timestamp())
        return {"daily_cap": t.tcfg.daily_cap, "used_last_24h": used,
                "remaining": t.tcfg.daily_cap - used}

    @tool("recommend(root_id: str) -> the scored action; settles at the post's "
          "reveal (see INSTRUCTION.md)", tags=("action",),
          schema={"additionalProperties": False, "properties": {"root_id": {"type": "string"}}, "required": ["root_id"], "type": "object"})
    async def recommend(self, args: dict) -> dict:
        sim = self.sim
        payload = {"root_id": args.get("root_id")}
        async with sim.lock:
            sim.task.record_notification(sim.clock.now, payload)
            sim.ledger.append("notify", sim.clock.now, payload=payload)
        return {"status": "accepted", "root_id": payload["root_id"],
                "at": iso(sim.clock.now)}


class RedditOracleApp(EnvApp):
    """Sig-oracle extra: per-post settled label lookup (free, universal).
    Provisioned only in sig=oracle runs."""

    def __init__(self, sim, task: "RedditPopularityTask"):
        super().__init__(sim)
        self.task = task

    @tool("get_post_popularity(root_id: str) -> free: the settled final "
          "`descendants` of ANY post already past its reveal", tags=("feedback",),
          schema={"additionalProperties": False, "properties": {"root_id": {"type": "string"}}, "required": ["root_id"], "type": "object"})
    async def get_post_popularity(self, args: dict) -> dict:
        t, sim = self.task, self.sim
        root_id = args.get("root_id")
        async with sim.lock:
            now_ts = _view_clock(sim, args).timestamp()
            if not isinstance(root_id, str) or not root_id:
                raise ToolError(
                    f"root_id must be a non-empty string, got {root_id!r}")
            r = t.roots.get(root_id)
            if r is None or r["created_utc"] > now_ts:
                raise ToolError(f"unknown or not yet posted root_id {root_id!r}")
            reveal_t = r["created_utc"] + t.tcfg.horizon_hours * 3600
            if reveal_t > now_ts:
                raise ToolError(f"post {root_id!r} is not settled yet "
                                f"(reveals at {_iso_ts(reveal_t)})")
        return {"root_id": root_id, "subreddit": r["subreddit"],
                "title": r["title"], "posted_at": _iso_ts(r["created_utc"]),
                "revealed_at": _iso_ts(reveal_t),
                "descendants": r["_label"]}
