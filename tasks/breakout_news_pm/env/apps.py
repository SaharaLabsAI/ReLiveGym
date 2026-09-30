"""Server-side tools of the breakout_news_pm task.

One MarketsApp declares every observation and action tool — name,
agent-facing doc, price, handler, metering. Agent programs discover these
through GET /tools; no task-specific client code exists on the agent side.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from harness.env_tools import EnvApp, ToolError, tool
from harness.limits import RateLimiter
from harness.timeutil import iso, parse_iso

if TYPE_CHECKING:
    from tasks.breakout_news_pm.task import BreakoutNewsPMTask


def _time(s, name: str) -> datetime:
    try:
        return parse_iso(str(s))
    except ValueError:
        raise ToolError(f"invalid datetime for {name!r}: {s!r}")


def _price_tag(usd: float, unit: str) -> str:
    return "free" if not usd else f"${usd:g} per {unit}"


class MarketsApp(EnvApp):
    def __init__(self, sim, task: "BreakoutNewsPMTask"):
        super().__init__(sim)
        self.task = task
        self.news_limiter = sim.limiters.setdefault(
            "news", RateLimiter("news api", task.tcfg.news_rate_limit))
        self.price_limiter = sim.limiters.setdefault(
            "prices", RateLimiter("market data api",
                                  task.tcfg.price_rate_limit))

    def tools(self):
        # The manifest's price tag comes from THIS run's cost table, so
        # tools.md / tools.json / the envkit stub agree with INSTRUCTION.md
        # (which renders the same numbers); the envkit stub tm=B agents
        # read carries the tag in brackets, so tm=B runs see e.g. "[free]"
        # for get_prices.
        import dataclasses

        cost = self.task.tcfg.cost
        tags = {"get_prices": _price_tag(cost.price_call, "call"),
                "search_news": _price_tag(cost.news_search_call,
                                          f"page of {self.task.tcfg.search_top_k}"),
                "get_article": _price_tag(cost.article_call, "call")}
        return [(dataclasses.replace(tdef, price=tags[tdef.name])
                 if tdef.name in tags else tdef, h)
                for tdef, h in super().tools()]

    @tool("get_markets() -> free: your monitoring requests (market_id, "
          "question, resolution criteria, window)",
          schema={"additionalProperties": False, "properties": {}, "type": "object"})
    async def get_markets(self, args: dict) -> list[dict]:
        t = self.task
        out = []
        for w in t.tcfg.markets:
            m = t.markets_meta[w.market_id]
            out.append({
                "market_id": w.market_id, "question": m["question"],
                "category": m["category"], "event_title": m.get("event_title"),
                "description": m["description"],
                "grid_minutes": m["grid_minutes"],
                "start": iso(w.start), "end": iso(w.end),
            })
        return out

    @tool("get_prices(market_id: str, start: iso, end: iso, "
          "grid_minutes?: N >= 1) -> price change-series (sparse: price is "
          "constant between points). grid_minutes coarsens the reply to "
          "the last change per N-minute bucket — default 1, the full "
          "minute grid", price="paid per call",
          schema={"additionalProperties": False, "properties": {"end": {"format": "date-time", "type": "string"}, "grid_minutes": {"minimum": 1, "type": "integer"}, "market_id": {"type": "string"}, "start": {"format": "date-time", "type": "string"}}, "required": ["market_id", "start", "end"], "type": "object"})
    async def get_prices(self, args: dict) -> dict:
        # Replies are clipped to now - price_delay_minutes (PriceStore's
        # visibility clamp). The actor-facing statement of that delay lives
        # in INSTRUCTION.md ("a point at time t becomes visible at t +
        # ${price_delay_minutes} minutes"); the tool doc stays contract-only
        # — "visibility-delayed" is experimenter/mechanics terminology.
        t, sim = self.task, self.sim
        market_id = args.get("market_id")
        if market_id not in t.markets_meta:
            raise ToolError(f"unknown market_id {market_id!r}")
        s, e = _time(args.get("start"), "start"), _time(args.get("end"), "end")
        gm = args.get("grid_minutes", 1)
        async with sim.lock:
            self.price_limiter.consume(sim.clock.now)
            try:
                result = t.prices.query(market_id, s, e, sim.clock.now, gm)
            except ValueError as err:
                raise ToolError(str(err))
            sim.bill(
                "price_call", t.tcfg.cost.price_call,
                market_id=market_id, start=iso(s), end=iso(e),
                grid_minutes=gm,
                returned_points=len(result["changes"]["time"]))
        return result

    @tool("search_news(q: str, date_from?: iso, date_to?: iso, offset?: int) "
          "-> PAID per page: BM25 search over published news (quoted "
          "phrases and AND/OR supported; date filters on publish time)",
          price="paid per page",
          schema={"additionalProperties": False, "properties": {"date_from": {"format": "date-time", "type": "string"}, "date_to": {"format": "date-time", "type": "string"}, "offset": {"type": "integer"}, "q": {"type": "string"}}, "required": ["q"], "type": "object"})
    async def search_news(self, args: dict) -> dict:
        t, sim = self.task, self.sim
        offset = args.get("offset", 0)
        if not isinstance(offset, int) or offset < 0:
            raise ToolError("offset must be an int >= 0")
        q = str(args.get("q") or "")
        if not q.strip():
            raise ToolError("q must be a non-empty query")
        date_from = args.get("date_from")
        date_to = args.get("date_to")
        async with sim.lock:
            self.news_limiter.consume(sim.clock.now)
            try:
                results = t.news.search(q, date_from, date_to, sim.clock.now,
                                        t.tcfg.search_top_k, offset)
            except ValueError as e:
                raise ToolError(f"bad query or date: {e}")
            sim.bill(
                "news_search", t.tcfg.cost.news_search_call,
                q=q, date_from=date_from, date_to=date_to, offset=offset,
                returned=len(results))
        return {"query": q, "offset": offset, "results": results}

    @tool("get_article(news_id: str) -> PAID: full text of one published "
          "article", price="paid per call",
          schema={"additionalProperties": False, "properties": {"news_id": {"type": "string"}}, "required": ["news_id"], "type": "object"})
    async def get_article(self, args: dict) -> dict:
        t, sim = self.task, self.sim
        news_id = args.get("news_id")
        if not isinstance(news_id, str) or not news_id:
            raise ToolError("news_id must be a non-empty string")
        async with sim.lock:
            self.news_limiter.consume(sim.clock.now)
            doc = t.news.get_article(news_id, sim.clock.now)
            sim.bill(
                "article_call", t.tcfg.cost.article_call,
                news_id=news_id, found=doc is not None)
        if doc is None:
            raise ToolError(f"unknown or not-yet-published news_id {news_id!r}")
        return doc

    @tool("notify(market_id: str, news_id: str, direction: 'up'|'down') -> "
          "the scored action: claim this market will start a breakout in "
          "this direction within the claim window, citing this article. "
          "One standing claim per market — a new claim is rejected until "
          "the previous one resolves (see INSTRUCTION.md)", tags=("action",),
          schema={"additionalProperties": False, "properties": {"direction": {"enum": ["up", "down"], "type": "string"}, "market_id": {"type": "string"}, "news_id": {"type": "string"}}, "required": ["market_id", "news_id", "direction"], "type": "object"})
    async def notify(self, args: dict) -> dict:
        sim = self.sim
        payload = {"market_id": args.get("market_id"),
                   "news_id": args.get("news_id"),
                   "direction": args.get("direction")}
        async with sim.lock:
            sim.task.record_notification(sim.clock.now, payload)
            sim.ledger.append("notify", sim.clock.now, payload=payload)
        return {"status": "accepted", "at": iso(sim.clock.now)}
