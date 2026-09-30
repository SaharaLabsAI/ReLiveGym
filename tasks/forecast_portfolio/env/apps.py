"""Server-side tools of the forecast_portfolio task.

One ForecastApp declares every observation and action tool. API-style
question surface: a compact index (`list_questions`), a per-question
detail call (`get_question`), and the agent's own submission history
(`get_forecasts`) — so the portfolio never floods a turn and a
context-poor agent can recover its state from the environment. News
tools mirror bnpm's (same billing types, same "news" limiter name).
There is no price surface anywhere.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from harness.env_tools import EnvApp, ToolError, tool
from harness.limits import RateLimiter
from harness.timeutil import iso, parse_iso

if TYPE_CHECKING:
    from tasks.forecast_portfolio.task import ForecastPortfolioTask

from harness.task import NotificationError


def _time(s, name: str) -> datetime:
    try:
        return parse_iso(str(s))
    except ValueError:
        raise ToolError(f"invalid datetime for {name!r}: {s!r}")


class ForecastApp(EnvApp):
    def __init__(self, sim, task: "ForecastPortfolioTask"):
        super().__init__(sim)
        self.task = task
        self.news_limiter = sim.limiters.setdefault(
            "news", RateLimiter("news api", task.tcfg.news_rate_limit))

    @tool("list_questions(status?: 'open'|'resolved', added_after?: iso) -> "
          "free: index of the questions added so far (id, question, "
          "added_at, scheduled_close, status; resolved ones include their "
          "outcome). added_after returns only questions added strictly "
          "after that time",
          schema={"additionalProperties": False, "properties": {"added_after": {"format": "date-time", "type": "string"}, "status": {"enum": ["open", "resolved"], "type": "string"}}, "required": [], "type": "object"})
    async def list_questions(self, args: dict) -> dict:
        status = args.get("status")
        if status not in (None, "open", "resolved"):
            raise ToolError("status must be 'open' or 'resolved'")
        added_after = (_time(args["added_after"], "added_after")
                       if args.get("added_after") is not None else None)
        async with self.sim.lock:
            rows = self.task.question_index(self.sim.clock.now, status,
                                            added_after)
        return {"questions": rows}

    @tool("get_question(question_id: str) -> free: full detail of one "
          "question: text, outcomes, resolution criteria, added_at, "
          "scheduled_close, status (+ outcome once resolved)",
          schema={"additionalProperties": False, "properties": {"question_id": {"type": "string"}}, "required": ["question_id"], "type": "object"})
    async def get_question(self, args: dict) -> dict:
        qid = args.get("question_id")
        if not isinstance(qid, str) or not qid:
            raise ToolError("question_id must be a non-empty string")
        async with self.sim.lock:
            try:
                return self.task.question_detail(qid, self.sim.clock.now)
            except NotificationError as e:
                raise ToolError(str(e))

    @tool("get_forecasts(question_id?: str) -> free: your own submission "
          "history — per question, every (at, forecast) you submitted and "
          "the current standing forecast",
          schema={"additionalProperties": False, "properties": {"question_id": {"type": "string"}}, "required": [], "type": "object"})
    async def get_forecasts(self, args: dict) -> dict:
        qid = args.get("question_id")
        if qid is not None and (not isinstance(qid, str) or not qid):
            raise ToolError("question_id must be a non-empty string")
        async with self.sim.lock:
            try:
                rows = self.task.forecast_history(qid, self.sim.clock.now)
            except NotificationError as e:
                raise ToolError(str(e))
        return {"forecasts": rows}

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

    @tool("submit_forecast(question_id: str, forecast: {outcome: prob}, "
          "news_id?: str) -> the scored action: set your standing forecast "
          "for this question. Give a probability for EVERY outcome, >= 0 "
          "and summing to 1 — a short sum is accepted, but the missing "
          "mass scores as zero on the outcomes you left out, NOT as "
          "abstention. It stays in force until you submit again or the "
          "question resolves. Optionally cite the article this forecast "
          "rests on", tags=("action",),
          schema={"additionalProperties": False, "properties": {"forecast": {"additionalProperties": {"type": "number"}, "type": "object"}, "news_id": {"type": "string"}, "question_id": {"type": "string"}}, "required": ["question_id", "forecast"], "type": "object"})
    async def submit_forecast(self, args: dict) -> dict:
        sim = self.sim
        payload = {"question_id": args.get("question_id"),
                   "forecast": args.get("forecast")}
        if args.get("news_id") is not None:
            payload["news_id"] = args.get("news_id")
        async with sim.lock:
            sim.task.record_notification(sim.clock.now, payload)
            if self.task.tcfg.cost.submit_call:
                sim.bill("submit_call", self.task.tcfg.cost.submit_call,
                         question_id=payload["question_id"])
            sim.ledger.append("submit_forecast", sim.clock.now,
                              payload=payload)
        return {"status": "accepted", "at": iso(sim.clock.now)}
