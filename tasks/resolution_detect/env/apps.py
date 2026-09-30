"""Server-side tools of the resolution_detect task.

One DetectApp declares every observation and action tool. API-style
question surface mirroring forecast_portfolio: a compact index
(`list_questions`), a per-question detail call (`get_question`), and the
agent's own standing claims (`get_marks`). News tools mirror bnpm's
(same billing types, same "news" limiter name). There is no price
surface anywhere. The action is `mark_outcome` — one claim per question
for the entire run, no retraction; invalid claims are rejected free of
charge and consume nothing.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from harness.env_tools import EnvApp, ToolError, tool
from harness.limits import RateLimiter
from harness.task import NotificationError
from harness.timeutil import iso, parse_iso

if TYPE_CHECKING:
    from tasks.resolution_detect.task import ResolutionDetectTask


def _time(s, name: str) -> datetime:
    try:
        return parse_iso(str(s))
    except ValueError:
        raise ToolError(f"invalid datetime for {name!r}: {s!r}")


class DetectApp(EnvApp):
    def __init__(self, sim, task: "ResolutionDetectTask"):
        super().__init__(sim)
        self.task = task
        self.news_limiter = sim.limiters.setdefault(
            "news", RateLimiter("news api", task.tcfg.news_rate_limit))

    @tool("list_questions(added_after?: iso) -> "
          "free: index of the questions added so far (id, question, "
          "added_at). added_after returns only questions added "
          "strictly after that time",
          schema={"additionalProperties": False, "properties": {"added_after": {"format": "date-time", "type": "string"}}, "required": [], "type": "object"})
    async def list_questions(self, args: dict) -> dict:
        added_after = (_time(args["added_after"], "added_after")
                       if args.get("added_after") is not None else None)
        async with self.sim.lock:
            rows = self.task.question_index(self.sim.clock.now,
                                            added_after)
        return {"questions": rows}

    @tool("get_question(question_id: str) -> free: full detail of one "
          "question: text, outcomes, resolution criteria, added_at",
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

    @tool("get_marks(question_id?: str) -> free: your own standing claims — "
          "per question, the (at, outcome) you committed",
          schema={"additionalProperties": False, "properties": {"question_id": {"type": "string"}}, "required": [], "type": "object"})
    async def get_marks(self, args: dict) -> dict:
        qid = args.get("question_id")
        if qid is not None and (not isinstance(qid, str) or not qid):
            raise ToolError("question_id must be a non-empty string")
        async with self.sim.lock:
            try:
                rows = self.task.mark_history(qid, self.sim.clock.now)
            except NotificationError as e:
                raise ToolError(str(e))
        return {"marks": rows}

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

    @tool("mark_outcome(question_id: str, outcome: str, news_id?: str) -> "
          "the scored action: claim that this question's outcome is "
          "already decided in the world. ONE claim per question for the "
          "entire run — it cannot be changed or withdrawn. Optionally cite "
          "the article your claim rests on. Invalid claims are rejected "
          "free of charge and do not consume your claim", tags=("action",),
          schema={"additionalProperties": False, "properties": {"news_id": {"type": "string"}, "outcome": {"type": "string"}, "question_id": {"type": "string"}}, "required": ["question_id", "outcome"], "type": "object"})
    async def mark_outcome(self, args: dict) -> dict:
        sim = self.sim
        payload = {"question_id": args.get("question_id"),
                   "outcome": args.get("outcome")}
        if args.get("news_id") is not None:
            payload["news_id"] = args.get("news_id")
        async with sim.lock:
            sim.task.record_notification(sim.clock.now, payload)
            if self.task.tcfg.cost.mark_call:
                sim.bill("mark_call", self.task.tcfg.cost.mark_call,
                         question_id=payload["question_id"])
            sim.ledger.append("mark_outcome", sim.clock.now, payload=payload)
        return {"status": "accepted", "at": iso(sim.clock.now)}
