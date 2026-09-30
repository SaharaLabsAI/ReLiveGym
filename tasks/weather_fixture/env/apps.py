"""Server-side tools of the weather task.

The weather API is free at real rates
(open-meteo) behind its documented daily quota, enforced as a 429 by the
shared limiter machinery. The scored action stays free."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from harness.env_tools import EnvApp, ToolError, tool
from harness.limits import RateLimiter
from harness.timeutil import iso, parse_iso

if TYPE_CHECKING:
    from tasks.weather_fixture.task import WeatherTask


class WeatherApp(EnvApp):
    def __init__(self, sim, task: "WeatherTask"):
        super().__init__(sim)
        self.task = task
        self.limiter = sim.limiters.setdefault(
            "weather", RateLimiter("weather api", task.tcfg.rate_limit))

    @tool("get_weather(start: iso, end: iso) -> hourly temperatures "
          "in [start, end], clipped to what is already observable "
          "(free; rate-limited — see INSTRUCTION.md)",
          schema={"additionalProperties": False, "properties": {"end": {"format": "date-time", "type": "string"}, "start": {"format": "date-time", "type": "string"}}, "required": ["start", "end"], "type": "object"})
    async def get_weather(self, args: dict) -> dict:
        t, sim = self.task, self.sim
        try:
            s = parse_iso(str(args.get("start")))
            e = parse_iso(str(args.get("end")))
        except ValueError as err:
            raise ToolError(str(err))
        async with sim.lock:
            self.limiter.consume(sim.clock.now)
            result = t.store.query(s, e, sim.clock.now)
            sim.bill("weather", 0.0, start=iso(s), end=iso(e),
                     returned_hours=len(result["hourly"]["time"]))
        return result

    @tool("notify(date: YYYY-MM-DD) -> the scored action: predict a "
          "threshold-exceedance day (see INSTRUCTION.md)", tags=("action",),
          schema={"additionalProperties": False, "properties": {"date": {"type": "string"}}, "required": ["date"], "type": "object"})
    async def notify(self, args: dict) -> dict:
        sim = self.sim
        payload = {"date": args.get("date")}
        async with sim.lock:
            sim.task.record_notification(sim.clock.now, payload)
            sim.ledger.append("notify", sim.clock.now, payload=payload)
        return {"status": "accepted", "at": iso(sim.clock.now)}
