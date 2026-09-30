"""Server-side tools of the crypto_price_consistency task.

One ExchangeApp: raw venue HTTP through the incident replay proxy, the
venue/endpoint contract, and the two scored actions. The fetch tool stays at
HTTP fidelity on purpose — status codes, headers, and unparsed bodies are
where the incident stream lives; a semantic get_price tool would absorb the
failures the task exists to measure.
"""

from __future__ import annotations

from harness.env_tools import EnvApp, ToolError, tool
from harness.timeutil import iso

# What a live venue's public docs would tell the client (and what the proxy
# implements). Exposed through get_exchange_docs so generated programs code
# against a documented contract, not guesswork.
ENDPOINT_DOCS = {
    "binance": {
        "path": "/api/v3/klines",
        "params": ("symbol (e.g. BTCUSDT), interval=5m, startTime?/endTime? "
                   "(ms), limit? (default 500, max 1000)"),
        "reply": ("JSON array of klines, ascending: [openTimeMs, open, high, "
                  "low, close, volume, closeTimeMs, quoteVolume, trades, "
                  "takerBase, takerQuote, ignore]"),
        "rate_limit": ("6000 request-weight per IP per minute (this endpoint "
                       "weighs 2); 429 carries Retry-After — requests before "
                       "it expires escalate to an HTTP 418 IP ban of "
                       "increasing duration"),
    },
    "okx": {
        "path": "/api/v5/market/candles (also /api/v5/market/history-candles)",
        "params": ("instId (e.g. BTC-USDT), bar=5m, after?/before? (ms, "
                   "exclusive), limit? (default 100, max 300)"),
        "reply": ("{code, msg, data}: rows newest-first [tsMs, open, high, "
                  "low, close, vol, volCcy, volCcyQuote, confirm]"),
        "rate_limit": "40 requests per 2 seconds per IP; 429 on exhaustion",
    },
    "kucoin": {
        "path": "/api/ua/v1/market/kline",
        "params": ("tradeType=SPOT, symbol (e.g. BTC-USDT), interval=5min, "
                   "startAt/endAt (seconds)"),
        "reply": ("{code, data: {tradeType, symbol, list}}: rows newest-first "
                  "[tsSeconds, open, high, low, close, volume, turnover]"),
        "rate_limit": ("2000 weight per IP per 30s public pool (this "
                       "endpoint weighs 3); 429 on exhaustion"),
    },
}


class ExchangeApp(EnvApp):
    def __init__(self, sim, task):
        super().__init__(sim)
        self.task = task

    @tool("get_exchange_docs() -> free: the venues, their candle endpoints, "
          "parameters, reply shapes, and documented rate limits — the "
          "contract for http_fetch",
          schema={"additionalProperties": False, "properties": {}, "type": "object"})
    async def get_exchange_docs(self, args: dict) -> dict:
        return {"venues": ENDPOINT_DOCS,
                "symbols": self.task.tcfg.symbols,
                "note": ("candles are 5-minute; the most recent completed "
                         "candle is the freshest data any venue serves")}

    @tool("http_fetch(venue: str, path: str, params?: {str: str|int}, "
          "timeout_s?: float) -> free (public endpoints; each venue's "
          "documented rate limits are enforced): raw HTTP against the "
          "venue as of now. Success and failure alike return what the "
          "wire saw: {status, headers, body, elapsed_ms} — body is "
          "an unparsed string — or {error: timeout|dns_error|"
          "connect_refused, elapsed_ms} when no response arrived",
          schema={"additionalProperties": False, "properties": {"params": {"type": "object"}, "path": {"type": "string"}, "timeout_s": {"type": "number"}, "venue": {"type": "string"}}, "required": ["venue", "path"], "type": "object"})
    async def http_fetch(self, args: dict) -> dict:
        task, sim = self.task, self.sim
        venue = args.get("venue")
        path = args.get("path")
        if not isinstance(venue, str) or not venue:
            raise ToolError("venue must be a non-empty string")
        if not isinstance(path, str) or not path.startswith("/"):
            raise ToolError("path must be a string starting with '/'")
        params = args.get("params") or {}
        if not isinstance(params, dict):
            raise ToolError("params must be an object")
        timeout_s = args.get("timeout_s", 10.0)
        if not isinstance(timeout_s, (int, float)) or timeout_s <= 0:
            raise ToolError("timeout_s must be a positive number")
        timeout_s = min(float(timeout_s), task.tcfg.timeout_max_s)

        async with sim.lock:
            response = task.proxy.request(venue, path, params,
                                          sim.clock.now, timeout_s)
            sim.bill(
                "fetch", 0.0, venue=venue, path=path,
                status=response.status, error=response.error,
                elapsed_ms=round(response.elapsed_ms, 1))
        if response.error is not None:
            return {"error": response.error,
                    "elapsed_ms": round(response.elapsed_ms, 1)}
        return {"status": response.status, "headers": response.headers,
                "body": response.body,
                "elapsed_ms": round(response.elapsed_ms, 1)}

    @tool("report(symbol: str, price: float) -> the scored action: your "
          "price for this symbol, this hour. The last report or abstain "
          "inside each hour is what gets scored (see INSTRUCTION.md)",
          tags=("action",),
          schema={"additionalProperties": False, "properties": {"price": {"type": "number"}, "symbol": {"type": "string"}}, "required": ["symbol", "price"], "type": "object"})
    async def report(self, args: dict) -> dict:
        return await self._notify({"kind": "report",
                                   "symbol": args.get("symbol"),
                                   "price": args.get("price")})

    @tool("abstain(symbol: str, reason?: str) -> the scored action: state "
          "that no trustworthy price is available this hour — allowed up "
          "to your abstention budget (see INSTRUCTION.md); silence is a "
          "protocol violation", tags=("action",),
          schema={"additionalProperties": False, "properties": {"reason": {"type": "string"}, "symbol": {"type": "string"}}, "required": ["symbol"], "type": "object"})
    async def abstain(self, args: dict) -> dict:
        return await self._notify({"kind": "abstain",
                                   "symbol": args.get("symbol"),
                                   "reason": args.get("reason")})

    async def _notify(self, payload: dict) -> dict:
        sim = self.sim
        async with sim.lock:
            sim.task.record_notification(sim.clock.now, payload)
            sim.ledger.append("notify", sim.clock.now, payload=payload)
            return {"status": "accepted", "at": iso(sim.clock.now)}
