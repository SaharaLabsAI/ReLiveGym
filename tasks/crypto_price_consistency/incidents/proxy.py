"""Replay proxy engine: recorded venue responses +
incident stream + limiter machines, as a pure library.

`ReplayProxy.request(venue, path, params, t)` answers exactly like the venue
would at simulated time `t`: candle rows are served **byte-identical** from
the recorded `raw/*.jsonl` backfill pages, re-wrapped in the venue's native
envelope and sliced to the request's own query parameters. Incidents from a
frozen scenario trace and the documented rate-limiter state machines
(`limiters.yaml`) are applied in composition order M -> E -> P: a
rate-limited client never observes the edge failure behind the 429, and an
edge failure masks whatever the provider is doing.

No wall-clock time, sockets, or filesystem writes: every response is a pure
function of (recorded data, trace, limiter state, sim time), with per-request
randomness seeded from (trace, venue, path, t) so replays are reproducible.

Fidelity caveats, deliberate and visible to the experimenter only:
- The recording holds completed 5-minute candles; a row becomes visible once
  its interval has closed, and the in-progress candle that live venue APIs
  include is never served (as if the venue omitted the forming candle).
- Only the recorded endpoints/symbols/interval exist; anything else gets a
  venue-shaped error, which is the contract INSTRUCTION.md documents.
- Envelopes are re-serialized compactly; rows keep their recorded JSON types
  verbatim.

The env layer (env/apps.py) owns clock/billing integration; it should bill
`Response.elapsed_ms` as sim time and apply `Trace.client_clock_skew(t)` to
any client-visible clock.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

from harness.limits import LimiterState, consume

INCIDENTS_DIR = Path(__file__).resolve().parent
DEFAULT_LIMITERS = INCIDENTS_DIR / "limiters.yaml"
DEFAULT_DATASETS_DIR = INCIDENTS_DIR.parent / "data" / "datasets"

CANDLE_SECONDS = 300
# indices of open/high/low/close inside a recorded row, all three venues
PRICE_INDICES = (1, 2, 3, 4)

BASE_LATENCY_MS = {"binance": 120.0, "okx": 180.0, "kucoin": 150.0}
DEFAULT_TIMEOUT_S = 10.0

JSON_HEADERS = {"content-type": "application/json"}
HTML_HEADERS = {"content-type": "text/html"}
CHALLENGE_BODY = ("<!DOCTYPE html><html><head><title>Just a moment...</title>"
                  "</head><body>Checking your browser before accessing."
                  "</body></html>")


@dataclass
class Response:
    """What the client observes. `error` set means no HTTP response arrived
    (timeout / dns_error / connect_refused); status/body are then None."""

    status: int | None
    headers: dict[str, str]
    body: str | None
    elapsed_ms: float
    error: str | None = None

    def json(self) -> Any:
        return json.loads(self.body) if self.body is not None else None


def _parse_t(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


class Trace:
    """A frozen scenario file: header + events, queryable by venue and time."""

    def __init__(self, path: Path):
        lines = Path(path).read_text().splitlines()
        self.header = json.loads(lines[0])
        self.name = self.header["trace"]
        self.events: list[dict[str, Any]] = []
        for line in lines[1:]:
            event = json.loads(line)
            event["start_dt"] = _parse_t(event["start"])
            event["end_dt"] = _parse_t(event["end"])
            self.events.append(event)
        self._by_venue: dict[str, list[dict[str, Any]]] = {}
        for event in self.events:
            self._by_venue.setdefault(event["venue"], []).append(event)

    def active(self, venue: str, layer: str, t: datetime) -> list[dict[str, Any]]:
        return [e for e in self._by_venue.get(venue, ())
                if e["layer"] == layer and e["start_dt"] <= t < e["end_dt"]]

    def client_clock_skew(self, t: datetime) -> float:
        """Seconds the client's clock is off at t (env applies to get_time)."""
        for event in self.active("client", "E", t):
            if event["mode"] == "clock_skew":
                return float(event["params"]["skew_seconds"])
        return 0.0


class CandleStore:
    """Recorded candle rows per (venue, symbol), deduped and sorted ascending
    by open time (seconds). Rows are kept exactly as recorded."""

    def __init__(self, dataset_dirs: list[Path] | None = None):
        if dataset_dirs is None:
            dataset_dirs = sorted(p for p in DEFAULT_DATASETS_DIR.iterdir()
                                  if (p / "raw").is_dir())
        self._rows: dict[tuple[str, str], list[tuple[int, list[Any]]]] = {}
        for dataset in dataset_dirs:
            for raw_file in sorted((dataset / "raw").glob("*.jsonl")):
                self._load(raw_file.stem, raw_file)

    def _load(self, venue: str, path: Path) -> None:
        buckets: dict[tuple[str, str], dict[int, list[Any]]] = {}
        for line in path.read_text().splitlines():
            record = json.loads(line)
            params = dict(p.split("=", 1) for p in
                          record["url"].split("?", 1)[1].split("&"))
            payload = record["payload"]
            if venue == "binance":
                symbol, rows = params["symbol"], payload
                unit = 1000
            elif venue == "okx":
                symbol, rows = params["instId"], payload["data"]
                unit = 1000
            elif venue == "kucoin":
                symbol, rows = params["symbol"], payload["data"]["list"]
                unit = 1
            else:
                continue
            bucket = buckets.setdefault((venue, symbol), {})
            for row in rows:
                bucket[int(row[0]) // unit] = row
        for key, by_ts in buckets.items():
            merged = self._rows.setdefault(key, [])
            merged.extend(sorted(by_ts.items()))
            merged.sort(key=lambda pair: pair[0])

    def symbols(self, venue: str) -> list[str]:
        return [s for v, s in self._rows if v == venue]

    def rows(self, venue: str, symbol: str, until: datetime,
             start_s: int | None = None,
             end_s: int | None = None) -> list[list[Any]] | None:
        """Completed rows visible at `until`, open time within [start_s,
        end_s] (inclusive, seconds), ascending. None = unknown symbol."""
        series = self._rows.get((venue, symbol))
        if series is None:
            return None
        visible = int(until.timestamp()) - CANDLE_SECONDS
        out = [row for ts, row in series
               if ts <= visible
               and (start_s is None or ts >= start_s)
               and (end_s is None or ts <= end_s)]
        return out


class LimiterBank:
    """Per-venue rate-limiter machines from limiters.yaml, driven by sim
    time. The window algorithms live in harness/limits.py; this bank layers the
    venue realism on top: ban escalation and capacity-modulation factors
    that multiply the budget for the request being checked (the
    "unexplained 429")."""

    def __init__(self, config_path: Path = DEFAULT_LIMITERS):
        self.config = yaml.safe_load(Path(config_path).read_text())["venues"]
        self.state: dict[str, LimiterState] = {}

    def check(self, venue: str, t: datetime,
              budget_factor: float = 1.0) -> tuple[bool, int | None, float | None]:
        """Consume one request. Returns (allowed, status, retry_after_s)."""
        spec = self.config.get(venue)
        if spec is None:
            return True, None, None
        state = self.state.setdefault(venue, LimiterState())
        now = t.timestamp()
        escalation = spec.get("ban_escalation")

        if state.ban_until is not None:
            if now < state.ban_until:
                return False, escalation["status"], state.ban_until - now
            state.ban_until = None
        if (state.last_violation is not None and escalation
                and now - state.last_violation >= escalation["decay_hours"] * 3600):
            state.violations = 0

        # a request inside a still-open Retry-After window is the documented
        # escalation trigger (any configured trigger is approximated as
        # request_while_429 — see limiters.yaml schedule_basis notes)
        if state.retry_until is not None and now < state.retry_until:
            if escalation:
                durations = escalation["durations_minutes"]
                ban_minutes = durations[min(state.violations, len(durations) - 1)]
                state.violations += 1
                state.last_violation = now
                state.ban_until = now + ban_minutes * 60
                state.retry_until = None
                return False, escalation["status"], ban_minutes * 60.0
            # no escalation configured: it is just another 429
            return False, spec["on_exhaustion"]["status"], state.retry_until - now

        weight = spec.get("request_weight", 1)
        allowed, retry_after = consume(spec, state, now, weight,
                                       budget_factor)
        if allowed:
            return True, None, None
        if spec["on_exhaustion"].get("retry_after_header") and retry_after:
            state.retry_until = now + retry_after
        return False, spec["on_exhaustion"]["status"], retry_after

# --- venue endpoint semantics ----------------------------------------------

def _error(venue: str, status: int, message: str,
           elapsed_ms: float) -> Response:
    bodies = {
        "binance": {"code": -1100, "msg": message},
        "okx": {"code": "51000", "msg": message, "data": []},
        "kucoin": {"code": "400100", "msg": message},
    }
    body = bodies.get(venue, {"error": message})
    return Response(status, dict(JSON_HEADERS),
                    json.dumps(body, separators=(",", ":")), elapsed_ms)


def _int_param(params: dict[str, Any], name: str) -> int | None:
    value = params.get(name)
    return None if value in (None, "") else int(value)


def _serve_binance(store: CandleStore, params: dict[str, Any],
                   t: datetime) -> tuple[int, Any] | str:
    if params.get("symbol") not in store.symbols("binance"):
        return "Invalid symbol."
    if params.get("interval") != "5m":
        return "Invalid interval (replay serves 5m only)."
    limit = min(_int_param(params, "limit") or 500, 1000)
    start_ms, end_ms = (_int_param(params, "startTime"),
                        _int_param(params, "endTime"))
    rows = store.rows("binance", params["symbol"], t,
                      start_ms // 1000 if start_ms is not None else None,
                      end_ms // 1000 if end_ms is not None else None)
    # startTime given: first `limit` from startTime; otherwise latest `limit`
    rows = rows[:limit] if start_ms is not None else rows[-limit:]
    return 200, rows


def _serve_okx(store: CandleStore, params: dict[str, Any],
               t: datetime) -> tuple[int, Any] | str:
    if params.get("instId") not in store.symbols("okx"):
        return "Instrument ID does not exist"
    if params.get("bar", "5m") != "5m":
        return "bar not available in replay (5m only)"
    limit = min(_int_param(params, "limit") or 100, 300)
    after, before = _int_param(params, "after"), _int_param(params, "before")
    rows = store.rows("okx", params["instId"], t,
                      before // 1000 + 1 if before is not None else None,
                      (after - 1) // 1000 if after is not None else None)
    rows = list(reversed(rows))[:limit]  # newest first
    return 200, {"code": "0", "msg": "", "data": rows}


def _serve_kucoin(store: CandleStore, params: dict[str, Any],
                  t: datetime) -> tuple[int, Any] | str:
    if params.get("tradeType", "SPOT") != "SPOT":
        return "unsupported tradeType"
    if params.get("symbol") not in store.symbols("kucoin"):
        return "symbol not found"
    if params.get("interval") != "5min":
        return "interval not available in replay (5min only)"
    start_s, end_s = _int_param(params, "startAt"), _int_param(params, "endAt")
    rows = store.rows("kucoin", params["symbol"], t, start_s, end_s)
    rows = list(reversed(rows))[:1500]  # newest first, venue cap
    return 200, {"code": "200000",
                 "data": {"tradeType": "SPOT", "symbol": params["symbol"],
                          "list": rows}}


ENDPOINTS = {
    ("binance", "/api/v3/klines"): _serve_binance,
    ("okx", "/api/v5/market/candles"): _serve_okx,
    ("okx", "/api/v5/market/history-candles"): _serve_okx,
    ("kucoin", "/api/ua/v1/market/kline"): _serve_kucoin,
}


# --- transforms ------------------------------------------------------------

def _multiply_prices(doc: Any, multiplier: int) -> Any:
    def scale(row: list[Any]) -> list[Any]:
        out = list(row)
        for index in PRICE_INDICES:
            out[index] = str(Decimal(str(out[index])) * multiplier)
        return out

    if isinstance(doc, list):  # binance
        return [scale(r) for r in doc]
    out = json.loads(json.dumps(doc))
    if isinstance(out.get("data"), list):  # okx
        out["data"] = [scale(r) for r in out["data"]]
    elif isinstance(out.get("data"), dict):  # kucoin
        out["data"]["list"] = [scale(r) for r in out["data"]["list"]]
    return out


def _rename_fields(doc: Any, renames: dict[str, str]) -> Any:
    if not isinstance(doc, dict):
        return doc
    return {renames.get(k, k): v for k, v in doc.items()}


class ReplayProxy:
    """One trace's worth of simulated venue HTTP, driven by sim time."""

    def __init__(self, store: CandleStore, trace: Trace,
                 limiters: LimiterBank | None = None):
        self.store = store
        self.trace = trace
        self.limiters = limiters if limiters is not None else LimiterBank()
        self._last_t: datetime | None = None

    def _rng(self, venue: str, path: str, t: datetime) -> random.Random:
        return random.Random(f"{self.trace.name}|{venue}|{path}|{t.isoformat()}")

    def request(self, venue: str, path: str, params: dict[str, Any],
                t: datetime, timeout_s: float = DEFAULT_TIMEOUT_S) -> Response:
        if t.tzinfo is None:
            t = t.replace(tzinfo=UTC)
        if self._last_t is not None and t < self._last_t:
            raise ValueError(f"sim time went backward: {t} < {self._last_t}")
        self._last_t = t
        rng = self._rng(venue, path, t)
        timeout_ms = timeout_s * 1000.0
        latency = (BASE_LATENCY_MS.get(venue, 150.0)
                   * rng.uniform(0.8, 1.3))

        # Layer M — limiter first: a banned client sees nothing else
        factor = 1.0
        for event in self.trace.active(venue, "M", t):
            if event["mode"] == "capacity_modulation":
                factor *= float(event["params"]["budget_factor"])
        allowed, status, retry_after = self.limiters.check(venue, t, factor)
        if not allowed:
            response = _error(venue, status or 429, "Too many requests.",
                              latency)
            if retry_after is not None:
                response.headers["retry-after"] = str(
                    max(1, math.ceil(retry_after)))
            return response

        # Layer E — edge failures mask the provider behind them
        edge = {e["mode"]: e for e in self.trace.active(venue, "E", t)}
        if "geoblock" in edge:
            return Response(edge["geoblock"]["params"]["status"],
                            dict(HTML_HEADERS), CHALLENGE_BODY, latency)
        if "dns_fail" in edge:
            return Response(None, {}, None, 50.0, error="dns_error")
        if "net_blip" in edge:
            return Response(None, {}, None, timeout_ms, error="timeout")
        if "cdn_challenge" in edge:
            return Response(edge["cdn_challenge"]["params"]["status"],
                            dict(HTML_HEADERS), CHALLENGE_BODY, latency)

        # Layer P — provider-side
        provider = self.trace.active(venue, "P", t)
        serve_time = t
        renames: dict[str, str] = {}
        multiplier: int | None = None
        for event in provider:
            mode, event_params = event["mode"], event["params"]
            if mode in ("outage", "maintenance"):
                return self._blocked(venue, event_params.get("behavior",
                                                            "http_503"),
                                     latency, timeout_ms)
            if mode == "degraded":
                if rng.random() < float(event_params.get("fail_fraction", 0.5)):
                    behavior = rng.choice(["http_503", "timeout"])
                    return self._blocked(venue, behavior, latency, timeout_ms)
                latency *= float(event_params.get("latency_multiplier", 1.0))
            elif mode == "slow_bleed":
                low, high = event_params["latency_ramp"]
                progress = ((t - event["start_dt"]).total_seconds()
                            / (event["end_dt"] - event["start_dt"]).total_seconds())
                latency *= low + (high - low) * progress
            elif mode == "stale_200":
                serve_time = min(serve_time, event["start_dt"])
            elif mode == "wrong_data":
                multiplier = int(event_params["price_multiplier"])
            elif mode == "schema_change":
                renames.update(event_params["field_renames"])

        if latency >= timeout_ms:
            return Response(None, {}, None, timeout_ms, error="timeout")

        handler = ENDPOINTS.get((venue, path))
        if handler is None:
            return _error(venue, 404, f"unknown endpoint {path!r}", latency)
        served = handler(self.store, params, serve_time)
        if isinstance(served, str):
            return _error(venue, 400, served, latency)
        status_code, doc = served
        if multiplier is not None:
            doc = _multiply_prices(doc, multiplier)
        if renames:
            doc = _rename_fields(doc, renames)
        return Response(status_code, dict(JSON_HEADERS),
                        json.dumps(doc, separators=(",", ":")), latency)

    def _blocked(self, venue: str, behavior: str, latency: float,
                 timeout_ms: float) -> Response:
        if behavior == "timeout":
            return Response(None, {}, None, timeout_ms, error="timeout")
        if behavior == "connect_refused":
            return Response(None, {}, None, 15.0, error="connect_refused")
        status = 502 if behavior == "http_502" else 503
        return Response(status, dict(JSON_HEADERS),
                        json.dumps({"msg": "Service Unavailable"},
                                   separators=(",", ":")), latency)
