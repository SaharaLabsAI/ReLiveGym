"""Replay proxy engine: recorded-row fidelity, limiter machines, and every
incident transform, on synthetic recordings (the real datasets are
git-ignored)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tasks.crypto_price_consistency.incidents.proxy import (
    CandleStore, LimiterBank, ReplayProxy, Trace)

T0 = datetime(2026, 3, 1, tzinfo=UTC)
T0_S = int(T0.timestamp())


def _binance_row(index: int) -> list:
    open_ms = (T0_S + index * 300) * 1000
    price = f"{50000 + index}.10000000"
    return [open_ms, price, price, price, price, "1.0", open_ms + 299999,
            "50000.0", 10, "0.5", "25000.0", "0"]


def _okx_row(index: int) -> list:
    open_ms = (T0_S + index * 300) * 1000
    price = f"{50000 + index}.1"
    return [str(open_ms), price, price, price, price, "1.0", "50000", "50000", "1"]


def _kucoin_row(index: int) -> list:
    price = f"{50000 + index}.1"
    return [T0_S + index * 300, price, price, price, price, "1.0", "50000.0"]


@pytest.fixture()
def store(tmp_path: Path) -> CandleStore:
    raw = tmp_path / "synth" / "raw"
    raw.mkdir(parents=True)
    # 24 candles = 2 hours from T0; overlapping pages exercise row dedup
    pages = {
        "binance": [("https://x/api/v3/klines?symbol=BTCUSDT&interval=5m",
                     [_binance_row(i) for i in rng])
                    for rng in (range(0, 16), range(12, 24))],
        "okx": [("https://x/api/v5/market/history-candles?instId=BTC-USDT&bar=5m",
                 [_okx_row(i) for i in reversed(rng)])
                for rng in (range(0, 16), range(12, 24))],
        "kucoin": [("https://x/api/ua/v1/market/kline?tradeType=SPOT&symbol=BTC-USDT&interval=5min",
                    [_kucoin_row(i) for i in reversed(rng)])
                   for rng in (range(0, 16), range(12, 24))],
    }
    for venue, fetches in pages.items():
        lines = [json.dumps({"fetched_at": "2026-08-04T00:00:00+00:00",
                             "url": url, "payload":
                             {"binance": rows,
                              "okx": {"code": "0", "msg": "", "data": rows},
                              "kucoin": {"code": "200000", "data":
                                         {"tradeType": "SPOT",
                                          "symbol": "BTC-USDT",
                                          "list": rows}}}[venue]})
                 for url, rows in fetches]
        (raw / f"{venue}.jsonl").write_text("\n".join(lines) + "\n")
    return CandleStore([tmp_path / "synth"])


def _trace(tmp_path: Path, events: list[dict]) -> Trace:
    header = {"trace": "test", "generator_version": "v1",
              "window": ["2026-03-01T00:00:00Z", "2026-08-01T00:00:00Z"]}
    lines = [json.dumps(header)]
    for index, event in enumerate(events, start=1):
        lines.append(json.dumps({"id": f"evt-{index:03d}",
                                 "provenance": "authored", **event}))
    path = tmp_path / "trace.jsonl"
    path.write_text("\n".join(lines) + "\n")
    return Trace(path)


def _proxy(store: CandleStore, tmp_path: Path,
           events: list[dict] | None = None) -> ReplayProxy:
    return ReplayProxy(store, _trace(tmp_path, events or []))


def _event(layer: str, venue: str, mode: str, start: str, end: str,
           params: dict) -> dict:
    return {"layer": layer, "venue": venue, "mode": mode,
            "start": start, "end": end, "params": params}


# --- base replay -----------------------------------------------------------

def test_binance_rows_byte_identical_and_visibility(store, tmp_path):
    proxy = _proxy(store, tmp_path)
    t = T0 + timedelta(minutes=32)  # candles 0..5 closed, 6 forming
    response = proxy.request("binance", "/api/v3/klines",
                             {"symbol": "BTCUSDT", "interval": "5m"}, t)
    assert response.status == 200
    rows = response.json()
    assert rows == [_binance_row(i) for i in range(6)]  # forming row absent


def test_binance_time_range_and_limit(store, tmp_path):
    proxy = _proxy(store, tmp_path)
    t = T0 + timedelta(hours=3)
    start_ms = (T0_S + 4 * 300) * 1000
    response = proxy.request(
        "binance", "/api/v3/klines",
        {"symbol": "BTCUSDT", "interval": "5m", "startTime": start_ms,
         "limit": 3}, t)
    assert [r[0] for r in response.json()] == [
        (T0_S + i * 300) * 1000 for i in (4, 5, 6)]
    # no startTime: latest `limit` completed rows
    response = proxy.request("binance", "/api/v3/klines",
                             {"symbol": "BTCUSDT", "interval": "5m",
                              "limit": 2}, t)
    assert [r[0] for r in response.json()] == [
        (T0_S + i * 300) * 1000 for i in (22, 23)]


def test_okx_descending_with_after_cursor(store, tmp_path):
    proxy = _proxy(store, tmp_path)
    t = T0 + timedelta(hours=3)
    after = (T0_S + 5 * 300) * 1000
    response = proxy.request("okx", "/api/v5/market/history-candles",
                             {"instId": "BTC-USDT", "bar": "5m",
                              "after": after, "limit": 3}, t)
    doc = response.json()
    assert doc["code"] == "0"
    assert [r[0] for r in doc["data"]] == [
        str((T0_S + i * 300) * 1000) for i in (4, 3, 2)]


def test_kucoin_descending_seconds_range(store, tmp_path):
    proxy = _proxy(store, tmp_path)
    t = T0 + timedelta(hours=3)
    response = proxy.request(
        "kucoin", "/api/ua/v1/market/kline",
        {"tradeType": "SPOT", "symbol": "BTC-USDT", "interval": "5min",
         "startAt": T0_S + 2 * 300, "endAt": T0_S + 5 * 300}, t)
    doc = response.json()
    assert doc["code"] == "200000"
    assert [r[0] for r in doc["data"]["list"]] == [
        T0_S + i * 300 for i in (5, 4, 3, 2)]


def test_unknown_symbol_and_endpoint_are_venue_shaped(store, tmp_path):
    proxy = _proxy(store, tmp_path)
    response = proxy.request("binance", "/api/v3/klines",
                             {"symbol": "DOGEUSDT", "interval": "5m"}, T0)
    assert response.status == 400 and response.json()["code"] == -1100
    response = proxy.request("okx", "/api/v5/nope", {}, T0)
    assert response.status == 404 and response.json()["code"] == "51000"


# --- layer P transforms ----------------------------------------------------

def test_outage_behaviors(store, tmp_path):
    start, end = "2026-03-01T01:00:00Z", "2026-03-01T02:00:00Z"
    t = datetime(2026, 3, 1, 1, 30, tzinfo=UTC)
    for behavior, expect in (("http_503", 503), ("http_502", 502)):
        proxy = _proxy(store, tmp_path, [_event(
            "P", "binance", "outage", start, end, {"behavior": behavior})])
        assert proxy.request("binance", "/api/v3/klines",
                             {"symbol": "BTCUSDT", "interval": "5m"},
                             t).status == expect
    proxy = _proxy(store, tmp_path, [_event(
        "P", "binance", "outage", start, end, {"behavior": "timeout"})])
    response = proxy.request("binance", "/api/v3/klines",
                             {"symbol": "BTCUSDT", "interval": "5m"}, t)
    assert response.error == "timeout" and response.status is None
    assert response.elapsed_ms == 10_000.0
    # outside the event window: normal service
    assert proxy.request("binance", "/api/v3/klines",
                         {"symbol": "BTCUSDT", "interval": "5m"},
                         datetime(2026, 3, 1, 2, 30, tzinfo=UTC)).status == 200


def test_stale_200_freezes_data(store, tmp_path):
    proxy = _proxy(store, tmp_path, [_event(
        "P", "binance", "stale_200", "2026-03-01T00:30:00Z",
        "2026-03-01T06:00:00Z", {})])
    frozen = proxy.request("binance", "/api/v3/klines",
                           {"symbol": "BTCUSDT", "interval": "5m"},
                           datetime(2026, 3, 1, 2, 0, tzinfo=UTC))
    assert frozen.status == 200
    rows = frozen.json()
    # last visible candle is the one closed by 00:30 (opens 00:25, closes
    # exactly at the freeze instant), despite t = 02:00
    assert rows[-1][0] == (T0_S + 5 * 300) * 1000


def test_wrong_data_multiplies_prices(store, tmp_path):
    proxy = _proxy(store, tmp_path, [_event(
        "P", "okx", "wrong_data", "2026-03-01T01:00:00Z",
        "2026-03-01T02:00:00Z", {"price_multiplier": 100})])
    doc = proxy.request("okx", "/api/v5/market/candles",
                        {"instId": "BTC-USDT", "bar": "5m", "limit": 1},
                        datetime(2026, 3, 1, 1, 30, tzinfo=UTC)).json()
    assert doc["data"][0][1] == "5001710.0"  # 50017.1 * 100, scale kept
    assert doc["data"][0][5] == "1.0"        # volume untouched


def test_schema_change_renames_envelope(store, tmp_path):
    proxy = _proxy(store, tmp_path, [_event(
        "P", "kucoin", "schema_change", "2026-03-01T01:00:00Z",
        "2026-08-01T00:00:00Z", {"field_renames": {"data": "items"}})])
    doc = proxy.request("kucoin", "/api/ua/v1/market/kline",
                        {"tradeType": "SPOT", "symbol": "BTC-USDT",
                         "interval": "5min", "startAt": T0_S,
                         "endAt": T0_S + 7200},
                        datetime(2026, 3, 1, 1, 30, tzinfo=UTC)).json()
    assert "data" not in doc and doc["items"]["list"]


def test_degraded_is_deterministic_per_request(store, tmp_path):
    events = [_event("P", "binance", "degraded", "2026-03-01T00:00:00Z",
                     "2026-03-02T00:00:00Z",
                     {"fail_fraction": 0.5, "latency_multiplier": 4.0})]
    args = ("binance", "/api/v3/klines",
            {"symbol": "BTCUSDT", "interval": "5m"})
    outcomes = []
    for hour in range(24):
        t = T0 + timedelta(hours=hour)
        first = _proxy(store, tmp_path, events).request(*args, t)
        second = _proxy(store, tmp_path, events).request(*args, t)
        assert (first.status, first.error) == (second.status, second.error)
        outcomes.append(first.status == 200)
    assert 3 < sum(outcomes) < 21  # both outcomes occur across the day
    success = next(r for r in
                   (_proxy(store, tmp_path, events).request(
                       *args, T0 + timedelta(hours=h)) for h in range(24))
                   if r.status == 200)
    assert success.elapsed_ms > 4 * 0.8 * 120  # latency multiplier applied


def test_slow_bleed_ramps_latency_into_timeout(store, tmp_path):
    events = [_event("P", "binance", "slow_bleed", "2026-03-01T00:00:00Z",
                     "2026-03-02T00:00:00Z", {"latency_ramp": [1, 300]})]
    proxy = _proxy(store, tmp_path, events)
    args = ("binance", "/api/v3/klines",
            {"symbol": "BTCUSDT", "interval": "5m"})
    early = proxy.request(*args, T0 + timedelta(minutes=10))
    late = proxy.request(*args, T0 + timedelta(hours=23))
    assert early.status == 200
    assert late.error == "timeout"  # ~300x base latency > 10s timeout


# --- layer E ---------------------------------------------------------------

def test_edge_modes(store, tmp_path):
    start, end = "2026-03-01T01:00:00Z", "2026-03-01T02:00:00Z"
    t = datetime(2026, 3, 1, 1, 30, tzinfo=UTC)
    args = ("okx", "/api/v5/market/candles", {"instId": "BTC-USDT", "bar": "5m"})
    blip = _proxy(store, tmp_path, [_event(
        "E", "okx", "net_blip", start, end, {"behavior": "timeout"})])
    assert blip.request(*args, t).error == "timeout"
    dns = _proxy(store, tmp_path, [_event(
        "E", "okx", "dns_fail", start, end, {"behavior": "dns_error"})])
    assert dns.request(*args, t).error == "dns_error"
    challenge = _proxy(store, tmp_path, [_event(
        "E", "okx", "cdn_challenge", start, end,
        {"status": 200, "body": "html_challenge"})])
    response = challenge.request(*args, t)
    assert response.status == 200
    assert response.headers["content-type"] == "text/html"
    with pytest.raises(json.JSONDecodeError):
        response.json()  # the naive-parser trap
    geo = _proxy(store, tmp_path, [_event(
        "E", "okx", "geoblock", start, end, {"status": 451})])
    assert geo.request(*args, t).status == 451


def test_clock_skew_exposed_for_env_layer(store, tmp_path):
    trace = _trace(tmp_path, [_event(
        "E", "client", "clock_skew", "2026-03-01T01:00:00Z",
        "2026-03-01T12:00:00Z", {"skew_seconds": -42.0})])
    assert trace.client_clock_skew(
        datetime(2026, 3, 1, 6, tzinfo=UTC)) == -42.0
    assert trace.client_clock_skew(
        datetime(2026, 3, 2, 6, tzinfo=UTC)) == 0.0


# --- layer M ---------------------------------------------------------------

def test_binance_429_then_418_escalation(store, tmp_path):
    proxy = _proxy(store, tmp_path)
    args = ("binance", "/api/v3/klines",
            {"symbol": "BTCUSDT", "interval": "5m"})
    t = T0
    # budget 6000, weight 2 -> 3000 requests fit in one minute window
    for _ in range(3000):
        assert proxy.request(*args, t).status == 200
    limited = proxy.request(*args, t)
    assert limited.status == 429 and "retry-after" in limited.headers
    # ignoring Retry-After escalates to the documented 418 ban (2 min first)
    banned = proxy.request(*args, t + timedelta(seconds=1))
    assert banned.status == 418
    still = proxy.request(*args, t + timedelta(seconds=90))
    assert still.status == 418
    # ban expires; fresh minute window serves again
    assert proxy.request(*args, t + timedelta(seconds=150)).status == 200


def test_respecting_retry_after_avoids_ban(store, tmp_path):
    proxy = _proxy(store, tmp_path)
    args = ("binance", "/api/v3/klines",
            {"symbol": "BTCUSDT", "interval": "5m"})
    for _ in range(3000):
        proxy.request(*args, T0)
    assert proxy.request(*args, T0).status == 429
    # waiting out the window costs only the wait
    assert proxy.request(*args, T0 + timedelta(seconds=61)).status == 200


def test_capacity_modulation_shrinks_budget(store, tmp_path):
    proxy = _proxy(store, tmp_path, [_event(
        "M", "okx", "capacity_modulation", "2026-03-01T00:00:00Z",
        "2026-03-01T02:00:00Z", {"budget_factor": 0.05})])
    args = ("okx", "/api/v5/market/candles", {"instId": "BTC-USDT", "bar": "5m"})
    # okx budget 40 per 2s -> factor 0.05 leaves 2
    assert proxy.request(*args, T0).status == 200
    assert proxy.request(*args, T0).status == 200
    assert proxy.request(*args, T0).status == 429
    # after the modulation window the full budget is back
    later = datetime(2026, 3, 1, 3, tzinfo=UTC)
    for _ in range(10):
        assert proxy.request(*args, later).status == 200


def test_limiter_hides_provider_outage(store, tmp_path):
    # composition M -> P: while rate-limited, the outage is invisible
    proxy = _proxy(store, tmp_path, [
        _event("M", "okx", "capacity_modulation", "2026-03-01T00:00:00Z",
               "2026-03-01T02:00:00Z", {"budget_factor": 0.0}),
        _event("P", "okx", "outage", "2026-03-01T00:00:00Z",
               "2026-03-01T02:00:00Z", {"behavior": "connect_refused"})])
    args = ("okx", "/api/v5/market/candles", {"instId": "BTC-USDT", "bar": "5m"})
    assert proxy.request(*args, T0).status == 429
    # after both end, service resumes
    assert proxy.request(
        *args, datetime(2026, 3, 1, 3, tzinfo=UTC)).status == 200


def test_time_cannot_go_backward(store, tmp_path):
    proxy = _proxy(store, tmp_path)
    proxy.request("binance", "/api/v3/klines",
                  {"symbol": "BTCUSDT", "interval": "5m"}, T0 + timedelta(hours=1))
    with pytest.raises(ValueError, match="backward"):
        proxy.request("binance", "/api/v3/klines",
                      {"symbol": "BTCUSDT", "interval": "5m"}, T0)


# --- frozen artifacts smoke ------------------------------------------------

def test_loads_frozen_trace_and_limiters():
    scenarios = Path("tasks/crypto_price_consistency/incidents/scenarios")
    trace = Trace(scenarios / "stress_ban_trap.jsonl")
    assert trace.events[0]["mode"] == "capacity_modulation"
    bank = LimiterBank()
    assert bank.config["binance"]["ban_escalation"]["status"] == 418
