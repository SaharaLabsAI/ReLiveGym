"""Task + env layer: truth medians, hourly scoring ladder, http_fetch
billing through a real Sim, and INSTRUCTION.md rendering — on a synthetic
mini-world (the real datasets are git-ignored)."""

from __future__ import annotations

import asyncio
import json
import string
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from harness.config import AgentSpec, CostConfig, RunConfig
from harness.runtime import Sim
from harness.task import NotificationError
from tasks.crypto_price_consistency.task import (
    TASK_DIR, CryptoPriceConsistencyTask, TruthStore)

T0 = datetime(2026, 3, 1, tzinfo=UTC)
T0_S = int(T0.timestamp())
N_CANDLES = 24 * 12  # two days of 5-minute candles
# venue closes differ so the median is testable: binance low, kucoin high
OFFSETS = {"binance": 0.0, "okx": 10.0, "kucoin": 20.0}


def _price(index: int) -> float:
    return 50_000.0 + index


def _build_world(tmp_path: Path) -> tuple[Path, Path]:
    dataset = tmp_path / "datasets" / "btc_usdt_spot_mar_jul"
    raw = dataset / "raw"
    if raw.is_dir():  # already built for this tmp_path
        return tmp_path / "datasets", tmp_path / "scenarios"
    raw.mkdir(parents=True)
    urls = {
        "binance": "https://x/api/v3/klines?symbol=BTCUSDT&interval=5m",
        "okx": "https://x/api/v5/market/history-candles?instId=BTC-USDT&bar=5m",
        "kucoin": ("https://x/api/ua/v1/market/kline?tradeType=SPOT"
                   "&symbol=BTC-USDT&interval=5min"),
    }
    for venue, offset in OFFSETS.items():
        rows = []
        csv_lines = ["open_time_ms,close"]
        for index in range(N_CANDLES):
            open_s = T0_S + index * 300
            close = _price(index) + offset
            csv_lines.append(f"{open_s * 1000},{close}")
            if venue == "binance":
                rows.append([open_s * 1000, str(close), str(close),
                             str(close), str(close), "1.0",
                             open_s * 1000 + 299999, "1", 1, "1", "1", "0"])
            elif venue == "okx":
                rows.append([str(open_s * 1000), str(close), str(close),
                             str(close), str(close), "1", "1", "1", "1"])
            else:
                rows.append([open_s, str(close), str(close), str(close),
                             str(close), "1", "1"])
        if venue == "okx":
            payload = {"code": "0", "msg": "", "data": rows[::-1]}
        elif venue == "kucoin":
            payload = {"code": "200000", "data": {
                "tradeType": "SPOT", "symbol": "BTC-USDT",
                "list": rows[::-1]}}
        else:
            payload = rows
        (raw / f"{venue}.jsonl").write_text(json.dumps(
            {"fetched_at": "2026-08-04T00:00:00+00:00",
             "url": urls[venue], "payload": payload}) + "\n")
        norm = (dataset / "normalized" / "cohort_btc_usdt_spot"
                / f"source_{venue}" / "interval_5m")
        norm.mkdir(parents=True)
        (norm / "candles.csv").write_text("\n".join(csv_lines) + "\n")

    scenarios = tmp_path / "scenarios"
    scenarios.mkdir()
    header = {"trace": "synthetic", "generator_version": "v1",
              "window": ["2026-03-01T00:00:00Z", "2026-08-01T00:00:00Z"]}
    outage = {"id": "evt-001", "layer": "P", "venue": "binance",
              "mode": "outage", "start": "2026-03-01T06:00:00Z",
              "end": "2026-03-01T07:00:00Z",
              "params": {"behavior": "http_503"}, "provenance": "authored"}
    (scenarios / "synthetic.jsonl").write_text(
        json.dumps(header) + "\n" + json.dumps(outage) + "\n")
    return tmp_path / "datasets", scenarios


def _config(tmp_path: Path, **task_overrides) -> RunConfig:
    datasets_dir, scenarios_dir = _build_world(tmp_path)
    task = {"name": "crypto_price_consistency", "trace": "synthetic",
            "symbols": ["BTC"], "datasets_dir": str(datasets_dir),
            "scenarios_dir": str(scenarios_dir), **task_overrides}
    return RunConfig(run_id="cpc-test", task=task, sim_start=T0,
                     sim_end=T0 + timedelta(days=2),
                     agent=AgentSpec(scaffold="none"))


def _task(tmp_path: Path, **overrides) -> CryptoPriceConsistencyTask:
    return CryptoPriceConsistencyTask.from_run_config(
        _config(tmp_path, **overrides), Path("."))


def _sim(tmp_path: Path, task: CryptoPriceConsistencyTask) -> Sim:
    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True)
    return Sim(_config(tmp_path), run_dir, tmp_path / "ws", task)


# --- construction ----------------------------------------------------------

def test_rejects_window_outside_recorded_data(tmp_path):
    cfg = _config(tmp_path)
    cfg = cfg.model_copy(update={
        "sim_start": datetime(2026, 2, 1, tzinfo=UTC)})
    with pytest.raises(ValueError, match="recorded window"):
        CryptoPriceConsistencyTask.from_run_config(cfg, Path("."))


def test_rejects_unknown_trace(tmp_path):
    with pytest.raises(ValueError, match="unknown trace"):
        _task(tmp_path, trace="nope")


# --- truth -----------------------------------------------------------------

def test_truth_is_cross_venue_median(tmp_path):
    datasets_dir, _ = _build_world(tmp_path)
    truth = TruthStore(datasets_dir, ["BTC"], "mar_jul")
    # candle 0 closes at 00:05: closes are 50000/50010/50020 -> median 50010
    assert truth.truth_at("BTC", T0 + timedelta(minutes=5)) == 50_010.0
    # between closes the last completed candle holds
    assert truth.truth_at("BTC", T0 + timedelta(minutes=9)) == 50_010.0
    assert truth.truth_at("BTC", T0 + timedelta(minutes=10)) == 50_011.0
    assert truth.truth_at("BTC", T0) is None


# --- scoring ladder --------------------------------------------------------

def test_hourly_scoring_ladder(tmp_path):
    task = _task(tmp_path)
    hour1 = T0 + timedelta(hours=1)
    truth_at = task.scorer.truth.truth_at

    # hour 0: accurate report (exactly the median at submission time)
    t_submit = T0 + timedelta(minutes=30)
    task.record_notification(t_submit, {
        "kind": "report", "symbol": "BTC",
        "price": truth_at("BTC", t_submit)})
    # hour 1: report 20 bps off
    t_submit = hour1 + timedelta(minutes=30)
    task.record_notification(t_submit, {
        "kind": "report", "symbol": "BTC",
        "price": truth_at("BTC", t_submit) * 1.0020})
    # hour 2: abstain; hour 3: nothing
    task.record_notification(T0 + timedelta(hours=2, minutes=10),
                             {"kind": "abstain", "symbol": "BTC"})

    events = task.close_due(T0 + timedelta(hours=4))
    assert [e.status for e in events] == ["ok", "priced", "abstain", "miss"]
    assert events[0].detail["excess_bps"] == 0.0
    assert events[1].detail["excess_bps"] == pytest.approx(20 - 5, rel=1e-4)
    assert events[2].detail == {} and events[3].detail == {}
    m = task.metrics()
    assert m["primary"]["name"] == "mean_excess_bps"
    assert m["primary"]["direction"] == "min"
    assert m["primary"]["value"] == pytest.approx(15 / 2, rel=1e-3)
    assert m["availability"] == 0.5 and m["misses"] == 1
    # one miss + abstain rate 25% > 10% budget -> both constraints violated
    assert set(task.constraint_violations()) == {"abstain_budget_exceeded",
                                                 "missed_hours"}


def test_last_submission_in_hour_wins(tmp_path):
    task = _task(tmp_path)
    truth_at = task.scorer.truth.truth_at
    wild = truth_at("BTC", T0 + timedelta(minutes=10)) * 100  # x100 bad tick
    task.record_notification(T0 + timedelta(minutes=10),
                             {"kind": "report", "symbol": "BTC", "price": wild})
    t_fix = T0 + timedelta(minutes=50)
    task.record_notification(t_fix, {"kind": "report", "symbol": "BTC",
                                     "price": truth_at("BTC", t_fix)})
    events = task.close_due(T0 + timedelta(hours=1))
    assert [e.status for e in events] == ["ok"]
    assert events[0].detail["excess_bps"] == 0.0


def test_wildly_wrong_report_is_capped_above_miss(tmp_path):
    task = _task(tmp_path)
    truth_at = task.scorer.truth.truth_at
    task.record_notification(T0 + timedelta(minutes=10), {
        "kind": "report", "symbol": "BTC",
        "price": truth_at("BTC", T0 + timedelta(minutes=10)) * 100})
    events = task.close_due(T0 + timedelta(hours=1))
    assert events[0].status == "priced"
    assert events[0].detail["excess_bps"] == 250.0  # capped at cap_bps


def test_rejects_bad_payloads(tmp_path):
    task = _task(tmp_path)
    t = T0 + timedelta(minutes=10)
    for payload in ({"kind": "report", "symbol": "ETH", "price": 1.0},
                    {"kind": "report", "symbol": "BTC", "price": -5},
                    {"kind": "report", "symbol": "BTC", "price": None},
                    {"kind": "guess", "symbol": "BTC", "price": 1.0}):
        with pytest.raises(NotificationError):
            task.record_notification(t, payload)


def test_oracle_outcomes_only_after_settlement(tmp_path):
    task = _task(tmp_path)
    t = T0 + timedelta(minutes=10)
    task.record_notification(t, {
        "kind": "report", "symbol": "BTC",
        "price": task.scorer.truth.truth_at("BTC", t)})
    assert task.oracle_outcomes(None, T0 + timedelta(minutes=59)) == []
    task.close_due(T0 + timedelta(hours=1))
    outcomes = task.oracle_outcomes(None, T0 + timedelta(hours=1))
    assert len(outcomes) == 1
    assert outcomes[0]["status"] == "ok" and outcomes[0]["truth"]
    # since-cursor excludes already-delivered records
    assert task.oracle_outcomes(T0 + timedelta(hours=1),
                                T0 + timedelta(hours=1)) == []


def test_close_all_settles_every_full_hour(tmp_path):
    task = _task(tmp_path)
    events = task.close_all()
    assert len(events) == 48  # 2 days x 24 h, one symbol, all misses
    m = task.metrics()
    assert m["hours_scored"] == 48
    assert m["by_status"] == {"miss": 48}
    assert m["availability"] == 0.0
    assert m["primary"]["value"] is None  # nothing reported
    assert task.constraint_violations() == ["missed_hours"]


# --- env app ---------------------------------------------------------------

def test_http_fetch_serves_and_bills(tmp_path):
    task = _task(tmp_path)
    sim = _sim(tmp_path, task)
    app = task.env_apps(sim)[0]
    sim.clock.advance_to(T0 + timedelta(hours=1))
    result = asyncio.run(app.http_fetch({
        "venue": "binance", "path": "/api/v3/klines",
        "params": {"symbol": "BTCUSDT", "interval": "5m", "limit": 2}}))
    assert result["status"] == 200
    rows = json.loads(result["body"])  # body arrives unparsed
    # latest visible candle opens 00:55 and closes exactly at t = 01:00
    assert len(rows) == 2 and rows[-1][0] == (T0_S + 11 * 300) * 1000
    fetch_events = [json.loads(line) for line in
                    (sim.run_dir / "ledger.jsonl").read_text().splitlines()]
    assert fetch_events[-1]["type"] == "fetch"
    # public endpoints are free at real rates; latency is still recorded
    assert fetch_events[-1]["cost"] == 0.0
    assert fetch_events[-1]["elapsed_ms"] > 0


def test_http_fetch_sees_trace_outage(tmp_path):
    task = _task(tmp_path)
    sim = _sim(tmp_path, task)
    app = task.env_apps(sim)[0]
    sim.clock.advance_to(T0 + timedelta(hours=6, minutes=30))
    result = asyncio.run(app.http_fetch({
        "venue": "binance", "path": "/api/v3/klines",
        "params": {"symbol": "BTCUSDT", "interval": "5m"}}))
    assert result["status"] == 503
    # other venues unaffected
    okx = asyncio.run(app.http_fetch({
        "venue": "okx", "path": "/api/v5/market/candles",
        "params": {"instId": "BTC-USDT", "bar": "5m"}}))
    assert okx["status"] == 200


def test_actions_route_through_notification(tmp_path):
    task = _task(tmp_path)
    sim = _sim(tmp_path, task)
    app = task.env_apps(sim)[0]
    sim.clock.advance_to(T0 + timedelta(minutes=30))
    accepted = asyncio.run(app.report(
        {"symbol": "BTC", "price": 50_016.0}))
    assert accepted["status"] == "accepted"
    asyncio.run(app.abstain({"symbol": "BTC", "reason": "checking"}))
    events = task.close_due(T0 + timedelta(hours=1))
    assert [e.status for e in events] == ["abstain"]  # last action wins


def test_get_exchange_docs_covers_all_venues(tmp_path):
    task = _task(tmp_path)
    sim = _sim(tmp_path, task)
    app = task.env_apps(sim)[0]
    docs = asyncio.run(app.get_exchange_docs({}))
    assert set(docs["venues"]) == {"binance", "okx", "kucoin"}
    assert docs["symbols"] == ["BTC"]


# --- instruction -----------------------------------------------------------

def test_instruction_renders_completely(tmp_path):
    task = _task(tmp_path)
    template = string.Template((TASK_DIR / "INSTRUCTION.md").read_text())
    rendered = template.substitute(
        {**task.instruction_context(), "budget_usd": "$10",
         "domain_caps": " (LLM capped at $20)",
         "llm_price_table": "provider list prices"})
    assert "${" not in rendered
    assert "BTC" in rendered and "http_fetch" in rendered
