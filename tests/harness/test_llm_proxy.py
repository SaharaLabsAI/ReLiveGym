"""M4 gate: proxy metering, budget cap, and error paths against a mocked
upstream."""

import json
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

import harness.llm_proxy as llm_proxy
from harness.api import make_app
from harness.runtime import Sim
from tests.conftest import make_config, make_weather_task, write_weather_csv

UTC = timezone.utc
START = datetime(2021, 6, 1, tzinfo=UTC)

CANNED = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "model": "test-model",
    "choices": [{"index": 0, "finish_reason": "stop",
                 "message": {"role": "assistant", "content": "hi"}}],
    "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
}

RATES = {"test-model": {"in": 1e-6, "out": 2e-6}}
# expected per-call: 100*1e-6 + 50*2e-6 = 0.0002 (real token cost only —
# no flat charge)
TOKEN_COST = 0.0002


def make_sim(tmp_path, **cfg_overrides) -> Sim:
    csv = write_weather_csv(tmp_path / "w.csv", START, [20.0] * 24 * 7)
    cfg = make_config(
        weather_csv=csv,
        cost=dict(llm_token_rates=RATES),
        **cfg_overrides,
    )
    return Sim(cfg, tmp_path, tmp_path, make_weather_task(cfg))


@pytest.fixture
def stack(tmp_path, monkeypatch):
    sim = make_sim(tmp_path)

    async def fake_upstream(path, body):
        assert sim.llm_inflight == 1  # watchdog suspended during the call
        return CANNED, None

    monkeypatch.setattr(llm_proxy, "_upstream", fake_upstream)
    client = TestClient(make_app(sim))
    headers = {"Authorization": f"Bearer {sim.token}"}
    return client, sim, headers


def post(client, headers, body=None):
    return client.post("/llm/chat/completions", headers=headers,
                       json=body or {"model": "test-model",
                                     "messages": [{"role": "user", "content": "q"}]})


def test_passthrough_and_metering(stack):
    client, sim, headers = stack
    res = post(client, headers)
    assert res.status_code == 200
    assert res.json()["choices"][0]["message"]["content"] == "hi"

    # full request/response persisted to llm_log.jsonl, linked to the ledger
    # event — and NOT retained in memory (keep_events=False)
    assert sim.llm_log is not None
    assert sim.llm_log.events == []
    call = json.loads(sim.llm_log._path.read_text().splitlines()[0])
    assert call["request"]["messages"][0]["content"] == "q"
    assert call["response"]["choices"][0]["message"]["content"] == "hi"
    assert call["ledger_seq"] == [e for e in sim.ledger.events
                                  if e["type"] == "llm"][0]["seq"]

    events = [e for e in sim.ledger.events if e["type"] == "llm"]
    assert len(events) == 1
    assert events[0]["cost"] == pytest.approx(TOKEN_COST)
    assert events[0]["prompt_tokens"] == 100
    assert sim.llm_spend == pytest.approx(TOKEN_COST)
    assert sim.wallet_spend == pytest.approx(TOKEN_COST)
    assert sim.llm_inflight == 0


def test_budget_cap_causes_llm_outage(tmp_path, monkeypatch):
    sim = make_sim(tmp_path, budget_usd=0.0001)

    async def fake_upstream(path, body):
        return CANNED, None

    monkeypatch.setattr(llm_proxy, "_upstream", fake_upstream)
    client = TestClient(make_app(sim))
    headers = {"Authorization": f"Bearer {sim.token}"}

    assert post(client, headers).status_code == 200  # spend 0 < budget
    res = post(client, headers)  # spend 0.0002 >= budget
    assert res.status_code == 503
    assert "budget_exhausted" in sim.flags
    assert any(e["type"] == "llm_rejected" for e in sim.ledger.events)
    # only the first call was booked
    assert len([e for e in sim.ledger.events if e["type"] == "llm"]) == 1


def test_llm_domain_cap_causes_outage_within_global_budget(tmp_path,
                                                          monkeypatch):
    """The llm domain cap (the experimenter's real provider bill) binds
    independently of the global wallet: LLM calls 503 once llm spend hits
    it, even with global budget left."""
    sim = make_sim(tmp_path, budget_usd=50.0,
                   domain_budgets={"llm": 0.0001})

    async def fake_upstream(path, body):
        return CANNED, None

    monkeypatch.setattr(llm_proxy, "_upstream", fake_upstream)
    client = TestClient(make_app(sim))
    headers = {"Authorization": f"Bearer {sim.token}"}

    assert post(client, headers).status_code == 200  # llm spend 0 < cap
    res = post(client, headers)
    assert res.status_code == 503
    assert "llm_budget_exhausted" in sim.flags
    assert "budget_exhausted" not in sim.flags  # global wallet untouched
    rej = [e for e in sim.ledger.events if e["type"] == "llm_rejected"]
    assert rej and "domain" in rej[0]["reason"]


def test_fee_domain_cap_refuses_only_that_type(tmp_path):
    """A fee-domain cap refuses that ledger type's paid calls once its
    tally hits the cap; other paid types keep drawing on the global
    wallet."""
    from harness.env_tools import PaymentRequired

    sim = make_sim(tmp_path, budget_usd=50.0,
                   domain_budgets={"news_search": 0.003})
    sim.bill("news_search", 0.002)
    sim.bill("news_search", 0.002)  # tally 0.002 < cap when checked
    with pytest.raises(PaymentRequired):
        sim.bill("news_search", 0.002)  # tally 0.004 >= cap
    assert "news_search_budget_exhausted" in sim.flags
    sim.bill("article_call", 0.002)  # other domains unaffected
    assert "budget_exhausted" not in sim.flags
    assert sim.wallet_spend == pytest.approx(0.006)


STREAM_CHUNKS = [
    {"id": "c", "object": "chat.completion.chunk", "model": "test-model",
     "choices": [{"index": 0, "finish_reason": None,
                  "delta": {"role": "assistant", "content": "hi "}}]},
    {"id": "c", "object": "chat.completion.chunk", "model": "test-model",
     "choices": [{"index": 0, "finish_reason": "stop",
                  "delta": {"content": "there"}}]},
    {"id": "c", "object": "chat.completion.chunk", "model": "test-model",
     "choices": [],
     "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}},
]


def test_streaming_relayed_and_billed(stack, monkeypatch, tmp_path):
    """stream=true is relayed as SSE (external harnesses' SDKs stream by
    default); usage from the final
    chunk is billed after the stream, the merged message is logged."""
    client, sim, headers = stack
    seen = {}

    async def fake_stream(path, body):
        seen["body"] = body

        async def gen():
            for c in STREAM_CHUNKS:
                yield dict(c)
        return gen()

    monkeypatch.setattr(llm_proxy, "_upstream_stream", fake_stream)
    with client.stream("POST", "/llm/chat/completions", headers=headers,
                       json={"model": "test-model", "messages": [],
                             "stream": True}) as res:
        assert res.status_code == 200
        assert res.headers["content-type"].startswith("text/event-stream")
        body = "".join(res.iter_text())
    lines = [l for l in body.split("\n") if l.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    events = [json.loads(l[6:]) for l in lines[:-1]]
    assert [e["choices"] for e in events] == [c["choices"] for c in STREAM_CHUNKS]
    assert seen["body"]["stream"] is True
    assert seen["body"]["stream_options"]["include_usage"] is True
    assert sim.wallet_spend == pytest.approx(TOKEN_COST)
    row = [e for e in sim.ledger.events if e["type"] == "llm"][-1]
    assert row["streamed"] is True and row["prompt_tokens"] == 100
    assert row["cost"] == pytest.approx(TOKEN_COST)
    assert sim.llm_inflight == 0
    logged = [json.loads(l) for l in
              (tmp_path / "llm_log.jsonl").read_text().splitlines()][-1]
    assert logged["response"]["choices"][0]["message"]["content"] == "hi there"
    assert logged["response"]["choices"][0]["finish_reason"] == "stop"


def test_streaming_responses_path_rejected(stack):
    client, sim, headers = stack
    res = client.post("/llm/responses", headers=headers,
                      json={"model": "test-model", "input": [], "stream": True})
    assert res.status_code == 400


def test_unknown_llm_path(stack):
    client, sim, headers = stack
    res = client.post("/llm/embeddings", headers=headers, json={})
    assert res.status_code == 404


def test_auth_required(stack):
    client, sim, _ = stack
    res = client.post("/llm/chat/completions", json={"model": "m", "messages": []})
    assert res.status_code == 401


def test_responses_endpoint_metered(tmp_path, monkeypatch):
    sim = make_sim(tmp_path)
    canned = {
        "id": "resp-1", "object": "response", "model": "test-model",
        "output": [{"type": "message", "content": [{"type": "output_text",
                                                    "text": "hi"}]}],
        "usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
    }

    async def fake_upstream(path, body):
        assert path == "responses"
        return canned, 0.123  # provider cost must be ignored when rates exist

    monkeypatch.setattr(llm_proxy, "_upstream", fake_upstream)
    client = TestClient(make_app(sim))
    headers = {"Authorization": f"Bearer {sim.token}"}
    res = client.post("/llm/responses", headers=headers,
                      json={"model": "test-model", "input": "q"})
    assert res.status_code == 200
    events = [e for e in sim.ledger.events if e["type"] == "llm"]
    assert events[0]["cost"] == pytest.approx(TOKEN_COST)
    assert events[0]["prompt_tokens"] == 100
    assert events[0]["endpoint"] == "responses"


def test_unconfigured_model_refused(tmp_path, monkeypatch):
    """A model in neither the run's rates nor the global table is an
    experimenter error: HTTP 400, nothing forwarded, nothing booked."""
    sim = make_sim(tmp_path)
    sim.cfg.cost.llm_token_rates.clear()

    async def fake_upstream(path, body):
        raise AssertionError("unpriced call must not reach the provider")

    monkeypatch.setattr(llm_proxy, "_upstream", fake_upstream)
    client = TestClient(make_app(sim))
    headers = {"Authorization": f"Bearer {sim.token}"}
    res = post(client, headers)
    assert res.status_code == 400
    assert "cost-config" in res.json()["detail"]
    assert "llm_model_unconfigured" in sim.flags
    rej = [e for e in sim.ledger.events if e["type"] == "llm_rejected"]
    assert rej and rej[0]["reason"] == "model not in cost config"
    assert not [e for e in sim.ledger.events if e["type"] == "llm"]
    assert sim.llm_spend == 0.0


def test_global_cost_table_prices_unpinned_model(tmp_path, monkeypatch):
    """A model absent from the run's rates but present in
    configs/model_costs.yaml books at the global rates."""
    import harness.model_costs as model_costs

    sim = make_sim(tmp_path)
    sim.cfg.cost.llm_token_rates.clear()
    monkeypatch.setattr(model_costs, "_cache",
                        {"test-model": {"in": 2e-6, "out": 4e-6}})

    async def fake_upstream(path, body):
        return CANNED, None

    monkeypatch.setattr(llm_proxy, "_upstream", fake_upstream)
    client = TestClient(make_app(sim))
    headers = {"Authorization": f"Bearer {sim.token}"}
    assert post(client, headers).status_code == 200
    events = [e for e in sim.ledger.events if e["type"] == "llm"]
    assert events[0]["cost"] == pytest.approx(100 * 2e-6 + 50 * 4e-6)


def test_cached_tokens_billed_at_cached_rate(tmp_path, monkeypatch):
    """Cached prompt tokens (a subset of prompt_tokens) book at
    cached_in; without cached_in they book at the full input rate."""
    sim = make_sim(tmp_path)
    sim.cfg.cost.llm_token_rates["test-model"]["cached_in"] = 1e-7
    canned = {**CANNED,
              "usage": {"prompt_tokens": 100, "completion_tokens": 50,
                        "total_tokens": 150,
                        "prompt_tokens_details": {"cached_tokens": 80}}}

    async def fake_upstream(path, body):
        return canned, None

    monkeypatch.setattr(llm_proxy, "_upstream", fake_upstream)
    client = TestClient(make_app(sim))
    headers = {"Authorization": f"Bearer {sim.token}"}
    assert post(client, headers).status_code == 200
    events = [e for e in sim.ledger.events if e["type"] == "llm"]
    expected = 20 * 1e-6 + 80 * 1e-7 + 50 * 2e-6
    assert events[0]["cost"] == pytest.approx(expected)
    assert events[0]["cached_tokens"] == 80
    assert sim.llm_spend == pytest.approx(expected)


def test_provider_cost_logged_not_booked(tmp_path, monkeypatch):
    """The provider-issued bill lands in the ledger (cost_provider) and
    the results total (llm_provider_usd) but never in the wallet."""
    sim = make_sim(tmp_path)

    async def fake_upstream(path, body):
        return CANNED, 0.123

    monkeypatch.setattr(llm_proxy, "_upstream", fake_upstream)
    client = TestClient(make_app(sim))
    headers = {"Authorization": f"Bearer {sim.token}"}
    assert post(client, headers).status_code == 200
    events = [e for e in sim.ledger.events if e["type"] == "llm"]
    assert events[0]["cost"] == pytest.approx(TOKEN_COST)
    assert events[0]["cost_provider"] == pytest.approx(0.123)
    assert events[0]["provider"] == "openai"
    assert sim.wallet_spend == pytest.approx(TOKEN_COST)
    assert sim.llm_provider_spend == pytest.approx(0.123)
    assert sim.results()["resources"]["llm_provider_usd"] == pytest.approx(0.123)


def test_openrouter_forwarding(tmp_path, monkeypatch):
    """provider=openrouter: the forwarded body gains the routing prefix
    and the usage-accounting flag; the agent-visible response and the
    ledger keep the bare model name."""
    sim = make_sim(tmp_path, agent=dict(scaffold="baseline_poller",
                                        model="openrouter:test-model"))
    assert sim.cfg.agent.provider == "openrouter"
    assert sim.cfg.agent.model == "test-model"
    seen = {}

    async def fake_upstream(path, body):
        seen.update(body)
        return {**CANNED, "model": "openrouter/test-model"}, 0.0004

    monkeypatch.setattr(llm_proxy, "_upstream", fake_upstream)
    client = TestClient(make_app(sim))
    headers = {"Authorization": f"Bearer {sim.token}"}
    res = post(client, headers)
    assert res.status_code == 200
    assert seen["model"] == "openrouter/test-model"
    assert seen["extra_body"] == {"usage": {"include": True}}
    assert res.json()["model"] == "test-model"  # prefix is server-side only
    events = [e for e in sim.ledger.events if e["type"] == "llm"]
    assert events[0]["model"] == "test-model"
    assert events[0]["provider"] == "openrouter"
    assert events[0]["cost"] == pytest.approx(TOKEN_COST)  # config rates
    assert events[0]["cost_provider"] == pytest.approx(0.0004)
    # the agent's own request body was logged unrewritten
    logged = json.loads(sim.llm_log._path.read_text().splitlines()[0])
    assert logged["request"]["model"] == "test-model"


def test_upstream_error_becomes_502(tmp_path, monkeypatch):
    sim = make_sim(tmp_path)

    async def broken_upstream(path, body):
        raise RuntimeError("provider down")

    monkeypatch.setattr(llm_proxy, "_upstream", broken_upstream)
    client = TestClient(make_app(sim))
    headers = {"Authorization": f"Bearer {sim.token}"}
    res = post(client, headers)
    assert res.status_code == 502
    assert any(e["type"] == "llm_error" for e in sim.ledger.events)
    assert sim.llm_inflight == 0


# -- the in-flight ceiling --------------------------------------------------


def test_upstream_timeout_is_a_504_unbilled_and_flagged(tmp_path, monkeypatch):
    import asyncio
    sim = make_sim(tmp_path, llm_timeout_seconds=0.2, llm_retries=1,
                   llm_retry_backoff_seconds=0.0)

    async def slow_upstream(path, body):
        await asyncio.sleep(5)
        return CANNED, None

    monkeypatch.setattr(llm_proxy, "_upstream", slow_upstream)
    client = TestClient(make_app(sim))
    headers = {"Authorization": f"Bearer {sim.token}"}
    r = post(client, headers)
    assert r.status_code == 504 and "2 attempt(s) of 0.2s" in r.text
    assert sim.ledger.count_by_type()["llm_retry"] == 1  # the first attempt
    assert sim.llm_inflight == 0 and sim.oldest_llm_inflight_seconds() == 0.0
    assert sim.wallet_spend == 0.0 and sim.llm_spend == 0.0
    assert sim.ledger.count_by_type()["llm_error"] == 1
    assert "llm_errors" in sim.flags

    async def ok_upstream(path, body):  # the run keeps going
        return CANNED, None

    monkeypatch.setattr(llm_proxy, "_upstream", ok_upstream)
    assert post(client, headers).status_code == 200
    assert sim.llm_spend == pytest.approx(TOKEN_COST)


def test_activity_reports_the_oldest_inflight_call(tmp_path, monkeypatch):
    import asyncio
    from harness.supervisor import Scheduler
    sim = make_sim(tmp_path, llm_timeout_seconds=5.0, llm_retries=0)
    sim.scheduler = Scheduler(sim)  # what harness.serve wires up
    seen = []

    async def watched_upstream(path, body):
        await asyncio.sleep(0.15)
        seen.append(sim.scheduler.activity())
        return CANNED, None

    monkeypatch.setattr(llm_proxy, "_upstream", watched_upstream)
    client = TestClient(make_app(sim))
    headers = {"Authorization": f"Bearer {sim.token}"}
    assert post(client, headers).status_code == 200
    (act,) = seen
    assert act["llm_inflight"] == 1
    assert 0.1 < act["oldest_llm_inflight_seconds"] < 5.0
    assert act["llm_timeout_seconds"] == 5.0
    idle = sim.scheduler.activity()
    assert idle["llm_inflight"] == 0 and idle["oldest_llm_inflight_seconds"] == 0.0


# -- server-side retries ------------------------------------------


class _Upstream(Exception):
    def __init__(self, status_code, msg="boom"):
        super().__init__(msg)
        self.status_code = status_code


def test_transient_upstream_failure_is_retried_and_recovers(tmp_path, monkeypatch):
    sim = make_sim(tmp_path, llm_retries=2, llm_retry_backoff_seconds=0.0)
    calls = []

    async def flaky_upstream(path, body):
        calls.append(1)
        if len(calls) == 1:
            raise _Upstream(503, "service unavailable")
        if len(calls) == 2:
            raise ConnectionError("reset by peer")
        return CANNED, None

    monkeypatch.setattr(llm_proxy, "_upstream", flaky_upstream)
    client = TestClient(make_app(sim))
    headers = {"Authorization": f"Bearer {sim.token}"}
    r = post(client, headers)
    assert r.status_code == 200 and len(calls) == 3
    counts = sim.ledger.count_by_type()
    assert counts["llm_retry"] == 2 and "llm_error" not in counts
    assert "llm_errors" not in sim.flags  # recovered: the run is not flagged
    assert sim.llm_spend == pytest.approx(TOKEN_COST)  # billed once
    assert sim.llm_inflight == 0


def test_non_transient_upstream_error_fails_at_once(tmp_path, monkeypatch):
    sim = make_sim(tmp_path, llm_retries=2, llm_retry_backoff_seconds=0.0)
    calls = []

    async def bad_request(path, body):
        calls.append(1)
        raise _Upstream(400, "context length exceeded")

    monkeypatch.setattr(llm_proxy, "_upstream", bad_request)
    client = TestClient(make_app(sim))
    headers = {"Authorization": f"Bearer {sim.token}"}
    r = post(client, headers)
    assert r.status_code == 502 and "after 1 attempt(s)" in r.text and len(calls) == 1
    counts = sim.ledger.count_by_type()
    assert counts["llm_error"] == 1 and "llm_retry" not in counts
    assert "llm_errors" in sim.flags


def test_retries_exhausted_reports_the_final_failure(tmp_path, monkeypatch):
    sim = make_sim(tmp_path, llm_retries=2, llm_retry_backoff_seconds=0.0)

    async def always_429(path, body):
        raise _Upstream(429, "rate limited")

    monkeypatch.setattr(llm_proxy, "_upstream", always_429)
    client = TestClient(make_app(sim))
    headers = {"Authorization": f"Bearer {sim.token}"}
    r = post(client, headers)
    assert r.status_code == 502 and "after 3 attempt(s)" in r.text
    counts = sim.ledger.count_by_type()
    assert counts["llm_retry"] == 2 and counts["llm_error"] == 1
    assert sim.wallet_spend == 0.0


def test_activity_allowance_covers_every_attempt(tmp_path):
    from harness.supervisor import Scheduler
    sim = make_sim(tmp_path, llm_timeout_seconds=300.0, llm_retries=2,
                   llm_retry_backoff_seconds=1.0)
    sim.scheduler = Scheduler(sim)
    # 3 attempts x 300 s + backoffs 1 s + 2 s
    assert sim.llm_allowance_seconds() == 903.0
    assert sim.scheduler.activity()["llm_timeout_seconds"] == 903.0
    (tmp_path / "b").mkdir()
    assert make_sim(tmp_path / "b", llm_retries=0).llm_allowance_seconds() == 300.0


def test_gemini_tool_call_ids_made_unique():
    """Two calls that the provider gave the same id leave the proxy with
    different ids (one tag per LLM call, streamed deltas and whole
    messages alike); only Gemini models are touched."""
    def tc():
        return {"index": 0, "id": "call_179100", "type": "function",
                "function": {"name": "bash", "arguments": "{}"}}

    first = {"choices": [{"index": 0, "message": {"tool_calls": [tc()]}}]}
    chunk = {"choices": [{"index": 0, "delta": {"tool_calls": [tc()]}}]}
    tail = {"choices": [{"index": 0, "delta": {"tool_calls": [
        {"index": 0, "function": {"arguments": "{}"}}]}}]}
    llm_proxy._unique_tool_call_ids(first, "aaaa")
    for c in (chunk, tail):
        llm_proxy._unique_tool_call_ids(c, "bbbb")
    assert first["choices"][0]["message"]["tool_calls"][0]["id"] == "call_179100_aaaa"
    assert chunk["choices"][0]["delta"]["tool_calls"][0]["id"] == "call_179100_bbbb"
    assert "id" not in tail["choices"][0]["delta"]["tool_calls"][0]
    assert llm_proxy._needs_unique_ids("google/gemini-3.5-flash")
    assert not llm_proxy._needs_unique_ids("gpt-5.6-luna")
