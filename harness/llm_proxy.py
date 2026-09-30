"""Metered LLM pass-through.

The agent's LLM client points at base_url = $ENV_URL/llm with the run
token as its (dummy) API key; real provider keys live only in the
helper's process environment, picked up by litellm. Both OpenAI-shape
endpoints are supported — `chat/completions` and `responses`. Requests
forward to the run's api provider (cfg.agent.provider: openai direct, or
openrouter with the routing prefix added server-side) — the agent only
ever sees the bare model name.

Every call books its cost from the pinned rate table — the per-run
cfg.cost.llm_token_rates override, else the repo-global
configs/model_costs.yaml — cache-aware: cached prompt tokens bill at the
model's cached_in rate. A model in neither table is refused with HTTP
400 (flag llm_model_unconfigured); nothing ever bills at a provider
list price. The provider-issued bill (OpenRouter's in-band usage.cost,
else litellm's completion_cost estimate) is NOT booked but logged per
call (`cost_provider`) and totaled in results (llm_provider_usd), so
config-vs-provider drift is visible in early runs.

`budget_usd` (the run's one wallet, LLM + API fees) and the "llm" domain
cap (`domain_budgets["llm"]` — the experimenter's REAL provider bill) are
enforced *before* forwarding: once either is spent, /llm returns 503 (the
agent experiences an LLM outage) and the run is flagged budget_exhausted /
llm_budget_exhausted. Sim time does not
advance during a call; the real-time watchdog is suspended while one is in
flight (sim.llm_inflight) — bounded: each attempt is cut off after
cfg.llm_timeout_seconds, transient failures (timeout, connection,
408/409/429/5xx) are retried cfg.llm_retries times with doubling backoff
(ledger `llm_retry` per recovered attempt), and only the final failure
reaches the actor — 504 when every attempt timed out, else 502 — with a
ledger `llm_error`, the flag `llm_errors` and nothing billed. The runner
treats an in-flight call older than the whole allowance (GET /activity's
llm_timeout_seconds = attempts x timeout + backoffs) plus a watchdog period
as a hang.

Streaming (for external harnesses whose SDKs stream by
default — OpenCode's does): `stream: true` on chat/completions is relayed
as server-sent events with `stream_options.include_usage` forced, so the
provider's final chunk carries the usage that is billed after the stream
ends; the merged message is what llm_log records. Admission (pricing,
budget), retries and the 502/504 mapping happen BEFORE the first byte is
sent; a failure mid-stream is delivered in-band as an `error` event and
ledgered `llm_error`. The `responses` path stays non-streaming.
"""

from __future__ import annotations

import asyncio
import json
import uuid

from fastapi import HTTPException
from starlette.responses import StreamingResponse

from harness.model_costs import resolve_rates
from harness.runtime import Sim

SUPPORTED_PATHS = {"chat/completions", "responses"}
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def _retryable(exc: BaseException) -> bool:
    """Transient upstream failures worth another attempt: timeouts and
    connection errors, or a provider status in RETRY_STATUS (litellm / openai
    exceptions carry `.status_code`). Anything else — bad request, auth,
    context length, unknown model, a parsing bug of ours — fails at once."""
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError)):
        return True
    code = getattr(exc, "status_code", None)
    if code is not None:
        try:
            return int(code) in RETRY_STATUS
        except (TypeError, ValueError):
            return False
    name = type(exc).__name__
    return any(k in name for k in ("Timeout", "Connection", "ServiceUnavailable",
                                   "RateLimit", "InternalServer"))


async def _upstream(path: str, body: dict) -> tuple[dict, float | None]:
    """Forward to the real provider. Returns (response dict, the
    provider-issued bill in $: OpenRouter's in-band usage.cost when
    present, else litellm's price-table estimate, else None). Patched in
    tests."""
    import litellm

    if path == "chat/completions":
        response = await litellm.acompletion(**body)
    else:
        response = await litellm.aresponses(**body)
    dumped = response.model_dump()
    provider_cost = (dumped.get("usage") or {}).get("cost")
    if provider_cost is None:
        try:
            provider_cost = float(
                litellm.completion_cost(completion_response=response))
        except Exception:
            provider_cost = None
    return dumped, provider_cost


async def _upstream_stream(path: str, body: dict):
    """Open a streamed chat completion at the real provider; returns an
    async iterator of chunk dicts. Errors opening the stream surface
    here (so admission-time retries and status codes still apply).
    Patched in tests / by the mock."""
    import litellm

    resp = await litellm.acompletion(**body)

    async def gen():
        async for chunk in resp:
            yield chunk.model_dump()
    return gen()


def _unique_tool_call_ids(payload: dict, tag: str) -> None:
    """Suffix every tool-call id in a chat completion (or one streamed
    chunk of it) with this call's tag. OpenRouter synthesizes Gemini's ids
    as `call_<small int>`, which collide within a long session; once two
    calls to different tools share an id, Google answers every later
    request of that session with 400 INVALID_ARGUMENT. The client echoes
    ids verbatim, so upstream sees the unique ones from then on."""
    for choice in payload.get("choices") or []:
        part = choice.get("delta") or choice.get("message") or {}
        for tc in part.get("tool_calls") or []:
            if tc.get("id"):
                tc["id"] = f"{tc['id']}_{tag}"


def _needs_unique_ids(model: str) -> bool:
    return model.startswith("google/")


def _merge_chunk(message: dict, chunk: dict) -> str | None:
    """Fold one streamed chunk into a chat-completion message (content and
    tool_calls); returns the finish_reason when the chunk carries one."""
    finish = None
    for choice in chunk.get("choices") or []:
        delta = choice.get("delta") or {}
        if delta.get("content"):
            message["content"] = (message.get("content") or "") + delta["content"]
        for tc in delta.get("tool_calls") or []:
            calls = message.setdefault("tool_calls", [])
            idx = tc.get("index", len(calls))
            while len(calls) <= idx:
                calls.append({"id": None, "type": "function",
                              "function": {"name": "", "arguments": ""}})
            slot = calls[idx]
            if tc.get("id"):
                slot["id"] = tc["id"]
            fn = tc.get("function") or {}
            if fn.get("name"):
                slot["function"]["name"] += fn["name"]
            if fn.get("arguments"):
                slot["function"]["arguments"] += fn["arguments"]
        if choice.get("finish_reason"):
            finish = choice["finish_reason"]
    return finish


def _usage_tokens(response: dict) -> tuple[int, int, int]:
    """Normalize usage across chat (prompt/completion) and responses
    (input/output) shapes; cached = the prompt-cache-read subset of the
    prompt count."""
    usage = response.get("usage") or {}
    prompt = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
    completion = usage.get("completion_tokens") or usage.get("output_tokens") or 0
    details = (usage.get("prompt_tokens_details")
               or usage.get("input_tokens_details") or {})
    cached = details.get("cached_tokens") or 0 if isinstance(details, dict) else 0
    return prompt, completion, min(cached, prompt)


async def _attempt(sim: Sim, model: str, fn, timeout: float, retries: int):
    """Run `fn()` (one upstream call) with the retry/backoff policy; maps
    the final failure to 504/502 after ledgering llm_error."""
    for attempt in range(1, retries + 2):
        try:
            return await asyncio.wait_for(fn(), timeout=timeout)
        except HTTPException:
            raise
        except Exception as e:
            timed_out = isinstance(e, asyncio.TimeoutError)
            err = f"timeout after {timeout:g}s" if timed_out else str(e)
            if attempt <= retries and (timed_out or _retryable(e)):
                sim.ledger.append("llm_retry", sim.clock.now, model=model,
                                  attempt=attempt, error=err)
                await asyncio.sleep(
                    sim.cfg.llm_retry_backoff_seconds * 2 ** (attempt - 1))
                continue
            sim.ledger.append("llm_error", sim.clock.now, model=model,
                              attempts=attempt, error=err)
            if "llm_errors" not in sim.flags:
                sim.flags.append("llm_errors")
            if timed_out:
                raise HTTPException(
                    504, f"upstream LLM timeout: {attempt} attempt(s) of "
                         f"{timeout:g}s; nothing was billed — retry")
            raise HTTPException(
                502, f"upstream LLM error after {attempt} attempt(s): {e}")


def _bill(sim: Sim, *, model: str, path: str, provider: str, rates: dict,
          request: dict, response: dict, provider_cost, streamed: bool) -> None:
    """Book one completed call (caller holds sim.lock, or runs with no
    await in between — asyncio makes this block atomic either way)."""
    prompt_tokens, completion_tokens, cached_tokens = _usage_tokens(response)
    token_cost = ((prompt_tokens - cached_tokens) * rates.get("in", 0.0)
                  + cached_tokens * rates.get("cached_in", rates.get("in", 0.0))
                  + completion_tokens * rates.get("out", 0.0))
    sim.wallet_spend += token_cost
    sim.llm_spend += token_cost
    if provider_cost:
        sim.llm_provider_spend += provider_cost
    if token_cost:
        sim.domain_spend["llm"] = sim.domain_spend.get("llm", 0.0) + token_cost
    detail = dict(model=model, endpoint=path, provider=provider,
                  prompt_tokens=prompt_tokens,
                  completion_tokens=completion_tokens,
                  cached_tokens=cached_tokens, cost_provider=provider_cost)
    if streamed:
        detail["streamed"] = True
    event = sim.ledger.append("llm", sim.clock.now, cost=token_cost, **detail)
    if sim.llm_log is not None:
        sim.llm_log.append(
            "llm_call", sim.clock.now, ledger_seq=event["seq"],
            endpoint=path, model=model, request=request, response=response)


async def _relay(sim: Sim, token: int, stream, *, model: str, path: str,
                 provider: str, rates: dict, request: dict):
    """The SSE body: forward chunks, fold the message, bill at the end
    (in `finally`, so a client that disconnects still books what the
    provider reported so far)."""
    message: dict = {"role": "assistant", "content": ""}
    usage = None
    provider_cost = None
    finish = None
    id_tag = uuid.uuid4().hex[:8] if _needs_unique_ids(model) else None
    try:
        async for chunk in stream:
            if chunk.get("usage"):
                usage = chunk["usage"]
                provider_cost = usage.get("cost")
            if id_tag:
                _unique_tool_call_ids(chunk, id_tag)
            finish = _merge_chunk(message, chunk) or finish
            if isinstance(chunk.get("model"), str):
                chunk["model"] = chunk["model"].removeprefix("openrouter/")
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        sim.ledger.append("llm_error", sim.clock.now, model=model, attempts=1,
                          error=f"mid-stream: {e}")
        if "llm_errors" not in sim.flags:
            sim.flags.append("llm_errors")
        yield f"data: {json.dumps({'error': {'message': str(e)}})}\n\n"
    finally:
        sim.llm_end(token)
        sim.touch()
        response = {"object": "chat.completion", "model": model,
                    "choices": [{"index": 0, "message": message,
                                 "finish_reason": finish}],
                    "usage": usage or {}}
        _bill(sim, model=model, path=path, provider=provider, rates=rates,
              request=request, response=response, provider_cost=provider_cost,
              streamed=True)


async def handle_llm(sim: Sim, path: str, body: dict):
    if path not in SUPPORTED_PATHS:
        raise HTTPException(404, f"unsupported LLM endpoint: /llm/{path}")
    streaming = bool(body.get("stream"))
    if streaming and path != "chat/completions":
        raise HTTPException(400, "streaming is supported for chat/completions "
                                 "only; use stream=false")
    model = body.get("model", "")

    rates = resolve_rates(sim.cfg, model)
    if rates is None:
        if "llm_model_unconfigured" not in sim.flags:
            sim.flags.append("llm_model_unconfigured")
        sim.ledger.append("llm_rejected", sim.clock.now, model=model,
                          reason="model not in cost config")
        raise HTTPException(400, f"model {model!r} has no cost-config entry")

    llm_cap = sim.cfg.domain_budgets.get("llm")
    if sim.wallet_spend >= sim.cfg.budget_usd:
        if "budget_exhausted" not in sim.flags:
            sim.flags.append("budget_exhausted")
        sim.ledger.append("llm_rejected", sim.clock.now, model=model,
                          reason="budget_usd exhausted")
        raise HTTPException(503, "LLM temporarily unavailable")
    if llm_cap is not None and sim.llm_spend >= llm_cap:
        if "llm_budget_exhausted" not in sim.flags:
            sim.flags.append("llm_budget_exhausted")
        sim.ledger.append("llm_rejected", sim.clock.now, model=model,
                          reason="llm domain budget exhausted")
        raise HTTPException(503, "LLM temporarily unavailable")

    provider = sim.cfg.agent.provider
    upstream_body = body
    if provider == "openrouter":
        # route via OpenRouter: litellm wants the routing prefix, and
        # usage accounting must be asked for to get the in-band bill
        upstream_body = {**body, "model": f"openrouter/{model}",
                         "extra_body": {"usage": {"include": True}}}

    timeout, retries = sim.cfg.llm_timeout_seconds, sim.cfg.llm_retries
    token = sim.llm_begin()
    if streaming:
        upstream_body = {**upstream_body, "stream": True,
                         "stream_options": {**(body.get("stream_options") or {}),
                                            "include_usage": True}}
        try:
            stream = await _attempt(
                sim, model, lambda: _upstream_stream(path, upstream_body),
                timeout, retries)
        except BaseException:
            sim.llm_end(token)
            sim.touch()
            raise
        return StreamingResponse(
            _relay(sim, token, stream, model=model, path=path,
                   provider=provider, rates=rates, request=body),
            media_type="text/event-stream")
    try:
        response, provider_cost = await _attempt(
            sim, model, lambda: _upstream(path, upstream_body), timeout, retries)
    finally:
        sim.llm_end(token)
        sim.touch()  # the completed call counts as activity

    if isinstance(response.get("model"), str):
        # the routing prefix is server-side only
        response["model"] = response["model"].removeprefix("openrouter/")
    if _needs_unique_ids(model):
        _unique_tool_call_ids(response, uuid.uuid4().hex[:8])

    async with sim.lock:
        _bill(sim, model=model, path=path, provider=provider, rates=rates,
              request=body, response=response, provider_cost=provider_cost,
              streamed=False)
    return response
