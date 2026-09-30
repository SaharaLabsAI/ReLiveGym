"""The runner's kill rule (scaffolds/runtime/actor.py::watchdog_verdict):
idle past the watchdog kills; an LLM call in flight suspends that — but
only until the call has outlived the server's LLM timeout by a further
watchdog period (a hung provider request would otherwise freeze a
resident run indefinitely, status alive, no error anywhere)."""

from scaffolds.runtime.actor import watchdog_verdict


def _act(**kw):
    base = {"idle_seconds": 0.0, "llm_inflight": 0, "watchdog_seconds": 120.0,
            "llm_timeout_seconds": 300.0, "oldest_llm_inflight_seconds": 0.0}
    return {**base, **kw}


def test_idle_past_watchdog_kills():
    assert watchdog_verdict(_act(idle_seconds=121.0)) == "idle"
    assert watchdog_verdict(_act(idle_seconds=119.0)) is None


def test_inflight_call_suspends_the_watchdog_until_the_ceiling():
    young = _act(idle_seconds=10_000.0, llm_inflight=1,
                 oldest_llm_inflight_seconds=419.0)
    assert watchdog_verdict(young) is None  # 300 + 120 not yet exceeded
    hung = _act(idle_seconds=10_000.0, llm_inflight=1,
                oldest_llm_inflight_seconds=421.0)
    assert watchdog_verdict(hung) == "llm_inflight"


def test_server_without_timeout_fields_keeps_the_unbounded_rule():
    old = {"idle_seconds": 10_000.0, "llm_inflight": 1, "watchdog_seconds": 120.0}
    assert watchdog_verdict(old) is None
    assert watchdog_verdict({**old, "llm_inflight": 0}) == "idle"
