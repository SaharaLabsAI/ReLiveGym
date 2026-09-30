"""Party episodes: one sim, N
waiter-bound bridges. `prepare` fans out sub-workspaces and declares the
roster; the bridge forces waiter identity on waits; the barrier advances
the clock only when every roster member waits; the stall watchdog evicts
a session that stops waiting while others are parked."""

from __future__ import annotations

import argparse
import asyncio
import json
import threading
from pathlib import Path

import pytest

from harness.mcp import _Http, _party_spec, main as mcp_main
from tests.control_plane.test_mcp_episode import (
    _session,
    payload_of,
    weather_base_yaml,
)

pytestmark = pytest.mark.filterwarnings("ignore::ResourceWarning")

SIM_END = "2021-06-03T00:00:00Z"


def prepare_party(tmp_path, capsys, tm="A"):
    base = weather_base_yaml(tmp_path, tm)
    rc = mcp_main(["prepare", "--task", "weather_fixture", "--tm", tm,
                   "--base", str(base), "--out", str(tmp_path / "eps"),
                   "--party", "w1,w2", "--stall-seconds", "0",
                   "--mock-llm"])
    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    ep_path = Path(summary["episode"])
    return ep_path, json.loads(ep_path.read_text())


def test_party_spec_per_market():
    args = argparse.Namespace(per_market=True, party=None)
    raw = {"task": {"markets": [{"market_id": "616902"},
                                {"market_id": "954518"}]}}
    assert _party_spec(args, raw) == [("m616902", "616902"),
                                      ("m954518", "954518")]
    assert _party_spec(argparse.Namespace(per_market=False, party="a, b"),
                       {}) == [("a", None), ("b", None)]
    assert _party_spec(argparse.Namespace(per_market=False, party=None),
                       {}) is None
    with pytest.raises(SystemExit):  # no roster in the base config
        _party_spec(argparse.Namespace(per_market=True, party=None),
                    {"task": {}})
    with pytest.raises(SystemExit):  # a party of one is a solo episode
        _party_spec(argparse.Namespace(per_market=False, party="a"), {})


def test_prepare_party_fanout(tmp_path, capsys):
    ep_path, ep = prepare_party(tmp_path, capsys, tm="B")
    assert sorted(ep["waiters"]) == ["w1", "w2"]
    assert ep["party"]["trigger_waiter"] == "w1"
    assert ep["evictions"] == []
    assert "watchdog_pid" not in ep  # --stall-seconds 0 disables it
    for w in ("w1", "w2"):
        sub = Path(ep["waiters"][w])
        assert sub.parent == Path(ep["workspace"])
        ins = (sub / "INSTRUCTION.md").read_text()
        assert "advances only when every agent is waiting" in ins
        assert "# Waiting by program" in ins  # tm=B kit per sub-workspace
        assert "# Your market" not in ins  # --party: no market scoping
        assert (sub / "envkit.py").exists() and (sub / "sleep.py").exists()
        mc = json.loads((sub / ".mcp.json").read_text())
        assert mc["mcpServers"]["env"]["args"][-2:] == ["--waiter", w]
        codex = (sub / ".codex" / "config.toml").read_text()
        assert f'"--waiter", "{w}"' in codex
    # the workspace root is only a container in party mode
    assert not (Path(ep["workspace"]) / "INSTRUCTION.md").exists()
    st = _Http(ep["env_url"], ep["token"]).get("/status")
    assert st["party"] == {"roster": ["w1", "w2"], "parked": [],
                           "trigger_waiter": "w1"}
    assert mcp_main(["settle", "--episode", str(ep_path)]) == 0
    capsys.readouterr()


def test_party_barrier_trigger_and_injection(tmp_path, capsys):
    ep_path, ep = prepare_party(tmp_path, capsys, tm="A")

    async def w1_fn(sess):  # the trigger_waiter: rides the barrier to the end
        wakes = []
        while True:
            r = payload_of(await sess.call_tool("sleep",
                                                {"until": SIM_END}))
            wakes.append(r)
            if r.get("experiment_over") or r.get("aborted"):
                return wakes

    async def w2_fn(sess):
        # the agent-supplied waiter_id is overridden by the bridge: were
        # it honored, "w1" would collide with the parked w1 and error
        return payload_of(await sess.call_tool(
            "sleep", {"until": SIM_END, "waiter_id": "w1"}))

    async def drive():
        return await asyncio.wait_for(asyncio.gather(
            _session(ep_path, w1_fn, waiter="w1"),
            _session(ep_path, w2_fn, waiter="w2")), timeout=90)

    w1_wakes, w2_last = asyncio.run(drive())
    assert w1_wakes[-1].get("experiment_over")
    assert w2_last.get("experiment_over")
    # both waiter identities on the sleep ledger events
    ledger = Path(ep["server_dir"]) / "ledger.jsonl"
    seen = {json.loads(line).get("waiter")
            for line in ledger.read_text().splitlines()
            if '"waiter"' in line}
    assert {"w1", "w2"} <= seen
    assert mcp_main(["settle", "--episode", str(ep_path)]) == 0
    capsys.readouterr()


@pytest.mark.slow
def test_stall_eviction(tmp_path, capsys):
    ep_path, ep = prepare_party(tmp_path, capsys, tm="A")
    wd = threading.Thread(target=mcp_main, args=(
        ["watchdog", "--episode", str(ep_path),
         "--stall-seconds", "1.5", "--poll-seconds", "0.3"],), daemon=True)
    wd.start()

    async def w1_fn(sess):
        return payload_of(await sess.call_tool(
            "sleep", {"until": "2021-06-01T02:00:00Z"}))

    async def drive():
        return await asyncio.wait_for(
            _session(ep_path, w1_fn, waiter="w1"), timeout=30)

    r = asyncio.run(drive())  # released only once w2 is evicted
    assert r.get("woke_for") == "sleep"
    assert r["now"].startswith("2021-06-01T02:00")
    epd = json.loads(ep_path.read_text())
    assert [e["waiter"] for e in epd["evictions"]] == ["w2"]
    assert "party_eviction" in epd["flags"]
    st = _Http(epd["env_url"], epd["token"]).get("/status")
    assert st["party"]["roster"] == ["w1"]
    assert mcp_main(["settle", "--episode", str(ep_path)]) == 0
    capsys.readouterr()
    wd.join(timeout=20)
    assert not wd.is_alive()
