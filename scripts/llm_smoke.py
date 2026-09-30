"""Manual gate: one real LLM round-trip through the metered proxy.

Path under test: raw HTTP client -> helper /llm proxy (run token as dummy
key) -> litellm -> real provider (key from .env, helper-side only).
Verifies the call round-trips through the meter, books at the pinned
cost-config rates, and logs the provider-issued bill next to the booked
cost so the two can be eyeballed. Costs a fraction of a cent.

Usage:
  python scripts/llm_smoke.py                    # openai:gpt-5.4-mini
  python scripts/llm_smoke.py openrouter:deepseek/deepseek-v4-pro
"""

import asyncio
import json
import sys
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import dotenv
import uvicorn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from harness.api import make_app  # noqa: E402
from harness.config import RunConfig  # noqa: E402
from harness.model_costs import apply_launch_model  # noqa: E402
from harness.runtime import Sim  # noqa: E402

dotenv.load_dotenv(REPO_ROOT / ".env")
UTC = timezone.utc


def build_sim(tmp: Path, model_spec: str) -> Sim:
    from harness.task import load_task_class

    cfg = RunConfig(
        run_id="llm-smoke",
        task=dict(name="weather_fixture", location="LA",
                  data_cutoff=datetime(2023, 8, 1, tzinfo=UTC),
                  threshold_c=33.0),
        sim_start=datetime(2023, 8, 1, tzinfo=UTC),
        sim_end=datetime(2023, 8, 15, tzinfo=UTC),
        budget_usd=0.10,
        agent=dict(scaffold="baseline_poller"),
    )
    apply_launch_model(cfg, model_spec)  # rates + provider key or SystemExit
    task = load_task_class("weather_fixture").from_run_config(cfg, REPO_ROOT)
    return Sim(cfg, tmp, tmp, task)


async def main() -> None:
    model_spec = sys.argv[1] if len(sys.argv) > 1 else "openai:gpt-5.4-mini"
    sim = build_sim(Path(tempfile.mkdtemp(prefix="llm-smoke-")), model_spec)
    server = uvicorn.Server(uvicorn.Config(
        make_app(sim), host="127.0.0.1", port=0, log_level="warning"))
    server_task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    proxy_url = f"http://127.0.0.1:{port}/llm"
    print(f"proxy at {proxy_url}, model {sim.cfg.agent.model} "
          f"via {sim.cfg.agent.provider}")

    def call_through_proxy():
        req = urllib.request.Request(
            f"{proxy_url}/chat/completions",
            data=json.dumps({
                "model": sim.cfg.agent.model,  # bare name, as ENV_MODEL is
                "messages": [{"role": "user",
                              "content": "Reply with exactly: OK"}],
            }).encode(),
            headers={"Authorization": f"Bearer {sim.token}",
                     "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())

    resp = await asyncio.to_thread(call_through_proxy)
    server.should_exit = True
    await server_task

    text = resp["choices"][0]["message"]["content"]
    events = [e for e in sim.ledger.events if e["type"] == "llm"]
    assert events, "no llm event booked in the ledger!"
    e = events[0]
    provider = (f"${e['cost_provider']:.6f}" if e["cost_provider"] is not None
                else "n/a")
    print(f"model reply: {text.strip()[:80]!r}")
    print(f"ledger booked: ${e['cost']:.6f} (config rates) vs "
          f"provider-issued {provider}; tokens {e['prompt_tokens']}+"
          f"{e['completion_tokens']} ({e['cached_tokens']} cached)")
    print(f"llm spend tracked: ${sim.llm_spend:.6f} booked / "
          f"${sim.llm_provider_spend:.6f} provider, budget "
          f"${sim.cfg.budget_usd}")
    print("manual gate: PASS")


if __name__ == "__main__":
    asyncio.run(main())
