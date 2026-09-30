"""The ext: TM-B base program `tmB-base`: per_market_main.py plus client-side
TM-B waiting (authored programs run inside the agent's jail)."""

from __future__ import annotations

import asyncio
import json
import re
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from harness import authored, llm_proxy
from harness.run import run_experiment
from tests.tasks.breakout_news_pm.test_cells_e2e import _resp, make_cfg, read_jsonl

REPO_ROOT = Path(__file__).resolve().parents[3]
TASK_DIR = REPO_ROOT / "tasks" / "breakout_news_pm"
EXT = TASK_DIR / "agent" / "ext_candidates" / "tmB-base"
RUNTIME = REPO_ROOT / "scaffolds" / "runtime"


def assemble_candidate(dst: Path) -> Path:
    dst.mkdir(parents=True)
    for f in ("main.py", "local_tools.py"):
        shutil.copyfile(EXT / f, dst / f)
    shutil.copytree(RUNTIME, dst / "runtime", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copyfile(REPO_ROOT / "harness" / "authored_skill.md", dst / "authored_skill.md")
    (dst / "jail_seed").mkdir()
    (dst / "jail_seed" / "sleep.py").write_text(authored.SLEEP_PROGRAM)
    shutil.copyfile(TASK_DIR / "agent" / "example_gatekeeper.py",
                    dst / "jail_seed" / "example_gatekeeper.py")
    return dst


@pytest.mark.skipif(
    not (TASK_DIR / "data" / "built" / "markets.jsonl").exists(),
    reason="built breakout_news_pm world absent (DATA.md)")
def test_ext_tmb_candidate_end_to_end(tmp_path, monkeypatch):
    async def fake_upstream(path, body):
        last = body["messages"][-1]["content"]
        if '"bytes"' in last:  # write_file result -> run what we wrote
            return _resp(json.dumps(
                {"tool": "run_program", "args": {"path": "wait.py"},
                 "thought": "park in my program"}), body["model"])
        text = " ".join(m["content"] for m in body["messages"])
        times = re.findall(r'(?:woke at |"now": ?")([0-9TZ:\-\.\+]+)', text)
        now = datetime.fromisoformat(times[-1].replace("Z", "+00:00"))
        until = (now + timedelta(hours=36)).strftime("%Y-%m-%dT%H:%M:%SZ")
        return _resp(json.dumps(
            {"tool": "write_file",
             "args": {"path": "wait.py",
                      "content": f'import envkit\nenvkit.wait("{until}")\n'},
             "thought": "monitor my market"}), body["model"])

    monkeypatch.setattr(llm_proxy, "_upstream", fake_upstream)
    cand = assemble_candidate(tmp_path / "cand")
    cfg = make_cfg(run_id="bnpm-ext-tmb",
                   sim_end="2026-03-02T00:00:00Z",
                   task={"name": "breakout_news_pm",
                         "markets": [
                             {"market_id": "616902", "start": "2026-03-01T00:00:00Z",
                              "end": "2026-03-02T00:00:00Z"},
                             {"market_id": "678777", "start": "2026-03-01T00:00:00Z",
                              "end": "2026-03-02T00:00:00Z"}]},
                   cell={"tm": "B", "tlrn": "none", "sig": "none", "alg": "none"},
                   agent={"scaffold": "ext:tmb-base", "model": "mock-luna"})
    results = asyncio.run(run_experiment(
        cfg, repo_root=tmp_path, run_dir=tmp_path / "run", scaffold_src=cand))

    assert "failed-degenerate" not in results["flags"]
    ws = tmp_path / "run" / "workspace"
    assert not list((ws / "logs").glob("crash-*.log"))
    # each market agent wrote and ran its own program from its jail
    assert (ws / "agents" / "m-616902" / "wait.py").exists()
    assert (ws / "agents" / "m-678777" / "wait.py").exists()
    # waits are ledgered per waiter; no server-side run_program exists
    ledger = read_jsonl(tmp_path / "run" / "ledger.jsonl")
    assert {e["waiter"] for e in ledger if e["type"] == "sleep"} >= {"m-616902", "m-678777"}
    assert not [e for e in ledger if e["type"] == "run_program"]
