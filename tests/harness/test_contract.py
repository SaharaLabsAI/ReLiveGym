"""The program contract: anything satisfying it is a valid
program. A minimal non-LLM program — read the manifest, install a cron,
call one tool, exit — runs e2e with zero runtime/ imports; the run records
what the program could see and call in workspace_manifest.json."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

from harness.run import run_experiment
from tests.conftest import make_config, write_weather_csv

START = datetime(2021, 6, 1, tzinfo=timezone.utc)

CONTRACT_PROGRAM = '''\
"""Minimal contract fixture: no LLM, no runtime/ imports."""
import json
import os
import urllib.request

BASE = os.environ["ENV_URL"]
HEADERS = {"Authorization": "Bearer " + os.environ["ENV_TOKEN"],
           "Content-Type": "application/json"}


def call(name, **args):
    req = urllib.request.Request(BASE + "/call/" + name,
                                 data=json.dumps(args).encode(),
                                 method="POST", headers=HEADERS)
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


req = urllib.request.Request(BASE + "/tools", headers=HEADERS)
with urllib.request.urlopen(req) as r:
    manifest = json.loads(r.read())["tools"]
with open("seen_manifest.json", "w") as f:
    json.dump(manifest, f)

call("set_crontab", entries=[{"id": "noon", "cron_expr": "0 12 * * *"}])
with open("times.jsonl", "a") as f:
    f.write(json.dumps(call("get_time")) + "\\n")
'''


def test_minimal_program_satisfies_contract(tmp_path):
    csv = write_weather_csv(tmp_path / "w.csv", START, [20.0] * 24 * 7)
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "main.py").write_text(CONTRACT_PROGRAM)
    cfg = make_config(weather_csv=csv, data_cutoff=START,
                      run_id="contract-test")
    results = asyncio.run(run_experiment(
        cfg, repo_root=tmp_path, run_dir=tmp_path / "run",
        scaffold_src=fixture))
    assert "failed-degenerate" not in results["flags"]

    ws = tmp_path / "run" / "workspace"
    # the program discovered its provisioned tools through the manifest
    seen = {t["name"] for t in
            json.loads((ws / "seen_manifest.json").read_text())}
    assert {"get_time", "set_crontab", "sleep", "get_weather",
            "notify"} <= seen
    assert "wait_until" not in seen  # tm=C default: sleep, not wait_until
    # it was re-invoked at its cron firings (start + daily noons)
    times = (ws / "times.jsonl").read_text().splitlines()
    assert len(times) >= 6

    # the run recorded exactly what the program could see and call
    manifest = json.loads(
        (tmp_path / "run" / "workspace_manifest.json").read_text())
    assert "main.py" in manifest["files"]
    assert {t["name"] for t in manifest["tools"]} == seen
