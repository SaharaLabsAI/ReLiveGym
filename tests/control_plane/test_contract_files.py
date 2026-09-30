"""gen_contract: the files an
author reads without a running sim equal what a started sim serves as
GET /contract; the envkit stub covers every program-callable tool and
none of the wait/schedule/party/jail tools."""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

from harness.contract import NOT_PROGRAM_TOOLS
from tests.control_plane.test_detached import Served, ext_config
from tests.control_plane.test_launcher import REPO_ROOT
from scaffolds.runtime.actor import _Api


def _gen():
    spec = importlib.util.spec_from_file_location(
        "gen_contract", REPO_ROOT / "scripts" / "gen_contract.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_files_equal_live_contract(tmp_path):
    cfg = ext_config(tmp_path / "x", days=1)
    gen = _gen()
    out = tmp_path / "contract"
    payload = gen.write_contract(cfg, out, "x", REPO_ROOT)
    files = json.loads((out / "tools.json").read_text())

    srv = Served(cfg, tmp_path / "x", tmp_path / "x" / "server")
    h = srv.handle
    api = _Api(h["env_url"], h["token"])
    live = api.get("/contract")
    assert files["tools"] == live["tools"] == payload["tools"]
    assert files["llm"] == live["llm"]
    assert (out / "envkit.py").read_text() == live["envkit_py"]
    assert (out / "INSTRUCTION.md").read_text() == live["instruction_md"]
    assert (out / "cell_config.py").read_text() == live["cell_config_py"]
    assert json.loads((out / "base.json").read_text())["id"] == "x"
    assert "| `get_weather` |" in (out / "tools.md").read_text()

    # the stub: one function per program-callable tool, none for the rest
    tree = ast.parse((out / "envkit.py").read_text())
    fns = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    names = {t["name"] for t in live["tools"]}
    callable_ = {t["name"] for t in live["tools"]
                 if t["name"] not in NOT_PROGRAM_TOOLS
                 and "wait" not in t["tags"]}
    assert callable_ <= fns
    assert not ((names & NOT_PROGRAM_TOOLS) & fns)
    assert {"wait", "handover", "now", "workspace", "deadline"} <= fns

    # finish the served sim
    from scaffolds.runtime.actor import Runner
    ws = tmp_path / "x" / "ws"
    ws.mkdir()
    (ws / "main.py").write_text("print('noop')\n")
    Runner(h["env_url"], h["token"], "", ws, ws.parent, quiet=True).run()
    srv.join()
