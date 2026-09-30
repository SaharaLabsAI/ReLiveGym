"""daily_reddit_digest scaffold wiring: react for tm A/B, task:cron_react
for tm C/D (no learning cells), smoke yamls validate, instruction renders."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from harness.config import load_config
from harness.contract import render_instruction
from scaffolds.compose import render_cell_config, render_main, workspace_files
from tasks.daily_reddit_digest.task import DailyRedditDigestTask
from tests.harness.test_compose import cfg_for
from tests.tasks.daily_reddit_digest.conftest import make_digest_config

TASK = "daily_reddit_digest"
NONE = {"tlrn": "none", "sig": "none", "alg": "none"}
CELLS = Path("tasks/daily_reddit_digest/configs/cells")


@pytest.mark.parametrize("tm", ["C", "D"])
def test_cron_react_mounts_react_set_plus_static_main(tm):
    cfg = cfg_for(TASK, {"tm": tm, **NONE}, "task:cron_react")
    files = workspace_files(cfg)
    react = workspace_files(cfg_for(TASK, {"tm": "A", **NONE}, "react"))
    assert set(files) == set(react) | {"main.py", "cell_config.py"} - {"cell_config.py"} | (
        {"cell_config.py"} if "cell_config.py" in files else set())
    assert files["main.py"].name == "cron_react_main.py"
    assert render_main(cfg) is None
    src = files["main.py"].read_text(encoding="utf-8")
    ast.parse(src)
    assert 'ACT_CRON = f"5 {DIGEST_HOUR} * * *"' in src  # daily at a:05
    cc = render_cell_config(cfg)
    assert "TASK_PARAMS = {" in cc
    assert "recommend" not in src.split("def write_instruction")[1].split(
        "def run_episode")[0]  # the schedule note names deliver, not recommend


def test_cron_react_rejects_react_tms_and_learning_algs():
    with pytest.raises(ValueError, match="supports tm"):
        workspace_files(cfg_for(TASK, {"tm": "A", **NONE}, "task:cron_react"))
    with pytest.raises(ValueError, match="supports alg"):
        workspace_files(cfg_for(
            TASK, {"tm": "C", "tlrn": "daily", "sig": "oracle", "alg": "memory"},
            "task:cron_react"))


@pytest.mark.parametrize("tm", ["A", "B"])
def test_react_cells_render(tm):
    cfg = cfg_for(TASK, {"tm": tm, **NONE}, "react")
    assert "main.py" not in workspace_files(cfg)
    ast.parse(render_main(cfg))


def test_authored_example_parses_and_is_mechanism_only(built):
    task = DailyRedditDigestTask.from_run_config(make_digest_config(built),
                                                built.parent)
    src = task.authored_example()
    ast.parse(src)
    assert "from envkit import" in src
    for leak in ("sort=", "sort_by", "12:00", "digest("):
        assert leak not in src, leak  # no experimenter-authored strategy


@pytest.mark.parametrize("path", sorted(CELLS.glob("*.yaml")),
                         ids=lambda p: p.stem)
def test_smoke_yamls_validate(path):
    cfg = load_config(path)
    assert cfg.task_name == TASK
    assert cfg.task_params["digest_hour_utc"] == 12
    assert cfg.budget_usd == 20.0  # sibling smoke economics, unchanged
    assert cfg.cell.label == path.stem
    workspace_files(cfg)  # the scaffold composes for this cell
    if cfg.cell.tm in "CD":
        ns = {}
        exec(render_cell_config(cfg), ns)
        assert ns["TASK_PARAMS"]["digest_hour_utc"] == 12


def test_instruction_renders_without_unfilled_placeholders(built):
    cfg = make_digest_config(built)
    task = DailyRedditDigestTask.from_run_config(cfg, built.parent)
    text = render_instruction(cfg, task)
    assert "${" not in text
    assert "12:00 UTC" in text and "2026-03-03" in text
    assert "2026-03-02T12:00:00Z" in text  # the worked example's window
    assert "5 scored days" in text
