"""Task-authored exploratory scaffolds: the task:<name> compose shape mounts the
checked-in program as main.py with the SAME learning file set as react
cells plus a generated cell_config.py, and the wait-party roster tool
is provisioned exactly for these cells."""

from __future__ import annotations

import ast
from datetime import datetime, timezone

import pytest

from harness.apps import harness_apps
from harness.runtime import Sim
from scaffolds.compose import (
    build_workspace,
    render_cell_config,
    render_main,
    workspace_files,
)
from tests.conftest import make_config, make_weather_task
from tests.harness.test_compose import RUNTIME, cfg_for

UTC = timezone.utc

CELL = {"tm": "B", "tlrn": "daily", "sig": "self", "alg": "skills"}


def test_per_market_mounts_react_learning_set_plus_static_main():
    cfg = cfg_for("breakout_news_pm", CELL, "task:per_market")
    files = workspace_files(cfg)
    react = workspace_files(cfg_for("breakout_news_pm", CELL, "react"))
    # exactly the react learning mounts + the checked-in program
    assert set(files) == set(react) | {"main.py"}
    assert files["main.py"].name == "per_market_main.py"
    assert render_main(cfg) is None  # static program, nothing generated
    ast.parse(files["main.py"].read_text(encoding="utf-8"))


def test_per_market_cron_mounts_react_learning_set_plus_static_main():
    cell = {"tm": "C", "tlrn": "daily", "sig": "oracle", "alg": "skills"}
    cfg = cfg_for("breakout_news_pm", cell, "task:per_market_cron")
    files = workspace_files(cfg)
    # same learning file set as per_market, just the TM-C main
    assert files["main.py"].name == "per_market_cron_main.py"
    assert {"records.py", "prompts/reflect.md"} <= set(files)
    assert render_main(cfg) is None  # static program, nothing generated
    ast.parse(files["main.py"].read_text(encoding="utf-8"))


def test_per_market_cron_rejects_agent_owned_timing_and_config_alg():
    with pytest.raises(ValueError, match="supports tm"):
        workspace_files(cfg_for("breakout_news_pm", CELL,
                                "task:per_market_cron"))  # CELL is tm=B
    with pytest.raises(ValueError, match="supports alg"):
        workspace_files(cfg_for(
            "breakout_news_pm",
            {"tm": "C", "tlrn": "daily", "sig": "oracle", "alg": "config"},
            "task:per_market_cron"))


def test_reddit_cron_react_mounts_react_learning_set_plus_static_main():
    cell = {"tm": "C", "tlrn": "daily", "sig": "oracle", "alg": "skills"}
    cfg = cfg_for("reddit_ai_popularity", cell, "task:cron_react")
    files = workspace_files(cfg)
    assert files["main.py"].name == "cron_react_main.py"
    assert {"records.py", "prompts/reflect.md"} <= set(files)
    assert render_main(cfg) is None  # static program, nothing generated
    ast.parse(files["main.py"].read_text(encoding="utf-8"))


def test_reddit_cron_react_rejects_agent_owned_timing_and_config_alg():
    with pytest.raises(ValueError, match="supports tm"):
        workspace_files(cfg_for(
            "reddit_ai_popularity",
            {"tm": "A", "tlrn": "none", "sig": "none", "alg": "none"},
            "task:cron_react"))
    with pytest.raises(ValueError, match="supports alg"):
        workspace_files(cfg_for(
            "reddit_ai_popularity",
            {"tm": "C", "tlrn": "daily", "sig": "oracle", "alg": "config"},
            "task:cron_react"))


def test_fp_cron_react_mounts_react_learning_set_plus_static_main():
    cell = {"tm": "C", "tlrn": "daily", "sig": "oracle", "alg": "skills"}
    cfg = cfg_for("forecast_portfolio", cell, "task:cron_react")
    files = workspace_files(cfg)
    assert files["main.py"].name == "cron_react_main.py"
    assert {"records.py", "prompts/reflect.md"} <= set(files)
    assert render_main(cfg) is None  # static program, nothing generated
    ast.parse(files["main.py"].read_text(encoding="utf-8"))


def test_cell_config_generated_only_for_task_scaffolds(tmp_path):
    cfg = cfg_for("breakout_news_pm", CELL, "task:per_market")
    assert render_cell_config(
        cfg_for("breakout_news_pm", CELL, "react")) is None
    manifest = build_workspace(cfg, tmp_path / "ws")
    assert "cell_config.py" in manifest and "main.py" in manifest
    ns: dict = {}
    exec((tmp_path / "ws" / "cell_config.py").read_text(encoding="utf-8"),
         ns)
    assert (ns["TM"], ns["SIG"], ns["ALG"]) == ("B", "self", "skills")
    assert ns["WAIT_TOOL"] == "run_program"  # CELL is tm=B: no sleep tool
    assert ns["WAIT_ARGS"] == {"path": "park.py"}
    assert ns["LEARN_CRON"] == "30 0 * * *"
    # the mounted program is the checked-in file, byte-identical
    src = workspace_files(cfg)["main.py"].read_bytes()
    assert (tmp_path / "ws" / "main.py").read_bytes() == src


def test_unknown_or_wrong_tm_task_scaffold_rejected():
    with pytest.raises(ValueError, match="not declared"):
        workspace_files(cfg_for("breakout_news_pm", CELL, "task:nope"))
    with pytest.raises(ValueError, match="supports tm"):
        workspace_files(cfg_for(
            "breakout_news_pm",
            {"tm": "C", "tlrn": "none", "sig": "none", "alg": "none"},
            "task:per_market"))
    with pytest.raises(ValueError, match="supports tm"):
        workspace_files(cfg_for(
            "breakout_news_pm",
            {"tm": "A", "tlrn": "none", "sig": "none", "alg": "none"},
            "task:per_market_cron"))


def _weather_sim(tmp_path, scaffold: str) -> Sim:
    from tests.conftest import write_weather_csv

    start = datetime(2021, 6, 1, tzinfo=UTC)
    csv = write_weather_csv(tmp_path / "w.csv", start, [20.0] * 24 * 7)
    cfg = make_config(weather_csv=csv, data_cutoff=start,
                      cell={"tm": "B", "tlrn": "daily", "sig": "self",
                            "alg": "memory"},
                      agent={"scaffold": scaffold})
    (tmp_path / f"run-{scaffold.replace(':', '_')}").mkdir()
    return Sim(cfg, tmp_path / f"run-{scaffold.replace(':', '_')}",
               tmp_path / "ws", make_weather_task(cfg))


def test_set_party_provisioned_only_for_task_scaffolds(tmp_path):
    names = {t.name for app in harness_apps(
        _weather_sim(tmp_path, "task:per_market")) for t, _ in app.tools()}
    assert "set_party" in names
    solo = {t.name for app in harness_apps(
        _weather_sim(tmp_path, "react")) for t, _ in app.tools()}
    assert "set_party" not in solo


# -- ext: external programs ------------------


def _ext_sim(tmp_path, tm: str) -> Sim:
    from tests.conftest import write_weather_csv

    start = datetime(2021, 6, 1, tzinfo=UTC)
    csv = write_weather_csv(tmp_path / f"w{tm}.csv", start, [20.0] * 24 * 7)
    cfg = make_config(weather_csv=csv, data_cutoff=start,
                      cell={"tm": tm, "tlrn": "none", "sig": "none",
                            "alg": "none"},
                      agent={"scaffold": "ext:cand"})
    (tmp_path / f"run-ext-{tm}").mkdir()
    return Sim(cfg, tmp_path / f"run-ext-{tm}", tmp_path / "ws",
               make_weather_task(cfg))


@pytest.mark.parametrize("tm", ["A", "B", "C"])
def test_ext_provisioning(tmp_path, tm):
    names = {t.name for app in harness_apps(_ext_sim(tmp_path, tm))
             for t, _ in app.tools()}
    # adhoc: the party roster; every tm: the sleep endpoint (an ext: tm=B
    # program waits by client-side authored code whose only clock is
    # sleep); never the server-side TM-B jail
    assert {"set_party", "sleep", "get_time", "get_costs"} <= names
    assert not ({"run_program", "write_file", "read_file", "ls",
                 "edit_file"} & names)
    assert ("set_crontab" in names) == (tm == "C")


def test_ext_cell_config_and_constructor():
    cfg = cfg_for("weather_fixture", {"tm": "B", "tlrn": "none", "sig": "none",
                              "alg": "none"}, "ext:cand")
    text = render_cell_config(cfg)
    assert 'TM = "B"' in text and 'WAIT_TOOL = "sleep"' in text
    with pytest.raises(ValueError, match="external program"):
        workspace_files(cfg)


def test_ext_instruction_has_no_tmb_appendix(tmp_path):
    from harness.contract import render_instruction

    ext = _ext_sim(tmp_path, "B")
    text = render_instruction(ext.cfg, ext.task)
    base = _weather_sim(tmp_path, "react")
    base_text = render_instruction(base.cfg, base.task)
    assert "run_program" in base_text
    assert "run_program" not in text


# -- TM-D: agent-managed schedules -------------

TMD_CELL = {"tm": "D", "tlrn": "none", "sig": "none", "alg": "none"}
TMD_LEARN = {"tm": "D", "tlrn": "daily", "sig": "oracle", "alg": "skills"}


@pytest.mark.parametrize("task,scaffold,main", [
    ("reddit_ai_popularity", "task:cron_react", "cron_react_main.py"),
    ("resolution_detect", "task:cron_react", "cron_react_main.py"),
    ("forecast_portfolio", "task:cron_react", "cron_react_main.py"),
    ("breakout_news_pm", "task:cron_react", "cron_react_main.py"),
    ("crypto_price_consistency", "task:cron_react", "cron_react_main.py"),
    ("breakout_news_pm", "task:per_market_cron", "per_market_cron_main.py"),
])
def test_tmd_mounts_the_same_main_as_tmc(task, scaffold, main):
    """The C/D contrast is the cell value alone: same checked-in main,
    same file set; cell_config carries TM=D to the program."""
    d = workspace_files(cfg_for(task, TMD_CELL, scaffold))
    c = workspace_files(cfg_for(
        task, {**TMD_CELL, "tm": "C"}, scaffold))
    assert d == c
    assert d["main.py"].name == main
    src = d["main.py"].read_text(encoding="utf-8")
    ast.parse(src)
    assert 'cell_config.TM not in ("C", "D")' in src
    assert "list_schedules" in src  # the D branch exists in this main
    assert 'TM = "D"' in render_cell_config(cfg_for(task, TMD_CELL, scaffold))


@pytest.mark.parametrize("task", ["reddit_ai_popularity", "resolution_detect",
                                  "forecast_portfolio", "breakout_news_pm"])
def test_tmd_cron_react_learning_cells_mount_like_tmc(task):
    d = workspace_files(cfg_for(task, TMD_LEARN, "task:cron_react"))
    c = workspace_files(cfg_for(task, {**TMD_LEARN, "tm": "C"},
                                "task:cron_react"))
    assert d == c
    assert {"records.py", "prompts/reflect.md"} <= set(d)


def test_tmd_scaffolds_keep_their_tmc_gates():
    with pytest.raises(ValueError, match="supports alg"):
        workspace_files(cfg_for("crypto_price_consistency", TMD_LEARN,
                                "task:cron_react"))
    with pytest.raises(ValueError, match="supports tm"):
        workspace_files(cfg_for(
            "breakout_news_pm", TMD_CELL, "task:per_market"))  # A/B only
    with pytest.raises(ValueError, match="supports tm"):
        workspace_files(cfg_for(
            "breakout_news_pm", {**TMD_CELL, "tm": "A"}, "task:cron_react"))


def test_tmd_provisions_agent_schedule_tools_beside_the_crontab(tmp_path):
    from tests.conftest import write_weather_csv

    start = datetime(2021, 6, 1, tzinfo=UTC)
    csv = write_weather_csv(tmp_path / "w.csv", start, [20.0] * 24 * 7)
    names = {}
    for tm in ("C", "D"):
        cfg = make_config(weather_csv=csv, data_cutoff=start,
                          cell={**TMD_CELL, "tm": tm},
                          agent={"scaffold": "baseline_poller"})
        (tmp_path / f"run-{tm}").mkdir()
        sim = Sim(cfg, tmp_path / f"run-{tm}", tmp_path / "ws",
                  make_weather_task(cfg))
        names[tm] = {t.name for app in harness_apps(sim)
                     for t, _ in app.tools()}
    crud = {"list_schedules", "create_schedule", "update_schedule",
            "delete_schedule"}
    assert names["D"] - names["C"] == crud
    assert {"get_crontab", "set_crontab", "run_at"} <= names["C"] & names["D"]
