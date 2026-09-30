"""Manifest provisioning: which tools exist derives from the
run config alone, and shared tool docs are byte-identical across cells and
tasks — tool-doc wording is part of instruction economics and has a single
server-side source."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from harness.api import tools_manifest
from harness.runtime import Sim
from tests.conftest import make_config, make_weather_task, write_weather_csv

START = datetime(2021, 6, 1, tzinfo=timezone.utc)

# schedule tools are provisioned only for program-managed-timing cells
# (TM-C); agent-owned-timing cells' wake is their wait tool
SCHEDULE_TOOLS = {"get_crontab", "set_crontab", "run_at"}
BASE_TOOLS = {"get_time", "get_costs"}
# the extended TM-B author + run-and-wait affordances ARE the A/B contrast
# (harness/authored.py; there is no condition grammar). TM-B holds NO
# sleep tool: waiting is
# run_program only (the seeded sleep.py is the blind-wait floor).
AUTHORED_TOOLS = {"ls", "read_file", "write_file", "edit_file",
                  "run_program"}
# TM-D: the agent's own schedule CRUD beside the program-owned crontab
AGENT_SCHEDULE_TOOLS = {"list_schedules", "create_schedule",
                        "update_schedule", "delete_schedule"}


@pytest.fixture(scope="module")
def weather_csv(tmp_path_factory):
    d = tmp_path_factory.mktemp("wx")
    return write_weather_csv(d / "w.csv", START, [20.0] * 24 * 7)


def manifest_for(tmp_path, weather_csv, cell) -> list[dict]:
    cfg = make_config(weather_csv=weather_csv, data_cutoff=START, cell=cell)
    task = make_weather_task(cfg)
    (tmp_path / "run").mkdir(exist_ok=True)
    sim = Sim(cfg, tmp_path / "run", tmp_path / "ws", task)
    return tools_manifest(sim)


PROVISIONING = [
    # (cell, expected tool names)
    ({"tm": "A", "sig": "none", "alg": "none"},
     BASE_TOOLS | {"sleep", "get_weather", "notify"}),
    ({"tm": "B", "sig": "none", "alg": "none"},
     BASE_TOOLS | AUTHORED_TOOLS | {"get_weather", "notify"}),
    ({"tm": "C", "sig": "none", "alg": "none"},
     BASE_TOOLS | SCHEDULE_TOOLS | {"sleep", "get_weather", "notify"}),
    ({"tm": "C", "tlrn": "daily", "sig": "oracle", "alg": "memory"},
     BASE_TOOLS | SCHEDULE_TOOLS
     | {"sleep", "get_feedback", "get_weather", "notify"}),
    ({"tm": "C", "tlrn": "daily", "sig": "self", "alg": "memory"},  # no env-side feedback
     BASE_TOOLS | SCHEDULE_TOOLS | {"sleep", "get_weather", "notify"}),
    ({"tm": "D", "sig": "none", "alg": "none"},
     BASE_TOOLS | SCHEDULE_TOOLS | AGENT_SCHEDULE_TOOLS
     | {"sleep", "get_weather", "notify"}),
    ({"tm": "D", "tlrn": "daily", "sig": "oracle", "alg": "skills"},
     BASE_TOOLS | SCHEDULE_TOOLS | AGENT_SCHEDULE_TOOLS
     | {"sleep", "get_feedback", "get_weather", "notify"}),
    # tlrn source=sim provisions schedule tools even under TM-A/B: the
    # generated program installs the learn cron (the react renderer pops
    # them from the actor's registry — program-owned, never actor-owned)
    ({"tm": "B", "tlrn": "daily", "sig": "oracle", "alg": "skills"},
     BASE_TOOLS | SCHEDULE_TOOLS | AUTHORED_TOOLS
     | {"get_feedback", "get_weather", "notify"}),
]


@pytest.mark.parametrize("cell,expected", PROVISIONING,
                         ids=[f"{c['tm']}-{c['sig']}-{c['alg']}"
                              for c, _ in PROVISIONING])
def test_provisioning_matrix(tmp_path, weather_csv, cell, expected):
    names = {t["name"] for t in manifest_for(tmp_path, weather_csv, cell)}
    assert names == expected


def test_shared_tool_docs_byte_identical_across_cells(tmp_path, weather_csv):
    """The same tool must read the same in every cell's manifest."""
    manifests = [manifest_for(tmp_path, weather_csv, c)
                 for c, _ in PROVISIONING]
    docs: dict[str, set[str]] = {}
    for m in manifests:
        for t in m:
            docs.setdefault(t["name"], set()).add(t["doc"])
    for name, variants in docs.items():
        assert len(variants) == 1, f"tool {name!r} has divergent docs"


def test_shared_tool_docs_byte_identical_across_tasks(tmp_path, weather_csv):
    """Harness tools read the same regardless of which task is mounted."""
    from tasks.reddit_ai_popularity.task import RedditPopularityTask
    from tests.tasks.reddit_ai_popularity.conftest import (
        make_reddit_config,
        write_world,
    )

    built = write_world(tmp_path)
    rd_cfg = make_reddit_config(built)
    rd_task = RedditPopularityTask.from_run_config(rd_cfg, built.parent)
    (tmp_path / "rd").mkdir()
    rd = tools_manifest(Sim(rd_cfg, tmp_path / "rd", tmp_path / "rdws",
                            rd_task))
    wx = manifest_for(tmp_path, weather_csv,
                      {"tm": "C", "sig": "none", "alg": "none"})
    rd_docs = {t["name"]: t["doc"] for t in rd}
    for t in wx:
        if t["name"] in rd_docs and t["name"] != "notify":  # task-owned
            assert t["doc"] == rd_docs[t["name"]], t["name"]
