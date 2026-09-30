"""Synthetic mini-world for reddit_ai_popularity tests.

Roots around March 2026 whose 24h `descendants` are powers of two so pop values
are exact. sim window Mar 2 - Mar 14, history_days=1, horizon 24h, daily_cap=3,
tail_min_desc=50 (default): the tail universe is {big, late}. Cascades carry per-comment timestamps so the growing-prefix reveal
is testable. All timestamps UTC. NB: source rows include `score`/`num_comments`
(as the real build does) — the task never surfaces `score`; the `num_comments`
it exposes is recomputed from the cascade prefix at now, not the source field.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from harness.config import RunConfig
from tasks.reddit_ai_popularity.task import (
    CascadeStore,
    RedditPopularityTask,
    RootStore,
    Scorer,
)

UTC = timezone.utc


def t(day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(2026, 3, day, hour, minute, tzinfo=UTC)


def ts(day: int, hour: int = 0, minute: int = 0) -> float:
    return t(day, hour, minute).timestamp()


# root_id -> (posted day, hour, descendants_within_24h, subreddit, selftext,
#             extra_late_comment?)  pop = log2(desc)+1
ROOTS = {
    "hist": (1, 6, 16, "OpenAI", "", False),   # pre-sim history, pop 5
    "big": (2, 6, 256, "OpenAI", "", True),    # the big hit, pop 9 (+1 comment >24h)
    "dud": (2, 9, 2, "grok", "", False),       # pop 2
    "self": (2, 12, 4, "Anthropic", "a self-text body", False),  # pop 3
    "mid": (2, 15, 8, "singularity", "", False),  # pop 4
    "late": (13, 12, 64, "OpenAI", "", False),  # reveals after sim_end, pop 7
}


def _cascade(root_id: str, post_ts: float, n_within: int, extra_late: bool) -> dict:
    nodes = [{"id": root_id, "kind": "root", "parent_id": None, "author": "op",
              "created_utc": int(post_ts), "body": "title text", "score": 999}]
    # comments spread across the first 12h (all within the 24h horizon),
    # deterministic times so prefix reveal is exact
    for i in range(n_within):
        off = 60 + i * 5  # seconds; all well within 24h even for 256
        nodes.append({"id": f"{root_id}_c{i}", "kind": "comment",
                      "parent_id": root_id if i % 3 else f"{root_id}_c{max(i-1,0)}",
                      "author": f"u{i}", "created_utc": int(post_ts + off),
                      "body": f"reply {i}", "score": 7})
    if extra_late:  # one comment beyond the 24h horizon -> excluded from label
        nodes.append({"id": f"{root_id}_late", "kind": "comment",
                      "parent_id": root_id, "author": "latecomer",
                      "created_utc": int(post_ts + 25 * 3600), "body": "late",
                      "score": 1})
    return {"root_id": root_id, "nodes": nodes}


def write_world(root: Path) -> Path:
    built = root / "built"
    built.mkdir(parents=True)
    root_rows, cascades = [], []
    for rid, (day, hour, desc, sub, selftext, late) in ROOTS.items():
        post = ts(day, hour)
        root_rows.append({
            "id": rid, "name": f"t3_{rid}", "subreddit": sub, "author": "op",
            "created_utc": int(post), "title": f"Post {rid}", "selftext": selftext,
            "url": f"https://x/{rid}", "is_self": bool(selftext), "over_18": False,
            "link_flair_text": None, "domain": "x",
            "score": 42, "num_comments": desc + (1 if late else 0),
            "retrieved_on": int(post)})
        cascades.append(_cascade(rid, post, desc, late))
    root_rows.sort(key=lambda r: (r["created_utc"], r["id"]))
    with open(built / "roots.jsonl", "w") as f:
        for r in root_rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    with open(built / "cascades.jsonl", "w") as f:
        for c in cascades:
            f.write(json.dumps(c) + "\n")
    (built / "build_stats.json").write_text(json.dumps({
        "window": {"after": int(ts(1)), "before": int(ts(15)),
                   "after_h": "2026-03-01", "before_h": "2026-03-15"},
        "horizon_days_collected": 7,
        "n_kept_roots": len(root_rows),
    }))
    return built


def make_reddit_config(built: Path, **overrides) -> RunConfig:
    task = dict(
        name="reddit_ai_popularity",
        data_dir=str(built),
        history_days=1,
        daily_cap=3,
        page_size=3,
    )
    task.update(overrides.pop("task", {}))
    base = dict(
        run_id="reddit-test",
        task=task,
        sim_start=t(2),
        sim_end=t(14),
        agent=dict(scaffold="baseline_recommender"),
    )
    base.update(overrides)
    return RunConfig(**base)


@pytest.fixture
def built(tmp_path) -> Path:
    return write_world(tmp_path)


@pytest.fixture
def task(built) -> RedditPopularityTask:
    cfg = make_reddit_config(built)
    return RedditPopularityTask.from_run_config(cfg, built.parent)


@pytest.fixture
def scorer(task) -> Scorer:
    return task.scorer


@pytest.fixture
def roots(task) -> RootStore:
    return task.roots


@pytest.fixture
def cascades(task) -> CascadeStore:
    return task.cascades
