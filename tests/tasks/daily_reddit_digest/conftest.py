"""Synthetic mini-world for daily_reddit_digest tests.

digest_hour 12, lookback 24 h, delivery 1 h, sim Mar 2 00:00 – Mar 8 00:00
(history_days 1) -> scored days Mar 3..Mar 7 (Mar 2's window would start
Mar 1 12:00, pre-run). Day Mar 3's candidate window is
[Mar 2 12:00, Mar 3 12:00); roots sit exactly on its edges plus ten popular
(50 comments, the bar) and three unpopular (8) posts inside it, and a
second popular batch on Mar 5. Comments arrive every 5 s from +60 s, so
every label is frozen well inside 24 h and prefix counts are exact.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from harness.config import RunConfig
from tasks.daily_reddit_digest.task import DailyRedditDigestTask, DigestScorer
from tests.tasks.reddit_ai_popularity.conftest import _cascade

UTC = timezone.utc


def t(day: int, hour: int = 0, minute: int = 0, second: int = 0) -> datetime:
    return datetime(2026, 3, day, hour, minute, second, tzinfo=UTC)


def ts(day: int, hour: int = 0, minute: int = 0, second: int = 0) -> float:
    return t(day, hour, minute, second).timestamp()


# root_id -> (posted datetime, descendants within 24h)
ROOTS = {
    "hist": (t(1, 18), 64),                   # pre-run history
    "edge_before": (t(2, 11, 59, 59), 64),    # 1 s before Mar 3's window
    "edge_lo": (t(2, 12), 64),                # first instant of the window
    "edge_hi": (t(3, 11, 59, 59), 64),        # last instant of the window
    "edge_at": (t(3, 12), 64),                # = a:00 -> Mar 4's window
    **{f"pop{i}": (t(3, 0, i), 50) for i in range(1, 11)},   # Mar 3, the bar
    **{f"unpop{i}": (t(3, 1, i), 8) for i in range(1, 4)},
    **{f"day5_{i}": (t(4, 20, i), 120) for i in range(1, 13)},  # Mar 5 window
}
POP = [f"pop{i}" for i in range(1, 11)]
UNPOP = [f"unpop{i}" for i in range(1, 4)]
DAY5 = [f"day5_{i}" for i in range(1, 13)]


def write_world(root: Path) -> Path:
    built = root / "built"
    built.mkdir(parents=True)
    rows, cascades = [], []
    for rid, (when, desc) in ROOTS.items():
        post = when.timestamp()
        rows.append({
            "id": rid, "name": f"t3_{rid}", "subreddit": "OpenAI", "author": "op",
            "created_utc": int(post), "title": f"Post {rid}", "selftext": "",
            "url": f"https://x/{rid}", "is_self": False, "over_18": False,
            "link_flair_text": None, "domain": "x",
            "score": 1, "num_comments": desc, "retrieved_on": int(post)})
        cascades.append(_cascade(rid, post, desc, False))
    rows.sort(key=lambda r: (r["created_utc"], r["id"]))
    with open(built / "roots.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    with open(built / "cascades.jsonl", "w") as f:
        for c in cascades:
            f.write(json.dumps(c) + "\n")
    (built / "build_stats.json").write_text(json.dumps({
        "window": {"after": int(ts(1)), "before": int(ts(20)),
                   "after_h": "2026-03-01", "before_h": "2026-03-20"},
        "horizon_days_collected": 7, "n_kept_roots": len(rows)}))
    return built


def make_digest_config(built: Path, **overrides) -> RunConfig:
    task = dict(name="daily_reddit_digest", data_dir=str(built),
                history_days=1, digest_hour_utc=12)
    task.update(overrides.pop("task", {}))
    base = dict(run_id="digest-test", task=task, sim_start=t(2),
                sim_end=t(8), agent=dict(scaffold="react"),
                cell=dict(tm="A", tlrn="none", sig="none", alg="none"))
    base.update(overrides)
    return RunConfig(**base)


@pytest.fixture
def built(tmp_path) -> Path:
    return write_world(tmp_path)


@pytest.fixture
def task(built) -> DailyRedditDigestTask:
    return DailyRedditDigestTask.from_run_config(make_digest_config(built),
                                                built.parent)


@pytest.fixture
def scorer(task) -> DigestScorer:
    return task.scorer
