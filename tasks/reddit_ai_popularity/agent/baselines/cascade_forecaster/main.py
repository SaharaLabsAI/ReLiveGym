"""Reddit popularity baseline: pure time-series cascade-growth forecasting
(no LLM, no text/static features — posts are ids with timestamps).

Task: recommend the posts whose discussion will turn out large, as early as
possible; full spec and price table in INSTRUCTION.md.

Strategy (the fixed anchor). Every POLL_MIN minutes, list newly posted
roots; every one of them is watched — there is no static prior, coverage is
indiscriminate. Each post is probed twice: at age A1 (get_cascade ->
comment count c1; posts with c1 < MIN_C1 are dropped — a dead cascade at 20
minutes is already conclusive) and at age A2 (-> c2). The forecast is a
Szabo-Huberman-style log-linear growth model with a velocity term,

    log1p(final) = w0 + w1*log1p(c1) + w2*log1p(c2 - c1)

refit daily by least squares on (c1, dc, final) triples sampled from the
trailing FIT_DAYS fully revealed days (labels free off revealed list pages,
counts from one cascade page per sampled post). Recommend when the
forecast final size clears TAIL_MIN — the task's tail threshold.

Under the task's economics this blanket observer is the
no-LLM reference ceiling: probing every post costs well under a dollar a
day at the real API rate and sits far inside the rate limit, so its
TWR@cap marks what pure mechanical forecasting achieves. The open margin
for smarter agents is precision (better use of the daily cap slots) and
earliness, not observation budget.
"""

import math
from datetime import datetime, timedelta, timezone

from runtime.env_client import Env, EnvError, iso
from runtime.state import load_state, save_state

POLL_MIN = 15  # fixed polling interval
A1 = 20.0  # first probe age, minutes
A2 = 45.0  # second probe age, minutes
MIN_C1 = 2  # drop posts with fewer comments than this at A1
FIT_DAYS = 3  # trailing fully-revealed days the daily refit uses
FIT_BIG = 40  # per fit day: all posts with final >= FIT_BIG_MIN, capped
FIT_BIG_MIN = 20
FIT_SMALL = 40  # per fit day: evenly-spaced sample of the smaller posts
TAIL_MIN = 50  # ${tail_min_desc} from INSTRUCTION.md

env = Env()


def list_between(since: str, until: str) -> list[dict]:
    out, offset = [], 0
    while True:
        res = env.call("list_posts", since=since, until=until, order="asc",
                       offset=offset)
        out.extend(res["posts"])
        offset += len(res["posts"])
        if not res["has_more"] or not res["posts"]:
            return out


def lstsq3(X: list[list[float]], y: list[float]) -> list[float]:
    """Normal-equations least squares for 3 coefficients."""
    k = 3
    A = [[sum(x[i] * x[j] for x in X) for j in range(k)] for i in range(k)]
    b = [sum(x[i] * yy for x, yy in zip(X, y)) for i in range(k)]
    for i in range(k):
        p = max(range(i, k), key=lambda r: abs(A[r][i]))
        A[i], A[p] = A[p], A[i]
        b[i], b[p] = b[p], b[i]
        for r in range(i + 1, k):
            f = A[r][i] / A[i][i]
            for j in range(i, k):
                A[r][j] -= f * A[i][j]
            b[r] -= f * b[i]
    w = [0.0] * k
    for i in range(k - 1, -1, -1):
        w[i] = (b[i] - sum(A[i][j] * w[j] for j in range(i + 1, k))) / A[i][i]
    return w


# -- daily refit ----------------------------------------------------------------------


def sample_day(day: str) -> list[list[float]]:
    """(c1, dc, final) triples for one fully revealed posted-day. One
    cascade page per sampled post; samples whose page ends before A2 are
    censored (count would be a lower bound) and skipped."""
    posts = list_between(f"{day}T00:00:00Z", f"{day}T23:59:59Z")
    big = sorted((p for p in posts if (p["descendants"] or 0) >= FIT_BIG_MIN),
                 key=lambda p: -p["descendants"])[:FIT_BIG]
    small = [p for p in posts if (p["descendants"] or 0) < FIT_BIG_MIN]
    picks = big + small[::max(1, len(small) // FIT_SMALL)][:FIT_SMALL]
    triples = []
    for p in picks:
        page = env.call("get_cascade", root_id=p["id"])
        nodes = page["nodes"]
        t2 = p["created_utc"] + A2 * 60
        if page["has_more"] and nodes and nodes[-1]["created_utc"] <= t2:
            continue
        c1 = sum(1 for n in nodes if n["kind"] == "comment"
                 and n["created_utc"] <= p["created_utc"] + A1 * 60)
        c2 = sum(1 for n in nodes if n["kind"] == "comment"
                 and n["created_utc"] <= t2)
        triples.append([float(c1), float(c2 - c1),
                        float(p["descendants"] or 0)])
    return triples


def refit(state: dict, now: datetime) -> None:
    latest = (now - timedelta(hours=48)).date()  # newest fully revealed day
    days = [str(latest - timedelta(days=i)) for i in range(FIT_DAYS)]
    cache = state.setdefault("days", {})
    for day in days:
        if day not in cache:
            cache[day] = sample_day(day)
    for stale in [d for d in cache if d not in days]:
        del cache[stale]
    X, y = [], []
    for triples in cache.values():
        for c1, dc, final in triples:
            X.append([1.0, math.log1p(c1), math.log1p(max(dc, 0.0))])
            y.append(math.log1p(final))
    state["w"] = lstsq3(X, y)


# -- polling + forecast ----------------------------------------------------------------


def poll(state: dict, now: datetime) -> None:
    w = state["w"]
    watch = state.setdefault("watch", {})
    cursor = state.get("cursor") or iso(now - timedelta(minutes=POLL_MIN))
    for p in list_between(cursor, iso(now)):
        watch[p["id"]] = {"t": p["created_utc"], "c1": None}
    state["cursor"] = iso(now)
    now_ts = now.timestamp()
    for root_id, ent in list(watch.items()):
        age_min = (now_ts - ent["t"]) / 60
        if ent["c1"] is None:
            if age_min < A1:
                continue
            c1 = env.call("get_cascade", root_id=root_id)["n_comments"]
            if c1 < MIN_C1:
                del watch[root_id]
                continue
            ent["c1"] = c1
            # fall through only once A2 is also due (rare long poll gaps)
        if age_min < A2:
            continue
        del watch[root_id]
        c2 = env.call("get_cascade", root_id=root_id)["n_comments"]
        forecast = math.expm1(w[0] + w[1] * math.log1p(ent["c1"])
                              + w[2] * math.log1p(max(c2 - ent["c1"], 0)))
        if forecast < TAIL_MIN:
            continue
        if env.call("quota")["remaining"] < 1:
            continue
        try:
            env.call("recommend", root_id=root_id)
        except EnvError:
            pass  # rejections are free (duplicate, cap race, ...)


# -- dispatch --------------------------------------------------------------------------

env.call("set_crontab", entries=[
    {"id": "poll", "cron_expr": f"*/{POLL_MIN} * * * *"},
    {"id": "refit", "cron_expr": "10 0 * * *"},
])
state = load_state()
now = env.now()
if "w" not in state:  # bootstrap fit on the very first invocation
    refit(state, now)
trigger_id = env.trigger.get("id")
if trigger_id == "refit":
    refit(state, now)
elif trigger_id == "poll":
    poll(state, now)
save_state(state)
