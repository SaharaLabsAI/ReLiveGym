"""broker_ops through a SERVED sim with its two hosts: the roster of duty
instances is fixed whatever the client does; a 25-minute reference
client scores 1.000; a client that never signs in scores 0 on the same
roster; a lockout zeroes the probes inside it; restore reproduces the
fold. Short window (3 days) — needs the crypto candle datasets."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

from harness.config import RunConfig
from harness.mcp import _Http
from harness.runtime import Sim
from harness.timeutil import iso, parse_iso
from tasks.broker_ops.task import DEFAULT_DATASETS, TASK
from tests.control_plane.test_detached import Served
from tests.control_plane.test_mcp_episode import settle

REPO_ROOT = Path(__file__).resolve().parents[3]
pytestmark = pytest.mark.skipif(
    not (DEFAULT_DATASETS / "btc_usdt_spot_mar_jul").exists(),
    reason="crypto candle datasets not built")
UTC = timezone.utc
START = datetime(2026, 4, 1, tzinfo=UTC)
END = datetime(2026, 4, 4, tzinfo=UTC)
BROWSER = {"Sec-Fetch-Mode": "navigate", "Sec-Fetch-Dest": "document"}
PASSWORD = "Spring-Portfolio-26!"


def make_cfg(run_id="bops-test", **task_over) -> RunConfig:
    task = {"name": "broker_ops", "username": "ops", "password": PASSWORD,
            "positions": {"BTC": 3.0, "ETH": 40.0}, "cash_usd": -190000,
            "maintenance_ratio": 0.30,
            "maintenance_schedule": [{"at": "2026-04-02T12:00:00Z", "ratio": 0.36}],
            "max_failed_logins": 3, **task_over}
    return RunConfig(run_id=run_id, task=task, sim_start=START, sim_end=END,
                     cell={"tm": "A", "tlrn": "none", "sig": "none", "alg": "none"},
                     budget_usd=20.0, agent={"scaffold": "ext:mcp"})


def roster_ids() -> list[str]:
    return [r["id"] for r in TASK.from_run_config(make_cfg(), REPO_ROOT).world.roster]


class Actor:
    def __init__(self, handle: dict):
        self.env = _Http(handle["env_url"], handle["agent_token"])
        self.broker = handle["hosts"]["broker"]
        self.mail = handle["hosts"]["mail"]
        self.api = handle["hosts"]["api"]
        self.http = httpx.Client(headers=BROWSER, follow_redirects=False, timeout=60)

    def sleep(self, until: datetime) -> dict:
        while True:
            r = self.env.call("sleep", {"until": iso(until)})
            if r.get("experiment_over") or parse_iso(r["now"]) >= until:
                return r

    def get(self, path: str) -> httpx.Response:
        return self.http.get(self.broker + path)

    def token(self, path: str) -> str:
        page = self.get(path)
        assert page.status_code == 200, (path, page.status_code)
        return re.search(r'name="form_token" value="([^"]+)"', page.text).group(1)

    def post(self, path: str, token_from: str, **fields) -> httpx.Response:
        tok = self.token(token_from)
        return self.http.post(self.broker + path, data={"form_token": tok, **fields})

    def signed_in(self) -> bool:
        return self.get("/positions").status_code == 200

    def login(self, password: str = PASSWORD) -> httpx.Response:
        r = self.post("/login", "/login", username="ops", password=password)
        if r.status_code != 303:
            return r
        msgs = self.http.get(self.mail + "/mail/api/messages").json()["messages"]
        code_msg = [m for m in msgs if m["kind"] == "code"][0]
        body = self.http.get(self.mail + f"/mail/api/messages/{code_msg['id']}").json()["body"]
        code = re.search(r"code is ([A-Z0-9]{6})", body).group(1)
        r = self.post("/login/code", "/login/code", code=code)
        assert r.status_code == 303, r.text[:200]
        return r

    def routine_pass(self) -> None:
        if not self.signed_in():
            self.login()
        page = self.get("/notices").text
        for nid, req in re.findall(r"<tr><td>(mc\d+)</td><td>margin_call</td><td>[^<]*</td>"
                                   r"<td>[^<]*</td><td>([\d.]+)</td><td>open", page):
            assert self.post(f"/notices/{nid}/respond", "/notices", amount_usd=req).status_code == 303
        page = self.get("/risk").text
        marks = dict(re.findall(r"(BTC|ETH) (\d+\.\d\d)", page))
        pend = page.split("<h3>Register")[0]
        for oid, sym, side, qty, price in re.findall(
                r"<tr><td>(o\d+)</td><td>(\w+)</td><td>(\w+)</td><td>([\d.]+)</td><td>([\d.]+)</td>", pend):
            want = "sell" if side == "buy" else "buy"
            trig = float(marks[sym]) * (0.95 if want == "sell" else 1.05)
            r = self.post("/risk", "/risk", order_id=oid, side=want, qty=qty,
                          trigger_price=f"{trig:.2f}")
            assert r.status_code == 303, r.text[:300]


def serve(tmp_path, name: str, **task_over):
    cfg = make_cfg(run_id=f"bops-{name}", **task_over)
    srv = Served(cfg, REPO_ROOT, tmp_path / name / "server")
    h = srv.handle
    ctl = _Http(h["env_url"], h["token"])
    return srv, h, ctl, ctl.get("/trigger/next")


def finish(srv, ctl, trig) -> dict:
    settle(ctl, trig)
    return srv.join()


def test_reference_client_scores_one_and_restores(tmp_path):
    srv, h, ctl, trig = serve(tmp_path, "ref")
    a = Actor(h)
    t = START
    while t < END:
        a.routine_pass()
        t += timedelta(minutes=25)
        a.sleep(t)
    ev = a.http.get(a.api + "/api/events").json()["events"]
    assert {e["kind"] for e in ev} >= {"fill", "margin_call", "treasury", "policy"}
    assert all(m["kind"] in ("code", "instructions") for m in
               a.http.get(a.mail + "/mail/api/messages").json()["messages"])
    res = finish(srv, ctl, trig)
    perf = res["performance"]
    ids = roster_ids()
    assert ids and list(res["task"]["outcomes"]) == ids
    assert perf["primary"]["value"] == 1.0, res["task"]["outcomes"]
    assert perf["by_routine"]["stop_after_fill"]["n"] >= 1
    assert perf["by_routine"]["margin_call"]["n"] >= 1 and perf["lockouts"] == 0
    assert 0 < perf["latency_frac"] < 1
    # the treasury wire, not the response, moved cash: the fold is the precompute
    assert res["task"]["account_end"]["cash_usd"] == pytest.approx(
        TASK.from_run_config(make_cfg(), REPO_ROOT).world.final_state["cash_usd"])
    events = [json.loads(l) for l in
              (tmp_path / "ref" / "server" / "ledger.jsonl").read_text().splitlines()]
    task = TASK.from_run_config(make_cfg(run_id="bops-restore"), REPO_ROOT)
    run_dir = tmp_path / "restore"
    run_dir.mkdir()
    Sim(make_cfg(run_id="bops-restore"), run_dir, run_dir / "ws", task)
    task.restore(events, END)
    assert task.metrics()["primary"]["value"] == 1.0
    assert task.report()["account_end"] == res["task"]["account_end"]


def test_never_signs_in_scores_zero_on_the_same_roster(tmp_path):
    srv, h, ctl, trig = serve(tmp_path, "lazy")
    a = Actor(h)
    assert a.get("/positions").status_code == 302
    r = a.http.post(a.broker + "/risk", data={"order_id": "o0001", "side": "sell",
                                              "qty": "1", "trigger_price": "1"})
    assert r.status_code == 200 and "not applied" in r.text
    a.sleep(END)
    res = finish(srv, ctl, trig)
    assert list(res["task"]["outcomes"]) == roster_ids()
    assert res["performance"]["primary"]["value"] == 0.0
    assert all(v["status"] == "miss" for v in res["task"]["outcomes"].values())
    assert res["task"]["account_end"]["cash_usd"] == pytest.approx(
        TASK.from_run_config(make_cfg(), REPO_ROOT).world.final_state["cash_usd"])


def test_lockout_zeroes_probes_inside_it(tmp_path):
    srv, h, ctl, trig = serve(tmp_path, "lock")
    a = Actor(h)
    for _ in range(3):
        r = a.login(password="wrong")
        assert r.status_code == 200
    assert "locked until" in r.text
    a.sleep(END)
    res = finish(srv, ctl, trig)
    outs = res["task"]["outcomes"]
    inside = [k for k, v in outs.items() if parse_iso(v["deadline"]) < START + timedelta(days=1)]
    assert inside and all(outs[k]["status"] == "locked_out" for k in inside)
    after = [k for k, v in outs.items() if parse_iso(v["deadline"]) >= START + timedelta(days=1)]
    assert after and all(outs[k]["status"] == "miss" for k in after)
    assert res["performance"]["lockouts"] == 1
