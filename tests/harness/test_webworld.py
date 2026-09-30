"""harness/webworld.py: journal tokens
and versions, timeline ordering with ranks and scans, session expiry in
sim time, lockouts."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from harness.webworld import (
    RANK_PROBE,
    RANK_REPLAY,
    RANK_SCRIPTED,
    RANK_TIMER,
    Journal,
    SessionStore,
    StaleVersion,
    Timeline,
    UnknownToken,
)

UTC = timezone.utc
T0 = datetime(2026, 4, 1, tzinfo=UTC)
H = timedelta(hours=1)


def test_journal_fold_tokens_versions():
    j = Journal("salt")
    tok = j.mint_token(T0)
    r1 = j.write(T0, actor="agent", app="sheet", kind="set", entity="A",
                 fields={"x": 1}, form_token=tok, expect_version=0)
    assert r1.version == 1 and j.get("sheet", "A")["fields"] == {"x": 1}
    dup = j.write(T0 + H, actor="agent", app="sheet", kind="set", entity="A",
                  fields={"x": 999}, form_token=tok)
    assert dup.duplicate and j.get("sheet", "A")["fields"] == {"x": 1}
    with pytest.raises(UnknownToken):
        j.write(T0, actor="agent", app="sheet", kind="set", entity="A",
                fields={}, form_token="never-minted")
    with pytest.raises(StaleVersion):
        j.write(T0, actor="agent", app="sheet", kind="update", entity="A",
                fields={"y": 2}, expect_version=0)
    r2 = j.write(T0 + 2 * H, actor="world", app="sheet", kind="update",
                 entity="A", fields={"y": 2}, expect_version=1)
    st = j.get("sheet", "A")
    assert st["fields"] == {"x": 1, "y": 2} and st["version"] == 2
    assert st["updated_by"] == "world" and st["updated_at"] == T0 + 2 * H
    j.write(T0 + 3 * H, actor="agent", app="sheet", kind="delete", entity="A")
    assert j.get("sheet", "A") is None and j.items("sheet", include_deleted=True)
    # restore: the ledgered payload replays verbatim, tokens trusted
    j2 = Journal("salt")
    for r in j.records:
        j2.replay(r.t, r.payload())
    assert [r.to_dict() for r in j2.records] == [r.to_dict() for r in j.records]
    # determinism: same salt, same tokens
    assert Journal("salt").mint_token(T0) == tok


def test_timeline_order_scan_and_cancel():
    tl = Timeline(T0)
    seen = []
    tl.add(T0 + 2 * H, "probe", rank=RANK_PROBE)
    tl.add(T0 + 2 * H, "notice", rank=RANK_SCRIPTED)
    e = tl.add(T0 + 1 * H, "timer", rank=RANK_TIMER, id="t1")
    tl.add(T0 + 3 * H, "later", rank=RANK_TIMER)
    assert tl.cancel("t1") and not tl.cancel("t1")

    def scan(t_from, t_to):
        # one replay interaction at 01:30, only when the window covers it
        fill = T0 + timedelta(minutes=90)
        if t_from < fill <= t_to:
            tl.add(fill, "fill", rank=RANK_REPLAY)
            return fill
        return None

    for ev in tl.drain(T0 + 2 * H, scan=scan):
        seen.append((ev.kind, ev.t))
        if ev.kind == "fill":  # a handler may add later events
            tl.add(T0 + 2 * H, "margin", rank=RANK_TIMER)
    assert seen == [("fill", T0 + timedelta(minutes=90)),
                    ("notice", T0 + 2 * H), ("margin", T0 + 2 * H),
                    ("probe", T0 + 2 * H)]
    assert tl.t_synced == T0 + 2 * H
    assert [e.kind for e in tl.pending()] == ["later"]
    with pytest.raises(ValueError):
        tl.add(T0, "past")
    assert list(tl.drain(T0 + 2 * H)) == []  # idempotent


def test_sessions_expire_in_sim_time():
    ss = SessionStore("salt", idle_minutes=30, cutoff_utc=(0, 0),
                      code_valid_minutes=10, max_failed=3, lockout_hours=24)
    s = ss.new_session(T0)
    ss.password_ok(s, "ops", T0)
    assert not ss.code_ok(s, "WRONG", T0)
    assert not ss.code_ok(s, s.code, T0 + timedelta(minutes=10))  # code dead
    ss.password_ok(s, "ops", T0)
    assert ss.code_ok(s, s.code, T0 + timedelta(minutes=9))
    assert s.authenticated and s.user == "ops"
    ss.touch(s.id, T0 + timedelta(minutes=20))
    assert ss.get(s.id, T0 + timedelta(minutes=49)).authenticated  # 29 idle
    assert not ss.get(s.id, T0 + timedelta(minutes=50)).authenticated  # 30 idle
    assert s.end_reason == "idle"
    s2 = ss.new_session(T0 + 20 * H)
    ss.password_ok(s2, "ops", T0 + 20 * H)
    ss.code_ok(s2, s2.code, T0 + 20 * H)
    ss.touch(s2.id, T0 + 23 * H + timedelta(minutes=59))
    assert not ss.get(s2.id, T0 + 24 * H).authenticated  # daily cutoff
    assert s2.end_reason == "cutoff"
    # lockout after 3 bad passwords, cleared after 24 h
    assert ss.password_failed("ops", T0) is None
    assert ss.password_failed("ops", T0) is None
    until = ss.password_failed("ops", T0)
    assert until == T0 + 24 * H and ss.locked_until("ops", T0 + H) == until
    assert ss.locked_until("ops", T0 + 24 * H) is None
    # deterministic ids and codes
    assert SessionStore("salt").new_session(T0).id == s.id
