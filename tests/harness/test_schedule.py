from datetime import datetime, timezone

import pytest

from harness.schedule import FALLBACK_ID, ScheduleStore, Trigger

UTC = timezone.utc


def t(day, hour=0, minute=0):
    return datetime(2021, 6, day, hour, minute, tzinfo=UTC)


def test_cron_fires_strictly_after_install():
    s = ScheduleStore()
    s.set_crontab([{"id": "daily", "cron_expr": "0 23 * * *"}], t(1, 0))
    assert s.peek_next(t(1, 0)).due_time == t(1, 23)
    # an unfired occurrence equal to `now` is still due (it must not be skipped)
    assert s.peek_next(t(1, 23)).due_time == t(1, 23)
    # but installing exactly at a matching instant fires at the NEXT occurrence
    s2 = ScheduleStore()
    s2.set_crontab([{"id": "daily", "cron_expr": "0 23 * * *"}], t(1, 23))
    assert s2.peek_next(t(1, 23)).due_time == t(2, 23)


def test_same_instant_entries_both_fire():
    # regression: an hourly entry and a weekly entry collide at Sunday 00:00
    # (2021-06-06 is a Sunday); before the fix the alphabetically-first entry
    # consumed the instant and the other starved forever
    s = ScheduleStore()
    s.set_crontab([
        {"id": "hourly_check", "cron_expr": "0 * * * *"},
        {"id": "weekly_reflection", "cron_expr": "0 0 * * 0"},
    ], t(5, 23, 30))
    sunday = t(6, 0)
    first = s.peek_next(t(5, 23, 30))
    assert first.due_time == sunday
    s.consume(first)
    second = s.peek_next(sunday)
    assert second.due_time == sunday
    assert {first.id, second.id} == {"hourly_check", "weekly_reflection"}
    s.consume(second)
    assert s.peek_next(sunday).due_time == t(6, 1)  # hourly resumes
    # ... and the weekly entry fires again NEXT Sunday, not never: replay the
    # supervisor loop for a week and collect what fires at June 13 00:00
    week_later = datetime(2021, 6, 13, tzinfo=UTC)
    now = sunday
    fired_at_week_later = set()
    while now <= week_later:
        trig = s.peek_next(now)
        s.consume(trig)
        now = trig.due_time
        if now == week_later:
            fired_at_week_later.add(trig.id)
        if len(fired_at_week_later) == 2:
            break
    assert fired_at_week_later == {"hourly_check", "weekly_reflection"}


def test_cron_consume_advances_to_next_occurrence():
    s = ScheduleStore()
    s.set_crontab([{"id": "daily", "cron_expr": "0 23 * * *"}], t(1, 0))
    trig = s.peek_next(t(1, 0))
    assert (trig.id, trig.kind) == ("daily", "cron")
    s.consume(trig)
    assert s.peek_next(trig.due_time).due_time == t(2, 23)


def test_idempotent_put_does_not_refire():
    s = ScheduleStore()
    entries = [{"id": "daily", "cron_expr": "0 23 * * *"}]
    s.set_crontab(entries, t(1, 0))
    trig = s.peek_next(t(1, 0))
    s.consume(trig)
    s.set_crontab(entries, t(1, 0))  # scaffold re-declares on every run
    assert s.peek_next(t(1, 23)).due_time == t(2, 23)


def test_oneshot_ordering_and_consumption():
    s = ScheduleStore()
    s.run_at("later", t(3, 12))
    s.run_at("sooner", t(2, 6))
    trig = s.peek_next(t(1))
    assert (trig.id, trig.due_time) == ("sooner", t(2, 6))
    s.consume(trig)
    assert s.peek_next(t(2, 6)).id == "later"
    s.consume(s.peek_next(t(2, 6)))
    assert s.peek_next(t(3, 12)).kind == "fallback"


def test_overdue_oneshot_fires_now():
    s = ScheduleStore()
    s.run_at("past", t(1, 5))
    assert s.peek_next(t(2, 9)).due_time == t(2, 9)


def test_run_at_same_id_replaces():
    s = ScheduleStore()
    s.run_at("job", t(5))
    s.run_at("job", t(3))
    assert s.peek_next(t(1)).due_time == t(3)
    s.consume(s.peek_next(t(1)))
    assert s.peek_next(t(3)).kind == "fallback"


def test_fallback_next_midnight_when_empty():
    s = ScheduleStore()
    trig = s.peek_next(t(1, 10, 30))
    assert (trig.id, trig.kind, trig.due_time) == (FALLBACK_ID, "fallback", t(2, 0))
    # exactly at midnight -> next midnight, strictly after
    assert s.peek_next(t(2, 0)).due_time == t(3, 0)


def test_no_fallback_when_cron_exists():
    s = ScheduleStore()
    s.set_crontab([{"id": "monthly", "cron_expr": "0 0 1 7 *"}], t(1))
    trig = s.peek_next(t(1))
    assert trig.kind == "cron"
    assert trig.due_time == datetime(2021, 7, 1, tzinfo=UTC)


def test_tiebreak_oneshot_before_cron():
    s = ScheduleStore()
    s.set_crontab([{"id": "daily", "cron_expr": "0 23 * * *"}], t(1, 0))
    s.run_at("adhoc", t(1, 23))
    assert s.peek_next(t(1)).id == "adhoc"


def test_invalid_cron_expr_rejected():
    s = ScheduleStore()
    with pytest.raises(ValueError):
        s.set_crontab([{"id": "bad", "cron_expr": "not a cron"}], t(1))


def test_duplicate_crontab_ids_rejected():
    s = ScheduleStore()
    with pytest.raises(ValueError):
        s.set_crontab([
            {"id": "x", "cron_expr": "0 1 * * *"},
            {"id": "x", "cron_expr": "0 2 * * *"},
        ], t(1))


def test_persistence_roundtrip(tmp_path):
    path = tmp_path / "schedule.json"
    s = ScheduleStore(path)
    s.set_crontab([{"id": "daily", "cron_expr": "0 23 * * *"}], t(1, 0))
    s.run_at("once", t(4, 12))
    s.consume(s.peek_next(t(1)))  # fires daily@1st 23:00

    s2 = ScheduleStore.load(path)
    assert s2.get_crontab() == [{"id": "daily", "cron_expr": "0 23 * * *"}]
    trig = s2.peek_next(t(1, 23))
    assert (trig.id, trig.due_time) == ("daily", t(2, 23))
    # replay the supervisor loop: daily on the 2nd and 3rd, then the one-shot
    # (which survived the reload) beats the daily entry on the 4th
    s2.consume(trig)
    trig = s2.peek_next(t(2, 23))
    assert (trig.id, trig.due_time) == ("daily", t(3, 23))
    s2.consume(trig)
    trig = s2.peek_next(t(3, 23))
    assert (trig.id, trig.due_time) == ("once", t(4, 12))


def test_trigger_env_payload():
    trig = Trigger("daily", "cron", t(1, 23))
    assert '"id": "daily"' in trig.to_env()
    assert '"kind": "cron"' in trig.to_env()
    assert "2021-06-01T23:00:00Z" in trig.to_env()


def test_unsatisfiable_cron_rejected_at_put():
    # "0 0 31 2 *" (Feb 31) is syntactically valid but never fires; it must be
    # a rejected PUT, not a scheduler crash at peek time (an agent can
    # write exactly this into its own crontab)
    s = ScheduleStore()
    with pytest.raises(ValueError, match="never fires"):
        s.set_crontab([{"id": "scan", "cron_expr": "0 0 31 2 *"}], t(1, 0))
    # the store is untouched: a later put of a valid entry works from clean state
    s.set_crontab([{"id": "scan", "cron_expr": "5 * * * *"}], t(1, 0))
    assert s.get_crontab() == [{"id": "scan", "cron_expr": "5 * * * *"}]


# -- agent-owned schedules -------------


def _tmd_store():
    s = ScheduleStore()
    s.set_crontab([{"id": "act", "cron_expr": "0 23 * * *"},
                   {"id": "learn", "cron_expr": "30 22 * * *"}], t(1, 0))
    return s


def test_agent_once_fires_with_note_and_is_removed():
    s = _tmd_store()
    row = s.agent_create("check", now=t(1, 0), at=t(1, 6), note="look again")
    assert row == {"id": "check", "owner": "agent", "type": "once",
                   "at": "2021-06-01T06:00:00Z", "note": "look again",
                   "next_fire": "2021-06-01T06:00:00Z"}
    trig = s.peek_next(t(1, 0))
    assert (trig.id, trig.kind, trig.owner, trig.note, trig.due_time) == \
        ("check", "at", "agent", "look again", t(1, 6))
    assert trig.payload() == {"id": "check", "kind": "at",
                              "due_time": "2021-06-01T06:00:00Z",
                              "owner": "agent", "note": "look again"}
    s.consume(trig)
    assert s.agent_list(t(1, 6)) == []
    assert s.peek_next(t(1, 6)).id == "learn"  # 22:30: the base resumes


def test_agent_recurring_fires_strictly_after_install_and_records_last_fire():
    s = _tmd_store()
    s.agent_create("hourly", now=t(1, 0), cron_expr="0 * * * *")
    trig = s.peek_next(t(1, 0))
    assert (trig.id, trig.kind, trig.owner, trig.due_time) == \
        ("hourly", "cron", "agent", t(1, 1))
    s.consume(trig)
    assert s.peek_next(t(1, 1)).due_time == t(1, 2)
    assert s.agent_list(t(1, 1))[0]["next_fire"] == "2021-06-01T02:00:00Z"


def test_system_rows_are_read_only_and_ids_reserved():
    s = _tmd_store()
    with pytest.raises(ValueError, match="read-only"):
        s.agent_update("act", now=t(1), cron_expr="0 1 * * *")
    with pytest.raises(ValueError, match="read-only"):
        s.agent_delete("learn")
    with pytest.raises(ValueError, match="reserved"):
        s.agent_create("act", now=t(1), cron_expr="0 1 * * *")
    with pytest.raises(ValueError, match="no such schedule"):
        s.agent_delete("ghost")
    assert s.get_crontab() == [{"id": "act", "cron_expr": "0 23 * * *"},
                               {"id": "learn", "cron_expr": "30 22 * * *"}]


def test_set_crontab_cannot_clobber_agent_rows():
    s = _tmd_store()
    s.agent_create("mine", now=t(1, 0), at=t(2))
    # the program re-declares its crontab every firing: agent rows survive
    s.set_crontab([{"id": "act", "cron_expr": "0 23 * * *"}], t(1, 1))
    assert [r["id"] for r in s.agent_list(t(1, 1))] == ["mine"]
    # and a program may not take over an agent id
    with pytest.raises(ValueError, match="agent-owned"):
        s.set_crontab([{"id": "mine", "cron_expr": "0 1 * * *"}], t(1, 1))
    with pytest.raises(ValueError, match="agent-owned"):
        s.run_at("mine", t(3))


def test_agent_update_is_put_and_changed_expr_resets_fire_state():
    s = _tmd_store()
    s.agent_create("r", now=t(1, 0), cron_expr="0 12 * * *")
    s.consume(s.peek_next(t(1, 0)))  # r @ 12:00 beats the base entries
    # unchanged expr keeps last fire; changed expr re-installs from now
    s.agent_update("r", now=t(1, 12), cron_expr="0 12 * * *")
    assert s.agent_list(t(1, 12))[0]["next_fire"] == "2021-06-02T12:00:00Z"
    s.agent_update("r", now=t(1, 12), cron_expr="0 * * * *", note="n")
    row = s.agent_list(t(1, 12))[0]
    assert (row["cron_expr"], row["note"], row["next_fire"]) == \
        ("0 * * * *", "n", "2021-06-01T13:00:00Z")
    # once <-> recurring conversion is a plain PUT
    s.agent_update("r", now=t(1, 12), at=t(1, 14))
    assert s.agent_list(t(1, 12))[0]["type"] == "once"
    with pytest.raises(ValueError, match="exactly one"):
        s.agent_update("r", now=t(1, 12), at=t(1, 14), cron_expr="0 * * * *")
    with pytest.raises(ValueError, match="already exists"):
        s.agent_create("r", now=t(1, 12), at=t(1, 15))


def test_agent_validation_and_caps():
    from harness.schedule import MAX_AGENT_SCHEDULES, NOTE_MAX_CHARS

    s = _tmd_store()
    with pytest.raises(ValueError, match="schedule id must match"):
        s.agent_create("Bad Id", now=t(1), at=t(2))
    with pytest.raises(ValueError, match="schedule id must match"):
        s.agent_create("__x", now=t(1), at=t(2))
    with pytest.raises(ValueError, match="invalid cron"):
        s.agent_create("x", now=t(1), cron_expr="nope")
    with pytest.raises(ValueError, match="never fires"):
        s.agent_create("x", now=t(1), cron_expr="0 0 31 2 *")
    with pytest.raises(ValueError, match="note longer"):
        s.agent_create("x", now=t(1), at=t(2), note="n" * (NOTE_MAX_CHARS + 1))
    for i in range(MAX_AGENT_SCHEDULES):
        s.agent_create(f"s{i}", now=t(1), at=t(2))
    with pytest.raises(ValueError, match="at most"):
        s.agent_create("one-more", now=t(1), at=t(2))


def test_agent_overdue_once_fires_now_and_ties_rank_with_kind():
    s = _tmd_store()
    s.agent_create("late", now=t(1, 0), at=t(1, 5))
    assert s.peek_next(t(1, 9)).due_time == t(1, 9)
    s.consume(s.peek_next(t(1, 9)))
    # agent once at 23:00 beats the system cron at 23:00 (once < cron)
    s.agent_create("tie", now=t(1, 9), at=t(1, 23))
    s.consume(s.peek_next(t(1, 9)))  # learn @ 22:30
    first = s.peek_next(t(1, 22, 30))
    assert (first.id, first.owner) == ("tie", "agent")
    s.consume(first)
    assert s.peek_next(t(1, 23)).id == "act"


def test_agent_rows_persist_and_old_files_load(tmp_path):
    path = tmp_path / "schedule.json"
    s = ScheduleStore(path)
    s.set_crontab([{"id": "act", "cron_expr": "0 23 * * *"}], t(1, 0))
    s.agent_create("r", now=t(1, 0), cron_expr="0 12 * * *", target="q-1")
    s.agent_create("o", now=t(1, 0), at=t(3), note="hi")
    s.consume(s.peek_next(t(1, 0)))  # r @ 1st 12:00
    s2 = ScheduleStore.load(path)
    rows = {r["id"]: r for r in s2.agent_list(t(1, 12))}
    assert rows["r"]["next_fire"] == "2021-06-02T12:00:00Z"
    assert rows["r"]["target"] == "q-1"
    assert rows["o"] == {"id": "o", "owner": "agent", "type": "once",
                         "at": "2021-06-03T00:00:00Z", "note": "hi",
                         "next_fire": "2021-06-03T00:00:00Z"}
    trig = s2.peek_next(t(3, 0))
    assert (trig.id, trig.note, trig.target) == ("o", "hi", None)
    # a pre-TM-D schedule.json has no "agent" section
    import json
    data = json.loads(path.read_text())
    del data["agent"]
    path.write_text(json.dumps(data))
    assert ScheduleStore.load(path).agent_list(t(1)) == []


def test_system_view_lists_base_entries():
    s = _tmd_store()
    s.run_at("__bootstrap__", t(1, 0))
    ids = {r["id"]: r for r in s.system_view(t(1, 0))}
    assert ids["act"] == {"id": "act", "owner": "system", "type": "recurring",
                          "cron_expr": "0 23 * * *",
                          "next_fire": "2021-06-01T23:00:00Z"}
    assert ids["__bootstrap__"]["type"] == "once"



# -- waits never see the fallback; TM-D default schedules ---------------


def test_peek_due_has_no_fallback():
    s = ScheduleStore()
    assert s.peek_due(t(1, 10)) is None            # what a wait sees
    assert s.peek_next(t(1, 10)).kind == "fallback"  # the supervisor's view
    s.run_at("x", t(1, 12))
    assert s.peek_due(t(1, 10)).id == s.peek_next(t(1, 10)).id == "x"


def test_default_schedule_is_seeded_once_and_agent_modifiable(tmp_path):
    path = tmp_path / "schedule.json"
    s = ScheduleStore(path)
    entries = [{"id": "act", "cron_expr": "0 0 * * *", "agent_owned": True},
               {"id": "learn", "cron_expr": "30 0 * * *"}]
    assert [r["id"] for r in s.set_crontab(entries, t(1))] == ["act"]
    assert s.get_crontab() == [{"id": "learn", "cron_expr": "30 0 * * *"}]
    assert [(r["id"], r["owner"]) for r in s.agent_list(t(1))] == [("act", "agent")]
    assert s.set_crontab(entries, t(1, 6)) == []   # re-declaring: a no-op
    s.agent_update("act", now=t(1, 6), cron_expr="0 13 * * *")
    assert s.agent_get("act", t(1, 6))["cron_expr"] == "0 13 * * *"
    s.agent_delete("act")
    assert s.set_crontab(entries, t(1, 7)) == []   # never brought back...
    assert ScheduleStore.load(path).set_crontab(entries, t(1, 8)) == []  # ...even after a reload
    assert s.agent_list(t(1, 8)) == []
    with pytest.raises(ValueError, match="reserved"):
        s.agent_create("act", now=t(1, 8), cron_expr="0 1 * * *")
