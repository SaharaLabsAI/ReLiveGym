from datetime import date, datetime, timezone

import pytest

from tasks.weather_fixture.task import NotificationError, Scorer, WeatherConfig, WeatherStore
from tests.conftest import make_config, write_weather_csv

UTC = timezone.utc
START = datetime(2021, 6, 1, tzinfo=UTC)


def t(day, hour, minute=0):
    return datetime(2021, 6, day, hour, minute, tzinfo=UTC)


@pytest.fixture
def scorer(tmp_path) -> Scorer:
    # 7-day window (June 1-7). Crossings: day 2 at 14:00, day 3 at 00:00,
    # day 5 at 23:00. Days 1, 4, 6, 7 stay below threshold.
    temps = []
    for day in range(1, 8):
        day_temps = [25.0] * 24
        if day == 2:
            day_temps[14] = 35.0
        elif day == 3:
            day_temps[0] = 34.0
        elif day == 5:
            day_temps[23] = 36.0
        temps += day_temps
    csv = write_weather_csv(tmp_path / "w.csv", START, temps)
    cfg = make_config()
    tcfg = WeatherConfig(**cfg.task_params)
    store = WeatherStore(csv, data_cutoff=tcfg.data_cutoff)
    return Scorer(tcfg, cfg.sim_start, cfg.sim_end, store)


def outcome(scorer, day):
    return {o.day: o for o in scorer.close_all()}[date(2021, 6, day)]


def test_credit_decays_linearly_with_delay(scorer):
    scorer.record_notification(t(2, 23), date(2021, 6, 2))  # h*=14:00, d=9h
    o = outcome(scorer, 2)
    assert o.status == "ok"
    assert o.delay_hours == 9.0
    assert o.credit == pytest.approx(1 - 9 / 24)


def test_earliest_possible_notification_delay_one_hour(scorer):
    scorer.record_notification(t(2, 15), date(2021, 6, 2))  # h*+1h exactly
    o = outcome(scorer, 2)
    assert o.delay_hours == 1.0
    assert o.credit == pytest.approx(1 - 1 / 24)


def test_miss_earns_zero_credit(scorer):
    o = outcome(scorer, 2)
    assert o.status == "miss"
    assert o.credit == 0.0


def test_very_late_notification_clamps_to_zero_credit(scorer):
    # day 5 crossing at 23:00; notified 25h later (within the 24h grace after
    # the day ends) -> past the credit window; still "ok" but worth nothing
    scorer.record_notification(t(7, 0), date(2021, 6, 5))
    o = outcome(scorer, 5)
    assert o.status == "ok"
    assert o.delay_hours == 25.0
    assert o.credit == 0.0


def test_premature_is_false_alarm_but_day_recoverable(scorer):
    # notification at h* itself: data not visible until h*+1h
    scorer.record_notification(t(2, 14), date(2021, 6, 2))
    scorer.record_notification(t(2, 16), date(2021, 6, 2))
    o = outcome(scorer, 2)
    assert o.status == "ok"
    assert o.false_alarms == 1
    assert o.delay_hours == 2.0
    assert o.credit == pytest.approx(1 - 2 / 24)


def test_premature_without_recovery_is_false_alarm_plus_miss(scorer):
    scorer.record_notification(t(2, 13), date(2021, 6, 2))
    o = outcome(scorer, 2)
    assert o.status == "miss"
    assert o.false_alarms == 1
    assert o.credit == 0.0


def test_false_alarm_on_non_crossing_day(scorer):
    scorer.record_notification(t(1, 10), date(2021, 6, 1))
    o = outcome(scorer, 1)
    assert o.status == "false_alarm"
    assert o.false_alarms == 1


def test_duplicates_counted_but_first_valid_wins(scorer):
    scorer.record_notification(t(2, 16), date(2021, 6, 2))
    scorer.record_notification(t(2, 18), date(2021, 6, 2))
    scorer.record_notification(t(2, 20), date(2021, 6, 2))
    o = outcome(scorer, 2)
    assert o.delay_hours == 2.0  # first valid one counts
    assert o.duplicates == 2


def test_quiet_day_counts_nothing(scorer):
    o = outcome(scorer, 4)
    assert o.status == "quiet"
    assert o.false_alarms == 0 and o.credit == 0.0


def test_notification_validation(scorer):
    with pytest.raises(NotificationError):  # future day
        scorer.record_notification(t(2, 10), date(2021, 6, 3))
    with pytest.raises(NotificationError):  # outside window
        scorer.record_notification(t(2, 10), date(2021, 5, 31))
    with pytest.raises(NotificationError):
        scorer.record_notification(t(2, 10), date(2021, 6, 30))


def test_closed_day_rejects_notification(scorer):
    # day 1 closes 24h after it ends = June 3 00:00
    scorer.close_days(t(3, 0))
    with pytest.raises(NotificationError):
        scorer.record_notification(t(3, 0), date(2021, 6, 1))


def test_close_days_timing_and_feedback(scorer):
    scorer.record_notification(t(2, 23), date(2021, 6, 2))
    assert scorer.close_days(t(2, 23, 59)) == []  # nothing closed yet
    closed = scorer.close_days(t(3, 0))  # day 1 closes exactly at June 3 00:00
    assert [o.day for o in closed] == [date(2021, 6, 1)]
    fb = scorer.feedback()
    assert len(fb) == 1 and fb[0]["date"] == "2021-06-01"
    closed = scorer.close_days(t(4, 0))  # day 2 closes
    assert [o.day for o in closed] == [date(2021, 6, 2)]
    assert scorer.feedback()[1]["status"] == "ok"
    # close_all picks up the rest exactly once
    rest = scorer.close_all()
    assert len(rest) == 5
    assert len(scorer.feedback()) == 7


def test_metrics_and_monthly_buckets(scorer):
    scorer.record_notification(t(2, 23), date(2021, 6, 2))  # ok, d=9h
    scorer.record_notification(t(1, 10), date(2021, 6, 1))  # false alarm
    scorer.close_all()
    m = scorer.metrics()
    credit = 1 - 9 / 24
    assert m["crossing_days"] == 3
    assert m["recall"] == pytest.approx(1 / 3, abs=1e-4)
    assert m["tc_recall"] == pytest.approx(credit / 3, abs=1e-4)
    assert m["precision"] == pytest.approx(1 / 2, abs=1e-4)
    p, r = 0.5, credit / 3
    assert m["primary"]["name"] == "tc_f1"
    assert m["primary"]["direction"] == "max"
    assert m["primary"]["value"] == pytest.approx(2 * p * r / (p + r), abs=1e-4)
    buckets = scorer.monthly_buckets()
    assert list(buckets) == ["2021-06"]
    b = buckets["2021-06"]
    assert b["days"] == 7
    assert b["crossing_days"] == 3
    assert b["ok"] == 1
    assert b["miss"] == 2
    assert b["false_alarm_days"] == 1
    assert b["credit"] == pytest.approx(credit, abs=1e-6)
