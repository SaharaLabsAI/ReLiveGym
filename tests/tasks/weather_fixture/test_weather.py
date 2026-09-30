from datetime import date, datetime, timezone

import pytest

from tasks.weather_fixture.task import WeatherStore
from tests.conftest import write_weather_csv

UTC = timezone.utc
START = datetime(2021, 6, 1, tzinfo=UTC)


@pytest.fixture
def store(tmp_path) -> WeatherStore:
    # 3 days: day 1 flat 20°, day 2 crosses 33° at 14:00-16:00, day 3 hot from 00:00
    temps = [20.0] * 24
    temps += [25.0] * 14 + [34.0, 35.0, 33.0] + [25.0] * 7
    temps += [33.5] * 24
    csv = write_weather_csv(tmp_path / "w.csv", START, temps)
    return WeatherStore(csv, data_cutoff=START)


def t(day, hour):
    return datetime(2021, 6, day, hour, tzinfo=UTC)


def test_load_counts(store):
    t0, t_last = store.coverage()
    assert t0 == t(1, 0)
    assert t_last == t(3, 23)


def test_availability_hour_visible_at_h_plus_1(store):
    # at 15:00 the last visible hour is 14:00
    res = store.query(t(2, 0), t(2, 23), now=t(2, 15))
    assert res["clipped_end"] == "2021-06-02T14:00:00Z"
    assert res["hourly"]["time"][-1] == "2021-06-02T14:00:00Z"
    assert res["hourly"]["temperature_2m"][-1] == 34.0
    # mid-hour: at 14:30 hour 14 is not complete yet -> last visible is 13:00
    res = store.query(t(2, 0), t(2, 23), now=t(2, 14).replace(minute=30))
    assert res["clipped_end"] == "2021-06-02T13:00:00Z"


def test_data_cutoff_clips_start(tmp_path):
    temps = [20.0] * 48
    csv = write_weather_csv(tmp_path / "w.csv", START, temps)
    store = WeatherStore(csv, data_cutoff=t(2, 0))
    res = store.query(t(1, 0), t(2, 23), now=t(3, 0))
    assert res["clipped_start"] == "2021-06-02T00:00:00Z"
    # at midnight of the 3rd, all 24 hours of the 2nd are visible
    assert len(res["hourly"]["time"]) == 24
    assert res["hourly"]["time"][0] == "2021-06-02T00:00:00Z"


def test_query_entirely_in_future_is_empty(store):
    res = store.query(t(3, 0), t(3, 23), now=t(2, 0))
    assert res["hourly"]["time"] == []
    assert res["clipped_start"] is None and res["clipped_end"] is None


def test_non_hourly_gap_rejected(tmp_path):
    csv = tmp_path / "gap.csv"
    from tests.conftest import CSV_HEADER
    csv.write_text(
        CSV_HEADER
        + "2021-06-01T00:00,20.0\n"
        + "2021-06-01T02:00,20.0\n",  # skips 01:00
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-hourly gap"):
        WeatherStore(csv, data_cutoff=START)


def test_first_crossing(store):
    assert store.first_crossing(date(2021, 6, 1), 33.0) is None
    assert store.first_crossing(date(2021, 6, 2), 33.0) == t(2, 14)
    assert store.first_crossing(date(2021, 6, 3), 33.0) == t(3, 0)
    # threshold is >=
    assert store.first_crossing(date(2021, 6, 2), 34.0) == t(2, 14)
    assert store.first_crossing(date(2021, 6, 2), 35.5) is None
