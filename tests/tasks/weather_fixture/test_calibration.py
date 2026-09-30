"""Calibration regression: pin the data pipeline to the numbers the
task was calibrated on (LA @ 33 °C, main window
2021-06-01 → 2024-06-01: 111 crossing days, Aug/Sep/Oct-heavy)."""

from datetime import date, timedelta


def crossing_days(store, start: date, end: date, threshold: float) -> list[date]:
    out = []
    d = start
    while d < end:
        if store.first_crossing(d, threshold) is not None:
            out.append(d)
        d += timedelta(days=1)
    return out


def test_main_window_crossing_count(la_store):
    days = crossing_days(la_store, date(2021, 6, 1), date(2024, 6, 1), 33.0)
    assert len(days) == 111


def test_crossings_concentrate_in_aug_sep_oct(la_store):
    days = crossing_days(la_store, date(2021, 6, 1), date(2024, 6, 1), 33.0)
    by_month: dict[int, int] = {}
    for d in days:
        by_month[d.month] = by_month.get(d.month, 0) + 1
    # 73/111 in this window (the design's 37/31/16% shares were measured over the
    # full 2016-2026 dataset, which is more Aug-heavy than these three years)
    aug_sep_oct = by_month.get(8, 0) + by_month.get(9, 0) + by_month.get(10, 0)
    assert aug_sep_oct == 73
    # no winter crossings at this threshold (design: Dec-Feb silence is safe)
    assert by_month.get(12, 0) == 0
    assert by_month.get(1, 0) == 0
    assert by_month.get(2, 0) == 0
