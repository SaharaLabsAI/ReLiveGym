"""Small datetime helpers. All simulation datetimes are timezone-aware UTC."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

UTC = timezone.utc
HOUR = timedelta(hours=1)


def as_utc(dt: datetime) -> datetime:
    """Attach UTC to naive datetimes, convert aware ones."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def parse_iso(s: str) -> datetime:
    """Parse ISO-8601 (accepts trailing 'Z') into an aware UTC datetime."""
    return as_utc(datetime.fromisoformat(s.replace("Z", "+00:00")))


def iso(dt: datetime) -> str:
    return as_utc(dt).strftime("%Y-%m-%dT%H:%M:%SZ")


def floor_hour(dt: datetime) -> datetime:
    return dt.replace(minute=0, second=0, microsecond=0)


def day_of(dt: datetime) -> date:
    """The UTC calendar day a sim instant belongs to."""
    return as_utc(dt).date()


def day_start(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, tzinfo=UTC)


def next_midnight(dt: datetime) -> datetime:
    """First midnight strictly after dt."""
    return day_start(day_of(dt)) + timedelta(days=1)
