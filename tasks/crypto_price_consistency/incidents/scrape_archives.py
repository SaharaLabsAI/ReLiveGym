"""Scrape public provider incident archives into normalized calibration events.

Keyless, like the price collector. Four sources are supported:

- ``coinbase``  — Statuspage JSON API (recent incidents + scheduled maintenances)
                  plus the paginated ``/history`` archive (3 months per page).
- ``bitstamp``  — same Statuspage machinery on status.bitstamp.net.
- ``coingecko`` — status.coingecko.com/incidents HTML listing; ``--deep``
                  additionally fetches each incident page to recover the
                  resolution timestamp and therefore a duration.
- ``okx``       — ``GET /api/v5/system/status?state=completed`` (machine-readable
                  recent events), the server-rendered status page lists, and the
                  help-center issue-description category whose date-slugged
                  postmortem articles reach back to 2020.

Every HTTP response is preserved under ``calibration/raw/`` before any parsing,
mirroring the raw/*.jsonl discipline of the price collector. Normalized events
land in ``calibration/events/<source>.json``; timestamps that cannot be parsed
confidently keep ``started_at=None`` alongside the verbatim ``raw_timestamp``.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx

USER_AGENT = "program-engineering-crypto-incident-archives/0.1"
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
SCHEMA_VERSION = 1

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "calibration"

# Statuspage history strings carry US-style timezone abbreviations. Unknown
# abbreviations leave the timestamp unparsed rather than guessing an offset.
TZ_OFFSETS_HOURS = {
    "UTC": 0, "GMT": 0,
    "PST": -8, "PDT": -7,
    "MST": -7, "MDT": -6,
    "CST": -6, "CDT": -5,
    "EST": -5, "EDT": -4,
    "CET": 1, "CEST": 2,
    "BST": 1,
}

MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"])}


class ScrapeError(RuntimeError):
    """An archive response could not be retrieved or parsed."""


@dataclass
class IncidentEvent:
    """One normalized archive entry. Nullable fields mean "not recoverable"."""

    source: str
    method: str  # statuspage_api | statuspage_maintenance | statuspage_history
    #             | listing_html | detail_html | status_api | status_html
    #             | postmortem_slug
    event_id: str
    title: str
    impact: str  # provider vocabulary, lowercased; "unknown" when absent
    scheduled: bool | None
    started_at: str | None  # ISO-8601 UTC
    ended_at: str | None
    duration_minutes: float | None
    url: str | None
    raw_timestamp: str | None
    extra: dict[str, Any] = field(default_factory=dict)


class ArchiveClient:
    """Retrying HTTP client that records every payload before parsing."""

    def __init__(self, *, raw_dir: Path, timeout_seconds: float = 30,
                 retries: int = 4, delay_seconds: float = 0.4) -> None:
        raw_dir.mkdir(parents=True, exist_ok=True)
        self._raw_dir = raw_dir
        self._log = (raw_dir / "requests.jsonl").open("w", encoding="utf-8")
        self._client = httpx.Client(
            headers={"User-Agent": USER_AGENT},
            timeout=timeout_seconds,
            follow_redirects=True,
        )
        self._retries = retries
        self._delay = delay_seconds
        self.request_count = 0

    def __enter__(self) -> "ArchiveClient":
        return self

    def __exit__(self, *_: object) -> None:
        self._client.close()
        self._log.close()

    def _record(self, entry: dict[str, Any]) -> None:
        entry["fetched_at"] = datetime.now(UTC).isoformat()
        self._log.write(json.dumps(entry, separators=(",", ":")) + "\n")
        self._log.flush()

    def _get(self, url: str, params: dict[str, object] | None) -> httpx.Response:
        if self.request_count:
            time.sleep(self._delay)
        last_error: Exception | None = None
        for attempt in range(self._retries + 1):
            try:
                response = self._client.get(url, params=params)
                self.request_count += 1
                if (response.status_code in RETRYABLE_STATUS_CODES
                        and attempt < self._retries):
                    retry_after = response.headers.get("retry-after")
                    delay = float(retry_after) if retry_after else min(2 ** attempt, 8)
                    time.sleep(delay)
                    continue
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError as exc:
                self._record({
                    "url": str(exc.request.url),
                    "status_code": exc.response.status_code,
                })
                raise ScrapeError(
                    f"HTTP {exc.response.status_code} for {exc.request.url}"
                ) from exc
            except httpx.TransportError as exc:
                last_error = exc
                if attempt < self._retries:
                    time.sleep(min(2 ** attempt, 8))
                    continue
                break
        raise ScrapeError(f"request failed for {url}: {last_error}") from last_error

    def get_json(self, url: str, params: dict[str, object] | None = None) -> Any:
        response = self._get(url, params)
        payload = response.json()
        self._record({"url": str(response.request.url), "kind": "json",
                      "payload": payload})
        return payload

    def get_html(self, url: str, save_as: str,
                 params: dict[str, object] | None = None) -> str:
        response = self._get(url, params)
        path = self._raw_dir / "html" / save_as
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(response.text, encoding="utf-8")
        self._record({"url": str(response.request.url), "kind": "html",
                      "html_file": str(path.relative_to(self._raw_dir))})
        return response.text


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat()


def _duration_minutes(start: str | None, end: str | None) -> float | None:
    if not start or not end:
        return None
    delta = datetime.fromisoformat(end) - datetime.fromisoformat(start)
    minutes = delta.total_seconds() / 60
    return round(minutes, 2) if minutes >= 0 else None


# --------------------------------------------------------------------------
# Statuspage (Coinbase, Bitstamp)
# --------------------------------------------------------------------------

def _statuspage_api_events(source: str, base_url: str,
                           client: ArchiveClient) -> list[IncidentEvent]:
    events: list[IncidentEvent] = []
    incidents = client.get_json(f"{base_url}/api/v2/incidents.json")
    for item in incidents.get("incidents", []):
        started = item.get("started_at") or item.get("created_at")
        resolved = item.get("resolved_at")
        started_iso = _iso(datetime.fromisoformat(started)) if started else None
        resolved_iso = _iso(datetime.fromisoformat(resolved)) if resolved else None
        events.append(IncidentEvent(
            source=source,
            method="statuspage_api",
            event_id=item["id"],
            title=item.get("name", ""),
            impact=(item.get("impact") or "unknown").lower(),
            scheduled=False,
            started_at=started_iso,
            ended_at=resolved_iso,
            duration_minutes=_duration_minutes(started_iso, resolved_iso),
            url=item.get("shortlink"),
            raw_timestamp=None,
            extra={"status": item.get("status"),
                   "components": [c.get("name") for c in item.get("components", [])],
                   "update_count": len(item.get("incident_updates", []))},
        ))
    maintenances = client.get_json(
        f"{base_url}/api/v2/scheduled-maintenances.json")
    for item in maintenances.get("scheduled_maintenances", []):
        started = item.get("scheduled_for") or item.get("started_at")
        ended = item.get("scheduled_until") or item.get("resolved_at")
        started_iso = _iso(datetime.fromisoformat(started)) if started else None
        ended_iso = _iso(datetime.fromisoformat(ended)) if ended else None
        events.append(IncidentEvent(
            source=source,
            method="statuspage_maintenance",
            event_id=item["id"],
            title=item.get("name", ""),
            impact=(item.get("impact") or "maintenance").lower(),
            scheduled=True,
            started_at=started_iso,
            ended_at=ended_iso,
            duration_minutes=_duration_minutes(started_iso, ended_iso),
            url=item.get("shortlink"),
            raw_timestamp=None,
            extra={"status": item.get("status")},
        ))
    return events


_VAR_TAG = re.compile(r"<var[^>]*>|</var>")
# "May 29, 16:22 - 20:40 PDT" | "Apr 30, 22:55 - May 1, 00:15 PDT"
# | "May 29, 16:22 PDT"
_HISTORY_RANGE = re.compile(
    r"^(?P<m1>[A-Za-z]{3})\s+(?P<d1>\d{1,2}),\s+(?P<t1>\d{1,2}:\d{2})"
    r"(?:\s*-\s*(?:(?P<m2>[A-Za-z]{3})\s+(?P<d2>\d{1,2}),\s+)?(?P<t2>\d{1,2}:\d{2}))?"
    r"\s+(?P<tz>[A-Z]{3,4})$"
)


def _parse_history_timestamp(raw: str, year: int,
                             month_name: str) -> tuple[str | None, str | None]:
    """Best-effort (started_at, ended_at) from a Statuspage history string."""
    text = _VAR_TAG.sub("", raw).strip()
    match = _HISTORY_RANGE.match(text)
    if not match:
        return None, None
    offset_hours = TZ_OFFSETS_HOURS.get(match.group("tz"))
    if offset_hours is None:
        return None, None
    tz = timezone(timedelta(hours=offset_hours))
    month_num = MONTHS.get(match.group("m1").lower())
    if month_num is None:
        return None, None

    def build(month: int, day: int, clock: str, year_: int) -> datetime:
        hour, minute = (int(p) for p in clock.split(":"))
        return datetime(year_, month, day, hour, minute, tzinfo=tz)

    start = build(month_num, int(match.group("d1")), match.group("t1"), year)
    end = None
    if match.group("t2"):
        end_month, end_year = month_num, year
        if match.group("m2"):
            end_month = MONTHS.get(match.group("m2").lower(), month_num)
            # Cross-year ranges (Dec 31 - Jan 1) roll the year forward.
            if end_month < month_num:
                end_year += 1
        end_day = int(match.group("d2")) if match.group("d2") else int(match.group("d1"))
        end = build(end_month, end_day, match.group("t2"), end_year)
    # The month block names the month the incident is filed under; a mismatch
    # between block month and parsed month is possible at page boundaries but
    # has not been observed, so the parsed month wins.
    del month_name
    return _iso(start), (_iso(end) if end else None)


_REACT_PROPS = re.compile(
    r'data-react-class="HistoryIndex"\s+data-react-props="([^"]*)"')


def _statuspage_history_events(source: str, base_url: str, client: ArchiveClient,
                               pages: int) -> list[IncidentEvent]:
    events: list[IncidentEvent] = []
    for page in range(1, pages + 1):
        text = client.get_html(f"{base_url}/history",
                               save_as=f"{source}_history_p{page}.html",
                               params={"page": page})
        match = _REACT_PROPS.search(text)
        if not match:
            raise ScrapeError(f"{source} history page {page}: no HistoryIndex props")
        props = json.loads(html.unescape(match.group(1)))
        months = props.get("months", [])
        if not months:
            break
        for month in months:
            for item in month.get("incidents", []):
                raw_ts = item.get("timestamp") or ""
                started, ended = _parse_history_timestamp(
                    raw_ts, int(month["year"]), month["name"])
                events.append(IncidentEvent(
                    source=source,
                    method="statuspage_history",
                    event_id=item["code"],
                    title=item.get("name", ""),
                    impact=(item.get("impact") or "unknown").lower(),
                    scheduled=None,
                    started_at=started,
                    ended_at=ended,
                    duration_minutes=_duration_minutes(started, ended),
                    url=f"{base_url}/incidents/{item['code']}",
                    raw_timestamp=_VAR_TAG.sub("", raw_ts).strip(),
                    extra={"month": f"{month['year']}-{month['name']}"},
                ))
    return events


def scrape_statuspage(source: str, base_url: str, client: ArchiveClient,
                      history_pages: int) -> list[IncidentEvent]:
    events = _statuspage_api_events(source, base_url, client)
    seen = {event.event_id for event in events}
    for event in _statuspage_history_events(source, base_url, client,
                                            history_pages):
        # The JSON API carries exact ISO timestamps; history entries only
        # repeat them with coarser strings, so the API version wins.
        if event.event_id not in seen:
            seen.add(event.event_id)
            events.append(event)
    return events


# --------------------------------------------------------------------------
# CoinGecko (status.coingecko.com)
# --------------------------------------------------------------------------

COINGECKO_BASE = "https://status.coingecko.com"
COINGECKO_TZ = timezone(timedelta(hours=8))  # page renders "+08" / "UTC+0800"

_CG_ROW = re.compile(
    r'data-js-date="(?P<date>[^"]+)"[^>]*>.*?'
    r'<a\s+href="(?P<url>[^"]*?/incidents/(?P<id>\d+))">\s*(?P<title>[^<]+)</a>',
    re.S,
)
_CG_DETAIL_DATES = re.compile(r'data-js-date="([^"]+)"')
# The listing paginates with an opaque Elixir-term cursor, not page numbers:
# <a href=".../incidents?after=g3QAAA...&page=1">Next</a>
_CG_NEXT = re.compile(r'<a[^>]*href="([^"]*?/incidents\?[^"]*)"[^>]*>\s*Next')
# Detail pages put the severity in the <title>:
# "Minor incident: [Ongoing] Degraded Performance | CoinGecko API Status"
_CG_DETAIL_TITLE = re.compile(r"<title>\s*([^|<]+?)\s*\|", re.S)


def _coingecko_impact(title: str) -> tuple[str, bool]:
    lowered = title.lower()
    if "scheduled maintenance" in lowered or "maintenance" in lowered:
        return "maintenance", True
    if lowered.startswith("major incident"):
        return "major", False
    if lowered.startswith("minor incident"):
        return "minor", False
    return "unknown", False


def scrape_coingecko(client: ArchiveClient, deep: bool,
                     max_pages: int = 10) -> list[IncidentEvent]:
    events: list[IncidentEvent] = []
    seen: set[str] = set()
    url = f"{COINGECKO_BASE}/incidents"
    for page in range(1, max_pages + 1):
        text = client.get_html(url, save_as=f"coingecko_incidents_p{page}.html")
        new_rows = 0
        for row in _CG_ROW.finditer(text):
            if row.group("id") in seen:
                continue
            seen.add(row.group("id"))
            new_rows += 1
            title = html.unescape(row.group("title")).strip()
            impact, scheduled = _coingecko_impact(title)
            filed = datetime.fromisoformat(row.group("date")).replace(
                tzinfo=COINGECKO_TZ)
            events.append(IncidentEvent(
                source="coingecko",
                method="listing_html",
                event_id=row.group("id"),
                title=title,
                impact=impact,
                scheduled=scheduled,
                started_at=_iso(filed),
                ended_at=None,
                duration_minutes=None,
                url=row.group("url"),
                raw_timestamp=row.group("date"),
                extra={},
            ))
        next_link = _CG_NEXT.search(text)
        if not next_link or new_rows == 0:
            break
        url = html.unescape(next_link.group(1))
        if url.startswith("/"):
            url = COINGECKO_BASE + url
    if deep:
        for event in events:
            try:
                detail = client.get_html(
                    event.url or f"{COINGECKO_BASE}/incidents/{event.event_id}",
                    save_as=f"coingecko_incident_{event.event_id}.html")
            except ScrapeError as exc:
                event.extra["detail_error"] = str(exc)
                continue
            title_match = _CG_DETAIL_TITLE.search(detail)
            if title_match:
                detail_title = html.unescape(title_match.group(1)).strip()
                impact, scheduled = _coingecko_impact(detail_title)
                if impact != "unknown":
                    event.impact, event.scheduled = impact, scheduled
                event.extra["detail_title"] = detail_title
            stamps = sorted(
                datetime.fromisoformat(s).replace(tzinfo=COINGECKO_TZ)
                for s in _CG_DETAIL_DATES.findall(detail)
            )
            if stamps:
                event.method = "detail_html"
                event.started_at = _iso(stamps[0])
                # Last update timestamp approximates resolution; an [Ongoing]
                # title means the archive page itself never closed the event.
                if "[ongoing]" not in event.title.lower():
                    event.ended_at = _iso(stamps[-1])
                    event.duration_minutes = _duration_minutes(
                        event.started_at, event.ended_at)
                event.extra["update_count"] = len(stamps)
    return events


# --------------------------------------------------------------------------
# OKX (status API + status page + help-center postmortems)
# --------------------------------------------------------------------------

OKX_BASE = "https://www.okx.com"
OKX_TZ = timezone(timedelta(hours=8))  # status page states "Time zone: UTC+8"

_OKX_LI = re.compile(
    r'<li[^>]*>.*?<h3>(?P<title>[^<]+)</h3>.*?<span>(?P<when>[^<]+)</span>'
    r'.*?(?:<span class="(?P<color>[a-z]+)">.*?</i>(?P<status>[^<]+)</span>)?'
    r'.*?(?:<div class="impact-description">(?P<desc>[^<]*)</div>)?.*?</li>',
    re.S,
)
_OKX_TIME = re.compile(
    r"(?P<m>[A-Za-z]{3})\s+(?P<d>\d{1,2}),\s+(?P<y>\d{4}),\s+"
    r"(?P<clock>\d{1,2}:\d{2})\s+(?P<ampm>[AP]M)"
)
_OKX_SLUG_DATE = re.compile(r"/([a-z]{3})-(\d{1,2})-(\d{4})-([a-z0-9-]+)$")


def _okx_parse_when(raw: str) -> tuple[str | None, str | None]:
    """Parse 'Aug 4, 2026, 5:25 PM ~ 5:35 PM' (UTC+8), or a cross-day range."""
    parts = [p.strip() for p in raw.split("~")]
    matches = [_OKX_TIME.search(p) for p in parts]
    if not matches or matches[0] is None:
        return None, None

    def build(match: re.Match[str]) -> datetime:
        hour, minute = (int(p) for p in match.group("clock").split(":"))
        hour = hour % 12 + (12 if match.group("ampm") == "PM" else 0)
        return datetime(int(match.group("y")), MONTHS[match.group("m").lower()],
                        int(match.group("d")), hour, minute, tzinfo=OKX_TZ)

    start = build(matches[0])
    end = None
    if len(parts) > 1:
        if matches[1] is not None:
            end = build(matches[1])
        else:
            clock = re.search(r"(\d{1,2}):(\d{2})\s+([AP]M)", parts[1])
            if clock:
                hour = int(clock.group(1)) % 12 + (12 if clock.group(3) == "PM" else 0)
                end = start.replace(hour=hour, minute=int(clock.group(2)))
                if end < start:
                    end += timedelta(days=1)
    return _iso(start), (_iso(end) if end else None)


def scrape_okx(client: ArchiveClient) -> list[IncidentEvent]:
    events: list[IncidentEvent] = []

    payload = client.get_json(f"{OKX_BASE}/api/v5/system/status",
                              params={"state": "completed"})
    for item in payload.get("data", []):
        begin = datetime.fromtimestamp(int(item["begin"]) / 1000, tz=UTC)
        end = (datetime.fromtimestamp(int(item["end"]) / 1000, tz=UTC)
               if item.get("end") else None)
        started_iso, ended_iso = _iso(begin), (_iso(end) if end else None)
        scheduled = item.get("maintType") == "1"
        events.append(IncidentEvent(
            source="okx",
            method="status_api",
            event_id=f"okx-api-{item['begin']}-{item.get('system', '')}",
            title=item.get("title", ""),
            impact="maintenance" if scheduled else "unscheduled",
            scheduled=scheduled,
            started_at=started_iso,
            ended_at=ended_iso,
            duration_minutes=_duration_minutes(started_iso, ended_iso),
            url=item.get("href") or None,
            raw_timestamp=None,
            extra={"serviceType": item.get("serviceType"),
                   "system": item.get("system"),
                   "maintType": item.get("maintType")},
        ))
    seen_starts = {event.started_at for event in events}

    text = client.get_html(f"{OKX_BASE}/en-us/status", save_as="okx_status.html")
    for match in _OKX_LI.finditer(text):
        raw_when = html.unescape(match.group("when")).strip()
        started, ended = _okx_parse_when(raw_when)
        if started is None or started in seen_starts:
            continue
        seen_starts.add(started)
        events.append(IncidentEvent(
            source="okx",
            method="status_html",
            event_id=f"okx-page-{started}",
            title=html.unescape(match.group("title")).strip(),
            impact="unknown",
            scheduled=None,
            started_at=started,
            ended_at=ended,
            duration_minutes=_duration_minutes(started, ended),
            url=f"{OKX_BASE}/en-us/status",
            raw_timestamp=raw_when,
            extra={"status": (match.group("status") or "").strip() or None},
        ))

    # Date-slugged postmortems ("mar-27-2024-rest-api-service-issues") reach
    # back to 2020. Date-only events: rate calibration, not duration.
    text = client.get_html(f"{OKX_BASE}/en-us/help/category/issue-description",
                           save_as="okx_issue_category.html")
    for href in dict.fromkeys(re.findall(r'href="(/en-us/help/[^"]+)"', text)):
        slug = _OKX_SLUG_DATE.search(href)
        if not slug:
            continue
        month = MONTHS.get(slug.group(1))
        if month is None:
            continue
        day_start = datetime(int(slug.group(3)), month, int(slug.group(2)),
                             tzinfo=OKX_TZ)
        events.append(IncidentEvent(
            source="okx",
            method="postmortem_slug",
            event_id=f"okx-help-{href.rsplit('/', 1)[-1]}",
            title=slug.group(4).replace("-", " "),
            impact="unscheduled",
            scheduled=False,
            started_at=_iso(day_start),
            ended_at=None,
            duration_minutes=None,
            url=f"{OKX_BASE}{href}",
            raw_timestamp=href,
            extra={"date_only": True},
        ))
    return events


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

SOURCES = ("coinbase", "bitstamp", "coingecko", "okx")
STATUSPAGE_BASES = {
    "coinbase": "https://status.coinbase.com",
    "bitstamp": "https://status.bitstamp.net",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", action="append", choices=SOURCES,
                        help="restrict to selected sources (repeatable)")
    parser.add_argument("--history-months", type=int, default=36,
                        help="Statuspage history depth in months (3 per page)")
    parser.add_argument("--deep", action="store_true",
                        help="fetch each CoinGecko incident page for durations")
    parser.add_argument("--delay-seconds", type=float, default=0.4,
                        help="politeness delay between requests")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    selected = args.source or list(SOURCES)
    events_dir = args.output_dir / "events"
    events_dir.mkdir(parents=True, exist_ok=True)
    history_pages = max(1, -(-args.history_months // 3))

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "requested_sources": selected,
        "history_months": args.history_months,
        "deep": args.deep,
        "sources": {},
    }

    with ArchiveClient(raw_dir=args.output_dir / "raw",
                       delay_seconds=args.delay_seconds) as client:
        for source in selected:
            started_requests = client.request_count
            try:
                if source in STATUSPAGE_BASES:
                    events = scrape_statuspage(source, STATUSPAGE_BASES[source],
                                               client, history_pages)
                elif source == "coingecko":
                    events = scrape_coingecko(client, deep=args.deep)
                else:
                    events = scrape_okx(client)
            except ScrapeError as exc:
                print(f"{source}: FAILED — {exc}")
                manifest["sources"][source] = {"status": "failed",
                                               "error": str(exc)}
                continue
            events.sort(key=lambda e: e.started_at or "")
            out_path = events_dir / f"{source}.json"
            out_path.write_text(json.dumps([asdict(e) for e in events], indent=1)
                                + "\n", encoding="utf-8")
            dated = [e for e in events if e.started_at]
            summary = {
                "status": "success",
                "event_count": len(events),
                "with_start": len(dated),
                "with_duration": sum(1 for e in events
                                     if e.duration_minutes is not None),
                "earliest": min((e.started_at for e in dated), default=None),
                "latest": max((e.started_at for e in dated), default=None),
                "requests": client.request_count - started_requests,
                "events_file": str(out_path.relative_to(args.output_dir)),
            }
            manifest["sources"][source] = summary
            print(f"{source}: {summary['event_count']} events "
                  f"({summary['with_duration']} with duration), "
                  f"{summary['earliest']} → {summary['latest']}")

    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    failed = [s for s, v in manifest["sources"].items() if v["status"] != "success"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
