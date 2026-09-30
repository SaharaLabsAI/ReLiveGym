"""Fit incident-generator parameters from scraped provider archives.

Consumes ``calibration/events/*.json`` (see scrape_archives.py) and emits
``generator_params_v1.json``: per-source empirical statistics plus a synthesized
per-venue parameter block for the six data venues. Every number carries a
provenance string; nothing here is tuned against any system under test — this
file is generated once and frozen.

Method summary:

- Provider severity vocabularies are harmonized into three classes:
  ``outage`` (unscheduled, severe), ``degradation`` (unscheduled, minor),
  ``maintenance`` (scheduled). Informational ``none`` entries are dropped.
- Coinbase's archive is dominated by per-asset wallet entries (delayed
  sends/receives) and fiat-rail issues; an explicit exclusion filter keeps only
  market-data/trading/platform events. Rule hits are counted in the output.
- Overlapping same-class intervals are merged (OKX publishes phase + overall
  rows for one maintenance) before rates or durations are computed.
- Durations are fit as a lognormal via robust quantile matching
  (mu = ln p50, sigma = (ln p75 - ln p25) / (2 * 0.67449)) because archive
  resolution timestamps are upper bounds (late postmortem updates inflate the
  tail, most visibly for CoinGecko).
- Rates divide merged event counts by each archive's observed span. OKX spans
  differ per retrieval method: postmortem slugs reach 2020 (outages), while the
  status API covers only recent months (maintenance).
- Binance has no archive; its outage rate/duration is anchored to the incident
  counts and downtime minutes in Binance's published H1 2024 / H1 2025 API
  uptime reports. KuCoin has no archive or anchor; it borrows the pooled
  exchange-class parameters and is flagged high-uncertainty (grid axis).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
PARAMS_VERSION = "v1"

DEFAULT_CALIBRATION_DIR = Path(__file__).resolve().parent / "calibration"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "generator_params_v1.json"

ARCHIVE_SOURCES = ("coinbase", "bitstamp", "coingecko", "okx")
CLASSES = ("outage", "degradation", "maintenance")

# Interval-merge tolerance: OKX publishes "Overall" plus per-phase rows for a
# single maintenance; statuspage occasionally splits one episode into
# back-to-back entries.
MERGE_GAP_MINUTES = 5.0

# Quantile-matching constant: ln p75 - ln p25 of a lognormal equals
# 2 * 0.67449 * sigma.
_QUANTILE_Z75 = 0.67449

# Coinbase entries excluded from market-data calibration: per-asset wallet
# operations and fiat rails do not touch the Exchange market-data API.
COINBASE_EXCLUDE = re.compile(
    r"send|receive|transaction|deposit|withdraw|staking|inscription|card|sepa"
    r"|payment|banking|wire|ach|onramp|cashout|custody|vault|reward|nft"
    r"|travel rule|futures transfer",
    re.IGNORECASE,
)

# Binance publishes no incident archive; its semi-annual API uptime reports
# give incident counts and downtime minutes for "essential trading APIs".
BINANCE_UPTIME_ANCHOR = {
    "observed_years": 1.0,
    "incidents": [
        {"period": "H1 2024", "downtime_minutes": 12,
         "note": "partial downtime on some query API services",
         "url": "https://www.binance.com/en/blog/tech/8233809493081423281"},
        {"period": "H1 2025", "downtime_minutes": 45,
         "note": "Futures UM order placement; Spot API reported 100%",
         "url": "https://www.binance.com/en/blog/tech/4885724370067176471"},
    ],
    "caveat": "self-reported; both incidents were partial, not full outages",
}


def _parse(ts: str | None) -> datetime | None:
    return datetime.fromisoformat(ts) if ts else None


def classify(event: dict[str, Any]) -> str | None:
    """Map one archive event to outage/degradation/maintenance, or drop."""
    source, impact = event["source"], event["impact"]
    if event["scheduled"] or impact == "maintenance":
        return "maintenance"
    if source == "coinbase" and COINBASE_EXCLUDE.search(event["title"]):
        return None
    if impact in ("critical", "major", "unscheduled"):
        return "outage"
    if source == "okx" and impact == "unknown":
        # status-page issue-list items ("trading service issue updates").
        return "outage"
    if impact == "minor":
        return "degradation"
    return None  # informational "none" and unclassified leftovers


def merge_intervals(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge overlapping/adjacent events of one (source, class) group."""
    dated = sorted((e for e in events if e["started_at"]),
                   key=lambda e: e["started_at"])
    undated = [e for e in events if not e["started_at"]]
    merged: list[dict[str, Any]] = []
    gap = timedelta(minutes=MERGE_GAP_MINUTES)
    for event in dated:
        start, end = _parse(event["started_at"]), _parse(event["ended_at"])
        if merged:
            prev = merged[-1]
            prev_end = _parse(prev["ended_at"])
            if prev_end is not None and start is not None and start <= prev_end + gap:
                if end is not None and end > prev_end:
                    prev["ended_at"] = event["ended_at"]
                prev["merged_ids"].append(event["event_id"])
                continue
        merged.append({
            "started_at": event["started_at"],
            "ended_at": event["ended_at"],
            "merged_ids": [event["event_id"]],
        })
    for interval in merged:
        s, e = _parse(interval["started_at"]), _parse(interval["ended_at"])
        interval["duration_minutes"] = (
            round((e - s).total_seconds() / 60, 2)
            if s is not None and e is not None and e >= s else None)
    return merged + [{"started_at": None, "ended_at": None,
                      "duration_minutes": None,
                      "merged_ids": [e["event_id"]]} for e in undated]


def quantile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolation quantile of pre-sorted values."""
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (pos - lo) * (sorted_values[hi] - sorted_values[lo])


def fit_lognormal(durations: list[float]) -> dict[str, Any] | None:
    """Robust lognormal fit from quantiles; None below 3 observations."""
    values = sorted(d for d in durations if d and d > 0)
    if len(values) < 3:
        return None
    p25, p50, p75 = (quantile(values, q) for q in (0.25, 0.50, 0.75))
    sigma = (math.log(p75) - math.log(p25)) / (2 * _QUANTILE_Z75)
    return {
        "mu": round(math.log(p50), 4),
        "sigma": round(max(sigma, 0.05), 4),
        "n": len(values),
        "quantiles_minutes": {
            f"p{int(q * 100)}": round(quantile(values, q), 1)
            for q in (0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
        },
    }


def okx_window_start(events: list[dict[str, Any]], cls: str) -> str | None:
    """OKX archive spans differ per retrieval method (see module docstring)."""
    methods = (("postmortem_slug", "status_html", "status_api")
               if cls != "maintenance" else ("status_api",))
    starts = [e["started_at"] for e in events
              if e["method"] in methods and e["started_at"]]
    return min(starts, default=None)


def fit_source(source: str, events: list[dict[str, Any]],
               window_end: datetime) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {cls: [] for cls in CLASSES}
    dropped = 0
    for event in events:
        cls = classify(event)
        if cls is None:
            dropped += 1
        else:
            grouped[cls].append(event)

    archive_start = min((e["started_at"] for e in events if e["started_at"]),
                        default=None)
    result: dict[str, Any] = {
        "archive_start": archive_start,
        "window_end": window_end.isoformat(),
        "event_count_raw": len(events),
        "dropped_informational_or_filtered": dropped,
        "classes": {},
    }
    for cls, members in grouped.items():
        window_start = (okx_window_start(members, cls) if source == "okx"
                        else archive_start)
        merged = merge_intervals(members)
        durations = [m["duration_minutes"] for m in merged
                     if m["duration_minutes"] is not None]
        years = None
        if window_start and merged:
            span = window_end - _parse(window_start)
            years = round(span.total_seconds() / (365.25 * 86400), 3)
        result["classes"][cls] = {
            "n_raw": len(members),
            "n_merged": len(merged),
            "n_with_duration": len(durations),
            "window_start": window_start,
            "window_years": years,
            "rate_per_year": (round(len(merged) / years, 3)
                              if years and years > 0 else None),
            "duration_lognormal": fit_lognormal(durations),
        }
        if cls == "maintenance":
            hours = [int(m["started_at"][11:13]) for m in merged
                     if m["started_at"]]
            result["classes"][cls]["start_hour_utc_histogram"] = [
                hours.count(h) for h in range(24)]
    return result


def _pool(sources: dict[str, Any], names: list[str], cls: str,
          key: str) -> list[float]:
    values = []
    for name in names:
        stats = sources[name]["classes"][cls]
        if key == "rate" and stats["rate_per_year"] is not None:
            values.append(stats["rate_per_year"])
        elif key == "sigma" and stats["duration_lognormal"]:
            values.append(stats["duration_lognormal"]["sigma"])
        elif key == "mu" and stats["duration_lognormal"]:
            values.append(stats["duration_lognormal"]["mu"])
    return values


def synthesize_venues(sources: dict[str, Any]) -> dict[str, Any]:
    """Per-venue parameters for the six data venues, with provenance."""
    exchange_pool = ["okx", "coinbase", "bitstamp"]

    def mean(values: list[float]) -> float | None:
        return round(sum(values) / len(values), 4) if values else None

    def fitted(name: str) -> dict[str, Any]:
        block: dict[str, Any] = {"provenance": f"fitted:{name}", "classes": {}}
        for cls in CLASSES:
            stats = sources[name]["classes"][cls]
            entry: dict[str, Any] = {
                "rate_per_year": stats["rate_per_year"],
                "duration_lognormal": (
                    {k: stats["duration_lognormal"][k] for k in ("mu", "sigma")}
                    if stats["duration_lognormal"] else None),
            }
            if cls == "maintenance":
                entry["start_hour_utc_histogram"] = stats.get(
                    "start_hour_utc_histogram")
            block["classes"][cls] = entry
        return block

    def borrowed(pool: list[str], note: str, uncertainty: str) -> dict[str, Any]:
        block: dict[str, Any] = {
            "provenance": f"borrowed:mean({','.join(pool)})",
            "uncertainty": uncertainty,
            "note": note,
            "classes": {},
        }
        okx_maint = sources["okx"]["classes"]["maintenance"]
        for cls in CLASSES:
            entry: dict[str, Any] = {
                "rate_per_year": mean(_pool(sources, pool, cls, "rate")),
                "duration_lognormal": {
                    "mu": mean(_pool(sources, pool, cls, "mu")),
                    "sigma": mean(_pool(sources, pool, cls, "sigma")),
                },
            }
            if cls == "maintenance":
                entry["start_hour_utc_histogram"] = okx_maint.get(
                    "start_hour_utc_histogram")
            block["classes"][cls] = entry
        return block

    # Binance: rate and typical duration anchored to published uptime reports;
    # duration spread (sigma) borrowed from the exchange pool since two
    # published incidents cannot constrain a distribution.
    anchor = BINANCE_UPTIME_ANCHOR
    downtimes = [i["downtime_minutes"] for i in anchor["incidents"]]
    binance = {
        "provenance": "anchored:binance_uptime_reports"
                      "+borrowed_sigma:exchange_pool",
        "uncertainty": "medium (self-reported, partial incidents)",
        "anchor": anchor,
        "classes": {
            "outage": {
                "rate_per_year": round(len(downtimes) / anchor["observed_years"], 3),
                "duration_lognormal": {
                    "mu": round(math.log(sum(downtimes) / len(downtimes)), 4),
                    "sigma": mean(_pool(sources, exchange_pool, "outage",
                                        "sigma")),
                },
            },
            "degradation": borrowed(exchange_pool, "", "")["classes"]["degradation"],
            "maintenance": borrowed(exchange_pool, "", "")["classes"]["maintenance"],
        },
    }

    return {
        "binance": binance,
        "okx": fitted("okx"),
        "kucoin": borrowed(
            exchange_pool,
            "no public archive or uptime report found (2026-08); "
            "rate uncertainty is a grid axis",
            "high"),
        "coinbase": fitted("coinbase"),
        "bitstamp": fitted("bitstamp"),
        "coingecko": fitted("coingecko"),
        "defillama": borrowed(
            ["coingecko"],
            "aggregator-class venue with no archive; borrows CoinGecko",
            "high"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--calibration-dir", type=Path,
                        default=DEFAULT_CALIBRATION_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    manifest = json.loads(
        (args.calibration_dir / "manifest.json").read_text(encoding="utf-8"))
    window_end = datetime.fromisoformat(manifest["created_at"])

    sources: dict[str, Any] = {}
    event_file_hashes: dict[str, str] = {}
    for name in ARCHIVE_SOURCES:
        path = args.calibration_dir / "events" / f"{name}.json"
        payload = path.read_bytes()
        event_file_hashes[name] = hashlib.sha256(payload).hexdigest()
        sources[name] = fit_source(name, json.loads(payload), window_end)

    venues = synthesize_venues(sources)

    # Validation checks, recorded alongside the parameters they qualify.
    sim_days = 153  # 2026-03-01 → 2026-08-01
    sparsity = {
        venue: round(block["classes"]["outage"]["rate_per_year"]
                     * sim_days / 365.25, 2)
        for venue, block in venues.items()
        if block["classes"]["outage"]["rate_per_year"] is not None
    }
    validation = {
        "exchange_outage_rates_per_year": {
            name: sources[name]["classes"]["outage"]["rate_per_year"]
            for name in ("okx", "coinbase", "bitstamp")},
        "exchange_outage_p50_minutes": {
            name: (sources[name]["classes"]["outage"]["duration_lognormal"]
                   or {}).get("quantiles_minutes", {}).get("p50")
            for name in ("okx", "coinbase", "bitstamp")},
        "expected_outages_per_153d_window": sparsity,
        "note": "cross-source spread bounds venue-level rate uncertainty; "
                "the trace grid sweeps rate x{0.5,1,4} around these fits",
    }

    output = {
        "schema_version": SCHEMA_VERSION,
        "params_version": PARAMS_VERSION,
        "fitted_at": datetime.now(window_end.tzinfo).isoformat(),
        "calibration": {
            "manifest_created_at": manifest["created_at"],
            "event_file_sha256": event_file_hashes,
        },
        "method": {
            "classes": list(CLASSES),
            "merge_gap_minutes": MERGE_GAP_MINUTES,
            "duration_fit": "lognormal via quantile matching "
                            "(mu=ln p50, sigma=(ln p75 - ln p25)/1.34898)",
            "coinbase_market_data_filter": COINBASE_EXCLUDE.pattern,
        },
        "sources": sources,
        "venues": venues,
        "validation": validation,
    }
    args.output.write_text(json.dumps(output, indent=1) + "\n",
                           encoding="utf-8")

    for name, stats in sources.items():
        parts = []
        for cls in CLASSES:
            c = stats["classes"][cls]
            fit = c["duration_lognormal"]
            p50 = fit["quantiles_minutes"]["p50"] if fit else None
            parts.append(f"{cls}: {c['n_merged']}ev "
                         f"{c['rate_per_year'] or '-'}/yr p50={p50 or '-'}m")
        print(f"{name:10s} " + " | ".join(parts))
    print(f"\nexpected outages per 153-day window: {sparsity}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
