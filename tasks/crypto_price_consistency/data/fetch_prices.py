#!/usr/bin/env python3
"""Fetch and compare keyless cross-venue cryptocurrency spot candles.

Run from the repository root, for example:

    python -m tasks.crypto_price_consistency.data.fetch_prices \
      --cohort btc_usdt_spot \
      --start 2026-08-01T00:00:00Z \
      --end 2026-08-02T00:00:00Z
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from tasks.crypto_price_consistency.data.artifacts import (
    write_comparison_artifacts,
    write_json,
    write_normalized,
)
from tasks.crypto_price_consistency.data.markets import COHORTS, get_cohort
from tasks.crypto_price_consistency.data.models import aggregate_candles, iso_utc
from tasks.crypto_price_consistency.data.sources import (
    ADAPTERS,
    FetchError,
    HttpJsonClient,
    fetch_market_source,
)


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "datasets"


def parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def timestamp_ms(value: datetime) -> int:
    return int(value.timestamp() * 1000)


def floor_datetime(value: datetime, interval_minutes: int) -> datetime:
    interval_seconds = interval_minutes * 60
    epoch_seconds = int(value.timestamp())
    floored = epoch_seconds - epoch_seconds % interval_seconds
    return datetime.fromtimestamp(floored, tz=UTC)


def compact_timestamp(value: datetime) -> str:
    return value.strftime("%Y%m%dT%H%M%SZ")


def resolve_window(args: argparse.Namespace) -> tuple[datetime, datetime]:
    now = datetime.now(UTC)
    end = (
        parse_datetime(args.end)
        if args.end
        else floor_datetime(now, args.output_interval_minutes)
    )
    start = (
        parse_datetime(args.start)
        if args.start
        else end - timedelta(hours=args.lookback_hours)
    )
    if start >= end:
        raise ValueError("start must be earlier than end")
    interval_ms = args.output_interval_minutes * 60_000
    for label, value in (("start", start), ("end", end)):
        if timestamp_ms(value) % interval_ms:
            raise ValueError(
                f"{label} must align to a {args.output_interval_minutes}-minute UTC boundary"
            )
    if args.output_interval_minutes % args.source_interval_minutes:
        raise ValueError("output interval must be a multiple of source interval")
    return start, end


def select_sources(args: argparse.Namespace):
    cohort = get_cohort(args.cohort)
    configured = {source.source: source for source in cohort.sources}
    requested = set(args.source or ())
    excluded = set(args.exclude_source or ())
    unknown = (requested | excluded) - configured.keys()
    if unknown:
        raise ValueError(
            f"sources not in cohort {cohort.name}: {', '.join(sorted(unknown))}"
        )
    overlap = requested & excluded
    if overlap:
        raise ValueError(
            "sources cannot be both included and excluded: "
            + ", ".join(sorted(overlap))
        )
    selected_names = requested or {
        source.source for source in cohort.sources if source.enabled_by_default
    }
    selected = [
        source
        for source in cohort.sources
        if source.source in selected_names and source.source not in excluded
    ]
    if not selected:
        raise ValueError("source selection is empty")
    return cohort, selected


def dataset_layout(dataset_dir: Path, cohort: str, source: str, interval: int) -> Path:
    return (
        dataset_dir
        / "normalized"
        / f"cohort={cohort}"
        / f"source={source}"
        / f"interval={interval}m"
        / "candles.csv"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohort", default="btc_usdt_spot", choices=sorted(COHORTS)
    )
    parser.add_argument(
        "--source",
        action="append",
        choices=sorted(ADAPTERS),
        help="fetch only this source; repeat to select multiple",
    )
    parser.add_argument(
        "--exclude-source",
        action="append",
        choices=sorted(ADAPTERS),
        help="omit this source; repeat to exclude multiple sources",
    )
    parser.add_argument("--start", help="inclusive ISO-8601 UTC start")
    parser.add_argument("--end", help="exclusive ISO-8601 UTC end")
    parser.add_argument(
        "--lookback-hours",
        type=int,
        default=24,
        help="used when --start is omitted (default: 24)",
    )
    parser.add_argument(
        "--source-interval-minutes",
        type=int,
        choices=(1, 3, 5, 60),
        default=None,
        help="source interval; defaults to the cohort's configured interval",
    )
    parser.add_argument(
        "--output-interval-minutes",
        type=int,
        default=None,
        help="UTC comparison interval; defaults to the cohort's configured interval",
    )
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT
    )
    parser.add_argument("--dataset-id", help="optional output directory name")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return a failure status if any requested source fails",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the collection plan only"
    )
    parser.add_argument(
        "--list-cohorts", action="store_true", help="print configured cohorts and exit"
    )
    return parser


def print_cohorts() -> None:
    for cohort in COHORTS.values():
        sources = ", ".join(
            f"{source.source}:{source.symbol}"
            + (" [opt-in]" if not source.enabled_by_default else "")
            for source in cohort.sources
        )
        print(f"{cohort.name}: {cohort.description}\n  {sources}")


def run(args: argparse.Namespace) -> tuple[Path | None, dict[str, object]]:
    start, end = resolve_window(args)
    cohort, selected_sources = select_sources(args)
    dataset_id = args.dataset_id or "__".join(
        (
            cohort.name,
            compact_timestamp(start),
            compact_timestamp(end),
            compact_timestamp(datetime.now(UTC)),
        )
    )
    dataset_dir = args.output_root / dataset_id
    plan = {
        "dataset_id": dataset_id,
        "cohort": cohort.name,
        "base": cohort.base,
        "quote": cohort.quote,
        "market_type": cohort.market_type,
        "start": start.isoformat().replace("+00:00", "Z"),
        "end": end.isoformat().replace("+00:00", "Z"),
        "source_interval_minutes": args.source_interval_minutes,
        "output_interval_minutes": args.output_interval_minutes,
        "requested_sources": [
            {
                "name": source.source,
                "symbol": source.symbol,
                "market_type": source.market_type or cohort.market_type,
                "collection_interval_minutes": ADAPTERS[
                    source.source
                ].collection_interval_minutes(
                    args.source_interval_minutes, args.output_interval_minutes
                ),
            }
            for source in selected_sources
        ],
        "output_directory": str(dataset_dir),
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return None, plan
    if dataset_dir.exists():
        raise FileExistsError(
            f"dataset directory already exists: {dataset_dir}; choose another --dataset-id"
        )
    dataset_dir.mkdir(parents=True)

    start_ms = timestamp_ms(start)
    end_ms = timestamp_ms(end)
    now_ms = timestamp_ms(datetime.now(UTC))
    candles_by_source = {}
    source_results: dict[str, object] = {}
    normalized_files: dict[str, str] = {}

    for market_source in selected_sources:
        source_name = market_source.source
        collection_interval_minutes = ADAPTERS[
            source_name
        ].collection_interval_minutes(
            args.source_interval_minutes, args.output_interval_minutes
        )
        raw_path = dataset_dir / "raw" / f"{source_name}.jsonl"
        try:
            with HttpJsonClient(raw_path=raw_path) as client:
                source_rows = fetch_market_source(
                    client,
                    cohort=cohort,
                    market_source=market_source,
                    interval_minutes=collection_interval_minutes,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    now_ms=now_ms,
                )
            if not source_rows:
                raise FetchError("source returned no candles in the requested window")
            output_rows = aggregate_candles(
                source_rows,
                output_interval_minutes=args.output_interval_minutes,
                start_ms=start_ms,
                end_ms=end_ms,
            )
            normalized_path = dataset_layout(
                dataset_dir,
                cohort.name,
                source_name,
                args.output_interval_minutes,
            )
            write_normalized(normalized_path, output_rows)
            candles_by_source[source_name] = output_rows
            normalized_files[source_name] = str(normalized_path.relative_to(dataset_dir))
            source_results[source_name] = {
                "status": "success",
                "symbol": market_source.symbol,
                "collection_interval_minutes": collection_interval_minutes,
                "source_row_count": len(source_rows),
                "output_row_count": len(output_rows),
                "complete_output_row_count": sum(row.complete for row in output_rows),
                "raw_file": str(raw_path.relative_to(dataset_dir)),
                "normalized_file": normalized_files[source_name],
            }
            print(
                f"{source_name}: {len(source_rows)} source candles -> "
                f"{len(output_rows)} normalized candles",
                file=sys.stderr,
            )
        except Exception as exc:  # keep independent public sources independent
            source_results[source_name] = {
                "status": "error",
                "symbol": market_source.symbol,
                "collection_interval_minutes": collection_interval_minutes,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "raw_file": str(raw_path.relative_to(dataset_dir)),
            }
            print(f"{source_name}: ERROR: {exc}", file=sys.stderr)

    expected_sources = [source.source for source in selected_sources]
    comparison_dir = (
        dataset_dir
        / "comparisons"
        / f"cohort={cohort.name}"
        / f"interval={args.output_interval_minutes}m"
    )
    summary = write_comparison_artifacts(
        comparison_dir,
        candles_by_source,
        expected_sources=expected_sources,
    )
    failed_sources = [
        source
        for source, result in source_results.items()
        if result["status"] == "error"
    ]
    manifest = {
        "schema_version": 1,
        "status": "partial" if failed_sources else "success",
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        **plan,
        "sources": source_results,
        "failed_sources": failed_sources,
        "normalized_files": normalized_files,
        "comparison_directory": str(comparison_dir.relative_to(dataset_dir)),
        "comparison_summary": summary,
        "notes": [
            "end is exclusive",
            "timestamps are UTC interval-open times",
            "missing source intervals are not forward-filled",
            "comparison statistics include only complete output candles",
        ],
    }
    write_json(dataset_dir / "manifest.json", manifest)
    if not candles_by_source:
        raise FetchError(f"all requested sources failed; see {dataset_dir / 'manifest.json'}")
    if args.strict and failed_sources:
        raise FetchError(
            f"strict collection failed for: {', '.join(failed_sources)}; "
            f"see {dataset_dir / 'manifest.json'}"
        )
    return dataset_dir, manifest


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_cohorts:
        print_cohorts()
        return 0
    cohort = get_cohort(args.cohort)
    if args.source_interval_minutes is None:
        args.source_interval_minutes = cohort.default_source_interval_minutes
    if args.output_interval_minutes is None:
        args.output_interval_minutes = cohort.default_output_interval_minutes
    try:
        dataset_dir, _ = run(args)
    except (ValueError, FileExistsError, FetchError) as exc:
        parser.exit(1, f"error: {exc}\n")
    if dataset_dir is not None:
        print(dataset_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
