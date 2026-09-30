#!/usr/bin/env python3
"""Regenerate comparison artifacts from a fetched dataset's normalized files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tasks.crypto_price_consistency.data.artifacts import (
    read_normalized,
    write_comparison_artifacts,
)


def compare_dataset(dataset_dir: Path) -> dict[str, object]:
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_sources = [source["name"] for source in manifest["requested_sources"]]
    candles_by_source = {
        source: read_normalized(dataset_dir / relative_path)
        for source, relative_path in manifest["normalized_files"].items()
    }
    comparison_dir = dataset_dir / manifest["comparison_directory"]
    return write_comparison_artifacts(
        comparison_dir,
        candles_by_source,
        expected_sources=expected_sources,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    summary = compare_dataset(args.dataset)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
