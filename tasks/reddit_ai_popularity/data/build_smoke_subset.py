"""Carve a small windowed subset of built_min0 for fast smoke runs.

The full built_min0 cascades.jsonl is ~1.1 GB; loading it (streamed twice per
run) dominates a smoke. The smoke cells only need a ~2-week slice, so this
writes data/built_smoke/ = every root posted in [LO, HI) plus its full reply tree,
with a build_stats.json whose coverage window spans the smoke run's
[history_start, sim_end]. One pass over the big files; run once.

    python tasks/reddit_ai_popularity/data/build_smoke_subset.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "built_min0"
DST = HERE / "built_smoke"

# Roots in [LO, HI). Must bracket the smoke run's history_start (sim_start -
# history_days) and sim_end. Quick-run cells: sim 2026-03-15..03-18, history 7d ->
# history_start 03-08; margin to 03-07 / 03-19.
LO = datetime(2026, 3, 7, tzinfo=timezone.utc).timestamp()
HI = datetime(2026, 3, 19, tzinfo=timezone.utc).timestamp()


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    keep: set[str] = set()
    n_roots = 0
    with open(SRC / "roots.jsonl") as f, open(DST / "roots.jsonl", "w") as out:
        for line in f:
            r = json.loads(line)
            if LO <= r["created_utc"] < HI:
                keep.add(r["id"])
                out.write(line)
                n_roots += 1
    n_casc = 0
    with open(SRC / "cascades.jsonl") as f, \
            open(DST / "cascades.jsonl", "w") as out:
        for line in f:
            # cheap prefilter before full parse
            rid = json.loads(line)["root_id"]
            if rid in keep:
                out.write(line)
                n_casc += 1
    src_stats = json.loads((SRC / "build_stats.json").read_text())
    stats = dict(src_stats)
    stats["window"] = {
        "after": LO, "before": HI,
        "after_h": datetime.fromtimestamp(LO, timezone.utc).isoformat(),
        "before_h": datetime.fromtimestamp(HI, timezone.utc).isoformat(),
    }
    stats["n_kept_roots"] = n_roots
    stats["note"] = "smoke subset of built_min0 (build_smoke_subset.py)"
    (DST / "build_stats.json").write_text(json.dumps(stats, indent=1))
    print(f"wrote {n_roots} roots, {n_casc} cascades -> {DST}")


if __name__ == "__main__":
    main()
