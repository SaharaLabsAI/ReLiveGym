"""Generate the frozen incident trace set.

Reads ``generator_params_v1.json`` (frozen fit) and the calibration event
files, writes ``scenarios/*.jsonl`` — one header line plus one event per
line — and ``scenarios/manifest.json`` with per-file SHA-256 hashes.

Trace set:
  zero                      no incidents (over-defensiveness cost)
  real_replay               archived incidents inside the window, true times
  fitted_dev_s001..s020     sampled from the frozen generator (development)
  fitted_held_s101..s120    held out; evaluated once per system version
  grid_r{05,1,4}_d{05,1,4}_c{off,on}   rate x duration x correlation sweep
  stress_*                  eight authored tail scenarios

Determinism: every random draw comes from ``random.Random(seed)`` with the
seed recorded in the trace header; no wall-clock time enters any scenario
file, so regeneration is byte-identical and the FREEZE hashes are stable.

Authored constants (Layer E rates, capacity modulation, correlation) have no
archive to fit by nature ; they are defined once in this module,
echoed into each trace header, and swept by the grid's rate axis — never
tuned against a system.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .fit_generator import classify, merge_intervals

GENERATOR_VERSION = "v1"
WINDOW_START = datetime(2026, 3, 1, tzinfo=UTC)
WINDOW_END = datetime(2026, 8, 1, tzinfo=UTC)
WINDOW_DAYS = (WINDOW_END - WINDOW_START).days  # 153
WINDOW_YEARS = WINDOW_DAYS / 365.25

INCIDENTS_DIR = Path(__file__).resolve().parent
DEFAULT_PARAMS = INCIDENTS_DIR / "generator_params_v1.json"
DEFAULT_CALIBRATION_DIR = INCIDENTS_DIR / "calibration"
DEFAULT_OUTPUT_DIR = INCIDENTS_DIR / "scenarios"
DATASETS_DIR = INCIDENTS_DIR.parent / "data" / "datasets"

DEV_SEEDS = list(range(1, 21))
HELD_SEEDS = list(range(101, 121))
GRID_SEED_BASE = 500
RATE_MULTS = {"05": 0.5, "1": 1.0, "4": 4.0}
DUR_MULTS = {"05": 0.5, "1": 1.0, "4": 4.0}

# Fitted Layer-P durations: archive resolution timestamps are upper bounds
# and the lognormal tail is unbounded, so samples are capped. Recorded in
# every header — a documented cap, not a silent one.
DURATION_CAP_MINUTES = 3 * 24 * 60.0

# Authored constants — no archive exists for these by nature.
EDGE = {
    "net_blip": {"rate_per_day_per_venue": 1.0,      # plan range 1-5/day; low
                 "duration_lognormal": {"mu": math.log(0.5), "sigma": 1.0},
                 "duration_cap_minutes": 15.0},       # end chosen for hourly cadence
    "dns_fail": {"rate_per_year_per_venue": 6.0,
                 "duration_uniform_minutes": [1.0, 5.0]},
    "cdn_challenge": {"rate_per_year_per_venue": 4.0,
                      "venues": ["okx", "kucoin", "bitstamp", "coingecko"],
                      "duration_uniform_minutes": [2.0, 20.0]},
    "clock_skew": {"prob_per_trace": 0.3,
                   "skew_abs_seconds": [5.0, 120.0],
                   "duration_uniform_hours": [6.0, 48.0]},
    "geoblock": {"prob_per_trace": 0.15,
                 "duration_lognormal": {"mu": math.log(7 * 24 * 60.0),
                                        "sigma": 1.0}},
}
CAPACITY_MODULATION = {
    "rate_per_year_per_venue": 6.0,
    "duration_lognormal": {"mu": math.log(60.0), "sigma": 1.0},
    "budget_factor_uniform": [0.0, 0.3],
}
CORRELATED = {
    "rate_per_year": 2.0,
    "venue_pool": ["binance", "okx", "kucoin", "bitstamp",
                   "coingecko", "defillama"],
    "venues_per_event": [2, 4],
    "duration_lognormal": {"mu": math.log(60.0), "sigma": 0.6},
}

OUTAGE_BEHAVIORS = [("http_503", 0.5), ("timeout", 0.3),
                    ("http_502", 0.1), ("connect_refused", 0.1)]


def iso(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def poisson(rng: random.Random, lam: float) -> int:
    """Knuth for small lambda, normal approximation above 30."""
    if lam <= 0:
        return 0
    if lam > 30:
        return max(0, round(rng.gauss(lam, math.sqrt(lam))))
    limit, k, product = math.exp(-lam), 0, rng.random()
    while product > limit:
        k += 1
        product *= rng.random()
    return k


def sample_duration(rng: random.Random, lognorm: dict[str, float],
                    mult: float, cap: float = DURATION_CAP_MINUTES) -> float:
    minutes = rng.lognormvariate(lognorm["mu"], lognorm["sigma"]) * mult
    return min(minutes, cap)


def uniform_start(rng: random.Random) -> datetime:
    offset = rng.uniform(0, (WINDOW_END - WINDOW_START).total_seconds())
    return WINDOW_START + timedelta(seconds=int(offset))


def weighted_choice(rng: random.Random, pairs: list[tuple[str, float]]) -> str:
    r, acc = rng.random() * sum(w for _, w in pairs), 0.0
    for value, weight in pairs:
        acc += weight
        if r <= acc:
            return value
    return pairs[-1][0]


def clip(start: datetime, end: datetime) -> tuple[datetime, datetime] | None:
    start, end = max(start, WINDOW_START), min(end, WINDOW_END)
    return (start, end) if end > start else None


def event(layer: str, venue: str, mode: str, start: datetime, end: datetime,
          params: dict[str, Any], provenance: str) -> dict[str, Any] | None:
    clipped = clip(start, end)
    if clipped is None:
        return None
    return {"layer": layer, "venue": venue, "mode": mode,
            "start": iso(clipped[0]), "end": iso(clipped[1]),
            "params": params, "provenance": provenance}


# --- fitted sampling -------------------------------------------------------

def sample_layer_p(rng: random.Random, venues: dict[str, Any],
                   rate_mult: float, dur_mult: float) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for venue, spec in venues.items():
        for cls, mode, scheduled in (("outage", "outage", False),
                                     ("degradation", "degraded", False),
                                     ("maintenance", "maintenance", True)):
            block = spec["classes"][cls]
            rate, lognorm = block["rate_per_year"], block["duration_lognormal"]
            if not rate or not lognorm or lognorm.get("mu") is None:
                continue
            # The grid's rate axis models incident-frequency uncertainty;
            # maintenance is announced and calendar-driven, so it stays at
            # the fitted rate.
            lam = rate * WINDOW_YEARS * (1.0 if scheduled else rate_mult)
            for _ in range(poisson(rng, lam)):
                start = uniform_start(rng)
                if scheduled:
                    histogram = block.get("start_hour_utc_histogram") or []
                    if sum(histogram) > 0:
                        hour = rng.choices(range(24), weights=histogram)[0]
                        start = start.replace(hour=hour, minute=0, second=0)
                minutes = sample_duration(rng, lognorm, dur_mult)
                end = start + timedelta(minutes=minutes)
                if mode == "outage":
                    params: dict[str, Any] = {
                        "behavior": weighted_choice(rng, OUTAGE_BEHAVIORS)}
                elif mode == "degraded":
                    params = {"fail_fraction": round(rng.uniform(0.2, 0.9), 2),
                              "latency_multiplier": round(rng.uniform(2, 10), 1)}
                else:
                    params = {"behavior": "http_503", "scheduled": True}
                events.append(event("P", venue, mode, start, end, params,
                                    "fitted"))
    return [e for e in events if e]


def sample_layer_m(rng: random.Random, venues: list[str],
                   rate_mult: float, dur_mult: float) -> list[dict[str, Any]]:
    spec = CAPACITY_MODULATION
    events = []
    for venue in venues:
        lam = spec["rate_per_year_per_venue"] * WINDOW_YEARS * rate_mult
        for _ in range(poisson(rng, lam)):
            start = uniform_start(rng)
            end = start + timedelta(minutes=sample_duration(
                rng, spec["duration_lognormal"], dur_mult))
            factor = round(rng.uniform(*spec["budget_factor_uniform"]), 2)
            events.append(event("M", venue, "capacity_modulation", start, end,
                                {"budget_factor": factor}, "authored"))
    return [e for e in events if e]


def sample_layer_e(rng: random.Random, venues: list[str],
                   rate_mult: float, dur_mult: float) -> list[dict[str, Any]]:
    events: list[dict[str, Any] | None] = []
    blip = EDGE["net_blip"]
    for venue in venues:
        lam = blip["rate_per_day_per_venue"] * WINDOW_DAYS * rate_mult
        for _ in range(poisson(rng, lam)):
            start = uniform_start(rng)
            minutes = sample_duration(rng, blip["duration_lognormal"],
                                      dur_mult, blip["duration_cap_minutes"])
            events.append(event("E", venue, "net_blip", start,
                                start + timedelta(minutes=max(minutes, 0.1)),
                                {"behavior": "timeout"}, "authored"))
        dns = EDGE["dns_fail"]
        lam = dns["rate_per_year_per_venue"] * WINDOW_YEARS * rate_mult
        for _ in range(poisson(rng, lam)):
            start = uniform_start(rng)
            minutes = rng.uniform(*dns["duration_uniform_minutes"]) * dur_mult
            events.append(event("E", venue, "dns_fail", start,
                                start + timedelta(minutes=minutes),
                                {"behavior": "dns_error"}, "authored"))
    cdn = EDGE["cdn_challenge"]
    for venue in cdn["venues"]:
        if venue not in venues:
            continue
        lam = cdn["rate_per_year_per_venue"] * WINDOW_YEARS * rate_mult
        for _ in range(poisson(rng, lam)):
            start = uniform_start(rng)
            minutes = rng.uniform(*cdn["duration_uniform_minutes"]) * dur_mult
            events.append(event("E", venue, "cdn_challenge", start,
                                start + timedelta(minutes=minutes),
                                {"status": rng.choice([200, 403]),
                                 "body": "html_challenge"}, "authored"))
    skew = EDGE["clock_skew"]
    if rng.random() < skew["prob_per_trace"]:
        start = uniform_start(rng)
        hours = rng.uniform(*skew["duration_uniform_hours"])
        seconds = rng.uniform(*skew["skew_abs_seconds"]) * rng.choice([-1, 1])
        events.append(event("E", "client", "clock_skew", start,
                            start + timedelta(hours=hours),
                            {"skew_seconds": round(seconds, 1)}, "authored"))
    geo = EDGE["geoblock"]
    if rng.random() < geo["prob_per_trace"]:
        venue = rng.choice(venues)
        start = uniform_start(rng)
        minutes = sample_duration(rng, geo["duration_lognormal"], 1.0,
                                  cap=math.inf)
        events.append(event("E", venue, "geoblock", start,
                            start + timedelta(minutes=minutes),
                            {"status": rng.choice([403, 451])}, "authored"))
    return [e for e in events if e]


def sample_correlated(rng: random.Random,
                      venues: list[str]) -> list[dict[str, Any]]:
    spec = CORRELATED
    events: list[dict[str, Any] | None] = []
    lam = spec["rate_per_year"] * WINDOW_YEARS
    for group_index in range(poisson(rng, lam)):
        pool = [v for v in spec["venue_pool"] if v in venues]
        count = rng.randint(*spec["venues_per_event"])
        hit = rng.sample(pool, min(count, len(pool)))
        start = uniform_start(rng)
        end = start + timedelta(minutes=sample_duration(
            rng, spec["duration_lognormal"], 1.0))
        for venue in hit:
            events.append(event("P", venue, "outage", start, end,
                                {"behavior": "timeout",
                                 "correlated_group": f"cg-{group_index + 1}"},
                                "authored"))
    return [e for e in events if e]


def sample_trace(seed: int, venues: dict[str, Any], rate_mult: float,
                 dur_mult: float, correlation: bool) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    names = list(venues)
    events = (sample_layer_p(rng, venues, rate_mult, dur_mult)
              + sample_layer_m(rng, names, rate_mult, dur_mult)
              + sample_layer_e(rng, names, rate_mult, dur_mult))
    if correlation:
        events += sample_correlated(rng, names)
    return events


# --- real replay -----------------------------------------------------------

def replay_events(calibration_dir: Path) -> tuple[list[dict[str, Any]], dict]:
    """Archived incidents whose interval overlaps the window, true times."""
    events: list[dict[str, Any] | None] = []
    skipped = {"undated": 0, "date_only": 0}
    for source in ("coinbase", "bitstamp", "coingecko", "okx"):
        raw = json.loads(
            (calibration_dir / "events" / f"{source}.json").read_text())
        by_class: dict[str, list[dict[str, Any]]] = {}
        titles = {e["event_id"]: e["title"] for e in raw}
        for entry in raw:
            cls = classify(entry)
            if cls is None:
                continue
            if entry.get("extra", {}).get("date_only"):
                skipped["date_only"] += 1  # midnight-anchored; too coarse
                continue
            by_class.setdefault(cls, []).append(entry)
        for cls, entries in by_class.items():
            for interval in merge_intervals(entries):
                start = interval["started_at"]
                if start is None:
                    skipped["undated"] += 1
                    continue
                start_dt = datetime.fromisoformat(start)
                end = interval["ended_at"]
                end_dt = (datetime.fromisoformat(end) if end
                          else start_dt + timedelta(minutes=30))
                if end_dt <= WINDOW_START or start_dt >= WINDOW_END:
                    continue
                mode = {"outage": "outage", "degradation": "degraded",
                        "maintenance": "maintenance"}[cls]
                # The archive records that it happened, not its HTTP surface;
                # defaults below are the proxy's rendering choice.
                params: dict[str, Any] = {
                    "behavior": "http_503",
                    "source_ids": interval["merged_ids"],
                    "source_title": titles.get(interval["merged_ids"][0]),
                }
                if mode == "degraded":
                    params.update({"fail_fraction": 0.5,
                                   "latency_multiplier": 3.0})
                if mode == "maintenance":
                    params["scheduled"] = True
                events.append(event("P", source, mode, start_dt, end_dt,
                                    params, "replayed"))
    return [e for e in events if e], skipped


# --- stress set ------------------------------------------------------------

def highest_vol_day() -> tuple[str, float]:
    """UTC day with max realized vol (sqrt sum of squared 5-min log returns)
    on Binance BTCUSDT closes — for the maintenance_in_vol scenario."""
    path = (DATASETS_DIR / "btc_usdt_spot_mar_jul" / "normalized"
            / "cohort_btc_usdt_spot" / "source_binance" / "interval_5m"
            / "candles.csv")
    daily: dict[str, float] = {}
    prev_close: float | None = None
    with path.open() as handle:
        header = handle.readline().rstrip("\n").split(",")
        time_i, close_i = header.index("open_time"), header.index("close")
        for line in handle:
            cells = line.rstrip("\n").split(",")
            close = float(cells[close_i])
            if prev_close is not None:
                day = cells[time_i][:10]
                ret = math.log(close / prev_close)
                daily[day] = daily.get(day, 0.0) + ret * ret
            prev_close = close
    day, variance = max(daily.items(), key=lambda kv: kv[1])
    return day, round(math.sqrt(variance), 6)


def stress_traces() -> dict[str, tuple[list[dict[str, Any]], str]]:
    """Eight authored tail scenarios: name -> (events, note)."""
    def at(spec: str) -> datetime:
        return datetime.fromisoformat(spec).replace(tzinfo=UTC)

    vol_day, vol = highest_vol_day()
    vol_start = datetime.fromisoformat(vol_day).replace(tzinfo=UTC)
    traces: dict[str, tuple[list[dict[str, Any]], str]] = {}

    traces["stress_stale_200"] = ([event(
        "P", "binance", "stale_200", at("2026-05-14T08:00"),
        at("2026-05-14T14:00"), {"note": "candles freeze, HTTP 200"},
        "authored")], "one venue's candles freeze 6h while returning 200")
    traces["stress_wrong_scale"] = ([event(
        "P", "okx", "wrong_data", at("2026-04-22T11:00"),
        at("2026-04-22T11:30"), {"price_multiplier": 100}, "authored")],
        "prices x100 for 30 min (decimal/symbol bug)")
    cdn_start, cdn_end = at("2026-06-18T13:00"), at("2026-06-18T14:30")
    traces["stress_correlated_cdn"] = ([
        event("P", venue, "outage", cdn_start, cdn_end,
              {"behavior": "timeout", "correlated_group": "cdn"}, "authored")
        for venue in ("binance", "okx")],
        "Binance+OKX simultaneously unreachable 90 min (Cloudflare-2022 shape)")
    traces["stress_schema_change"] = ([event(
        "P", "kucoin", "schema_change", at("2026-05-01T00:00"), WINDOW_END,
        {"field_renames": {"data": "items"}}, "authored")],
        "response envelope field renamed, permanent from start date")
    traces["stress_geoblock_flip"] = ([event(
        "E", "kucoin", "geoblock", at("2026-06-10T00:00"), WINDOW_END,
        {"status": 403}, "authored")],
        "403 from day X, never recovers within the window")
    traces["stress_maintenance_in_vol"] = ([event(
        "P", "binance", "outage", vol_start, vol_start + timedelta(days=1),
        {"behavior": "http_503", "chosen_day": vol_day,
         "realized_vol_5m": vol}, "authored")],
        f"median-defining venue out on the max-realized-vol day ({vol_day})")
    traces["stress_slow_bleed"] = ([event(
        "P", "coinbase", "slow_bleed", at("2026-07-03T00:00"),
        at("2026-07-04T00:00"), {"latency_ramp": [1, 30]}, "authored")],
        "latency ramps 1x->30x over 24h, then recovers")
    traces["stress_ban_trap"] = ([event(
        "M", "binance", "capacity_modulation", at("2026-04-09T14:00"),
        at("2026-04-09T18:00"), {"budget_factor": 0.02}, "authored")],
        "budget ~0; retry-without-backoff walks into the 418 escalating ban")
    return {name: ([e for e in evts if e], note)
            for name, (evts, note) in traces.items()}


# --- output ----------------------------------------------------------------

def write_trace(path: Path, name: str, events: list[dict[str, Any]],
                header_extra: dict[str, Any]) -> dict[str, Any]:
    events = sorted(events, key=lambda e: (e["start"], e["venue"], e["mode"]))
    for index, evt in enumerate(events, start=1):
        evt["id"] = f"evt-{index:03d}"
    header = {"trace": name, "generator_version": GENERATOR_VERSION,
              "window": [iso(WINDOW_START), iso(WINDOW_END)], **header_extra}
    lines = [json.dumps(header, separators=(", ", ": "))]
    lines += [json.dumps({"id": e["id"], "layer": e["layer"],
                          "venue": e["venue"], "mode": e["mode"],
                          "start": e["start"], "end": e["end"],
                          "params": e["params"],
                          "provenance": e["provenance"]},
                         separators=(", ", ": "))
              for e in events]
    text = "\n".join(lines) + "\n"
    path.write_text(text)
    return {"sha256": hashlib.sha256(text.encode()).hexdigest(),
            "events": len(events)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS)
    parser.add_argument("--calibration-dir", type=Path,
                        default=DEFAULT_CALIBRATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    params_text = args.params.read_text()
    params_sha = hashlib.sha256(params_text.encode()).hexdigest()
    venues = json.loads(params_text)["venues"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fitted_header = {
        "params_sha256": params_sha,
        "rate_multiplier": 1.0, "duration_multiplier": 1.0,
        "correlation": False,
        "duration_cap_minutes": DURATION_CAP_MINUTES,
        "authored_constants": {"edge": EDGE,
                               "capacity_modulation": CAPACITY_MODULATION,
                               "correlated": CORRELATED},
    }
    manifest: dict[str, Any] = {}

    manifest["zero"] = write_trace(
        args.output_dir / "zero.jsonl", "zero", [],
        {"note": "no incidents; measures over-defensiveness cost"})

    replayed, skipped = replay_events(args.calibration_dir)
    manifest["real_replay"] = write_trace(
        args.output_dir / "real_replay.jsonl", "real_replay", replayed,
        {"provenance": "replayed", "skipped": skipped,
         "note": ("archived incidents overlapping the window on their true "
                  "timestamps; binance/kucoin/defillama have no archive so "
                  "carry no events here")})

    for seed in DEV_SEEDS + HELD_SEEDS:
        split = "dev" if seed in DEV_SEEDS else "held"
        name = f"fitted_{split}_s{seed:03d}"
        events = sample_trace(seed, venues, 1.0, 1.0, correlation=False)
        manifest[name] = write_trace(args.output_dir / f"{name}.jsonl", name,
                                     events, {"seed": seed, **fitted_header})
        manifest[name]["seed"] = seed

    grid_index = 0
    for r_key, r_mult in RATE_MULTS.items():
        for d_key, d_mult in DUR_MULTS.items():
            for c_key, corr in (("coff", False), ("con", True)):
                name = f"grid_r{r_key}_d{d_key}_{c_key}"
                seed = GRID_SEED_BASE + grid_index
                grid_index += 1
                events = sample_trace(seed, venues, r_mult, d_mult, corr)
                header = {**fitted_header, "seed": seed,
                          "rate_multiplier": r_mult,
                          "duration_multiplier": d_mult, "correlation": corr}
                manifest[name] = write_trace(
                    args.output_dir / f"{name}.jsonl", name, events, header)
                manifest[name]["seed"] = seed

    for name, (events, note) in stress_traces().items():
        manifest[name] = write_trace(args.output_dir / f"{name}.jsonl", name,
                                     events,
                                     {"provenance": "authored", "note": note})

    manifest_doc = {"schema_version": 1,
                    "generator_version": GENERATOR_VERSION,
                    "params_sha256": params_sha,
                    "window": [iso(WINDOW_START), iso(WINDOW_END)],
                    "traces": manifest}
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest_doc, indent=1) + "\n")

    total = sum(t["events"] for t in manifest.values())
    print(f"wrote {len(manifest)} traces, {total} events total "
          f"-> {args.output_dir}")
    for name in ("real_replay", "fitted_dev_s001", "grid_r4_d4_con"):
        print(f"  {name}: {manifest[name]['events']} events")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
