"""Pause, checkpoint and resume of one sim.

A run served with a pause instant stops its clock there (harness/clock.py
clamp): the actor sees `experiment_over`, exactly as at the real end, and
the server writes a CHECKPOINT instead of results —

    checkpoint.json        {run_id, seed, paused_at, code_sha, ledger_rows, spent_usd, flags, parent}
    results_partial.json   sim.results() as of the pause: nothing force-settled,
                           pending claims still pending, closed breakpoints only

— on top of what every run dir has (config.json, ledger.jsonl,
schedule.json, run.json). A later run RESUMES from that directory:
`prepare_resume` copies the parent's ledger and schedule store into the
new (empty) run dir before the Sim is built, `restore` rebuilds the Sim on
them — ledger preloaded, scoring state replayed through Task.restore,
wallet from the cost column, clock at the pause instant, the resume
trigger for resident disciplines — and appends one `resume` row. The
new run's ledger is therefore the whole chain, and its results.json (or
next checkpoint) covers the episode from sim_start.

Why exact: the scorer is a function of (its data, the notify sequence,
the clock); the wallet is a sum over the ledger; the schedule store is
persisted on every mutation; the wait party is per process and the
program rebuilds it. Rate-limiter windows are the one thing NOT restored
(every bnpm cell runs `window: none`); a limited cell that is resumed
starts its windows empty — say so, do not silently re-count.

The actor side is a plain copy of the snapshot the experimenter (or an
improver) leaves; nothing here touches it.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

from harness.config import RunConfig
from harness.schedule import ScheduleStore
from harness.task import NotificationError
from harness.timeutil import as_utc, iso, parse_iso

CHECKPOINT = "checkpoint.json"
PARTIAL = "results_partial.json"
RESUME = "resume.json"
RESUME_TRIGGER = "__resume__"  # the one-shot that restarts a resident program at the cut
RESIDENT_TMS = ("A", "B")  # disciplines whose program stays resident and must be restarted
COMPARE_KEYS = ("task", "sim_start", "sim_end", "budget_usd", "domain_budgets",
                "cell", "cost")  # what a resumed config must share with its parent


def parse_pause_at(cfg: RunConfig, value) -> datetime | None:
    """A `--pause-at` argument: None (no pause), or an instant strictly
    inside the run window."""
    if value is None or value == "":
        return None
    t = as_utc(value) if isinstance(value, datetime) else parse_iso(str(value))
    if not (cfg.sim_start < t < cfg.sim_end):
        raise ValueError(f"pause_at {iso(t)} must lie strictly inside the run "
                         f"window {iso(cfg.sim_start)} .. {iso(cfg.sim_end)}")
    return t


# -- writing a checkpoint -----------------------------------------------------------------


def write_checkpoint(sim, scheduler, parent: Path | None = None) -> dict:
    """The paused server's artefacts (caller: serve/run after the scheduler
    reports paused). Returns the checkpoint dict."""
    run_dir = sim.run_dir
    partial = sim.results()
    partial["paused_at"] = iso(sim.clock.now)
    (run_dir / PARTIAL).write_text(json.dumps(partial, indent=1))
    ck = {"run_id": sim.cfg.run_id, "seed": sim.cfg.seed,
          "paused_at": iso(sim.clock.now),
          "sim_start": iso(sim.cfg.sim_start), "sim_end": iso(sim.cfg.sim_end),
          "code_sha": getattr(scheduler, "_last_code_hash", None),
          "ledger_rows": len(sim.ledger.events),
          "spent_usd": round(sim.ledger.total_cost(), 8),
          "flags": list(sim.flags),
          "parent": str(parent) if parent else None,
          "written_at": datetime.now().astimezone().isoformat(timespec="seconds")}
    (run_dir / CHECKPOINT).write_text(json.dumps(ck, indent=1))
    return ck


# -- reading one ---------------------------------------------------------------------------


def checkpoint_of(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    p = run_dir / CHECKPOINT
    if not p.is_file():
        raise FileNotFoundError(f"{run_dir} is not a paused run (no {CHECKPOINT})")
    ck = json.loads(p.read_text())
    for f in ("ledger.jsonl", "config.json"):
        if not (run_dir / f).is_file():
            raise FileNotFoundError(f"{run_dir}: checkpoint without {f}")
    return ck


def config_mismatch(parent_cfg: dict, cfg: RunConfig) -> list[str]:
    """Keys on which a resumed run's config differs from its parent's
    (seed, run_id and the model name may differ: every seed resumes from
    the main branch's checkpoint; the provider/model must not)."""
    ours = cfg.model_dump(mode="json")
    bad = [k for k in COMPARE_KEYS
           if json.dumps(ours.get(k), sort_keys=True, default=str)
           != json.dumps(parent_cfg.get(k), sort_keys=True, default=str)]
    pa, oa = parent_cfg.get("agent") or {}, ours.get("agent") or {}
    for k in ("model", "provider", "scaffold"):
        if pa.get(k) != oa.get(k):
            bad.append(f"agent.{k}")
    return bad


# -- resuming ------------------------------------------------------------------------------


def prepare_resume(run_dir: Path, parent: Path, cfg: RunConfig) -> dict:
    """Before the Sim is built on `run_dir` (empty or absent): validate the
    parent checkpoint against `cfg`, copy its ledger and schedule store in.
    Returns the checkpoint dict."""
    parent = Path(parent).resolve()
    ck = checkpoint_of(parent)
    if "failed-degenerate" in (ck.get("flags") or []):
        raise ValueError(f"cannot resume {parent.name}: the run is failed-degenerate")
    parent_cfg = json.loads((parent / "config.json").read_text())
    bad = config_mismatch(parent_cfg, cfg)
    if bad:
        raise ValueError(f"cannot resume {parent.name}: config differs on {bad}")
    paused_at = parse_iso(ck["paused_at"])
    if not (cfg.sim_start < paused_at < cfg.sim_end):
        raise ValueError(f"checkpoint instant {ck['paused_at']} outside the window")
    run_dir = Path(run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(parent / "ledger.jsonl", run_dir / "ledger.jsonl")
    if (parent / "schedule.json").is_file():
        shutil.copyfile(parent / "schedule.json", run_dir / "schedule.json")
    (run_dir / RESUME).write_text(json.dumps(
        {"parent": str(parent), "paused_at": ck["paused_at"],
         "parent_run_id": ck.get("run_id"), "parent_seed": ck.get("seed"),
         "run_id": cfg.run_id, "seed": cfg.seed}, indent=1))
    return ck


def restore(sim, scheduler, parent: Path, ck: dict) -> dict:
    """After the Sim is built on a prepared run dir: rebuild it as the
    continuation of `parent` at the checkpoint instant. Returns a summary
    (also the fields of the `resume` ledger row)."""
    parent = Path(parent).resolve()
    paused_at = parse_iso(ck["paused_at"])
    n_rows = sim.ledger.load_existing()
    events = sim.ledger.events
    if n_rows != int(ck.get("ledger_rows") or n_rows):
        raise ValueError(f"cannot resume {parent.name}: ledger has {n_rows} rows, "
                         f"checkpoint says {ck.get('ledger_rows')}")
    # scoring state: replay the accepted notifications through the task
    try:
        n_notify = sim.task.restore(events, paused_at)
    except NotificationError as e:
        raise ValueError(f"cannot resume {parent.name}: the ledger contradicts "
                         f"the task on replay ({e})") from None
    # wallet: sums over the copied rows
    by_type = sim.ledger.cost_by_type()
    sim.wallet_spend = sim.ledger.total_cost()
    sim.domain_spend = {k: v for k, v in by_type.items() if v}
    sim.llm_spend = by_type.get("llm", 0.0)
    sim.llm_provider_spend = round(sum(float(e.get("cost_provider") or 0.0)
                                       for e in events if e.get("type") == "llm"), 8)
    flags = []
    if sim.wallet_spend >= sim.cfg.budget_usd:
        flags.append("budget_exhausted")
    llm_cap = sim.cfg.domain_budgets.get("llm")
    if llm_cap is not None and sim.llm_spend >= llm_cap:
        flags.append("llm_budget_exhausted")
    for t, cap in sim.cfg.domain_budgets.items():
        if t != "llm" and sim.domain_spend.get(t, 0.0) >= cap:
            flags.append(f"{t}_budget_exhausted")
    if any(e.get("type") == "llm_error" for e in events):
        flags.append("llm_errors")
    if any(e.get("type") == "llm_rejected" and e.get("reason") == "model not in cost config"
           for e in events):
        flags.append("llm_model_unconfigured")
    sim.flags = flags
    # schedule store: the parent's, persisted on every mutation
    sim.schedule = ScheduleStore.load(sim.run_dir / "schedule.json")
    # clock: at the cut; the daily brief state stays fresh, so the first
    # wake at the cut delivers that date's brief as a midnight wake would
    sim.clock.advance_to(paused_at)
    if sim.cfg.cell.tm in RESIDENT_TMS:
        # a resident program must be restarted AT the cut: without this
        # the next trigger is the fallback midnight a day later. Cron
        # disciplines resume from their restored crontab (the cron mains
        # run every market on an unknown trigger id: no one-shot for them)
        sim.schedule.run_at(RESUME_TRIGGER, paused_at)
    scheduler.seed_code_hash(ck.get("code_sha"))
    summary = {"parent": str(parent), "parent_run_id": ck.get("run_id"),
               "paused_at": ck["paused_at"], "seed": sim.cfg.seed,
               "ledger_rows": n_rows, "notifications_replayed": n_notify,
               "spent_usd": round(sim.wallet_spend, 8),
               "resume_trigger": sim.cfg.cell.tm in RESIDENT_TMS}
    sim.ledger.append("resume", paused_at, **summary)
    return summary
