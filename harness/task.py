"""Task interface: everything task-specific lives behind one Task object.

A task is a self-contained directory `tasks/<name>/` providing:

  task.py         module exposing TASK, a Task subclass (this interface)
  INSTRUCTION.md  agent-facing task spec `u`; a string.Template rendered into
                  the workspace at run init (`${...}` placeholders come from
                  Task.instruction_context())
  README.md       experimenter-facing documentation
  data/           replayable historical data + provenance README
  env/apps.py     the task's server-side EnvApps (observation + action tools)
  agent/          agent-side task material the constructor provisions into
                  workspaces (handlers, sig_self/, records.py, baselines/)
  configs/        run YAMLs for this task

Division of labor: the harness (harness/*) owns the clock, schedule,
ledger, wallet + LLM metering, rate-limiter registry, the agent
supervisor, and the harness-side tools (get_time, crontab, sleep,
get_feedback, /llm). The task owns its observation tools, its action
tool(s), the ground truth, and the scoring — declared once as EnvApps and
served through the manifest.

Objective contract: every task is a
constrained optimization — maximize/minimize the task's primary metric
over the run window, subject to the run's money budget (`budget_usd`,
enforced by the wallet) and the task's rate limits (enforced via
sim.limiters). Outcomes are settled as cost-free OutcomeEvents; there are
no penalty dollars anywhere.
"""

from __future__ import annotations

import abc
import importlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from harness.timeutil import parse_iso

if TYPE_CHECKING:
    from harness.config import RunConfig
    from harness.runtime import Sim


class NotificationError(ValueError):
    """Rejected notification (invalid payload/target). Maps to HTTP 400 in the
    API layer — rejection is the error-handling path and costs nothing."""


@dataclass
class OutcomeEvent:
    """One settled outcome to book into the ledger (cost-free)."""

    ref: str  # what settled, e.g. the calendar day or the post id
    status: str  # e.g. ok | miss | false_alarm | tail | nontail
    detail: dict = field(default_factory=dict)


class Task(abc.ABC):
    """One task instance, bound to a run's config and data."""

    name: str  # must match the tasks/<name>/ directory

    # The cron arms' cadence when the task is acted as an episode
    # (harness.mcp prepare --tm C|D; the checked-in cron mains carry their own
    # ACT_CRON): the fixed schedule under C, the default schedule under D.
    # A checked-in constant, not a launch flag — it is the instrument.
    episode_act_cron: str | None = None

    # -- construction ---------------------------------------------------------------

    @classmethod
    @abc.abstractmethod
    def from_run_config(cls, cfg: RunConfig, repo_root: Path) -> "Task":
        """Validate cfg.task with the task's own config model and build the
        data store / scorer. Raise ValueError on inconsistent config."""

    # -- environment API -------------------------------------------------------------

    @abc.abstractmethod
    def env_apps(self, sim: Sim) -> list:
        """The task's EnvApps (observation + action tools; see
        harness/env_tools.py). Handlers must take sim.lock, book real-rate
        fees via sim.bill, consume the task's registered rate limiters,
        and never leak ground truth. Provisioning decisions (e.g.
        oracle-only tools) key on sim.cfg, never on agent code."""

    @abc.abstractmethod
    def record_notification(self, sim_time: datetime, payload: dict) -> None:
        """Validate and record one action payload (task-defined shape).
        Raise NotificationError to reject it free of charge."""

    # -- scoring ---------------------------------------------------------------------

    @abc.abstractmethod
    def close_due(self, now: datetime) -> list[OutcomeEvent]:
        """Settle every scoring period whose close time has passed."""

    @abc.abstractmethod
    def close_all(self) -> list[OutcomeEvent]:
        """End of run: settle everything still open."""

    @abc.abstractmethod
    def oracle_outcomes(self, since: datetime | None, now: datetime) -> list[dict]:
        """Sig-B payload (/oracle/outcomes): settled outcome records with
        t_settled in (since, now], oldest first. Universal ground truth for
        everything already settled — own actions and gold events — but must
        never reveal anything for periods still open at `now`."""

    @abc.abstractmethod
    def metrics(self) -> dict:
        """results.performance: regular ML metrics over everything settled
        so far. MUST contain "primary": {"name": str, "value": float|None,
        "direction": "max"|"min"} — the one number cells are ranked by."""

    def restore(self, events: list[dict], now: datetime) -> int:
        """Rebuild the scoring state from a ledger — a checkpoint resume
        (harness/checkpoint.py). Replays every accepted `notify` row in order, settling what
        was due before each, then settles up to `now`; the outcome events
        the replay regenerates are discarded (the copied ledger already
        holds them). The default suits every event-sourced task — its
        state is a function of the notifications and the clock; a task
        with state outside them overrides. Returns the number of
        notifications replayed; raises NotificationError when the ledger
        contradicts this task (a rejected replay = an inconsistent chain)."""
        n = 0
        for e in events:
            if e.get("type") != "notify":
                continue
            t = parse_iso(e["sim_time"])
            self.close_due(t)
            self.record_notification(t, dict(e.get("payload") or {}))
            n += 1
        self.close_due(now)
        return n

    def constraint_violations(self) -> list[str]:
        """Task-level behavioral-constraint violations (e.g. an abstention
        budget overrun). Resource constraints (money budget, rate limits)
        are harness-owned; default: none."""
        return []

    @abc.abstractmethod
    def report(self) -> dict:
        """Task section of results.json (full outcome tables, per-period
        breakdowns, ...)."""

    # -- authored wait programs (TM-B) -------------------------------------------------

    def authored_example(self) -> str | None:
        """example_gatekeeper.py content shipped into a TM-B actor's
        authored-program workspace (harness/authored.py) — a runnable
        template the actor copies instead of cold-starting from a blank
        file. None ships no example."""
        return None

    # -- binding ---------------------------------------------------------------------

    def bind(self, sim: Sim) -> None:
        """Called once by Sim.__init__: the task may keep the sim for
        ledger-backed grading rules. Default: keep a reference."""
        self._sim = sim

    # -- web hosts -------------------

    def web_apps(self, sim: Sim) -> list:
        """The task's browser-facing hosts: [harness.web.WebHostSpec]. One
        loopback listener per host, URLs in sim.hosts and as the
        `${<name>_url}` INSTRUCTION placeholders. Read-only hosts serve
        replay data (a watcher program may poll them); writable hosts
        carry sessions and forms and are the agent's only write path.
        Default: none."""
        return []

    # -- instruction -----------------------------------------------------------------

    @abc.abstractmethod
    def instruction_context(self) -> dict[str, object]:
        """Values for the ${...} placeholders in INSTRUCTION.md. The harness
        adds its own (budget_usd, llm_price_table)."""


def load_task_class(name: str) -> type[Task]:
    """Import tasks/<name>/task.py and return its TASK class."""
    try:
        module = importlib.import_module(f"tasks.{name}.task")
    except ModuleNotFoundError as e:
        raise ValueError(f"unknown task {name!r}: {e}") from None
    try:
        return module.TASK
    except AttributeError:
        raise ValueError(f"tasks/{name}/task.py defines no TASK class") from None


def task_dir(name: str) -> Path:
    """Directory of tasks/<name>/ (anchored on the module, not the cwd)."""
    module = importlib.import_module(f"tasks.{name}.task")
    return Path(module.__file__).resolve().parent
