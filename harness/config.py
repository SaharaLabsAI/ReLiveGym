"""Run configuration , loaded from YAML into pydantic models.

Harness-level config only. Task-specific parameters (thresholds, data files,
the task's price table) live in the `task:` section, which the harness treats
as an opaque dict validated by the task's own config model
(tasks/<name>/task.py, see harness/task.py).
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

import yaml
from pydantic import BaseModel, field_validator, model_validator

from harness.timeutil import as_utc


class CostConfig(BaseModel):
    """Realistic LLM billing only: no
    flat per-call charge. `llm_token_rates` is a per-run OVERRIDE of the
    repo-global table (configs/model_costs.yaml via harness/model_costs)
    in $/token: model -> {"in", "out", optional "cached_in"}. Models in
    neither table are refused by the proxy — nothing bills at a provider
    list price."""

    llm_token_rates: dict[str, dict[str, float]] = {}


# Registered learning-trigger levels (the tlrn axis).
# The single place tlrn parameters exist: the composer compiles program
# text from the resolved entry, provisioning (harness/apps.py) reads the
# source, and the mirror row lives in the parameters appendix. Adding a
# cadence is one line here — renderers read {source, params}, never the
# tag. Future levels: {"source": "agent"} (actor-owned tools),
# {"source": "turns", "n": N} (workspace-event, A/B only).
TLRN_LEVELS: dict[str, dict | None] = {
    "none": None,                                     # no learning trigger
    "daily": {"source": "sim", "cron": "30 0 * * *"},     # the default
}


class CellSpec(BaseModel):
    """One cell of the TM x TLRN x Sig x Alg design matrix (TM owns acting
    timing, tlrn owns learning timing). Provisioning —
    which routes are mounted, which scaffold layers are copied, which
    tools exist — derives from this spec alone."""

    tm: Literal["A", "B", "C", "D"] = "C"  # D = TM-C + agent-managed
    #   schedules
    tlrn: str = "none"
    sig: Literal["none", "oracle", "self"] = "none"
    alg: Literal["none", "memory", "skills", "vskills", "vskills2",
                 "vskills3", "config", "full"] = "none"  # vskills = skills +
    #   replay-verified adoption;
    #   vskills2 = trajectory-aware curation of a per-wake REMINDER;
    #   vskills3 = vskills2 verified on day ROLLOUTS under a virtual clock

    @model_validator(mode="after")
    def _consistent(self) -> "CellSpec":
        if self.tlrn not in TLRN_LEVELS:
            raise ValueError(
                f"cell: unregistered tlrn level {self.tlrn!r} "
                f"(registered: {sorted(TLRN_LEVELS)})")
        if (self.sig == "none") != (self.alg == "none"):
            raise ValueError(
                "cell: sig and alg must be 'none' together (Sig-A is the "
                "no-learning wiring; every signal cell pairs with an alg)")
        if (self.tlrn == "none") != (self.alg == "none"):
            raise ValueError(
                "cell: tlrn and alg must be 'none' together (a learner "
                "that never triggers and a trigger with no learner are "
                "both dead code, not treatments)")
        if self.alg in ("vskills", "vskills2", "vskills3") and self.sig != "oracle":
            raise ValueError(
                f"cell: alg={self.alg!r} requires sig='oracle' (the replay "
                "scorer needs settled outcomes in the feed)")
        if self.tm != "C" and self.alg in ("config", "full"):
            raise ValueError(
                f"cell: alg={self.alg!r} requires tm=C (ReACT arms have no "
                f"deterministic trigger config / editable program loop)")
        return self

    @property
    def tlrn_spec(self) -> dict | None:
        """The resolved {source, params} of this cell's tlrn level."""
        return TLRN_LEVELS[self.tlrn]

    @property
    def label(self) -> str:
        return f"tm{self.tm}-tlrn{self.tlrn}-sig{self.sig}-alg{self.alg}"


class AgentSpec(BaseModel):
    scaffold: str  # directory name under tasks/<task>/scaffolds/
    model: str | None = None  # bare model name, or 'api_provider:model'
    #   (normalized below). Usually unset in yamls and supplied at launch
    #   (--model); the run dir then carries the model name.
    provider: Literal["openai", "openrouter"] = "openai"  # api route the
    #   proxy forwards through — server-side only, never agent-visible
    prompt_variant: str | None = None
    seed_skills: str | None = None  # EXPLORATORY (wait grammar v2 plan):
    #   file under tasks/<task>/agent/ copied to memory/skills.md at
    #   workspace build — an expert-curated initial skill block. Requires
    #   a cell whose learned block renders skills.md (alg skills/config/
    #   full); reflection may overwrite it like any learned block.
    seed_monitor_plans: str | None = None  # EXPLORATORY: task-agent JSON
    #   copied to memory/seed_monitor_plans.json. Task-authored scaffolds may
    #   validate and install it; generic scaffolds do not interpret it.
    replay_max_iter: int = 5  # alg=vskills: proposer attempts per learn
    #   firing
    replay_max_items: int = 40  # alg=vskills: the replay set is the most
    #   recent N settled decision points (older ones are dropped and the
    #   drop is logged on the vskills_start row)
    rollout_days: int = 1  # alg=vskills3: settled day segments rolled out
    #   per candidate (the newest; each starts at the actor's earliest
    #   live wake of the day) — the v3 meaning of replay_max_items
    # reflection render caps (scaffolds/runtime/reflect.py; defaults = the
    # values every cell has run with, so unchanged yamls render identically)
    reflect_own_history_cap: int = 30  # newest own-action records shown
    reflect_digest_cap: int = 40  # newest non-own outcomes in the digest
    reflect_examples_per_stratum: int = 4  # sampled exemplars per status
    replay_max_workers: int = 1  # alg=vskills: decision points replayed
    #   concurrently per block (episodes are independent; 1 = sequential)
    reflect_max_edits: int = 3  # alg=vskills: edits accepted per reply
    reflect_max_edit_words: int = 100  # alg=vskills: new words per reply
    reflect_diff_rows: int = 60  # alg=vskills: diff rows per candidate
    reflect_transcript_tokens: int = 60_000  # alg=vskills: the curation
    #   transcript budget (oldest learning cycles dropped first);
    #   alg=vskills2 never compacts (no context limit)
    reflect_trajectory_wakes: int = 6  # alg=vskills2: newest wakes shown
    #   verbatim to the curator (the first wake always is)
    reflect_trajectory_tokens: int = 20_000  # alg=vskills2: trajectory
    #   render budget (digest lines never dropped before verbatim wakes)
    reflect_trajectory_result_chars: int = 300  # alg=vskills2: tool
    #   result clip inside verbatim wakes
    block_tokens: int = 5000  # learned-block budget
    context_tokens: int = 60_000  # per-agent transcript truncation budget
    #   (scaffolds/runtime/agent.py compactor). Substrate constant, pinned
    #   across cells within an experiment; override only in dedicated
    #   context-size probes (wgv3 ctx20k). The default leaves generated
    #   react programs byte-identical to pre-knob runs.

    @model_validator(mode="after")
    def _split_model_spec(self) -> "AgentSpec":
        if self.model and ":" in self.model:
            from harness.model_costs import parse_model_spec

            self.provider, self.model = parse_model_spec(self.model)
        return self


class RunConfig(BaseModel):
    run_id: str
    task: dict = {"name": "weather_fixture"}  # {"name": ..., **task params}
    sim_start: datetime
    sim_end: datetime
    cell: CellSpec = CellSpec()  # design-matrix cell; default = TM-C Sig-A
    window_id: str | None = None  # Stage-1 window bookkeeping (run manifest)
    budget_usd: float = 50.0  # the run's ONE wallet (real USD): LLM tokens
    #   at real rates + API fees at real rates. Advertised in
    #   INSTRUCTION.md; when spent, /llm returns 503 and paid tools 402
    #   (flag budget_exhausted) — the run keeps going and performance
    #   degrades naturally (constrained-optimization contract).
    domain_budgets: dict[str, float] = {"llm": 20.0}  # optional per-domain
    #   caps INSIDE the global wallet, keyed by ledger spend type
    #   ("llm", "news_search", ...). The llm cap is the experimenter's
    #   REAL provider bill — simulated fees only draw
    #   on the global wallet. Hitting a cap refuses that domain's paid
    #   calls (503 for llm, 402 for tools; flag <domain>_budget_exhausted)
    #   while other domains keep going.
    log_llm_traffic: bool = True  # persist full LLM request/response bodies
    #                               to runs/<id>/llm_log.jsonl (replay/analysis)
    watchdog_seconds: float = 120.0  # real-time inactivity limit
    llm_timeout_seconds: float = 300.0  # real-time ceiling on ONE ATTEMPT of an
    #   upstream LLM call (a hung provider request would otherwise freeze a
    #   resident run with no error anywhere; litellm's own default is 6000 s).
    llm_retries: int = 2  # transient failures (timeout, connection, 408/409/
    #   429/5xx) are retried this many times server-side before /llm answers
    #   504 (all attempts timed out) or 502 — ledger llm_retry per recovered
    #   attempt, llm_error + flag llm_errors on the final failure; nothing is
    #   billed for failed attempts. Same-day addition: a 504 had crashed
    #   4 of 12 evaluation runs for a sim-day each.
    llm_retry_backoff_seconds: float = 1.0  # real-time wait before retry k:
    #   backoff * 2**k. The watchdog is suspended while a call is in flight;
    #   the runner kills an invocation whose oldest in-flight call exceeds
    #   the whole allowance (attempts x timeout + backoffs) + watchdog.
    max_consecutive_crashes: int = 5  # K: then the run is marked failed-degenerate
    cost: CostConfig = CostConfig()
    agent: AgentSpec
    seed: int = 0

    @field_validator("sim_start", "sim_end")
    @classmethod
    def _utc(cls, v: datetime) -> datetime:
        return as_utc(v)

    @model_validator(mode="after")
    def _check(self) -> "RunConfig":
        if self.sim_end <= self.sim_start:
            raise ValueError("sim_end must be after sim_start")
        if not self.task.get("name"):
            raise ValueError("task section needs a 'name'")
        return self

    @property
    def task_name(self) -> str:
        return self.task["name"]

    @property
    def task_params(self) -> dict:
        return {k: v for k, v in self.task.items() if k != "name"}


def load_config(path) -> RunConfig:
    with open(path) as f:
        return RunConfig(**yaml.safe_load(f))
