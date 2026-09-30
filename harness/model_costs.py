"""The repo-global LLM cost table and the launch-time model spec.

configs/model_costs.yaml pins $/Mtok rates per bare model name; booked
LLM cost is a pure function of that file plus any per-run override
(cfg.cost.llm_token_rates), never of a provider price table. A model
with no entry is refused — at launch (apply_launch_model) and per call
(harness/llm_proxy.py).

A launch-time model is written `api_provider:model_name`
(openai:gpt-5.6-luna, openrouter:deepseek/deepseek-v4-pro). The api
provider stays server-side: agents, the contract and the cost table see
only the bare model name, so the same model reads identically across
providers and provider choice is never a treatment.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
COSTS_PATH = REPO_ROOT / "configs" / "model_costs.yaml"
LIMITS_PATH = REPO_ROOT / "configs" / "model_limits.yaml"

# api providers the proxy can route to, and the .env key each one needs
PROVIDER_KEYS = {"openai": "OPENAI_API_KEY",
                 "openrouter": "OPENROUTER_API_KEY"}

_RATE_KEYS = {"in", "out", "cached_in"}
_cache: dict[str, dict[str, float]] | None = None


def parse_costs_file(path: Path) -> dict[str, dict[str, float]]:
    """model -> {"in", "out", "cached_in"?} in $/token (the file is $/Mtok)."""
    raw = (yaml.safe_load(path.read_text()) or {}).get("models") or {}
    table: dict[str, dict[str, float]] = {}
    for model, rates in raw.items():
        unknown = set(rates) - _RATE_KEYS
        if unknown or not {"in", "out"} <= set(rates):
            raise ValueError(
                f"{path}: entry {model!r} must have keys 'in' and 'out' "
                f"(optional 'cached_in'), got {sorted(rates)}")
        table[model] = {k: float(v) / 1e6 for k, v in rates.items()}
    return table


def load_model_costs() -> dict[str, dict[str, float]]:
    global _cache
    if _cache is None:
        _cache = parse_costs_file(COSTS_PATH)
    return _cache


_LIMIT_KEYS = {"context", "output"}
_limits_cache: dict[str, dict[str, int]] | None = None


def parse_limits_file(path: Path) -> dict[str, dict[str, int]]:
    """model -> {"context", "output"} in tokens (configs/model_limits.yaml)."""
    raw = (yaml.safe_load(path.read_text()) or {}).get("models") or {}
    table: dict[str, dict[str, int]] = {}
    for model, lim in raw.items():
        if set(lim) != _LIMIT_KEYS or any(int(v) <= 0 for v in lim.values()):
            raise ValueError(
                f"{path}: entry {model!r} must have positive 'context' and "
                f"'output', got {lim}")
        table[model] = {k: int(v) for k, v in lim.items()}
    return table


def load_model_limits() -> dict[str, dict[str, int]]:
    global _limits_cache
    if _limits_cache is None:
        _limits_cache = parse_limits_file(LIMITS_PATH)
    return _limits_cache


def resolve_limits(model: str) -> dict[str, int] | None:
    """Context/output token limits for `model`, or None when the table
    has no entry. OpenCode's auto-compaction needs `limit.context`; an
    episode with a model missing here is refused at prepare."""
    return load_model_limits().get(model)


def resolve_rates(cfg, model: str) -> dict[str, float] | None:
    """$/token rates for `model`: the run's own llm_token_rates entry
    wins over the global table; None means the model is unpriced (and
    every unpriced call is refused)."""
    rates = cfg.cost.llm_token_rates.get(model)
    if rates is not None:
        return rates
    return load_model_costs().get(model)


def parse_model_spec(spec: str) -> tuple[str, str]:
    """'api_provider:model_name' -> (provider, bare model). A bare spec
    (no colon) defaults to openai."""
    provider, sep, model = spec.partition(":")
    if not sep:
        return "openai", spec
    if provider not in PROVIDER_KEYS or not model:
        raise ValueError(
            f"bad model spec {spec!r}: want api_provider:model_name with "
            f"api_provider in {sorted(PROVIDER_KEYS)}")
    return provider, model


def model_slug(model: str) -> str:
    """The model's run-dir form: path separators flattened."""
    return model.replace("/", "-")


def apply_launch_model(cfg, model_spec: str | None) -> None:
    """Launch-time model resolution, shared by every entrypoint: a
    --model spec overrides cfg.agent.model; a set model must be priced
    and its provider key present (call after dotenv.load_dotenv), and
    the model name lands in run_id — hence the run dir — ahead of any
    -s<seed> suffix. A run with no model launches (silent baselines);
    the proxy refuses any LLM call it would make."""
    if model_spec:
        cfg.agent.provider, cfg.agent.model = parse_model_spec(model_spec)
    if cfg.agent.model is None:
        return
    if resolve_rates(cfg, cfg.agent.model) is None:
        raise SystemExit(
            f"model {cfg.agent.model!r} has no entry in {COSTS_PATH} "
            "(and no per-run llm_token_rates override) — add its rates "
            "before launching")
    key = PROVIDER_KEYS[cfg.agent.provider]
    if not os.environ.get(key):
        raise SystemExit(
            f"provider {cfg.agent.provider!r} needs {key} in the "
            "environment (.env) — the run's LLM calls cannot be forwarded")
    cfg.run_id = f"{cfg.run_id}-{model_slug(cfg.agent.model)}"
