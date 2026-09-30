"""Env-side tool framework.

Every capability the environment offers an agent program is declared exactly
once, server-side, as a tool method on an EnvApp — the name, the doc (the
agent-facing contract text), the price tag, and the handler live next to the
metering. The API layer serves the manifest (GET /tools) and dispatches
calls (POST /call/{name}). Which tools exist in a given run is decided by
*provisioning* — which apps harness/apps.py and the task instantiate for
this run's config — never by branches in agent code.

The doc string of a tool is part of instruction economics: it is the single
source of the wording every cell sees, and a harness test asserts shared
tools render byte-identically across cells.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Awaitable, Callable

if TYPE_CHECKING:
    from harness.runtime import Sim


class ToolError(ValueError):
    """Rejected tool call (bad args / not applicable). Maps to HTTP 400 in
    the API layer — rejection is the free error-handling path."""


class PaymentRequired(ToolError):
    """Cost-bearing call refused because the run's budget_usd is spent.
    Maps to HTTP 402 in the API layer."""


# Args schema when a tool declares none: any JSON object. Declared schemas
# are machine-readable structure ONLY —
# the doc string stays the single agent-facing wording (instrument
# boundary; the manifest doc-identity tests keep enforcing that).
GENERIC_SCHEMA: dict = {"type": "object", "additionalProperties": True}


@dataclass(frozen=True)
class ToolDef:
    """One manifest entry: what the agent program can see about a tool."""

    name: str
    doc: str
    price: str  # "free" | short price note, e.g. "paid per call"
    tags: frozenset[str]
    input_schema: dict | None = None  # JSON Schema of the args object

    def to_dict(self) -> dict:
        return {"name": self.name, "doc": self.doc, "price": self.price,
                "tags": sorted(self.tags),
                "input_schema": self.input_schema or GENERIC_SCHEMA}


def tool(doc: str, price: str = "free", tags: tuple[str, ...] = (),
         schema: dict | None = None):
    """Declare a method of an EnvApp as an agent-callable tool. The method
    name is the tool name; the handler signature is `async def name(self,
    args: dict) -> jsonable`. `schema` (optional) is a JSON Schema for the
    args object, served in the manifest as `input_schema` (generic object
    when absent); validation stays in the handler — the schema is contract
    data for MCP clients / stub generators, not an enforcement layer."""

    def mark(fn):
        fn.__tool__ = {"doc": doc, "price": price, "tags": frozenset(tags),
                       "schema": schema}
        return fn

    return mark


Handler = Callable[[dict], Awaitable[Any]]


class EnvApp:
    """Base class for one environment capability area. Subclasses declare
    tools with @tool; handlers take sim.lock themselves and book costs to
    sim.ledger — metering stays server-side by construction."""

    def __init__(self, sim: "Sim"):
        self.sim = sim

    def tools(self) -> list[tuple[ToolDef, Handler]]:
        out: list[tuple[ToolDef, Handler]] = []
        for cls in type(self).__mro__:
            for name, attr in vars(cls).items():
                meta = getattr(attr, "__tool__", None)
                if meta is None or any(t.name == name for t, _ in out):
                    continue
                out.append((ToolDef(name=name, doc=meta["doc"],
                                    price=meta["price"], tags=meta["tags"],
                                    input_schema=meta.get("schema")),
                            getattr(self, name)))
        return out


def build_registry(apps: list[EnvApp]) -> dict[str, tuple[ToolDef, Handler]]:
    registry: dict[str, tuple[ToolDef, Handler]] = {}
    for app in apps:
        for tdef, handler in app.tools():
            if tdef.name in registry:
                raise ValueError(f"duplicate tool name {tdef.name!r}")
            registry[tdef.name] = (tdef, handler)
    return registry


def manifest(registry: dict[str, tuple[ToolDef, Handler]]) -> list[dict]:
    return [tdef.to_dict() for tdef, _ in registry.values()]
