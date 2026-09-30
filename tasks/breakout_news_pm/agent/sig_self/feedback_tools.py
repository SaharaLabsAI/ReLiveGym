"""Agent tools for sig-self feedback compilation (breakout_news_pm).

NOT provisioned by any registered cell: under the tlrn axis, learning is
triggered by the compiled learn step, not by actor-owned tools. This
registry is kept as the compiled body of a future `tlrn: agent` level
. The `records` key declares which part of a
result is outcome records — runtime.memory's auto_append wraps such tools
so records are appended on arrival."""

from __future__ import annotations

from datetime import datetime

import feedback_fn


def _parse(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def feedback_tools(env) -> dict[str, dict]:
    def hindsight(args):
        return feedback_fn.hindsight_verdicts(
            env, args["market_id"], _parse(args["since"]),
            _parse(args["until"]), args["candidates"],
            question=args.get("question", ""))

    return {
        "hindsight_verdicts": {
            "doc": "hindsight_verdicts(market_id, since: iso, until: iso,"
                   " candidates: [{news_id, title, published}], "
                   "question?) -> detector + LLM attribution verdicts "
                   "over a closed period (paid prices + metered LLM)",
            "fn": hindsight,
            "records": lambda out: out["verdicts"],
        },
    }
