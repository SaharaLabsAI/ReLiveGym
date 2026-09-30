"""Token counting for the learned-block budget and transcript compaction.
Program library.

tiktoken (o200k_base) is a hard requirement, so the same counter runs in
every cell of a run — which is what the experiment control requires.
"""

from __future__ import annotations

import tiktoken

BUDGET_TOKENS = 5000  # learned-block budget

_enc = tiktoken.get_encoding("o200k_base")


def count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def truncate(text: str, budget: int) -> str:
    toks = _enc.encode(text)
    if len(toks) <= budget:
        return text
    return _enc.decode(toks[:budget])
