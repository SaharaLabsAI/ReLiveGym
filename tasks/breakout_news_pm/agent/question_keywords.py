"""Canonical query construction for scaffolds that build keyword searches
from a market question (the TM-C scan handler — ReACT cells compose
queries free-form and don't use this).

Query policy, not env contract: the env's BM25 index (tantivy) indexes
every token, and the wake sidecar matches raw substrings, so nothing is
dropped server-side. Excluding stopwords and generic question tokens
("win", years, question words) here is a deliberate choice — they match
half the corpus and dilute the ranked results — protecting the k-slot
query budget.
"""

import re

STOPWORDS = frozenset(
    "the a an of to in on for and or is are was were be been will with by at "
    "as it its this that from not no yes than then over under after before "
    "who what when which how much many win wins won 2024 2025 2026 he she his "
    "her they them their but about amid says said first new back more most "
    "up down out off just now today why all can could would should".split())

_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")


def tokens(text: str) -> list[str]:
    """Non-stop tokens in text order (duplicates kept, for frequency use)."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def keywords(question: str, k: int = 8) -> list[str]:
    """First k distinct non-stop tokens of a market question."""
    seen, out = set(), []
    for t in tokens(question):
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out[:k]
