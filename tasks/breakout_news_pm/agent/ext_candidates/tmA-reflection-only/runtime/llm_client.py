"""Minimal client for the environment's metered /llm proxy. Program library.

Calls are booked server-side (flat fee + tokens); editing this file cannot
change what is measured. One convention for every call in every cell:
the model comes from ENV_MODEL (required — no fallback model),
there is no completion cap, and JSON replies are parsed strictly
(fence-tolerant, no salvage) — a None return means the caller logs and
skips.
"""

from __future__ import annotations

import json
import os


def chat(env, messages: list[dict], json_mode: bool = False) -> str:
    """One chat completion through the metered proxy; returns the text.
    No sampling overrides: gpt-5.6-luna accepts only its default
    temperature, so the convention is provider defaults everywhere.
    json_mode asks the provider for a JSON object (response_format):
    without it Gemini tends to emit the {"tool": ...} protocol as a
    native function call, which Google rejects (MALFORMED_FUNCTION_CALL
    -> empty reply, zero usage)."""
    body = {
        "model": os.environ["ENV_MODEL"],
        "messages": messages,
        "stream": False,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    resp = env.llm("chat/completions", body)
    return resp["choices"][0]["message"]["content"] or ""


def chat_json(env, messages: list[dict], **kw) -> dict | list | None:
    """chat(), then parse the reply as JSON (tolerating ``` fences).
    Returns None when the reply is not parseable — callers must degrade
    gracefully rather than crash the run."""
    text = chat(env, messages, json_mode=True, **kw).strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
