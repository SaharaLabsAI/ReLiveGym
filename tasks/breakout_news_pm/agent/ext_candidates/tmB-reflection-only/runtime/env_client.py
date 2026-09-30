"""Client for the experiment environment. Program library.

The environment serves a manifest of its tools (GET /tools) and one uniform
dispatch route (POST /call/{name}); this client is a thin proxy over them.
What tools exist, their docs, and their prices are environment data — read
`tools()`, don't hard-code. Metering happens server-side: editing this file
cannot change what is measured.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone


class EnvError(RuntimeError):
    """The environment rejected a request (HTTP 4xx/5xx)."""


class Env:
    def __init__(self):
        self.base_url = os.environ["ENV_URL"]
        self._headers = {
            "Authorization": "Bearer " + os.environ["ENV_TOKEN"],
            "Content-Type": "application/json",
        }
        # Why this process was launched: {"id": ..., "kind": "cron"|"at"|
        # "fallback"|"crash_recovery", "due_time": ...}
        self.trigger = json.loads(os.environ.get("ENV_TRIGGER", "{}"))
        self._manifest: list[dict] | None = None

    def _request(self, method: str, path: str, body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base_url + path, data=data,
                                     method=method, headers=self._headers)
        try:
            with urllib.request.urlopen(req) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            raise EnvError(
                f"{method} {path} -> {e.code}: {e.read().decode()}") from None

    # -- the contract ------------------------------------------------------------

    def tools(self) -> list[dict]:
        """The manifest of tools this run provisions: [{name, doc, price,
        tags}, ...]. Cached after the first read."""
        if self._manifest is None:
            self._manifest = self._request("GET", "/tools")["tools"]
        return self._manifest

    def call(self, name: str, **args):
        """Invoke one environment tool by name. Wrong names/args fail loudly
        with a free 4xx (EnvError)."""
        return self._request("POST", f"/call/{name}", args)

    def llm(self, path: str, body: dict) -> dict:
        """The metered LLM proxy (used by llm_client; not a manifest tool)."""
        return self._request("POST", f"/llm/{path}", body)

    def contract(self) -> dict:
        """The published contract of this run (GET /contract): tools, llm
        terms, INSTRUCTION.md, cell_config.py, the envkit stub."""
        return self._request("GET", "/contract")

    # -- convenience -------------------------------------------------------------

    def now(self) -> datetime:
        """Current simulated time (UTC). Constant between waits: sim time
        advances only inside the wait tools."""
        return parse_iso(self.call("get_time")["now"])


def parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def iso(t: datetime | str) -> str:
    if isinstance(t, datetime):
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return t
