"""The actor runner: the process half of the program contract's lifecycle
. Program library — copied into
every workspace as runtime/actor.py; stdlib only, no harness imports.

    cd <workspace> && ENV_URL=... ENV_TOKEN=... ENV_MODEL=... \\
        python runtime/actor.py run

drives one run against a live sim server: fetch the contract (writes
INSTRUCTION.md + cell_config.py for this sim), then loop
`GET /trigger/next` -> spawn `python main.py` with ENV_TRIGGER -> real-time
watchdog -> `POST /trigger/<id>/exit` until the server says done. The
server owns sim time, the ledger and every decision (crash streak,
rollback); this runner owns the process, the crash logs, the code_history
snapshots and the git reset a rollback asks for. It is the standalone
harness program: given a server and a workspace it runs anywhere — the
combined mode (harness.run) runs this same loop in a thread of the server
process, so both modes produce the same ledger.

    python runtime/actor.py check [<dir>]

is the offline preflight: main.py present, every .py parses, runtime/ in
place. No server needed.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

OUTPUT_TAIL_BYTES = 1 << 20


def watchdog_verdict(act: dict) -> str | None:
    """Why the runner should kill the invocation now, or None.

    `act` is GET /activity. The watchdog fires after `watchdog_seconds` of
    real time without an environment call ("idle") — but not while an LLM
    call is in flight: a long call is not a hang. That suspension is
    bounded: once the oldest in-flight call has outlived the server's own
    LLM timeout by a further watchdog period ("llm_inflight") it can no
    longer complete normally, so the invocation is treated as hung. A
    server that reports no timeout fields gets the unbounded rule."""
    wd = float(act["watchdog_seconds"])
    if not act.get("llm_inflight"):
        return "idle" if float(act["idle_seconds"]) > wd else None
    timeout = act.get("llm_timeout_seconds")
    oldest = act.get("oldest_llm_inflight_seconds")
    if timeout is not None and oldest is not None \
            and float(oldest) > float(timeout) + wd:
        return "llm_inflight"
    return None


class ServerError(RuntimeError):
    pass


class _Api:
    def __init__(self, base_url: str, token: str):
        self.base_url = base_url.rstrip("/")
        self._headers = {"Authorization": "Bearer " + token,
                         "Content-Type": "application/json"}

    def _request(self, method: str, path: str, body=None, params=None):
        url = self.base_url + path
        if params:
            q = {k: v for k, v in params.items() if v is not None}
            if q:
                url += "?" + urllib.parse.urlencode(q)
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method,
                                     headers=self._headers)
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            raise ServerError(
                f"{method} {path} -> {e.code}: {e.read().decode()}") from None

    def get(self, path: str, **params):
        return self._request("GET", path, params=params)

    def post(self, path: str, body: dict):
        return self._request("POST", path, body=body)


# -- helpers ----------------------------------------------------------------------------


def _git_head(workspace: Path) -> str | None:
    if not (workspace / ".git").is_dir():
        return None
    r = subprocess.run(["git", "-C", str(workspace), "rev-parse", "HEAD"],
                       capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def _code_sha(workspace: Path) -> tuple[str | None, bytes | None]:
    program = workspace / "main.py"
    if not program.exists():
        return None, None
    content = program.read_bytes()
    return hashlib.sha256(content).hexdigest()[:12], content


def _stamp(iso_time: str) -> str:
    return iso_time.replace(":", "")


def _list_files(workspace: Path) -> list[str]:
    return sorted(str(p.relative_to(workspace)) for p in workspace.rglob("*")
                  if p.is_file() and "__pycache__" not in p.parts
                  and ".git" not in p.parts)


def _log(msg: str) -> None:
    print(f"[actor] {msg}", flush=True)


# -- the runner ----------------------------------------------------------------------------


class Runner:
    def __init__(self, env_url: str, token: str, model: str,
                 workspace: Path, out_dir: Path,
                 python_exe: str = sys.executable, quiet: bool = False):
        self.api = _Api(env_url, token)
        self.env_url = env_url
        self.token = token
        self.model = model
        self.workspace = Path(workspace).resolve()
        self.out_dir = Path(out_dir).resolve()
        self.python_exe = python_exe
        self.quiet = quiet
        self._last_code_hash: str | None = None
        self.watchdog_seconds: float | None = None
        self._kill_reason: str | None = None  # watchdog_verdict of the last kill

    def log(self, msg: str) -> None:
        if not self.quiet:
            _log(msg)

    # -- setup ------------------------------------------------------------------------

    def prepare(self) -> dict:
        """Fetch the contract; write the sim's INSTRUCTION.md and
        cell_config.py into the workspace; record the workspace manifest
        (files + tools) next to the run artefacts if nobody has yet."""
        ws = self.workspace
        contract = self.api.get("/contract")
        if contract.get("instruction_md") is not None:
            (ws / "INSTRUCTION.md").write_text(contract["instruction_md"],
                                               encoding="utf-8")
        if contract.get("cell_config_py") is not None:
            (ws / "cell_config.py").write_text(contract["cell_config_py"],
                                               encoding="utf-8")
        (ws / "logs").mkdir(exist_ok=True)
        (ws / "memory").mkdir(exist_ok=True)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        manifest = self.out_dir / "workspace_manifest.json"
        if not manifest.exists():
            manifest.write_text(json.dumps(
                {"files": _list_files(ws), "tools": contract["tools"]},
                indent=1))
        return contract

    # -- the loop --------------------------------------------------------------------

    def run(self) -> dict:
        self.prepare()
        while True:
            code_sha, content = self._code()
            trig = self.api.get("/trigger/next", code_sha=code_sha,
                                head_sha=self._head())
            if trig.get("done"):
                self.log(f"done at {trig.get('now')}")
                return trig
            self._snapshot_code(code_sha, content, trig["now"])
            self.log(f"trigger {trig['id']} kind={trig['kind']} "
                     f"due={trig['due_time']}")
            code, killed, output = self._spawn(trig)
            resp = self.api.post(f"/trigger/{trig['id']}/exit", {
                "code": code, "killed": killed, "output": output,
                "head_sha": self._head()})
            outcome = "watchdog_killed" if killed else f"exit={code}"
            self.log(f"  {outcome} at {resp['now']}")
            if resp.get("crashed"):
                self._write_crash_log(trig, code, killed, output, resp["now"])
            if resp.get("rollback_to"):
                self._rollback(resp["rollback_to"])
            if resp.get("failed_degenerate"):
                self.log("  run marked failed-degenerate")

    # what the server tracks of the program between invocations (code_change,
    # rollback); a runner whose workspace holds no program overrides both
    def _code(self) -> tuple[str | None, bytes | None]:
        return _code_sha(self.workspace)

    def _head(self) -> str | None:
        return _git_head(self.workspace)

    # -- one invocation ---------------------------------------------------------------

    def _spawn(self, trig: dict) -> tuple[int, bool, str]:
        env = {
            **os.environ,
            "ENV_URL": self.env_url,
            "ENV_TOKEN": self.token,
            "ENV_TRIGGER": json.dumps({
                k: trig[k] for k in ("id", "kind", "due_time", "owner",
                                     "note", "target", "costs") if k in trig}),
            "ENV_MODEL": self.model or "",
        }
        proc = subprocess.Popen(
            [self.python_exe, "main.py"], cwd=str(self.workspace), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        chunks: list[bytes] = []

        def reader():
            assert proc.stdout is not None
            for chunk in iter(lambda: proc.stdout.read(65536), b""):
                chunks.append(chunk)

        t = threading.Thread(target=reader, daemon=True)
        t.start()
        killed = False
        poll = 0.1
        while proc.poll() is None:
            time.sleep(poll)
            if proc.poll() is not None:
                break
            act = self.api.get("/activity")
            wd = float(act["watchdog_seconds"])
            self.watchdog_seconds = wd
            poll = max(0.05, min(0.5, wd / 4))
            verdict = watchdog_verdict(act)
            if verdict is not None:
                self._kill_reason = verdict
                proc.kill()
                killed = True
        code = proc.wait()
        t.join()
        output = b"".join(chunks)[-OUTPUT_TAIL_BYTES:].decode(errors="replace")
        return code, killed, output

    # -- actor-side artefacts -----------------------------------------------------------

    def _snapshot_code(self, digest: str | None, content: bytes | None,
                       now: str) -> None:
        """code_history/: every distinct main.py the run executed."""
        if digest is None or digest == self._last_code_hash:
            return
        history = self.out_dir / "code_history"
        history.mkdir(parents=True, exist_ok=True)
        (history / f"{_stamp(now)}-{digest}.py").write_bytes(content or b"")
        self._last_code_hash = digest

    def _write_crash_log(self, trig: dict, code: int, killed: bool,
                         output: str, now: str) -> None:
        """workspace/logs/crash-<simtime>.log — a deliberate affordance:
        the agent can read its own crash logs."""
        logs = self.workspace / "logs"
        logs.mkdir(exist_ok=True)
        wd = self.watchdog_seconds
        if not killed:
            reason = f"exit code {code}"
        elif self._kill_reason == "llm_inflight":
            reason = ("watchdog kill: an LLM call stayed in flight past the "
                      f"environment's LLM timeout plus {wd:g}s")
        else:
            reason = f"watchdog kill after {wd:g}s without helper interaction"
        trigger = json.dumps({"id": trig["id"], "kind": trig["kind"],
                              "due_time": trig["due_time"]})
        (logs / f"crash-{_stamp(now)}.log").write_text(
            f"sim_time: {now}\n"
            f"trigger: {trigger}\n"
            f"reason: {reason}\n"
            f"--- output (stdout+stderr) ---\n{output}",
            encoding="utf-8")

    def _rollback(self, sha: str) -> None:
        subprocess.run(["git", "-C", str(self.workspace), "reset", "--hard",
                        sha], capture_output=True)
        self.log(f"  rolled back workspace to {sha[:12]}")


def run_loop(env_url: str, token: str, model: str, workspace, out_dir,
             python_exe: str = sys.executable, quiet: bool = False) -> dict:
    return Runner(env_url, token, model, Path(workspace), Path(out_dir),
                  python_exe, quiet).run()


# -- preflight ----------------------------------------------------------------------------


def check(root: Path) -> list[str]:
    """Offline checks; returns the list of problems (empty = ok)."""
    problems = []
    if not (root / "main.py").is_file():
        problems.append("no main.py")
    if not (root / "runtime" / "env_client.py").is_file():
        problems.append("no runtime/ library (copy runtime/ into the program)")
    for p in sorted(root.rglob("*.py")):
        if "__pycache__" in p.parts or ".git" in p.parts:
            continue
        try:
            ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
        except SyntaxError as e:
            problems.append(f"{p.relative_to(root)}: syntax error line "
                            f"{e.lineno}: {e.msg}")
    return problems


# -- CLI ------------------------------------------------------------------------------------


def _load_handle(path: str | None) -> dict:
    handle = {}
    if path:
        handle = json.loads(Path(path).read_text())
    return {
        "env_url": os.environ.get("ENV_URL") or handle.get("env_url"),
        "token": os.environ.get("ENV_TOKEN") or handle.get("token"),
        "model": os.environ.get("ENV_MODEL") or handle.get("model") or "",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="actor runner: run a program against a live sim server")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run", help="drive one run to its end")
    p_run.add_argument("--run", help="run.json handle {env_url, token, "
                                     "model}; env vars ENV_URL/ENV_TOKEN/"
                                     "ENV_MODEL take precedence")
    p_run.add_argument("--workspace", default=".", help="the program dir "
                       "(default: cwd)")
    p_run.add_argument("--out", default=None, help="where code_history/ "
                       "and workspace_manifest.json go (default: the "
                       "workspace's parent)")
    p_run.add_argument("--python", default=sys.executable)
    p_check = sub.add_parser("check", help="offline preflight of a program dir")
    p_check.add_argument("dir", nargs="?", default=".")
    args = parser.parse_args(argv)

    if args.cmd == "check":
        problems = check(Path(args.dir).resolve())
        for p in problems:
            print(f"[check] {p}")
        print("[check] ok" if not problems else f"[check] {len(problems)} problem(s)")
        return 0 if not problems else 1

    handle = _load_handle(args.run)
    if not handle["env_url"] or not handle["token"]:
        parser.error("need ENV_URL and ENV_TOKEN (env vars or --run run.json)")
    workspace = Path(args.workspace).resolve()
    out_dir = Path(args.out).resolve() if args.out else workspace.parent
    result = run_loop(handle["env_url"], handle["token"], handle["model"],
                      workspace, out_dir, args.python)
    return 0 if result.get("done") else 1


if __name__ == "__main__":
    sys.exit(main())
