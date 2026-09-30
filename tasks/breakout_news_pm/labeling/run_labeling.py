#!/usr/bin/env python3
"""Hindsight attribution labeling runner.

One episode per breakpoint in sample_v1.jsonl: litellm tool loop with
search_news / get_article over the CC-NEWS tantivy index, ending when the
model calls `submit`. The mandatory verbatim-question search is enforced
mechanically: a submit before any search whose query equals the full market
question is rejected back to the model, and cited news_ids must have appeared
in this episode's search results.

Resume-safe: one JSON per episode under --out (skip-if-exists), containing
the label, the full transcript, and token usage.

Usage:
  python3 run_labeling.py --pilot 30            # stratified pilot
  python3 run_labeling.py                       # all 1,525 episodes
  python3 run_labeling.py --model gpt-5.6-sol --concurrency 4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import litellm

HERE = Path(__file__).resolve().parent

# repo-root .env (OPENAI_API_KEY etc.) — loaded if not already in the env
_env = HERE.parents[2] / ".env"
if _env.exists():
    import os
    for _line in _env.read_text().splitlines():
        if "=" in _line and not _line.lstrip().startswith("#"):
            k, v = _line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
sys.path.insert(0, str(HERE.parent / "news"))
from search import NewsSearch  # noqa: E402

from prompts import (SYSTEM_PROMPT, SEARCH_TOOLS, SUBMIT_TOOL,  # noqa: E402
                     build_episode_prompt)

DEFAULT_DATA = HERE.parent / "market"
DEFAULT_INDEX = HERE.parent / "news" / "tantivy_index"
MAX_TURNS = 24
MAX_ARTICLE_CHARS = 10_000
PILOT_SEED = 42


def norm(q: str) -> str:
    return re.sub(r"\s+", " ", q).strip().strip('"').lower()


class Episode:
    def __init__(self, ns: NewsSearch, market: dict, bp: dict,
                 points: list[list[float]], model: str):
        self.ns, self.market, self.bp, self.model = ns, market, bp, model
        # Responses-API input item list; grows with model output items and
        # function_call_output items each turn (stateless, store=False)
        self.input = [
            {"role": "user", "content": build_episode_prompt(market, bp, points)},
        ]
        self.seen_ids: set[str] = set()
        self.verbatim_done = False
        self.n_searches = 0
        self.usage = Counter()

    # ---- tool implementations -------------------------------------------
    def search_news(self, query: str, date_from: str | None = None,
                    date_to: str | None = None, top_k: int = 10) -> str:
        if norm(query) == norm(self.market["question"]):
            self.verbatim_done = True
        self.n_searches += 1
        try:
            hits = self.ns.search(query, date_from, date_to,
                                  top_k=min(int(top_k), 25))
        except ValueError as e:
            return f"query error: {e}"
        self.seen_ids.update(h["news_id"] for h in hits)
        if not hits:
            return "no results — try fewer/other terms or a wider date range"
        return "\n".join(
            f"[{h['news_id'][:16]}] {h['published'][:16]} {h['domain']}\n"
            f"  {h['title']}\n  {h['snippet'][:200]}"
            for h in hits)

    def get_article(self, news_id: str) -> str:
        matches = [i for i in self.seen_ids if i.startswith(news_id)]
        if not matches:
            return "unknown news_id — use ids from your search results"
        d = self.ns.get_article(matches[0])
        if d is None:
            return "article not found"
        text = d["text"][:MAX_ARTICLE_CHARS]
        if len(d["text"]) > MAX_ARTICLE_CHARS:
            text += " [...truncated]"
        return (f"title: {d['title']}\ndomain: {d['domain']}\n"
                f"url: {d['url']}\npublished: {d['pub_date']}\n\n{text}")

    def check_submit(self, args: dict) -> str | None:
        """Return an error string to bounce back to the model, or None."""
        if not self.verbatim_done:
            return ("rejected: you have not yet searched the full market "
                    "question verbatim — do that first, then submit")
        groups = args.get("groups", [])
        if bool(args.get("no_attribution")) != (len(groups) == 0):
            return "rejected: no_attribution must be true iff groups is empty"
        for g in groups:
            bad = [i for i in g.get("news_ids", [])
                   if not any(s.startswith(i) for s in self.seen_ids)]
            if bad:
                return (f"rejected: news_ids not from your search results: "
                        f"{bad}")
        return None

    # ---- litellm loop (OpenAI Responses API) -----------------------------
    async def _call(self, force_submit: bool):
        for attempt in range(4):
            try:
                return await litellm.aresponses(
                    model=self.model,
                    instructions=SYSTEM_PROMPT,
                    input=self.input,
                    tools=SEARCH_TOOLS + [SUBMIT_TOOL],
                    tool_choice=({"type": "function", "name": "submit"}
                                 if force_submit else "auto"),
                    store=False,
                    include=["reasoning.encrypted_content"])
            except Exception:
                if attempt == 3:
                    raise
                await asyncio.sleep(5 * 2 ** attempt)

    async def run(self) -> dict:
        label = None
        for turn in range(MAX_TURNS):
            resp = await self._call(force_submit=turn == MAX_TURNS - 1)
            u = resp.usage
            self.usage["input_tokens"] += u.input_tokens
            self.usage["output_tokens"] += u.output_tokens
            calls = []
            for item in resp.output:
                d = (item.model_dump(exclude_none=True)
                     if hasattr(item, "model_dump") else dict(item))
                self.input.append(d)
                if d.get("type") == "function_call":
                    calls.append(d)
            if not calls:
                self.input.append({
                    "role": "user",
                    "content": "Use the tools; finish by calling submit."})
                continue
            for c in calls:
                name = c["name"]
                try:
                    args = json.loads(c["arguments"])
                except json.JSONDecodeError:
                    result = "invalid JSON arguments"
                    args = None
                if args is None:
                    pass
                elif name == "search_news":
                    result = await asyncio.to_thread(self.search_news, **args)
                elif name == "get_article":
                    result = await asyncio.to_thread(self.get_article, **args)
                elif name == "submit":
                    err = self.check_submit(args)
                    if err is None:
                        label = args
                        result = "accepted"
                    else:
                        result = err
                else:
                    result = f"unknown tool {name}"
                self.input.append({"type": "function_call_output",
                                   "call_id": c["call_id"],
                                   "output": result})
            if label is not None:
                break
        return {
            "market_id": self.market["market_id"],
            "question": self.market["question"],
            "category": self.market["category"],
            "bp": self.bp,
            "label": label,  # None => episode hit MAX_TURNS without a valid submit
            "n_searches": self.n_searches,
            "verbatim_done": self.verbatim_done,
            "usage": dict(self.usage),
            "model": self.model,
            "transcript": self.input,
        }


def load_episodes(sample_path: Path, data_dir: Path) -> list[tuple[dict, dict, list]]:
    eps = []
    for line in open(sample_path):
        m = json.loads(line)
        pts = json.loads((data_dir / "raw" / "prices"
                          / f"{m['market_id']}.json").read_text())["points"]
        for bp in m["breakpoints"]:
            eps.append((m, bp, pts))
    return eps


def pilot_subset(eps, n):
    """~n episodes, stratified across categories, seeded."""
    rng = random.Random(PILOT_SEED)
    by_cat: dict[str, list] = {}
    for e in eps:
        by_cat.setdefault(e[0]["category"], []).append(e)
    per = max(1, n // len(by_cat))
    out = []
    for cat in sorted(by_cat):
        out.extend(rng.sample(by_cat[cat], min(per, len(by_cat[cat]))))
    return out[:n] if len(out) > n else out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default=str(HERE / "sample_v1.jsonl"))
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA))
    ap.add_argument("--index", default=str(DEFAULT_INDEX))
    ap.add_argument("--out", default=str(HERE / "out_v1"))
    ap.add_argument("--model", default="gpt-5.6-sol")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--pilot", type=int, default=0,
                    help="run only ~N stratified episodes")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ns = NewsSearch(args.index)
    eps = load_episodes(Path(args.sample), Path(args.data_dir))
    if args.pilot:
        eps = pilot_subset(eps, args.pilot)
    todo = [(m, bp, pts) for m, bp, pts in eps
            if not (out_dir / f"{m['market_id']}_{bp['date']}.json").exists()]
    print(f"{len(eps)} episodes, {len(eps) - len(todo)} done, "
          f"{len(todo)} to run (model={args.model})")

    sem = asyncio.Semaphore(args.concurrency)
    stats = Counter()
    t0 = time.time()

    async def one(m, bp, pts):
        async with sem:
            try:
                res = await Episode(ns, m, bp, pts, args.model).run()
            except Exception as e:
                stats["failed"] += 1
                print(f"  FAIL {m['market_id']}_{bp['date']}: {e}", flush=True)
                return
            (out_dir / f"{m['market_id']}_{bp['date']}.json").write_text(
                json.dumps(res))
            stats["done"] += 1
            stats["input_tokens"] += res["usage"].get("input_tokens", 0)
            stats["output_tokens"] += res["usage"].get("output_tokens", 0)
            if res["label"] is None:
                stats["no_submit"] += 1
            elif res["label"]["no_attribution"]:
                stats["no_attribution"] += 1
            n = stats["done"] + stats["failed"]
            if n % 10 == 0:
                rate = (time.time() - t0) / n
                print(f"  [{n}/{len(todo)}] no_attr={stats['no_attribution']} "
                      f"no_submit={stats['no_submit']} fail={stats['failed']} "
                      f"tok={stats['input_tokens'] / 1e6:.1f}M/"
                      f"{stats['output_tokens'] / 1e3:.0f}k "
                      f"eta={rate * (len(todo) - n) / 60:.0f}m", flush=True)

    await asyncio.gather(*(one(m, bp, pts) for m, bp, pts in todo))
    print(f"\ndone: {dict(stats)}")


if __name__ == "__main__":
    asyncio.run(main())
