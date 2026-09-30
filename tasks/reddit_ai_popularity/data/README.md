# Data: reddit_ai_popularity

Raw, replayable Reddit cascades from AI subreddits, for early-trending
prediction. Method follows MMG-Pop (arXiv:2606.27539): each thread is a
tree-structured cascade rooted at a submission; every node keeps its exact
`created_utc`. **No quantization, no observation windows, no splits, no baked
label** — the "continuously evolving world" is the task's time-gated
visibility rule. Comment timestamps are kept (they are the observable early
signal); the visibility rule hides only the future.

The raw cache and the built files are not shipped; this README,
`subreddits.yaml` and `build.py` are the deterministic rebuild.

## Source

[Arctic Shift](https://github.com/ArthurHeitmann/arctic_shift) — the active
Pushshift successor. We use its
**API** (`https://arctic-shift.photon-reddit.com`), not the ~30 GB/month full
dumps, since we want only ten subreddits:

- `/api/posts/search` — root submissions per subreddit, paginated asc by `created_utc`.
- `/api/comments/search?link_id=<root>` — the full comment tree for a root (all comments).

Free, no auth, **no uptime/rate guarantees** — the builder is polite (~1 req/s,
backoff on `422 "slow down"`/`429`/`5xx`) and fully resumable (on-disk cache +
`.done` sidecars).

### Coverage / window

Full Mar–Jul 2026 build: the API ingests near-real-time, so the data frontier
(latest comment) was **2026-08-04** — all of July is covered. **24 h labels
are safe for 100 %** of roots; **7 d labels for 97.7 %** — only **1,745
late-July roots (2.3 %, all in the final week)** sit within 7 d of the
frontier and may have a still-growing tree. Flag/exclude those if using the
7 d horizon.

## Rebuild

```bash
# the world the run configs read: every root, 10 subreddits, Mar–Jul 2026
python tasks/reddit_ai_popularity/data/build.py \
    --after 2026-03-01 --before 2026-08-01 --min-comments 0 \
    --out tasks/reddit_ai_popularity/data/built_min0

# a two-week slice of built_min0 for quick runs
python tasks/reddit_ai_popularity/data/build_smoke_subset.py
```

Build-only deps: `requests`, `pyyaml` (not runtime deps of the task).

## Outputs (`built_min0/`)

- `roots.jsonl` — one row per root: `id, subreddit (the topic
  label), author, created_utc, title, selftext, url, is_self, over_18,
  link_flair_text, domain, score, num_comments, retrieved_on`.
- `cascades.jsonl` — one row per root: `{root_id, nodes[]}` where each node is
  `{id, kind (root|comment), parent_id, author, created_utc, body, score}`.
  Raw tree; parent edges resolve within the cascade; timestamps unrounded.
- `build_stats.json` — the window, per-subreddit posts scanned, kept roots,
  cascades-per-subreddit, cascade-size percentiles.

## Topic model (subreddit = base unit)

`subreddits.yaml` is a flat list of subreddits. **Each subreddit is its own
topic label** — no company grouping (r/OpenAI and r/ChatGPT are separate
topics, never merged) and no alias / fuzzy matching. A root is labeled by the
subreddit it was posted in. Only the listed subreddits are crawled.

Inclusion rule: AI-relevant subs with decent volume (~≥350 cascades/mo at
min 5 comments) — a mix of company-flavored subs and general AI topic
communities (`LocalLLaMA`, `singularity`, `artificial`). Excluded: low-volume
subs (deepseek, midjourney, Bard, mistral, machinelearning, huggingface) and
non-AI-specific ones (`technology`, `google`).

## Known caveats (experimenter-facing; keep OUT of any INSTRUCTION.md)

- **Score is an ingestion-time snapshot** (`retrieved_on`), and Reddit fuzzes
  votes — treat structural signals (cascade size, unique users, depth, width,
  virality) as primary, `score` as noisy/auxiliary. The task never serves it.
- **Deleted/removed content** — `body`/`author` become `[deleted]`/`[removed]`
  but the node's timestamp + tree position are retained (kept as structural nodes).
- **Topic label = subreddit, not a resolved entity** — the label is the
  community, not "which company the post is about". Company-flavored subs skew
  to their company; topic subs (`LocalLLaMA`, `singularity`, `artificial`) span
  many companies by design.
- **Archive completeness** — best-effort; spot-check large threads vs live Reddit.
- **Licensing** — community redistribution; fine for research, not officially
  licensed.

## Dataset (`built_min0`, measured)

10 subreddit topics, `min_comments=0`: **229,810 roots** (ClaudeAI 65.1k,
ChatGPT 34.1k, grok 29.3k, LocalLLaMA 23.3k, GeminiAI 20.3k, OpenAI 16.6k,
StableDiffusion 12.6k, artificial 11.4k, Anthropic 9.5k, singularity 7.6k).
Cascade size: median 3, p90 28, p99 180, max 6,365. Reddit cascades saturate
fast (median tree span ~1.3 d), so +7 d is near-final. About 1.3 GB built;
the build takes a few hours at ~5 req/s and is fully resumable.
