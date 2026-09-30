# Data: what each task reads and how it is curated

No dataset is shipped in this folder. Every task world is either rebuilt from
public sources with the scripts kept next to it (this page), or installed from
a released bundle with `scripts/data_sync/setup_data.py` (last section). All
worlds are sampled after March 1, 2026; the windows the paper uses are fixed
in the run configs.

## Substrates shared across tasks

### Polymarket markets and prices (`tasks/breakout_news_pm/market/`)

Used by breakout news detection, forecast portfolio and resolution detection.
Public Polymarket APIs, no key.

```sh
cd tasks/breakout_news_pm/market
python fetch_markets.py                                     # gamma-api keyset crawl of every market
                                                            # overlapping 2026-03-01..07-01 -> raw/markets.jsonl
python fetch_prices.py --min-volume 10000 --min-days 7 --exclude-sports   # clob batch-prices-history, hourly grid
python fetch_prices.py --merge                              # shards -> raw/prices/<market_id>.json
python fetch_prices.py --fetch-start 2026-02-15 --fetch-end 2026-03-01    # February warm-up for the detector
python fetch_prices.py --fidelity 1 ...                     # minute grid (raw/prices_1m/), what breakout news replays
python detect_breakpoints.py                                # hindsight breakpoints -> raw/breakpoints.jsonl
```

`market/README.md` pins the counts of the crawl the paper used (fetched
2026-07-21). Roughly 1.6 GB raw.

### CC-NEWS corpus and BM25 index (`tasks/breakout_news_pm/news/`)

The news world of the three news tasks: articles from Common Crawl's CC-NEWS
crawl for 2026-03-01..07-07, full text, indexed with tantivy (BM25). Three
passes over the WARC dumps, run on EC2 in us-east-1 (`news/ccnews/README.md`
has the exact commands and instance sizing), then the index build:

```sh
cd tasks/breakout_news_pm/news
python ccnews/census.py     --start 20260301 --end 20260707 --out <dir>   # every response record: url, domain, warc_date, offset
python ccnews/extract.py    --start 20260301 --end 20260707 --out <dir>   # English gate + trafilatura/htmldate -> corpus_<month>.parquet
python ccnews/extract.py    --out <dir> --merge
python ccnews/timestamps.py --start 20260301 --end 20260707 --out <dir>   # <head> publish timestamps -> published_at.parquet
python ccnews/timestamps.py --out <dir> --merge
python build_index.py --corpus corpus/ --pubtimes ccnews/pubtime/published_at.parquet --ts-v3 --out tantivy_index_v3
```

Extra build dependencies: `boto3 warcio pyarrow trafilatura py3langid
lxml_html_clean`. The index build assigns each article one publish clock
(`pub_ts`, the "v3" rule documented at the top of `build_index.py`: a
self-reported time is trusted only when the crawler corroborates it within
two hours, otherwise crawl time minus two hours), which is the visibility
clock of every news search. About 9.5 M documents, 31 GB; the corpus parquet
files are 12 GB. `news/search.py` is the query engine the environment wraps.

### Reddit cascades (`tasks/reddit_ai_popularity/data/`)

Used by popularity prediction and daily digest. Source: the Arctic Shift API
(`https://arctic-shift.photon-reddit.com`), free, no key, no rate guarantee.
`build.py` harvests root submissions per subreddit for a window, then the
full comment tree of every kept root, preserving every `created_utc`
(no quantization, no observation windows, no baked labels):

```sh
python tasks/reddit_ai_popularity/data/build.py \
    --after 2026-03-01 --before 2026-08-01 --min-comments 0 \
    --out tasks/reddit_ai_popularity/data/built_min0
```

Output: `roots.jsonl`, `cascades.jsonl`, `build_stats.json` (~1.3 GB); the
API cache under `data/cache/` is resumable. The run configs point at
`built_min0`; `build_smoke_subset.py` carves a two-week slice for quick
runs. Ten AI subreddits; the subreddit is the topic label. Build-only deps:
`requests`, `pyyaml`.

### Exchange candles (`tasks/crypto_price_consistency/data/`)

Used by market price check and web broker ops. Keyless public candle
endpoints of Binance, OKX and KuCoin (Bybit opt-in), 5-minute BTC/USDT and
ETH/USDT spot, recorded live and stored as raw responses plus a normalized
table per venue:

```sh
python -m tasks.crypto_price_consistency.data.fetch_prices --cohort btc_usdt_spot \
    --start 2026-03-01T00:00:00Z --end 2026-08-01T00:00:00Z --dataset-id btc_usdt_spot_mar_jul
python -m tasks.crypto_price_consistency.data.fetch_prices --cohort eth_usdt_spot \
    --start 2026-03-01T00:00:00Z --end 2026-08-01T00:00:00Z --dataset-id eth_usdt_spot_mar_jul
```

The task reads `data/datasets/<cohort>_mar_jul/` (`dataset_suffix` in
`task.py`); `data/README.md` describes cohorts, layout and provenance
(~140 MB). The proxy the agent sees replays the raw HTTP bodies.

### Incident stream (`tasks/crypto_price_consistency/incidents/`)

The incidents injected into the candle proxy come from a generator fitted
to public status-page archives and frozen before any agent existed
(`incidents/README.md`, `FREEZE_v1.md`). The fitted parameters
(`generator_params_v1.json`), the normalized calibration events
(`calibration/events/`), the rate-limiter definitions (`limiters.yaml`) and
the traces the paper's configs use (`scenarios/fitted_dev_s001.jsonl`,
`zero.jsonl`) are included. To re-scrape, refit, or regenerate the full
trace set (development, held-out, grid and stress scenarios; byte-identical
under the recorded seeds):

```sh
python -m tasks.crypto_price_consistency.incidents.scrape_archives --deep   # -> calibration/raw, calibration/events
python -m tasks.crypto_price_consistency.incidents.fit_generator            # -> generator_params_v1.json
python -m tasks.crypto_price_consistency.incidents.generate_traces          # -> scenarios/*.jsonl + manifest.json
```

### SEC EDGAR snapshot (`tasks/edgar_portfolio/data/`)

Used by web filing tracking. One snapshot of three public data.sec.gov
endpoints for the roster's tickers: the ticker map, each CIK's `submissions`
index (accession numbers, forms, acceptance times to the second) and each
CIK's XBRL `companyfacts`. The replayed host serves these files filtered to
what was accepted at the sim instant; the ground truth is the same files.
SEC fair-access policy: a declared `User-Agent` with a contact address, at
most 10 requests per second (the script enforces both).

```sh
SEC_USER_AGENT="<your name> <contact-email>" \
python -m tasks.edgar_portfolio.data.fetch_edgar     # -> data/raw/ (~170 MB for the 35-ticker roster)
```

The paper's snapshot was taken 2026-09-16 (`raw/manifest.json`). A later
snapshot changes nothing inside a window that has already closed, except
occasional SEC corrections of acceptance timestamps.

## Per-task builds

### Breakout news detection (`tasks/breakout_news_pm/`)

```sh
cd tasks/breakout_news_pm
python labeling/make_sample_v1.py                 # the 350 frozen markets (stratified per domain, seeded) -> labeling/sample_v1.jsonl
python labeling/run_labeling.py --model <model>   # hindsight attribution: one LLM episode per breakpoint over the
                                                  # news index, citing the articles that explain the move -> labeling/out_v1/
python data/build.py                              # -> data/built/{markets.jsonl, prices/, breakpoints.jsonl, attributions.jsonl, build_stats.json}
```

The world compiles the sample's markets, their minute-grid prices, the
minute-localized breakpoints and the attribution labels. The coverage-F1
metric of the paper needs only the breakpoints; the attribution labels feed
the citation-aware reporting metrics and the `sig: oracle` feedback of
learning cells. `data/built/build_stats.json` (included) pins the counts of
the build the paper used; the run configs list the market rosters of the
three episodes (w10, w13, w17).

### Forecast portfolio (`tasks/forecast_portfolio/`)

```sh
cd tasks/forecast_portfolio/data
python build_sample.py        # 300 markets, stratified on open-time features only, from the Polymarket crawl -> sample_v1.jsonl
python build.py               # -> built/questions.jsonl (agent block + scorer block), built/prices/ (scorer-side anchor only)
```

Resolution facts are recorded as outcome data, never used for selection
(`data/README.md`). Prices come from the shared Polymarket store or are
fetched for the few missing markets. The run configs name the 60-question
roster.

### Resolution detection (`tasks/resolution_detect/`)

```sh
cd tasks/resolution_detect/data
python build_sample.py        # 700 far-deadline markets (scheduled end >= sim_end + 30 d) -> sample_detect_v1.jsonl
python audit_sample.py        # gate audit: scored set, Yes/No split, detection instants, trap census
python build.py               # -> built/questions.jsonl with t_det pinned under the frozen settlement constants
```

`data/README.md` records the admission rule and the two post-selection
exclusions (easy-at-activation rows and full-deadline-value markets) that
give the 686-row roster; the configs name the 60-question rosters of the
three episodes.

### Daily digest, web broker ops

No build of their own: daily digest reads the Reddit world
(`data_dir` in its configs), broker ops reads the exchange candles
(`DEFAULT_DATASETS` in its `task.py`). Their worlds are otherwise generated
from the config (the broker's scripted book, sessions and notices).

## Installing a released bundle

`scripts/data_sync/manifest.yaml` lists every data path above in tiers
(`required`: what the tasks read at run time, about 33 GB of which 31 GB is
the news index; `optional`: alternate builds and labeler outputs; `raw`:
rebuild-only inputs). `pack_data.py` zips the entries with repo-relative
members and writes an `index.json` with sizes and checksums;
`setup_data.py` downloads from a shared folder, verifies and extracts at the
repository root, where every `task.py` expects the files:

```sh
pip install gdown
python scripts/data_sync/setup_data.py download --folder "<shared folder link>"          # required tier
python scripts/data_sync/setup_data.py download --folder "<link>" --only fp-built,rd-built
python scripts/data_sync/setup_data.py status
```

Run any task's calibration pins afterwards (`pytest tests/tasks/<task>`;
the tests that need built data skip until it is present).
