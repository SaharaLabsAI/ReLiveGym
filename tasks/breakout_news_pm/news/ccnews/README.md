# CC-NEWS corpus build

Build our own news corpus from Common Crawl's CC-NEWS WARC dumps for the
2026-03-01 → 2026-07-01 market window (crawl range extended to 2026-07-07 to
catch late crawls of late-June articles).

Two passes, both on EC2 in **us-east-1**:

| pass | script | what | cost / time (64 vCPU) |
|---|---|---|---|
| 1. census | `census.py` | url/domain/warc_date/offset for ALL ~34.5M response records; no HTML parsing | ~$3–5, <1h |
| 2. extraction | `extract.py` | English gate + trafilatura/htmldate → article parquet (~9M rows, ~20–25GB) | ~$10–20, ~2–4h |
| 3. publish timestamps | `timestamps.py` | regex over document `<head>` → `(id, published_at, source)` sidecar for records carrying a full timestamp (~79% of raw records in validation); joins onto the corpus on `id` | ~$5–10, ~1–2h (gunzip-bound, no trafilatura) |

Census results and the corpus policy: `census_findings.md`. Extraction adds
deps: `python3.12 -m pip install trafilatura py3langid lxml_html_clean`
(the last one because `lxml.html.clean` is a separate package since lxml 5.2
and a trafilatura code path imports it).

```bash
# validation first, then full run (same flags/resume semantics as census.py)
python3.12 extract.py --limit 3 --out /home/ec2-user/corpus
nohup python3.12 extract.py --start 20260301 --end 20260707 \
    --workers 60 --out /home/ec2-user/corpus > extract.out 2>&1 &
python3.12 extract.py --out /home/ec2-user/corpus --merge   # dedup → corpus_<month>.parquet
```

Pass 3 (publish timestamps) needs only `boto3 warcio pyarrow`:

```bash
python3.12 timestamps.py --limit 3 --out /home/ec2-user/pubtimes
nohup python3.12 timestamps.py --start 20260301 --end 20260707 \
    --workers 60 --out /home/ec2-user/pubtimes > pubtimes.out 2>&1 &
python3.12 timestamps.py --out /home/ec2-user/pubtimes --merge  # → published_at.parquet
# pull published_at.parquet down (~1–2GB); it covers ALL response records —
# join onto the corpus locally on id. Values with no time-of-day (midnight)
# are rejected at extraction, so every row is genuinely hour+ precision, UTC.
```

## Scale facts

- `crawl-data/CC-NEWS/2026/{03..06}` = 1,470 WARCs, ~1.07GB each ≈ 1.6TB;
  ~12 WARCs/day. No CDX index exists for CC-NEWS → full sequential scan.
- ~19.6 response records per MB compressed (request records are the other
  ~half of each WARC) → ~250k articles/day, ~31M fetches in-window.
- Raw CC-NEWS is broad + multilingual (302 domains in a single 100MB slice).
  Presence of major fast outlets (Reuters/AP/BBC/Guardian/...) in 2026 is
  UNKNOWN — many block CCBot since ~2023. The census answers this.

## AWS setup

Nothing exotic — one EC2 instance; the `commoncrawl` S3 bucket is public
(AWS Open Data), read anonymously, free from within us-east-1.

- **Region**: us-east-1. Non-negotiable — cross-region reads of 1.6TB would
  cost ~$32 and run slower.
- **EC2**: `c7i.16xlarge` (64 vCPU, $2.82/h on-demand; spot ≈ 60% off) or
  `c7i.8xlarge` (32 vCPU, $1.41/h, ~2× wall clock). Amazon Linux 2023.
- **EBS**: 100GB gp3 root — census shards are ~5GB; extraction pass output
  ~30GB.
- **IAM**: the commoncrawl bucket requires **authenticated** requests
  (anonymous S3 reads were disabled in 2024). Attach an instance role with a
  minimal policy — no special grants beyond your own account:

  ```json
  {"Version": "2012-10-17", "Statement": [{"Effect": "Allow",
    "Action": ["s3:GetObject", "s3:ListBucket"],
    "Resource": ["arn:aws:s3:::commoncrawl", "arn:aws:s3:::commoncrawl/*"]}]}
  ```

  (For local validation without AWS credentials, `--source https` streams
  from data.commoncrawl.org instead — rate-limited, validation only.)
- **Egress**: only the merged outputs leave AWS (census ~1–2GB, corpus
  ~20–30GB) at $0.09/GB ≈ $3 total. scp/rsync them down, then terminate the
  instance.

```bash
# on the instance
sudo dnf install -y python3.12 python3.12-pip
python3.12 -m pip install boto3 warcio pyarrow

# validation: 3 WARCs — sanity-check output before the full run
python3.12 census.py --limit 3 --out /home/ec2-user/census
# (locally, without AWS credentials: add --source https)

# full run (nohup: survives ssh drop; resumable — rerun skips done shards)
nohup python3.12 census.py --start 20260301 --end 20260707 \
    --workers 60 --out /home/ec2-user/census > census.out 2>&1 &
tail -f census.out

# merge shards, then pull census.parquet down and terminate
python3.12 census.py --out /home/ec2-user/census --merge
```

## Analyzing the census (locally, duckdb or pandas)

```sql
-- per-domain volume: the menu for the corpus policy
SELECT domain, count(*) n FROM 'census.parquet' GROUP BY 1 ORDER BY n DESC LIMIT 100;
-- are the fast outlets present at all?
SELECT domain, count(*) FROM 'census.parquet'
WHERE domain SIMILAR TO '%(reuters|apnews|bbc|theguardian|bloomberg|cnbc|cnn|politico|axios)%'
GROUP BY 1;
-- daily volume sanity
SELECT substr(warc_date, 1, 10) d, count(*) FROM 'census.parquet' GROUP BY 1 ORDER BY 1;
```

English share isn't in the census (needs content); estimate it by sampling
records via Range requests before committing the extraction policy.

## Random access to any article

`(warc_file, offset, length)` lets you fetch a single record without a
rescan — used by the sampling probes and the extraction pass's unit tests:

```python
body = s3.get_object(Bucket="commoncrawl", Key=warc_file,
                     Range=f"bytes={offset}-{offset + length - 1}")["Body"]
rec = next(r for r in ArchiveIterator(body) if r.rec_type == "response")
html = rec.content_stream().read()
```

## Downstream (after extraction)

Corpus parquet feeds `../build_index.py` (tantivy BM25 index + the search
API shared by the hindsight labeler and the tasks). Canonical article clock
= `pub_ts`, assigned at index build (`--ts-v3`: the pass-3 publish time when
the crawl corroborates it within 2 h, otherwise `warc_date` − 2 h; the rule
is documented at the top of `build_index.py`); search date-filtering runs
on it and agent-facing
surfaces expose only its ISO form `pub_date` (provenance fields stay stored,
hidden from agents). Syndication near-duplicates are kept and tagged with a
cluster id, not deleted.
