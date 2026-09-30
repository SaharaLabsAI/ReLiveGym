# CC-NEWS census findings

Census run on EC2 (us-east-1), analyzed from
`tasks/breakout_news_pm/news/ccnews/census/census.parquet`. Crawl range 2026-03-01 → 2026-07-07.

## Headline numbers

- **34,525,265 response records**, 34,438,845 distinct URLs (0.25% dup —
  negligible), **16,060 distinct domains**, 130 calendar days with **no
  missing days**.
- Daily volume ~250–320k typical (one spike day at 739k).
- **English share ≈ 26%** (html `lang` attr on a 240-record random sample;
  ±~6%) → **~9M English articles** in-window. Next: es 12%, fr 6%, ru/de/ja
  5% each, fa 3% (Iran), ar 2%.

## Fast/elite outlets: confirmed ABSENT

Reuters, AP, BBC, Guardian, Bloomberg, CNBC, NYT, WaPo, Politico, Axios,
The Hill, Newsweek, dw, france24, CoinDesk, TechCrunch, The Verge,
MarketWatch, Telegraph/Independent/Mail (UK), SCMP, Nikkei — all zero.
Post-2023 CCBot blocking. **No corpus built from CC-NEWS can contain the
fastest wires.** First-report latency in our corpus will lag the true break
for many events; this is now a documented property of the world, not a bug.

## Useful present tier (by our market categories)

- **Geopolitics**: aa.com.tr (Anadolu) 62k, aljazeera.com 18.5k (+43k
  Arabic .net), tass.com 14k, skynewsarabia 8.9k, euronews 3.5k, semafor 3k,
  ynetnews 2.8k, en.mehrnews.com 2.6k (Iran), kyivindependent 2.3k,
  themoscowtimes 949, jpost 309, ukrinform 165.
- **US politics**: westernjournal 102k, cbsnews 80k, foxnews 13.2k,
  thedailybeast 9.9k, breitbart 9.7k, nbcnews 555.
- **Finance**: marketbeat 121k, prnewswire 102k + einpresswire 87k +
  globenewswire 81k (press-release wires — genuinely useful for corporate
  events), nasdaq.com 80k, business-standard 80k, benzinga 17.5k, fxstreet 451.
- **Crypto**: cointelegraph 2.7k, u.today 1.3k (thin — decrypt/theblock/
  coindesk all absent).
- **Tech**: engadget 3.8k, tomshardware 2.1k, theregister 108 (thin).
- **Asia/India bulk**: timesofindia 196k, zeenews 123k, ndtv 37k,
  straitstimes 27k, chosun 140k(ko), koreaherald 9.1k.
- Global top domains are dominated by non-English (livedoor.jp 1.3M,
  zazoom.it 544k, agazeta.com.br 343k, mehrnews.com 309k fa, …) and sports
  (goal.com 412k).

## Corpus policy (as built)

1. **English only** (v1): html `lang` attr gate (`en*`), py3langid check on
   extracted text when the attr is missing/ambiguous. ~9M articles.
2. **All domains** — no whitelist. Domain is a column; tiering/relevance is
   the retrieval layer's job, not a collection-time decision.
3. Extraction: trafilatura (+htmldate metadata) per record; drop records
   where extraction yields no text (nav/video/tag pages).
4. Dedup: exact-URL keep-earliest-`warc_date` at merge. Syndication
   near-dups KEPT (tagged with cluster ids later, at index build).
5. Visibility clock assigned at index build (`../build_index.py`): the
   parsed publish time when `warc_date` corroborates it, else
   `warc_date` − 2 h.
6. Provenance: every row keeps `(warc_file, offset, length)` for exact
   re-fetch.

Extraction-probe caveat (30-record sample, 5 kept, all with clean
title/text/date): CC-NEWS recrawls **evergreen pages** — 2 of the 5 kept had
`date_publish` of 2022/2024 vs a 2026 `warc_date` (study-guide pages, not
news). Handle at index build, not collection: flag/exclude rows with
`date_publish` far before `warc_date` (e.g. > 30 days stale) from the news
index; visibility clock stays `warc_date` regardless.
