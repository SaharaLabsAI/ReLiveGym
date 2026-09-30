# Task: keep the fund's filings sheet current from EDGAR

You are the junior analyst of a fund. Whenever one of the fund's holdings
files quarterly results with the SEC, you record the filing in the fund's
portfolio sheet **before the next NYSE open**. Nothing to forecast: the
numbers are in the filing's XBRL data. The job is being there in time and
copying the right values into the right row.

## Where things are

- **EDGAR mirror** (read-only, same paths as data.sec.gov, served as of the
  current time): `${sec_url}`
  - `${sec_url}/files/company_tickers.json` — ticker → CIK map
  - `${sec_url}/submissions/CIK##########.json` — a company's filing index
    (`filings.recent`: `accessionNumber`, `form`, `items`,
    `acceptanceDateTime`, `reportDate`, …; newest first; years of history)
  - `${sec_url}/api/xbrl/companyfacts/CIK##########.json` — all XBRL facts
    (large)
  - `${sec_url}/api/xbrl/companyconcept/CIK##########/us-gaap/<Tag>.json`
    — one tag's facts (small), e.g. `Revenues`,
    `RevenueFromContractWithCustomerExcludingAssessedTax`,
    `EarningsPerShareDiluted`
  - filing documents (`/Archives/...`) are not served by the mirror.
  Fetch these with your shell or browser as you normally would. The public
  sec.gov is not reachable from this machine; the mirror is the source.
- **Portfolio sheet** (the fund's internal web app): `${portal_url}/portfolio/`
  — the holdings page lists every position; a ticker's page lists its
  recorded filings and has the form to record one. Writing to the sheet
  is done through that form in your browser.
${api_arm_section}
## Holdings at the start (${n_holdings} positions)

| ticker | CIK | shares |
|---|---|---|
${holdings_table}

The holdings page is authoritative: the desk trades during the run, so
positions can be added or sold. A filing only matters while the position
is held at the time the filing is accepted. Some holdings report outside
this run's window.

## What to record

For every held company, two filings per quarter count:

1. the **earnings press release** — an `8-K` whose `items` include `2.02`;
2. the **periodic report** — the `10-Q` (or `10-K`).

Record each as its own row on the ticker's page, keyed by its accession
number:

| field | 8-K | 10-Q / 10-K |
|---|---|---|
| accession number | required | required |
| form | `8-K` | `10-Q` / `10-K` |
| acceptance time (UTC, ISO-8601) | required, to the minute | required, to the minute |
| fiscal year, fiscal period | — | as in the XBRL facts of that accession (`fy`, `fp`) |
| revenue, USD | — | the single-quarter (10-Q) or full-year (10-K) value reported by that accession |
| diluted EPS | — | same |

XBRL facts of a filing are the rows whose `accn` equals its accession
number; a 10-Q reports both the quarter and the year-to-date — enter the
quarter (the row whose period is about three months). Where a filer does
not report a revenue or EPS tag, leave the field blank.

## Deadline and credit

Each recorded filing is checked once, at the **first NYSE open after its
acceptance** (09:30 ET on a trading day; weekends and NYSE holidays are
skipped; the sheet's header shows the market status and the next open).
A filing earns credit when, at that instant, its row exists and:

- form, acceptance time (within ${acceptance_tolerance_s} s), and — for
  periodic reports — fiscal year, fiscal period, revenue (within
  ${revenue_tolerance_pct}%) and diluted EPS (within ${eps_tolerance})
  match the filing;
- the row's last save happened **after the filing was accepted** and before
  the deadline (a row saved before the filing existed earns nothing, even if
  its numbers turn out right);
- between the filing's acceptance and that save you fetched that company's
  filing index or XBRL facts from the mirror (the mirror logs every request).

Score = credited filings / filings that count, over the whole run. Higher
is better. Filings accepted after 16:00 ET have until the next morning's
open; filings accepted before the open have only until 09:30 ET the same
day — these are the tight ones.

## Time, budget, cost

The run covers ${sim_start} to ${sim_end}. Simulated time is frozen while
you work and passes only ${time_passes}; a page or file loaded
${stale_since} is stale until fetched again. You have **${budget_usd} for the
entire run**.${domain_caps} LLM calls bill real token cost against it
(${llm_price_table}); fetching the mirror and using the sheet are free. When
the budget is spent, LLM calls fail and the run continues without you.
Spend your attention where it changes decisions: filings cluster around
quarter-end deadlines, most land after 16:00 ET, and a company's own
history in `submissions` shows when it usually files.
