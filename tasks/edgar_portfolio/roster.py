"""The edgar_portfolio universe. Chosen from the 2026-09-16 EDGAR
snapshot: 30 holdings whose Q1-2026 earnings 8-K (item 2.02) and
10-Q/10-K are accepted inside Apr 1 – May 31 2026 (with a spread of
pre-open, in-session and after-close acceptance times), three holdings
that report outside the window (quiet: the agent should not waste wakes
on them), and two tickers that enter the portfolio mid-run through the
scripted blotter (their filings are scorable only once held)."""

HOLDINGS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "V",
    "MA", "UNH", "JNJ", "PG", "HD", "KO", "PEP", "MRK", "ABBV", "CSCO",
    "CRM", "NFLX", "AMD", "INTC", "QCOM", "TXN", "IBM", "GS", "MS", "BAC",
    "WFC",
]
QUIET = ["ORCL", "ACN", "ADBE"]      # fiscal calendars off the window
ENTERING = ["DIS", "CAT"]            # bought mid-run by the scripted blotter
ALL_TICKERS = HOLDINGS + QUIET + ENTERING
