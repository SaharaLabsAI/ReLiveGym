# Task: keep the desk's broker account in good standing

You are the operations analyst for a small fund. The portfolio manager is
away and their book keeps trading through the desk's order algorithm; you
do not trade. Your job is to answer, in time, every event the account
produces. Nothing to forecast — each duty is a known rule with a deadline.
The challenge is being there in time, every time, without getting locked
out.

## Where things are

- **Broker portal** (login required): `${broker_url}/login`
  - username `${username}`, password `${password}`
  - after the password, the portal mails you a one-time code (valid
    ${code_minutes} minutes) — read it in your inbox and enter it
  - pages: Positions, Orders (the PM's book, read-only), Notices, Risk desk
- **Broker account API** (read-only, no login): `${api_url}/api/events`
  — every fill, margin call, requirement change and treasury wire the
  moment it happens (`?since=<iso>`; `/api/events/<id>`); also
  `/api/account`, `/api/orders`, `/api/notices`. JSON, readable with your
  browser or shell.
- **Ops inbox** (read-only): `${mail_url}/mail/` — the PM's note and your
  login codes. JSON: `${mail_url}/mail/api/messages` (`?since=<iso>`),
  `/mail/api/messages/<id>`.

Everything on the portal is done through its pages and forms in your
browser. The API and the inbox you may read with your browser or shell.

## The account

Symbols ${symbols}; positions at the start:

| symbol | quantity |
|---|---|
${positions_table}

Cash: ${cash_usd} USD (the account is levered). The portal shows
equity = cash + positions value, gross exposure, and the **margin
ratio** = equity / exposure, against the broker's **maintenance
requirement** (${maintenance_pct} at the start; the broker may raise it
by notice). The desk algorithm enters the PM's orders at ${pm_hours} UTC;
they fill whenever the market trades through their price.

## Duties

1. **Margin calls** — when the margin ratio falls below the maintenance
   requirement the broker issues a margin call (an API event and a
   notice on the portal) naming a required deposit and a deadline ${call_hours} hours
   out. Send a funding instruction on the Notices page for at least the
   required amount before the deadline. Calls come when prices move or
   when the broker raises the requirement — at no fixed time.
2. **Protective stops** — whenever one of the PM's orders fills (an API
   event; the Orders page shows the fill),
   register a protective stop for that fill with the risk desk (Risk desk
   page: a sell stop after a buy fill, a buy stop after a sell fill, with a
   quantity and a trigger price on the protective side of the market)
   within ${stop_hours} hours of the fill.

Each duty instance is checked once, at its deadline: the margin call at
its deadline, the stop ${stop_hours} h after the fill. Score = duty
instances met / duty instances, over the whole run. Higher is better.

## Sessions, codes and lockout

- A session ends after ${idle_minutes} idle minutes and, whatever you do,
  at ${cutoff_utc} UTC every day (20:00 New York time in this season). A
  form submitted on a dead session is **not applied** — the portal shows
  the sign-in page instead; sign in and redo it.
- ${max_failed} consecutive failed sign-ins lock the account for
  ${lockout_hours} hours. Every duty whose deadline falls inside a
  lockout counts as missed.

## Time, budget, cost

The run covers ${sim_start} to ${sim_end}. Simulated time is frozen while
you work and passes only ${time_passes}; a page loaded
${stale_since} is stale until reloaded. You have **${budget_usd} for the entire
run**.${domain_caps} LLM calls bill real token cost against it
(${llm_price_table}); the portal and the inbox are free. When the budget
is spent, LLM calls fail and the run continues without you. Wake for what
has a deadline; the duties announce themselves on the account API and on
the Notices and Orders pages.
