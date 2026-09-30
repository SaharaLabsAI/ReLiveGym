>>> Expert curated skill - DO NOT MODIFY, KEEP FROZEN
Market news monitoring playbook.

TIME AND WAITING
- Wait tools take an ABSOLUTE time (`until`, ISO), not a duration. Read
  the current time from your last wake's `now` (or the free get_time)
  and pass now + intended delay. Time only passes while you wait.
- `wait_until` with `any` empty (or omitted) is a plain wait until
  `until` and costs nothing. You do not need conditions to wait.

DO THE COST ARITHMETIC BEFORE ARMING ANYTHING
- Every armed news_match condition bills one search PER CHECK while
  armed, whether or not anything matches. At the default 5-minute
  checks and $0.01/search that is $2.88/day per condition; three
  stacked conditions are $8.64/day per market. An unmatchable query
  still pays full price — a quiet condition is not a cheap condition.
- Conditions bill independently: prefer ONE precise query over several
  overlapping broad ones.
- Set "poll_minutes" to choose the check cadence (>= 5; coarser is
  cheaper): poll_minutes 60 bills 24 searches/day (~$0.24), a good
  default for steady background vigilance. Reserve 5-minute checks for
  a market in a clearly elevated-risk state, and disarm when it calms.
- The cheap alternative to standing conditions: wait free (empty
  `any`), wake on your own schedule, and run one targeted search_news
  ($0.01). A daily check on a quiet market costs cents per month.

VALIDATE A QUERY BEFORE ARMING IT
- First run the candidate `q` as a plain search_news over the past few
  days. If it returns more than a handful of hits per day, the standing
  version will fire near-constantly and every wake costs you thinking
  time. Tighten with +required entity terms and "quoted phrases" from
  the market's resolution criteria until the historical rate is a few
  hits/day, then arm.

USE THE PRICE CHANNEL AS A THERMOMETER
- price_level / price_move conditions bill at the price rate (check
  the tool prices — often far cheaper than news checks). Keep a
  price_move armed per market as a state detector: a move you did not
  alert on first means you missed news. Then run ONE search over the
  preceding 24 hours, find the article you should have caught, and
  rewrite your standing query using its actual wording. Do not alert
  on that move — cited news must precede the move, so spend the lesson
  on the next one.

WAKE TRIAGE
- A condition wake includes the result page. Judge TITLES first;
  re-arm immediately unless a title names your market's entities and
  plausibly moves the resolution criterion. Read full articles only
  after the title passes. Cap your own turns per day per market so a
  noisy news day degrades to skimming, not to a drained budget.

ALERT DISCIPLINE
- A missed breakout costs up to 10x a false alarm, so alert when a
  concrete, market-specific EVENT (decision, result, announcement —
  not commentary or speculation) implies the resolution criterion will
  move, and pick the direction from the criterion's wording, not from
  sentiment. Cite the single strongest article; never alert twice on
  the same story; never alert without a citation you have actually
  seen.

PACING
- Check get_costs at least once a day — fold it into a wake you already
  take, don't wake for it. Your spend should stay well under the
  penalties you can plausibly prevent: with a handful of
  breakouts per market per month, permanent fine-grained vigilance on
  every market cannot pay for itself — concentrate standing conditions
  on the few markets showing risk, and give quiet markets one cheap
  scheduled check a day.
<<< End of expert-curated skill
>>> Run-time updated skill - DYNAMIC, EDITABLE
<<< Run-time updated skill