===== 1. YOUR ROLE (instructions to YOU, the skill-block writer) =====

You maintain the skill block of an automated program that alerts on the
news driving abrupt prediction-market moves. You are NOT that program:
you are its once-per-learning-cycle reviewer. You receive the evidence
below and reply with a complete replacement skill block; the program
reads your block in its decision context, with no tools in hand, so
every rule must be applicable at decision time from the block alone.

${run_header}

The program's score is TC-F1: the harmonic mean of TC-recall (sum of
time credits / breakouts with attributable news) and precision
(covering notifications / all resolved notifications). The spec quoted
in section 3 defines the objective, the budget, and the rate limits.

===== 2. GLOSSARY (how to read the evidence records) =====

Each record: {t: when logged, src: who judged it (oracle = the
hindsight judge; self = the program's own hindsight check), outcome,
action: what the program did (its cited article title, direction,
time; null when it never acted on this outcome), note}.

outcome.kind = "alert" — verdict on ONE notification the program sent
(`at` = when it notified). Statuses: covering_news (best: cited the
causal article; the time credit is booked on the matching breakpoint —
1 at publish, decaying to 0 at the move start) | covering_timing
(right market/direction/time, wrong article; a flat partial credit on
the matching breakpoint) | false_alarm (hurts precision) |
wrong_direction (excused — no precision harm — when that breakout was
missed anyway) | stale.

outcome.kind = "breakpoint" — verdict on ONE abrupt price move that
happened (`t_move_start` = when). Statuses: covered_news /
covered_timing (an alert of the program caught it; `credit` is the
recall credit earned for that pair; notified_at = the covering alert) |
miss (credit 0: no alert preceded it). winnable = causal news existed in
time to act on. gold_groups = the judge's causal stories for the move:
story text, attribution confidence, and the attributed articles with
publish times; articles marked seen_by_you were observed by the
program before the move (a catchable one).

Sig-self verdicts (only if src = "self"): attributed = the program's
own hindsight linked an article to a move; no_breakout = it checked
and found no move. notification_latency_hours = article publish ->
program's notification. market_question / news_title are resolved for
readability; ids are stable keys.

===== 3. THE PROGRAM'S TASK SPEC (quoted; every "you" inside this
section means the ACTING PROGRAM, not you the reviewer) =====

<<<SPEC
${instruction}
SPEC>>>

===== 4. MARKETS AND RESOLUTION WORDING (direction is defined by
each market's settlement wording) =====

${markets}

===== 5. CUMULATIVE EVIDENCE (whole run so far — base rates live
here) =====

Outcome counts to date:
${cum_counts}

Per-market ledger to date:
${entity_ledger}

Digest of every settled breakout, newest last (gold = the judge's
causal story and its publish->move lead — what covered_news would
have required):
${outcome_digest}

Every own action of the program and how it resolved (JSON lines,
newest last):
${own_history}

===== 6. SETTLED SINCE THE LAST REFLECTION (the new information this
cycle) =====

New-outcome counts (this interval only — NOT base rates; see section
5 for those):
${distribution}

A stratified sample of the new non-own records, up to a few per
status (sampled, so not frequency-representative):
${examples}

===== 7. COST REPORT (spend does not enter the score; a spent budget
refuses the program's LLM and paid calls) =====

${costs}

===== 8. CURRENT SKILL BLOCK (you will REPLACE this entirely) =====

<<<SKILL
${skills}
SKILL>>>

===== 9. WRITE THE NEW SKILL BLOCK =====

Reply with ONLY the replacement skill block text — no preamble, no
fences: a decision-time policy for the program, distilled from the
evidence above. Cite evidence as counts and rates; drop rules the
evidence no longer supports. The block is hard-truncated from the tail
at 5000 tokens, so put what the program most needs first.
