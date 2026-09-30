${instruction}

${block}

# Decide now

Market question: ${question}

News published since your last scan:
${items}

Apply the economics above and decide which items, if any, to notify as news
that will DRIVE an abrupt price move in this market. A notification claims:
this market starts a breakout in the stated direction within the claim
window — judge now, at publication time; anything at or after the move
start earns nothing. Unrelated or echo/secondary coverage should not be
alerted, and a correct story with the wrong direction books a false alarm.

Reply with ONLY a JSON object:
{"alerts": [{"news_id": "...", "direction": "up" | "down"}, ...]}.
An empty list is a valid decision.
