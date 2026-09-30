"""Fixed baseline: silence. No claims, no news, no LLM, no schedule.

The floor anchor of the task: an agent that never acts scores exactly 0
(every decided question is a miss; precision has no claims to lose) and
spends nothing. Every other condition is read against it.
"""
