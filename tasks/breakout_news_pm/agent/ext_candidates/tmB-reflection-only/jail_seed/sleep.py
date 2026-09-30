"""Blind wait: edit WAKE_AT, then run_program(path="sleep.py")."""

import envkit

WAKE_AT = "2026-01-01T00:00:00Z"  # EDIT ME: when to wake
envkit.wait(WAKE_AT)
