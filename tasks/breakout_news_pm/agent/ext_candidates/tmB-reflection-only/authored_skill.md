# Waiting by program

You have no sleep tool. Sim time passes only inside `run_program`: you run a
Python program from your workspace, and the program decides — in code — when
to return control to you.

The contract:

- A program observes the world ONLY through `envkit`'s fetch functions; each
  call is billed exactly like the same-named tool and consumes the same rate
  limits.
- `envkit.wait(until)` is the only way sim time advances, and it is free.
  Your program states its real wake times here — poll cadences, event
  times, backoffs are all just datetimes you compute and wait to.
- `envkit.handover(payload)` ends the program; `payload` comes back as the
  `run_program` result. A due scheduled trigger also ends it, as does the
  `until` deadline of `run_program` — a hard timeout backstop (default:
  experiment end), not where your wake logic lives.
- Local compute and file IO in your workspace are free: they happen at a
  frozen sim instant.

Your workspace ships three files — read before you write:

1. `envkit.py` — the exact API: one function per tool, each with its price.
2. `example_gatekeeper.py` — a runnable watcher template for this task.
3. `sleep.py` — the minimal program, a blind wait with the wake time in
   code:

   ```python
   import envkit

   WAKE_AT = "2026-01-01T00:00:00Z"  # EDIT ME: when to wake
   envkit.wait(WAKE_AT)
   ```

   `edit_file` the `WAKE_AT` line, then `run_program(path="sleep.py")` —
   the plain-sleep floor, no `until` needed. A real gatekeeper instead
   loops: fetch, decide, and `envkit.wait(envkit.now() + <interval>)`
   until something is worth a `handover`.

The working loop:

1. `read_file` `envkit.py` and `example_gatekeeper.py` first.
2. `write_file` / `edit_file` your own gatekeeper: fetch cheaply (window
   date filters to the interval since your last poll), loop on
   `envkit.wait`, and `handover` a compact payload once something needs
   your judgment. Persist anything bulky to workspace files.
3. `run_program(path, until, validate=true)` dry-runs it at zero cost and
   zero sim time, surfacing errors; then run it without `validate` to wait
   for real.

While a program runs you spend no LLM tokens: delegating routine watching to
code and saving your own turns for judgment is what the mechanism is for.
