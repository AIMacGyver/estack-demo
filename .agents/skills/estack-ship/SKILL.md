---
name: estack-ship
description: >
  Execute a locked Estack job locally in this repo: smallest diff, pytest,
  stop at Done. Use when implementing, shipping, or merging a scoped change.
  The human reviews the diff and test output. Do not spawn a Cloud Agent
  unless the user asks.
---

# Estack ship

Run the locked Goal / Scope / Done in this checkout. Verification over vibes.

## Before edits

1. If this turn has no lock, write the same three fields as `estack-lock`, then implement only that card.
2. Do the work here. Do not spawn a Cursor Cloud Agent unless the user asks.
3. Do not open a next slice or ask Jev what to build next unless the user asked.

## Execute

- Smallest diff that meets Done.
- No new required dependencies. Optional extras only (`viewer`, `typesafe`).
- Do not change `simulate()`, payoffs, or the golden 12×12 / 30 gens / seed `20260316` / mutation `0.02` / final `2/144` unless the lock says so.
- TypeSafe tests use a fake client. Do not commit `.env`.
- Jev judges; code owns PD math.

## Prove

```bash
uv run pytest
```

Same via `make test`. If you touched linted Python, also `uv run ruff check` on those files.

## Stop

Done is met, or you report the blocker. Cite the test output. Do not stack the next job.

The review step is the human looking at the diff and that output — not a Cloud Agent.
