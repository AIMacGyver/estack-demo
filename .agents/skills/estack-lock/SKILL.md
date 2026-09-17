---
name: estack-lock
description: >
  Lock an Estack job before coding: Goal, Scope (in/out), and a verifiable
  Done Condition. Use when starting a repo task, planning a PR, scoping a
  next slice, or the user says Estack / make a plan. Do not write application
  code in this skill.
---

# Estack lock

Token-bounded steward. No application edits until this card is written in
the chat. TypeSafe/Jev may score options; it does not write the lock.

The [Estack Grok Bot](https://x.ai/bot/R0acF6Pmp8YewSZm6fA-D) may hand work
to a Cloud Agent. **In this repo, stay in the current Cursor session** unless
the user asks for a Cloud Agent.

## Card (write all three)

**Goal** — one end state. Not steps, files, or APIs.

**Scope** — what is in, and explicit out. One PR. No surprise runtime deps.
Out unless the card names them: `simulate()`, payoffs, golden 2/144, pygame,
web, pandas, a new *required* package.

**Done Condition** — observable. Name the command (`uv run pytest` / `make test`)
and the artifact (flag, test, README line). Words like "better" or "polished"
are not Done.

## Rules

1. Show the card before any repo edit.
2. If Goal or Done is fuzzy, ask one question or state the tighter reading. Do not start a second job.
3. Jev can pick among candidates. Code still owns PD math. The lock is still written in this repo's words.
4. One job. "And also" is a later lock.
5. Stop after the card. Implement only if the user already said to ship in the same message — then follow `estack-ship`.
