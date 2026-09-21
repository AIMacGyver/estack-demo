# Guard tournament — 2026-09-21

This experiment ranks the five canonical arena policies plus both information
guards. It does not change frequencies. Information is play-derived: the
informed pass replays the same pairings after first-pass cooperation becomes
each identity's reputation, and a warning is published iff that cooperation
was below 0.5.

## Method

- roster: Always Cooperate, Always Defect, TFT, Pavlov, Forgiving TFT,
  Reputation Guard, Communication Guard
- 20 rounds per match
- every unordered pair, including self-play (28 matches per pass)
- uninformed pass: no reputation or warning signals
- informed pass: constant identity signals from the uninformed pass

Default `python -m spatial_ipd.arena` remains the canonical five-policy
round robin.

## Rankings

Uninformed total payoff (rank):

1. Always Defect — 452
2. Tit-for-Tat — 439
3. Forgiving TFT — 438
4. Pavlov — 430
5. Always Cooperate — 420
6. Communication Guard — 420
7. Reputation Guard — 420

Without signals both guards are unconditional cooperators and share last place
with Always Cooperate. Always Defect wins.

Informed total payoff (rank):

1. Communication Guard — 440
2. Reputation Guard — 440
3. Tit-for-Tat — 439
4. Forgiving TFT — 438
5. Pavlov — 430
6. Always Cooperate — 420
7. Always Defect — 292

Always Defect is the only identity with first-pass cooperation below 0.5, so
it is the only warned / low-reputation opponent. Guards defect from the first
informed round against it and move from last to first. Always Defect's payoff
falls from 452 to 292.

Reciprocal policies are unchanged: they never read the signals. Always
Cooperate remains exploitable because it also ignores them.

## Interpretation

A static tournament can hide the same information effect that selection
amplifies. Uninformed ranks repeat the Axelrod-style finding that defection
wins against naive cooperators. Informed ranks show that identity-level
truthful signals let the guards refuse that exploitation without changing
payoffs or pairing.

This is a two-pass pairwise result, not an evolving population. Signals are
global identity averages from the first pass, not within-match updates, local
messages, or lies.

## Reproduce

```bash
uv run python -m spatial_ipd.arena \
  --rounds 20 --roster guards --information off \
  --jsonl evidence/2026-09-21-guard-tournament/uninformed.jsonl \
  --csv evidence/2026-09-21-guard-tournament/uninformed.csv

uv run python -m spatial_ipd.arena \
  --rounds 20 --roster guards --information on \
  --jsonl evidence/2026-09-21-guard-tournament/informed.jsonl \
  --csv evidence/2026-09-21-guard-tournament/informed.csv
```
