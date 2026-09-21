# Judged arena — 2026-09-21

This run puts a backend-chosen action into the fixed population. It does not
add selection, reputation, or communication. The committed backend is seeded
random. Jev and the local model use the same manifest and the same 0.5 gate;
tests cover those clients without a live service.

## Rule

Each round, a judged agent receives the round index, both action histories, and
any reputation or warning. Those signals are null here. The backend returns
one cooperate Noul. Code cooperates when that Noul is at least 0.5 and defects
otherwise.

Random draws Uniform[0, 1] and uses the draw as both the Noul and the recorded
confidence. A local model must return a boolean `cooperate` plus a separate
confidence. The boolean becomes 1 or 0 before the same gate. Confidence does
not choose the action. Jev's Noul is the probability and the confidence.

## Method

- 8 agents: 2 each Always Cooperate, Always Defect, Tit-for-Tat, and judged
- seed 20260921
- 20 encounter shuffles
- 10 rounds per match
- no memory limit, reputation, communication, or selection

Pair order depends on the seed, not on which backend answers.

## Random result

Average payoff per round and cooperation rate:

- Always Defect: 3.020, cooperation 0.000
- Judged (random): 2.505, cooperation 0.5325
- Tit-for-Tat: 2.130, cooperation 0.595
- Always Cooperate: 1.4625, cooperation 1.000

The random gate cooperates on about half its actions, so it is exploited by
Always Defect and does not match Tit-for-Tat. That is the null behavior. A
later live Jev or local run on this manifest can be compared with these
numbers. This single seed is not a ranking of the backends.

## Reproduce

```bash
uv run python -m spatial_ipd.judged \
  --manifest examples/judged.json \
  --backend random \
  --jsonl evidence/2026-09-21-judged-arena/random.jsonl \
  --csv evidence/2026-09-21-judged-arena/random.csv \
  --decisions evidence/2026-09-21-judged-arena/random-decisions.jsonl
```
