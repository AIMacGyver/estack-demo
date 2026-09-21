# Payoff-proportional evolution evidence — 2026-09-21

This experiment adds deterministic selection to the canonical mixed population.
It does not add mutation, crossover, learning, reputation, or communication.

## Rule

Each generation runs a fresh seeded mixed-population evaluation. A policy's
expected offspring share is its total payoff divided by total population
payoff. Largest remainder converts expected shares to integer counts while
keeping population size fixed; policy name breaks exact remainder ties.

Policies with zero allocated offspring become extinct and cannot return because
this slice has no mutation.

## Method

- initial population: 100 agents, 20 of each canonical policy
- 10 declared base seeds (20260921–20260930)
- 10 generations
- 50 encounter shuffles per generation
- 10 rounds per match
- generation seed = base seed + generation index
- no memory limit or information mechanisms

A 10-agent pilot rounded every policy back to two agents each generation. The
committed evidence uses 100 agents so proportional differences survive integer
allocation.

## First replicate trajectory

Counts after each generation:

- Always Cooperate: 20, 19, 18, 17, 16, 16, 15, 15, 15, 15
- Always Defect: 19, 19, 18, 17, 16, 14, 13, 11, 10, 8
- Forgiving TFT: 21, 21, 22, 23, 24, 25, 26, 27, 27, 28
- Pavlov: 20 in every generation
- Tit-for-Tat: 20, 21, 22, 23, 24, 25, 26, 27, 28, 29

First-replicate final population:

- Tit-for-Tat: 29
- Forgiving TFT: 28
- Pavlov: 20
- Always Cooperate: 15
- Always Defect: 8

Conditional reciprocal strategies expanded while unconditional strategies
contracted. Always Defect's strong one-generation payoff in the original fixed
composition did not persist as a long-run frequency advantage under
frequency-dependent matching.

## Ten-seed final frequencies

Mean final count (minimum–maximum):

- Tit-for-Tat: 29.7 (29–30)
- Forgiving TFT: 25.5 (22–28)
- Pavlov: 20.2 (20–21)
- Always Cooperate: 15.1 (13–17)
- Always Defect: 9.5 (8–11)

The ordering was stable across these ten schedule sequences. Tit-for-Tat and
forgiving TFT expanded in every replicate; Always Cooperate and Always Defect
contracted; Pavlov remained nearly unchanged.

## Limits

This is one initial composition, population size, selection rule, and
no-mutation horizon. Largest-remainder selection is deterministic but not a
biological claim. Other compositions, longer runs, and mutation are needed
before discussing stable equilibria or fixation.

## Reproduce

```bash
uv run python -m spatial_ipd.evolution \
  --manifest evidence/2026-09-21-evolution/manifest.json \
  --jsonl evidence/2026-09-21-evolution/generations.jsonl \
  --csv evidence/2026-09-21-evolution/generations.csv
```
