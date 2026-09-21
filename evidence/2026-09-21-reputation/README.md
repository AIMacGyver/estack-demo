# Cumulative-reputation evidence — 2026-09-21

This experiment isolates one identity-level reputation signal while preserving
the population, seed, pair schedule, rounds, and payoffs.

## Mechanism

Each agent's reputation is its cumulative cooperation fraction across completed
encounters. Reputation is fixed during a match and updated afterward.

`reputation_guard`:

- cooperates when an opponent is unknown
- cooperates when opponent reputation is at least 0.5
- defects below 0.5

The disabled control passes no reputation, making the guard cooperate
unconditionally. Other policies ignore the optional signal.

## Method

- two agents each: Always Cooperate, Always Defect, Reputation Guard, TFT
- 50 seeded encounter shuffles
- 200 matches, 10 rounds per match
- identical disabled/enabled pair schedules
- no memory limit, selection, mutation, or communication

## Results

Reputation Guard:

- cooperation rate: 1.000 → 0.790
- average payoff per round: 2.340 → 2.550

Always Defect:

- cooperation rate: 0.000 in both runs
- average payoff per round: 3.128 → 2.288

Always Cooperate and Tit-for-Tat summaries were unchanged.

The mechanism protected reputation-aware agents from repeated exploitation and
substantially reduced the defector payoff. It did not reform Always Defect.
Replacing C/D exploitation with D/D punishment also reduced total population
payoff from 9,956 to 9,326 and lowered mean cooperation across policy groups
from 0.678 to 0.626.

## Limits

This is one cumulative metric, threshold, composition, and schedule seed.
Reputation is truthful code-owned history; there is no gossip, deception,
forgetting, or strategic reputation management. The evidence supports a causal
mechanism result for this roster, not a universal claim that reputation
increases cooperation or welfare.

## Reproduce

```bash
for mode in disabled enabled; do
  uv run python -m spatial_ipd.population \
    --manifest "evidence/2026-09-21-reputation/reputation-$mode.json" \
    --jsonl "evidence/2026-09-21-reputation/encounters-$mode.jsonl" \
    --csv "evidence/2026-09-21-reputation/summary-$mode.csv"
done
```
