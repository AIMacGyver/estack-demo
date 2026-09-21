# Reputation under selection — 2026-09-21

This experiment couples the already-isolated cumulative-reputation mechanism
with payoff-proportional selection. The disabled and enabled conditions use the
same ten base seeds, initial composition, schedules, rounds, and no-mutation
selection rule.

## Method

- 100 agents: 25 each Always Cooperate, Always Defect, Reputation Guard, TFT
- ten base seeds
- ten generations
- 50 encounter shuffles per generation
- 10 rounds per match
- mutation rate 0
- reputation disabled versus enabled

Reputation resets at the start of each generation and accumulates across
completed encounters within that generation. Reputation Guard cooperates with
unknown/reputable opponents and defects below cumulative cooperation 0.5.

## Final counts across ten seeds

Mean final count (minimum–maximum):

- Always Cooperate: 1.7 (1–2) disabled; 23.3 (21–25) enabled
- Always Defect: 61.5 (57–67) disabled; 3.6 (3–5) enabled
- Reputation Guard: 1.6 (1–2) disabled; 38.7 (36–40) enabled
- Tit-for-Tat: 35.2 (30–40) disabled; 34.4 (33–36) enabled

Without reputation, Reputation Guard behaves like Always Cooperate and both are
nearly eliminated while Always Defect becomes the majority.

With reputation, guards avoid repeated exploitation. Always Defect contracts to
3–5 agents in every replicate, Reputation Guard becomes the largest policy, and
Always Cooperate persists because the cooperative environment is less
exploitative.

At generation 10, mean population-weighted cooperation rose from about 0.167 to
0.922. Mean total payoff per current agent also rose substantially in this
schedule (about 691 to 1,429), unlike the fixed-composition reputation run where
information reduced welfare by replacing exploitation with D/D.

## Interpretation

Selection changes the reputation conclusion. In a fixed population, reputation
protected informed agents but reduced cooperation and total payoff. Under
selection, that protection changes frequencies: persistent defectors lose
access to exploitable cooperators, conditional/reputation-aware policies expand,
and a cooperative population becomes self-reinforcing.

This is a coupled-mechanism result, not evidence that reputation always improves
evolution. It depends on truthful cumulative reputation, threshold 0.5, reset
each generation, this roster, and payoff-proportional selection.

## Reproduce

```bash
for mode in disabled enabled; do
  uv run python -m spatial_ipd.evolution \
    --manifest "evidence/2026-09-21-reputation-selection/reputation-$mode.json" \
    --jsonl "evidence/2026-09-21-reputation-selection/$mode.jsonl" \
    --csv "evidence/2026-09-21-reputation-selection/$mode.csv"
done
```
