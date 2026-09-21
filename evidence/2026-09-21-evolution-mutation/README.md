# Evolution mutation evidence — 2026-09-21

This experiment compares identical ten-seed evolutionary manifests with policy
mutation rates 0 and 0.02. Selection remains payoff-proportional and population
size remains 100.

## Mutation rule

After selection and integer largest-remainder allocation, each offspring mutates
independently with the declared probability. A mutation chooses uniformly among
the other policies in the initial roster. A separate deterministic RNG stream
uses the replicate seed and generation, so mutation does not alter encounter
scheduling.

There is no crossover, learned mutation, or adaptive rate.

## Method

- ten base seeds
- ten generations per replicate
- 100 agents, initially 20 of each canonical policy
- 50 encounter shuffles per generation
- 10 rounds per match
- matched mutation rates 0 and 0.02

The mutation condition produced 217 policy mutations across all replicates and
generations.

## Final counts across ten seeds

Mean final count (minimum–maximum):

- Always Cooperate: 15.1 (13–17) at rate 0; 15.1 (11–18) at rate 0.02
- Always Defect: 9.5 (8–11) at rate 0; 10.8 (8–15) at rate 0.02
- Forgiving TFT: 25.5 (22–28) at rate 0; 25.9 (21–30) at rate 0.02
- Pavlov: 20.2 (20–21) at rate 0; 21.1 (16–23) at rate 0.02
- Tit-for-Tat: 29.7 (29–30) at rate 0; 27.1 (22–30) at rate 0.02

Mutation reduced TFT's mean dominance and widened every policy's range. It
slightly raised mean counts for Always Defect, forgiving TFT, and Pavlov while
leaving Always Cooperate's mean unchanged. The broad conditional-strategy
advantage remained, but final ordering varied more across seeds.

## Limits

Only one nonzero mutation rate and one initial composition were tested. No
policy went extinct in the ten-generation control, so reintroduction after
actual extinction remains a tested code capability rather than an observed
event in this evidence run. These results do not identify an optimal mutation
rate or equilibrium.

## Reproduce

```bash
for mode in control mutation-002; do
  uv run python -m spatial_ipd.evolution \
    --manifest "evidence/2026-09-21-evolution-mutation/$mode.json" \
    --jsonl "evidence/2026-09-21-evolution-mutation/$mode.jsonl" \
    --csv "evidence/2026-09-21-evolution-mutation/$mode.csv"
done
```
