# Bounded-memory population evidence — 2026-09-21

This experiment isolates within-match history length while holding population,
pair schedule, seed, rounds, payoffs, and policy implementations constant.

## Method

- same 10-agent canonical population as the first mixed run
- seed 20260921 and byte-identical pair schedules
- 50 encounter shuffles, 250 matches, 10 rounds per match
- memory windows 1, 2, 4, and full history
- no selection, mutation, reputation, or communication

The memory wrapper truncates both own and opponent histories before delegating
to the existing policy. It does not change policy code or arena payoffs.

## Results

Windows 2, 4, and full history produced identical policy summaries. This is
expected: no current canonical policy reads more than two prior rounds.

Window 1 changed forgiving TFT:

- cooperation rate: 0.808 → 1.000
- average payoff per round: 2.472 → 2.280

With only one visible round, forgiving TFT can never observe the two
consecutive defections required to retaliate, so it becomes behaviorally
equivalent to Always Cooperate.

That increased Always Defect's average payoff:

- 2.564 → 3.332

Always Cooperate, Pavlov, and Tit-for-Tat summaries were unchanged. The direct
memory effect is therefore localized and its exploitation consequence is
visible in the opponent aggregate.

## Interpretation

For this roster, one round of memory is insufficient for a policy whose rule
requires a two-round pattern. More than two rounds adds no value because no
policy consumes it. This does not establish a universal optimal memory length;
it validates the population mechanism and shows that memory must be interpreted
relative to policy requirements.

## Reproduce

```bash
for window in 1 2 4 full; do
  uv run python -m spatial_ipd.population \
    --manifest "evidence/2026-09-21-bounded-memory/memory-$window.json" \
    --jsonl "evidence/2026-09-21-bounded-memory/encounters-$window.jsonl" \
    --csv "evidence/2026-09-21-bounded-memory/summary-$window.csv"
done
```
