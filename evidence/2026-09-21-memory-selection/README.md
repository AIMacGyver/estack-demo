# Memory under selection — 2026-09-21

This experiment couples the already-isolated memory-window wrapper with
payoff-proportional selection. Windows 1, 2, 4, and full history use the same
ten base seeds, canonical roster, schedules, rounds, and no-mutation rule.

## Method

- 100 agents: 20 each Always Cooperate, Always Defect, Forgiving TFT, Pavlov,
  TFT
- ten base seeds
- ten generations
- 50 encounter shuffles per generation
- 10 rounds per match
- mutation rate 0
- reputation and communication disabled
- memory windows 1, 2, 4, and full (`null`)

The wrapper truncates both own and opponent histories before delegating to the
existing policy. Encounter seeds remain `base + generation`, so pair schedules
stay matched across windows.

## Final counts across ten seeds

Mean final count (minimum–maximum):

| Policy | Window 1 | Windows 2 / 4 / full |
| --- | --- | --- |
| Always Cooperate | 2.9 (2–4) | 15.1 (13–17) |
| Always Defect | 57.4 (53–62) | 9.5 (8–11) |
| Forgiving TFT | 3.3 (2–4) | 25.5 (22–28) |
| Pavlov | 11.3 (9–14) | 20.2 (20–21) |
| Tit-for-Tat | 25.1 (22–28) | 29.7 (29–30) |

Windows 2, 4, and full history produced identical generation-policy tables
except for the recorded `memory_window` field. That matches the fixed-population
result: no canonical policy reads more than two prior rounds.

Window 1 makes forgiving TFT unable to see two consecutive defections, so it
behaves as Always Cooperate. Under selection that naive extra cooperation is
harvested: Always Defect becomes the majority, forgiving TFT nearly vanishes,
and mean population-weighted cooperation falls from 0.829 to 0.278.

The window-2 / full finals reproduce the earlier no-mutation evolution
evidence exactly, so adding the memory field did not disturb the default
full-history rule.

## Interpretation

Selection amplifies the same one-round memory failure seen in the fixed
population. Insufficient memory does not just lower one policy's payoff; it
changes frequencies until defectors dominate. Extra memory beyond the policy
requirement still adds nothing.

This depends on the current canonical rules, payoff-proportional selection,
and no mutation. It is not a general claim about optimal memory length.

## Reproduce

```bash
for mode in memory-1 memory-2 memory-4 memory-full; do
  uv run python -m spatial_ipd.evolution \
    --manifest "evidence/2026-09-21-memory-selection/${mode}.json" \
    --jsonl "evidence/2026-09-21-memory-selection/${mode}.jsonl" \
    --csv "evidence/2026-09-21-memory-selection/${mode}.csv"
done
```
