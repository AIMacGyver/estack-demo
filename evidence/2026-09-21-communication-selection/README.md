# Communication under selection — 2026-09-21

This experiment couples the already-isolated truthful warning mechanism with
payoff-proportional selection. The disabled and enabled conditions use the same
ten base seeds, initial composition, schedules, rounds, and no-mutation
selection rule.

## Method

- 100 agents: 25 each Always Cooperate, Always Defect, Communication Guard, TFT
- ten base seeds
- ten generations
- 50 encounter shuffles per generation
- 10 rounds per match
- mutation rate 0
- communication disabled versus enabled
- reputation disabled

Warnings reset at the start of each generation and accumulate across completed
encounters within that generation. Communication Guard cooperates when no report
exists or warning rate is below 0.5, and defects otherwise.

## Final counts across ten seeds

Mean final count (minimum–maximum):

- Always Cooperate: 1.7 (1–2) disabled; 24.6 (23–26) enabled
- Always Defect: 61.5 (57–67) disabled; 4.8 (3–6) enabled
- Communication Guard: 1.6 (1–2) disabled; 36.5 (35–38) enabled
- Tit-for-Tat: 35.2 (30–40) disabled; 34.1 (33–36) enabled

Without communication, Communication Guard behaves like Always Cooperate and
both are nearly eliminated while Always Defect becomes the majority.

With communication, guards avoid repeated exploitation. Always Defect contracts
to 3–6 agents in every replicate, Communication Guard becomes the largest
policy, and Always Cooperate persists because the cooperative environment is
less exploitative.

At generation 10, mean population-weighted cooperation rose from about 0.167 to
0.893. Mean total payoff per current agent also rose substantially in this
schedule (about 691 to 1,402).

## Comparison with reputation under selection

The matched reputation-selection experiment used the same seeds, size,
schedule, and no-mutation rule, swapping only the information channel and
guard. Disabled-control finals were identical. Enabled reputation produced
slightly fewer remaining defectors (3.6 vs 4.8) and slightly higher cooperation
(0.922 vs 0.893). These are close coupled-mechanism results, not a ranking of
the two channels.

## Interpretation

Selection changes the communication conclusion. In a fixed population, truthful
warnings protected informed agents but reduced cooperation and total payoff.
Under selection, that protection changes frequencies: persistent defectors lose
access to exploitable cooperators, warning-aware policies expand, and a
cooperative population becomes self-reinforcing.

This is a coupled-mechanism result, not evidence that communication always
improves evolution. It depends on truthful globally visible warnings, threshold
0.5, reset each generation, this roster, and payoff-proportional selection.

## Reproduce

```bash
for mode in disabled enabled; do
  uv run python -m spatial_ipd.evolution \
    --manifest "evidence/2026-09-21-communication-selection/communication-$mode.json" \
    --jsonl "evidence/2026-09-21-communication-selection/$mode.jsonl" \
    --csv "evidence/2026-09-21-communication-selection/$mode.csv"
done
```
