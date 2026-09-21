# Noisy communication under selection — 2026-09-21

This experiment keeps the truthful-warning rule and flips each published bit
with a fixed probability. Rate 0 and rate 0.25 use the same ten base seeds,
roster, schedules, rounds, and no-mutation selection rule as the clean
communication-selection run.

## Mechanism

After each match, both participants still compute the truthful warning: warn
when the opponent's cooperation rate was below 0.5. When
`communication_error_rate` is positive, each published bit is inverted if an
independent draw falls below that rate. Pair shuffling uses the encounter RNG.
Warning noise uses `Random(seed * 1000003 + 17)` and is not drawn at rate 0.

`warning_about_*` is the bit that entered the warning store.
`truthful_warning_about_*` is the pre-noise bit. Rate 0 leaves them equal.
The omitted field defaults to 0, so earlier communication evidence is unchanged.

## Method

- 100 agents: 25 each Always Cooperate, Always Defect, Communication Guard, TFT
- ten base seeds
- ten generations
- 50 encounter shuffles per generation
- 10 rounds per match
- mutation rate 0
- reputation disabled
- communication enabled
- error rates 0 and 0.25

On the initial 100-agent generation at seed 20260921, 1,261 of 5,000 published
reports flipped (realized rate 0.252).

## Final counts across ten seeds

Mean final count (minimum–maximum):

- Always Cooperate: 24.6 (23–26) at rate 0; 25.9 (24–28) at rate 0.25
- Always Defect: 4.8 (3–6) at rate 0; 16.2 (10–19) at rate 0.25
- Communication Guard: 36.5 (35–38) at rate 0; 24.5 (23–26) at rate 0.25
- Tit-for-Tat: 34.1 (33–36) at rate 0; 33.4 (31–36) at rate 0.25

Rate 0 reproduces the clean communication-selection finals exactly.

At generation 10, mean population-weighted cooperation fell from 0.893 to
0.677. Mean total payoff per current agent fell from about 1,402 to 1,212.

## Interpretation

A 25% independent flip rate does not erase the communication effect. Always
Defect stays well below the no-communication majority (61.5) and cooperation
stays well above 0.167. It does weaken the channel: guards shrink, defectors
roughly triple, and their replicate range widens from 3–6 to 10–19.

False warnings and missed warnings both matter. Guards sometimes defect against
cooperators and sometimes keep cooperating with defectors. Selection then
rewards the defectors who slip through.

This is one symmetric error rate on globally visible boolean warnings. It is
not strategic lying, sender reputation, or a claim about an optimal noise level.

## Reproduce

```bash
for mode in error-0 error-025; do
  uv run python -m spatial_ipd.evolution \
    --manifest "evidence/2026-09-21-noisy-communication/${mode}.json" \
    --jsonl "evidence/2026-09-21-noisy-communication/${mode}.jsonl" \
    --csv "evidence/2026-09-21-noisy-communication/${mode}.csv"
done
```
