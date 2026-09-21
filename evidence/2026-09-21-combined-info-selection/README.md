# Combined information under selection — 2026-09-21

This experiment asks whether the two already-isolated truthful channels are
redundant once frequencies adapt. Reputation-only, communication-only, and
both-on use the same ten base seeds, roster, schedules, rounds, and
no-mutation selection rule.

## Method

- 100 agents: 20 each Always Cooperate, Always Defect, Communication Guard,
  Reputation Guard, and TFT
- ten base seeds
- ten generations
- 50 encounter shuffles per generation
- 10 rounds per match
- mutation rate 0
- three conditions: reputation only, communication only, both enabled

Each information store resets at the start of a generation and accumulates
across completed encounters. The unused channel leaves its guard cooperating
unconditionally.

## Final counts across ten seeds

Mean final count (minimum–maximum):

| Policy | Reputation only | Communication only | Both |
| --- | --- | --- | --- |
| Always Cooperate | 12.4 (12–14) | 12.7 (12–15) | 20.9 (20–23) |
| Always Defect | 17.9 (15–20) | 24.1 (19–31) | 1.6 (1–2) |
| Communication Guard | 13.1 (12–15) | 26.7 (23–29) | 27.0 (26–29) |
| Reputation Guard | 29.9 (28–33) | 12.7 (9–15) | 27.2 (25–29) |
| Tit-for-Tat | 26.7 (25–28) | 23.8 (20–27) | 23.3 (22–25) |

A single channel protects its own guard and leaves the other guard as extra
unconditional-cooperate biomass. Defectors remain common: 15–20 under
reputation-only and 19–31 under communication-only.

Both channels together remove that exploitable pool. Always Defect contracts
to 1–2 agents in every replicate. The two guards persist at similar high
counts, Always Cooperate recovers, and mean population-weighted cooperation
rises from 0.688 / 0.609 to 0.963. Mean total payoff per current agent rises
from about 1,219 / 1,149 to 1,466.

## Interpretation

The channels are complementary under this roster, not interchangeable. Each
alone still leaves a naive cooperating type that defectors can harvest.
Enabling both makes naive cooperation safer because each identified defector
is visible on at least one channel used by a large informed class.

This is a coupled-mechanism result for truthful global reputation plus truthful
global warnings, threshold 0.5, generation-reset stores, this five-policy
roster, and payoff-proportional selection. It does not rank the channels in
general or test lying, local range, or memory.

## Reproduce

```bash
for mode in reputation-only communication-only both; do
  uv run python -m spatial_ipd.evolution \
    --manifest "evidence/2026-09-21-combined-info-selection/${mode}.json" \
    --jsonl "evidence/2026-09-21-combined-info-selection/${mode}.jsonl" \
    --csv "evidence/2026-09-21-combined-info-selection/${mode}.csv"
done
```
