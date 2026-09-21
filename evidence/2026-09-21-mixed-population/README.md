# Mixed canonical population — 2026-09-21

This API-free run establishes deterministic identified-agent scheduling and
per-policy accounting before adding selection, memory limits, reputation, or
communication.

## Method

- two agents of each canonical policy
- 50 seeded encounter shuffles
- five pair matches per encounter
- 10 rounds per match
- 100 matches and 1,000 played rounds per policy
- no reproduction, elimination, mutation, or policy adaptation

The same manifest produces byte-identical JSONL and CSV artifacts. Changing the
seed changes encounter order.

## Results

Average payoff per played round:

1. Tit-for-Tat: 2.622
2. Always Defect: 2.564
3. Forgiving TFT: 2.472
4. Pavlov: 2.425
5. Always Cooperate: 2.370

Cooperation rates:

- Always Cooperate: 1.000
- Pavlov: 0.885
- Tit-for-Tat: 0.838
- Forgiving TFT: 0.808
- Always Defect: 0.000

Tit-for-Tat had the highest payoff and the most mutual-cooperation rounds
(820). Always Defect outscored the other three cooperative conditional
strategies in this finite schedule, but did not rank first.

## Limits

This is descriptive evidence for one population composition and pairing seed.
Policies do not reproduce, remember beyond their built-in rule, observe
reputation, or communicate. Payoff rank is not evolutionary fitness until a
selection mechanism is explicitly defined and separately tested.

The next ready mechanism is bounded memory: the arena already passes complete
history, so memory windows can be isolated without adding population selection.

## Reproduce

```bash
uv run python -m spatial_ipd.population \
  --manifest evidence/2026-09-21-mixed-population/manifest.json \
  --jsonl evidence/2026-09-21-mixed-population/encounters.jsonl \
  --csv evidence/2026-09-21-mixed-population/summary.csv
```
