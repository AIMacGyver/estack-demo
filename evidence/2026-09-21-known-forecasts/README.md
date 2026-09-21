# Known-outcome forecasts — 2026-09-21

This run scores three backends on committed arena facts. It does not use
lattice ties, and it does not relabel them.

## Catalog

Sixty yes/no statements, thirty true and thirty false:

- Reputation selection, final Always Defect count below 10. Enabled is true
  (the committed counts are 3–5). Disabled is false (57–67).
- Communication selection, the same below-10 question. Enabled is true (3–6).
  Disabled is false (57–67).
- Memory selection, final Always Defect count above 40. Window 1 is true
  (53–62). Full memory is false (8–11).

Each item is one replicate seed from those tables. The prompt state carries
the mechanism, condition, seed, roster, and the question. It does not carry
the final count, payoff, or cooperation rate. Code attaches the label after
the backend answers.

## Scores

`calibration_claim_supported` is true because the sample has 60 resolved
events and both classes. That flag means the sample is large enough to
compare. It does not mean a backend is calibrated. Random is the null.
A score closer to Jev is agreement with Jev, not proof of calibration.

| Backend | Brier | Log loss | Decisions matching the label |
| --- | ---: | ---: | ---: |
| Random, seed 1 | 0.3269 | 0.9641 | 29/60 |
| Jev | 0.1804 | 0.5449 | 44/60 |
| Local `qwen3:8b` | 0.2900 | 0.7803 | 30/60 |

Random draws Uniform[0, 1] as P(true). Its class means sit near one half
(0.494 on true items, 0.473 on false items).

Jev returns a Noul for “the statement is true.” Mean P(true) is 0.490 on true
items and 0.284 on false items. Every false item was decided false. Fourteen
of the thirty true items cleared 0.5.

The local model returns a boolean plus a separate self-reported confidence.
P(true) is that confidence when the boolean is true, and one minus the
confidence when it is false. On this catalog every answer was true at
confidence 0.7, so every probability is 0.7. The self-report does not move
with the label.

## Reproduce

```bash
uv run python -m spatial_ipd.forecast \
  --backend random --seed 1 \
  --jsonl evidence/2026-09-21-known-forecasts/random.jsonl

uv run python -m spatial_ipd.forecast \
  --backend jev \
  --jsonl evidence/2026-09-21-known-forecasts/jev.jsonl

uv run python -m spatial_ipd.forecast \
  --backend local --local-model qwen3:8b --local-reasoning-effort none \
  --jsonl evidence/2026-09-21-known-forecasts/local.jsonl
```

Live Jev and local answers can change between runs. The random file is
deterministic for seed 1.
