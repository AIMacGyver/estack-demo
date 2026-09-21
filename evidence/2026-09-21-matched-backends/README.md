# Matched backend evidence — 2026-09-21

This is the first live, repeated-seed comparison produced by the experiment
runner. It is evidence, not a backend ranking or calibration claim.

## Method

- 8×8 lattice, 10 generations, mutation 0.02
- seeds 1, 2, and 3
- thinker generations 5 and 10
- one frontier seat per thinker call
- no sticky hold
- identical settings for plain imitation, seeded random control, Jev, and
  local `qwen3:8b`
- Qwen used `cluster_guard_v1`, temperature 0, 256 max tokens, and
  `reasoning_effort=none`

The original two-seat pilot produced the same schema failure twice: Qwen
returned one decision when two were required. The committed manifest uses one
seat per call and completed all 12 runs. This is a measured limitation of the
current local structured-output prompt, not silently discarded noise.

## Results

Across three seeds:

- Plain mean final cooperation: 0.01042
- Jev mean final cooperation: 0.01042
- Local LLM mean final cooperation: 0.01042
- Random-control mean final cooperation: 0.01563
- Plain/Jev/local trajectories were identical for every seed.
- Random matched plain on seeds 1 and 2. One threshold-clearing random hold on
  seed 3 raised final cooperation from 2/64 to 3/64 and slightly raised AUC.

Decision evidence:

- Jev applied zero holds. On the four C→D seats, resist Nouls were 0.37–0.38,
  below the 0.6 policy gate.
- Qwen applied zero holds. It returned `resist=false` at self-reported
  confidence 0.8 on all four C→D seats, equivalent to an oriented
  self-reported `P(resist=true)=0.2`.
- Both therefore agreed on the action while disagreeing in probability
  magnitude. Agreement does not establish calibration.

Operational evidence:

- Mean Jev backend call time was about 0.35 seconds.
- Mean local backend call time was about 3.20 seconds, roughly 9× Jev.
- All three final one-seat local runs were schema-valid; the earlier two-seat
  pilot was not.

## Limits

Three seeds and four C→D decisions are far below the checked-in reliability
threshold. The random improvement is one chance event, not evidence that random
is better. These records do not contain objective counterfactual labels, so
Brier/log-loss comparisons are not yet justified.

Next evidence should expand deterministic counterfactual motifs to include more
resolved positive and negative hold outcomes before rerunning reliability
reports.

## Reproduce

```bash
uv sync --extra typesafe
uv run python -m spatial_ipd.experiment \
  --manifest evidence/2026-09-21-matched-backends/manifest.json \
  --jsonl evidence/2026-09-21-matched-backends/runs.jsonl \
  --csv evidence/2026-09-21-matched-backends/summary.csv
```

Successful stable run IDs are skipped on rerun.
