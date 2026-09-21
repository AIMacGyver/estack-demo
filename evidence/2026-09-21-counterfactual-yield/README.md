# Counterfactual label-yield spike — 2026-09-21

This spike tested whether naturally occurring C→D imitation events can supply
the resolved positive and negative labels required for backend calibration.
It did not change label semantics or application code.

## Context

The [matched backend run](../2026-09-21-matched-backends/README.md) produced
action agreement but no objective correctness labels. The checked-in
reliability gate requires at least 50 resolved events and both outcome classes.

The existing target branches a post-imitation grid into:

- **imitate** — leave the focal C→D result at D
- **hold** — restore that focal cell to C once

Both branches then run the same horizon with identical RNG. A label is positive
when hold raises mean future cooperation, negative when it lowers it, and
unresolved when outcomes tie exactly.

## Deterministic yield

For seeds 1–30 on 8×8 lattices over 10 generations with a three-generation
horizon:

- mutation 0.00: 963 C→D events; 0 positive, 0 negative, 963 ties
- mutation 0.02: 1,306 C→D events; 0 positive, 0 negative, 1,306 ties

A separate seeded sample of 3,000 random 5×5 states selected one C→D seat per
state:

- 0 positive, 0 negative, 3,000 ties

The existing handcrafted `pivotal_cluster_hold` remains a valid positive
example, but it is rare and selected rather than naturally sampled.

A bounded 50,000-state search for a harmful hold was stopped after roughly 100
seconds without finding a negative example. This does **not** prove that a
negative is impossible; it shows that brute-force random search is not an
efficient next slice.

## Jev judgment on exact ties

Jev evaluated three policies for exact counterfactual ties:

- keep unresolved: probability 0.43
- separate no-effect class: probability 0.40
- label negative: probability 0.17

The Noul for “a no-benefit tie is valid evidence resist should be false” was
0.29. Treating ties as negative received a scientific-usefulness score of
0.21/3 with 0.84 probability on the lowest level.

Code therefore keeps ties unresolved. Relabeling them as negatives would create
volume by changing the target, not by collecting evidence.

## Conclusion

The current one-shot focal hold does not yield a natural binary calibration
dataset at this scale. Brier/log-loss comparisons remain diagnostic only.

Next work should move to the larger active-agent roadmap and revisit
calibration only after a mechanism creates meaningful action/outcome variation.
Memory, mixed populations, reputation, and communication are candidates;
Jev should rank them against current architecture and evidence.
