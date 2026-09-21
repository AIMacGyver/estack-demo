# Component-hold label yield — 2026-09-21

This spike asks whether restoring a focal cooperator's whole pre-imitation
component creates the missing negative class. It does not relabel ties or
change the one-cell hold.

## Intervention

On each imitation C→D seat, the one-cell hold sets only that seat back to C.
The component hold sets every cell in that seat's Moore-connected cooperating
component, measured on the pre-imitation lattice, back to C. Both branches
then advance for three generations with the same RNG. A higher mean future
cooperation is positive, a lower mean is negative, and an exact tie stays
unresolved.

## Sample

- seeds 1–30
- 8×8 lattices
- 10 generations
- horizon 3
- mutation rate 0

The one-cell recount is 963 events, 0 positive, 0 negative, 963 ties. That
matches the earlier [label-yield spike](../2026-09-21-counterfactual-yield/README.md).

## Component result

The same 963 events:

- positive: 63
- negative: 0
- tie: 900

Every positive event had a component of 27–39 cells (mean about 34). None was
a lone cooperator. Mean cooperation gains were small: 0.0052 to 0.0104, which
is one or two extra cooperating cells for part of the three-generation window
on a 64-cell lattice.

## Conclusion

Component restoration can produce real positive labels. It still produces no
negative labels in this sample, so the checked-in reliability gate remains
unmet. Ties stay unresolved. Calibration claims stay deferred.

## Reproduce

```bash
uv run python -c "from spatial_ipd.calibrate import count_hold_yield; print(count_hold_yield(seeds=tuple(range(1, 31)), intervention='cell')); print(count_hold_yield(seeds=tuple(range(1, 31)), intervention='component'))"
```
