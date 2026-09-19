# Analysis tooling decision

This spike selected an optional evidence stack for refactoring. None of these
metrics is a standing instruction to change code.

## Selected

- **Pyinstrument 5.1.3** — readable statistical call stacks for short local
  commands; supports Python 3.12 and async code.
- **Memray 1.20.0** — Python/native allocation and peak-memory reports; current
  macOS arm64 support.
- **pytest-benchmark 5.3.0** — repeatable before/after microbenchmarks integrated
  with the existing test stack.
- **Vulture 2.16** — conservative dead-code review at 100% confidence.
- **Radon 6.0.1** — cyclomatic complexity, Maintainability Index, and raw
  metrics. The release cadence is slow, but it supports this repo's Python 3.12
  runtime and produced useful output.
- **Complexipy 8.0.1** — actively maintained cognitive complexity as a second,
  human-readability-oriented signal.

All are development-only members of the `analysis` dependency group.

## Refactor gates

`.agents/skills/refactor-evidence/thresholds.toml` contains checked-in defaults
for evidence eligibility, repeated measurements, minimum improvement, and
cross-metric regression. The skill requires two corroborating signals and
defaults to a 10% median improvement ship gate.

Users may override a numeric gate only by naming the key, replacement value,
and workload-specific rationale in the Estack lock. Vague blanket permission
is invalid. Observable behavior and side-effect preservation cannot be
overridden under the refactoring skill.

## Deferred

- **py-spy 0.4.2** is excellent for attaching to an already-running process.
  This repository currently runs short, reproducible CLIs, so Pyinstrument is
  the more direct default.
- **Scalene 2.3.0** combines CPU, memory, and native/Python attribution, but
  overlaps the selected Pyinstrument and Memray workflow and brings a heavier
  dependency set. Add it only if line-level native/Python separation becomes a
  concrete need.
- **pyperf/ASV** provide more rigorous or historical benchmark orchestration.
  Start with pytest-benchmark; graduate when stable benchmark history or
  multi-environment tracking is required.

## Baseline observations

- Vulture reported no 100%-confidence dead code across `src` and `tests`.
- Radon's average cyclomatic complexity was **A (4.14)**.
- Radon highlighted `compare_local_ablations` (D/28), `reliability_report`
  (C/20), `viewer.run` (C/18), and `think_after_step` (C/16).
- Complexipy independently highlighted `compare_local_ablations` (51),
  `viewer.run` (26), and `think_after_step` (22).
- The agreement between complexity tools makes those functions reasonable
  review candidates. It does not prove that splitting them will improve speed,
  memory use, correctness, or maintainability.
- On one local 64×64 / 40-generation Pyinstrument run, `score_cells` accounted
  for about 0.59s of 0.83s total and `adopt_best` about 0.17s. This points to
  the engine loops—not the highest-complexity orchestration functions—as the
  measured CPU hotspot.
- Initial pytest-benchmark means on this machine were approximately 5.17ms for
  one 64×64 step, 8.26ms for final-grid analytics, and 154ms for a 30-generation
  simulation. These are a local baseline, not portable performance targets.
- Memray with Python allocator tracing confirmed the workload is small enough
  that import allocations are prominent; larger/repeated workloads are needed
  before making a memory refactor claim.

The next refactor should begin with CPU/memory/benchmark evidence for a real
workflow and choose only one corroborated hotspot.
