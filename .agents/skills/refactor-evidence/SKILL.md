---
name: refactor-evidence
description: Profiles CPU and memory, benchmarks representative workloads, and reviews dead-code and complexity evidence before refactoring. Use when investigating performance, maintainability, hotspots, memory growth, or planning a measured refactor in this repository.
---

# Refactor from evidence

Use measurements to select one refactor. Metrics are leads, not instructions.
Do not change behavior, thresholds, payoffs, `simulate()`, or the golden run
unless the Estack lock explicitly allows it.

## Workflow

1. Write the Estack Goal / Scope / Done card.
2. State one hypothesis and choose a representative command or benchmark.
3. Run `uv run pytest` before changing code.
4. Gather only the evidence relevant to the hypothesis.
5. Select one hotspot where runtime evidence and maintainability evidence agree.
6. Make the smallest behavior-preserving change.
7. Re-run correctness and the exact same measurement.
8. Report before/after values, variance, and anything that did not improve.

Install tools with:

```bash
uv sync --group analysis
mkdir -p .profiles
```

## Static evidence

Run conservative dead-code analysis. Do not delete a Vulture finding without
checking dynamic imports, CLI entry points, tests, and public APIs.

```bash
uv run --group analysis vulture src tests --min-confidence 100
```

Review cyclomatic complexity and maintainability:

```bash
uv run --group analysis radon cc src -s -a
uv run --group analysis radon mi src -s
```

Use cognitive complexity as a second perspective. The threshold of 20 is a
review queue, not a required mass cleanup:

```bash
uv run --group analysis complexipy src --max-complexity-allowed 20
```

Prefer candidates corroborated by tests, profiling, duplication, or frequent
change. A complexity score alone does not authorize a refactor.

## CPU evidence

Pyinstrument is the default for this repo's short CLI workloads:

```bash
uv run --group analysis pyinstrument \
  -r html -o .profiles/cpu.html \
  .agents/skills/refactor-evidence/scripts/profile_workload.py \
  --size 64 --generations 40 --repeats 3
```

Use text output for quick inspection by replacing `-r html -o ...` with
`-r text`. Profile a real user command when the fixed workload is not
representative.

`py-spy` is intentionally not installed. Add it only when profiling an
already-running or production-like process is the actual requirement.

## Memory evidence

Record allocations with Memray, then inspect summary and flame graph:

```bash
uv run --group analysis memray run \
  --trace-python-allocators \
  -o .profiles/memray.bin \
  .agents/skills/refactor-evidence/scripts/profile_workload.py \
  --size 64 --generations 40 --repeats 3

uv run --group analysis memray summary .profiles/memray.bin
uv run --group analysis memray flamegraph \
  -o .profiles/memray.html .profiles/memray.bin
```

Use a fresh output path or Memray's overwrite option for subsequent runs.
Track retained/peak allocations, not one transient allocation in isolation.

Scalene is intentionally deferred because its CPU/memory role overlaps the
selected Pyinstrument and Memray workflow. Reconsider it only when line-level
Python/native time separation is needed.

## Benchmarks

The benchmark suite is outside default `testpaths`; run it explicitly:

```bash
uv run --group analysis pytest benchmarks --benchmark-only \
  --benchmark-sort=mean
```

Save a baseline before editing and compare on the same quiet machine:

```bash
uv run --group analysis pytest benchmarks --benchmark-only \
  --benchmark-autosave
uv run --group analysis pytest benchmarks --benchmark-only \
  --benchmark-compare
```

Do not claim an improvement from a single noisy run. Check correctness,
median/mean, spread, input size, Python version, and machine conditions.

## Report

Return:

- Hypothesis and representative workload
- Correctness baseline
- CPU, memory, benchmark, dead-code, and complexity evidence actually used
- Selected hotspot and why other findings were deferred
- Smallest refactor made
- Before/after measurements using identical commands
- Test/Ruff results and remaining uncertainty
