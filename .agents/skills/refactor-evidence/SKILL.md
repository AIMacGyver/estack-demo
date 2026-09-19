---
name: refactor-evidence
description: Profiles CPU and memory, benchmarks representative workloads, and reviews dead-code and complexity evidence before refactoring. Use when investigating performance, maintainability, hotspots, memory growth, or planning a measured refactor in this repository.
---

# Refactor from evidence

Use measurements to select one refactor. Metrics are leads, not instructions.
This skill is only for structural, maintainability, CPU, and memory changes.

## Non-overridable behavior boundary

Do not change observable behavior under this skill. This includes:

- public APIs, CLI flags/output, file formats, ordering, and exception behavior
- RNG seeds, streams, tie-breaking, thresholds, prompts, decisions, and defaults
- filesystem, network, subprocess, environment, or logging side effects
- dependencies visible to runtime users
- payoffs, `simulate()`, and the locked golden run

Capture at least one representative end-to-end command before editing and run
the identical command afterward. Compare exit status and complete observable
output/artifacts. Add characterization tests where intent is not already locked.
Any unexplained difference requires reverting the refactor. Behavioral work must
use a separate task, branch, and skill; an Estack lock cannot override this rule.

## Default evidence gates

Read [`thresholds.toml`](thresholds.toml) before writing the Estack lock. By
default:

- require 2 independent corroborating signals
- investigate CPU/allocation sites at or above 10% share
- review cyclomatic complexity at or above 11, or cognitive complexity at or above 20
- repeat baseline and after measurements at least 3 times
- ship only with at least 10% median improvement
- reject regressions above 5% in another tracked metric

If eligibility gates are not met, record the evidence and move on without
editing code. Complexity alone is never enough.

Users may override numeric defaults, but the request must name each threshold,
provide its numeric replacement, and explain why it fits the workload. Copy the
override into the Estack lock. Vague permission such as “YOLO,” “use your
judgment,” “ignore the gates,” or “optimize whatever” is invalid. Ask for
specific values instead. The behavior boundary is never overridable.

Valid override format:

```text
Override min_median_improvement from 0.10 to 0.07 because this
platform-bound workload has three-run median noise below 0.01.
```

## Workflow

1. Read `thresholds.toml`; write its applicable gates and any valid overrides
   into the Estack Goal / Scope / Done card.
2. State one hypothesis and choose a representative command or benchmark.
3. Run `uv run pytest` and the representative end-to-end command before changing code.
4. Gather only the evidence relevant to the hypothesis.
5. Select one hotspot only when the eligibility gates pass.
6. Make the smallest behavior-preserving change.
7. Re-run correctness, end-to-end behavior, and the exact same measurement.
8. Revert if behavior differs or ship gates fail; otherwise report the proof.

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
