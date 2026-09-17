# Spatial IPD

A small **Spatial Iterated Prisoner's Dilemma** engine (Nowak & May 1992): cooperation evolving on a 2-D lattice. This repo is an Estack dry-run demo — clone it, run the tests, export a cooperation-rate CSV, optionally open a Pygame window. The engine is stdlib-only; pygame is an optional extra.

**Share / clone:** [github.com/aimacgyver/estack-demo](https://github.com/aimacgyver/estack-demo)

## Try it

Python 3.12+ and [uv](https://docs.astral.sh/uv/). From a clone of this repo:

### 1. Install and test

```bash
uv sync --group dev
uv run pytest
```

Same thing via `make sync` and `make test`.

### 2. Headless CSV export

No display required. Writes `generation,cooperation_rate` (generation `0` is the initial lattice) and prints one summary line. This seed is the locked golden (final cooperation `2/144`):

```bash
uv run python -m spatial_ipd.export \
  --height 12 --width 12 --generations 30 \
  --seed 20260316 --mutation-rate 0.02 \
  --out coop_rates.csv
```

### 3. Optional viewer

Needs a display. Blue = Cooperate, red = Defect. Space pauses, `r` resets, `q` / Esc quits.

```bash
uv sync --extra viewer
uv run python -m spatial_ipd.viewer \
  --height 32 --width 32 --seed 42 \
  --mutation-rate 0.01 --cell-size 12 --fps 8
```

## Model

Each cell is Cooperate (`C = 1`) or Defect (`D = 0`). Every generation the cell plays one-shot PD against its eight Moore neighbors on a torus, then imitates the strict highest scorer among itself and those neighbors (focal wins ties; scan NW → W). Optional mutation then flips each cell with probability `mutation_rate` via `random.Random(seed)`.

|            | vs C | vs D |
|------------|------|------|
| **C**      | 3, 3 | 0, 5 |
| **D**      | 5, 0 | 1, 1 |

`T = 5`, `R = 3`, `P = 1`, `S = 0`. `simulate` and `step` are deterministic for a fixed seed, size, mutation rate, and generation count.

## Optional TypeSafe labels

After a run, [TypeSafe](https://docs.typesafe.ai) (Jev) can label the cooperation-rate *shape*. It does not change payoffs or update rules. Needs `TYPESAFE_API_KEY` (env var or a local `.env`, which is gitignored).

```bash
uv sync --extra typesafe
uv run python -m spatial_ipd.label \
  --height 12 --width 12 --generations 30 \
  --seed 20260316 --mutation-rate 0.02
```

Questions and the survival threshold live in `src/spatial_ipd/judgments.py`.

## Optional Jev thinkers

Most cells still imitate. Every `--think-every` generations, a few seeded seats ask Jev whether to **hold** (resist a copy) or **flip**. Payoffs and the default `simulate()` golden do not change. Needs `TYPESAFE_API_KEY`.

```bash
uv sync --extra typesafe
uv run python -m spatial_ipd.think \
  --height 16 --width 16 --generations 30 \
  --seed 20260316 --mutation-rate 0.02 \
  --think-every 5 --thinkers 4
```

Viewer outlines thinker seats in gold when `--think-every` is set.

```python
from spatial_ipd import simulate

print(simulate(20, 20, 50, seed=42, mutation_rate=0.01).final_cooperation_rate)
```

## Development

Same layout as SkillFlow: uv, Ruff (Google docstrings), pre-commit, Makefile. Planning and execution stay Estack-shaped (small scope, no surprise runtime deps, prove with pytest).

```bash
make sync
make hooks
make lint
make test
```

```bash
uv sync --group dev
uv run pre-commit install
uv run ruff check .
uv run ruff format .
uv run pytest
```

Optional extras: `uv sync --extra typesafe` and/or `--extra viewer`. Copy `.env.example` to `.env` for a TypeSafe key (gitignored).
