# Spatial IPD

Headless engine for the **Spatial Iterated Prisoner's Dilemma**: evolutionary cooperation on a 2-D lattice. An optional Pygame viewer can paint the same engine; it is not a required runtime dependency.

Each cell is a player that is either Cooperate (`C = 1`) or Defect (`D = 0`). Every generation:

1. The cell plays the one-shot Prisoner's Dilemma against its **Moore neighborhood** (8 neighbors).
2. Payoffs are accumulated.
3. The cell **imitates** the highest-scoring strategy among itself and those neighbors.
4. An optional **mutation** then flips each cell independently with a small probability.

## Payoff matrix

|            | vs C | vs D |
|------------|------|------|
| **C**      | 3, 3 | 0, 5 |
| **D**      | 5, 0 | 1, 1 |

Temptation `T = 5`, Reward `R = 3`, Punishment `P = 1`, Sucker `S = 0`.

## Lattice rules

- **Topology:** the grid is a **torus** (edges wrap). Every cell has exactly eight neighbors, including corners.
- **Scoring:** a cell does **not** play itself; it only plays the eight neighbors.
- **Imitation:** candidates are the focal cell plus its eight neighbors.
- **Ties:** the focal cell is considered first. A neighbor replaces it only with a *strictly* higher score. Neighbors are scanned NW → N → NE → E → SE → S → SW → W, so the first strict maximum wins.
- **Mutation:** after imitation, each cell flips with probability `mutation_rate` using `random.Random(seed)`.

The model is the Nowak & May (1992) spatial PD, with optional noise.

## Install and test

Requires Python 3.10+. The core engine is the standard library only. Tests need pytest. Pygame is an **optional extra** for the local viewer and is not imported by the engine modules.

```bash
python3 -m pip install -e ".[dev]"
python3 -m pytest
```

Or, without an editable install (pytest still needed):

```bash
python3 -m pip install pytest
PYTHONPATH=src python3 -m pytest
```

Import the engine:

```python
from spatial_ipd import simulate

result = simulate(height=20, width=20, generations=50, seed=42, mutation_rate=0.01)
print(result.final_cooperation_rate)
```

`simulate` and `step` are deterministic for a fixed seed, size, mutation rate, and generation count.

## Optional Pygame viewer

The viewer is a thin window over the existing `random_grid` / `step` API. It does not change payoffs or update rules. Install pygame as an extra, then launch from a machine **with a display** (the Cloud Agent VM has no interactive window):

```bash
python3 -m pip install -e ".[viewer]"
python3 -m spatial_ipd.viewer
```

Useful flags (defaults are small enough to run comfortably):

```bash
python3 -m spatial_ipd.viewer \
  --height 32 --width 32 --seed 42 \
  --mutation-rate 0.01 --cell-size 12 --fps 8
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--height` / `--width` | 32 | Lattice size |
| `--seed` | 42 | Initial grid and mutation RNG |
| `--mutation-rate` | 0.01 | Per-cell flip after imitation |
| `--cell-size` | 12 | Pixel size of each cell |
| `--fps` | 8 | Auto-step rate |

Keys: **space** pause/resume, **r** reset with the same seed, **q** or **Esc** quit. Cooperate cells are blue; Defect cells are red.

Headless tests cover color mapping, CLI parsing, and the pygame-import guard. They do not open a window.
