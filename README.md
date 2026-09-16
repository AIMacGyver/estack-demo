# Spatial IPD

Headless engine for the **Spatial Iterated Prisoner's Dilemma**: evolutionary cooperation on a 2-D lattice. No GUI.

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

Requires Python 3.10+. Runtime code is the standard library only. Tests need pytest:

```bash
python3 -m pip install -e ".[dev]"
python3 -m pytest
```

Or, without an editable install (pytest still needed):

```bash
python3 -m pip install pytest
PYTHONPATH=src python3 -m pytest
```

There is no CLI and no visualization. Import the engine:

```python
from spatial_ipd import simulate

result = simulate(height=20, width=20, generations=50, seed=42, mutation_rate=0.01)
print(result.final_cooperation_rate)
```

`simulate` and `step` are deterministic for a fixed seed, size, mutation rate, and generation count.
