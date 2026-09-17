"""Spatial IPD step: score the Moore neighborhood, then imitate the best.

Update rule
-----------
1. Each cell plays the one-shot PD against its eight Moore neighbors
   (toroidal wrap). Payoffs are summed. A cell does **not** play itself.
2. Each cell then adopts the strategy of the highest-scoring cell among
   itself and its eight neighbors.
3. Ties: the focal cell is considered first. A neighbor replaces it only
   on a *strictly* higher score. Neighbors are scanned in ``MOORE_OFFSETS``
   order (NW, N, NE, E, SE, S, SW, W), so the first strict maximum wins.
4. Optional mutation: after imitation, each cell independently flips
   strategy with probability ``mutation_rate`` using a seeded ``random.Random``.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random

from spatial_ipd.neighborhood import moore_neighbors
from spatial_ipd.payoffs import COOPERATE, DEFECT, payoff

Grid = list[list[int]]


@dataclass(frozen=True)
class SimulationResult:
    """Outcome of a seeded multi-generation run."""

    grid: tuple[tuple[int, ...], ...]
    cooperation_rates: tuple[float, ...]
    final_cooperation_rate: float
    generations: int
    seed: int


def _require_grid(grid: Grid) -> tuple[int, int]:
    if not grid or not grid[0]:
        raise ValueError("grid must be non-empty")
    height = len(grid)
    width = len(grid[0])
    if any(len(row) != width for row in grid):
        raise ValueError("grid rows must be rectangular")
    return height, width


def _copy_grid(grid: Grid) -> Grid:
    return [row[:] for row in grid]


def cooperation_rate(grid: Grid) -> float:
    """Fraction of cells playing Cooperate."""
    height, width = _require_grid(grid)
    cooperators = sum(1 for row in grid for cell in row if int(cell) == COOPERATE)
    return cooperators / (height * width)


def random_grid(
    height: int,
    width: int,
    *,
    seed: int,
    cooperate_p: float = 0.5,
) -> Grid:
    """Bernoulli lattice: each cell is C with probability ``cooperate_p``."""
    if height < 1 or width < 1:
        raise ValueError("grid dimensions must be positive")
    rng = Random(seed)
    return [[COOPERATE if rng.random() < cooperate_p else DEFECT for _ in range(width)] for _ in range(height)]


def score_cells(grid: Grid) -> list[list[int]]:
    """Accumulate PD payoffs against each cell's eight Moore neighbors."""
    height, width = _require_grid(grid)
    scores = [[0] * width for _ in range(height)]
    for r in range(height):
        for c in range(width):
            focal = int(grid[r][c])
            total = 0
            for nr, nc in moore_neighbors(r, c, height, width):
                total += payoff(focal, int(grid[nr][nc]))
            scores[r][c] = total
    return scores


def adopt_best(grid: Grid, scores: list[list[int]]) -> Grid:
    """Return a new grid after one imitation update (no mutation)."""
    height, width = _require_grid(grid)
    if len(scores) != height or any(len(row) != width for row in scores):
        raise ValueError("scores must match grid shape")

    next_grid = [[DEFECT] * width for _ in range(height)]
    for r in range(height):
        for c in range(width):
            best_score = scores[r][c]
            best_strategy = int(grid[r][c])
            for nr, nc in moore_neighbors(r, c, height, width):
                neighbor_score = scores[nr][nc]
                if neighbor_score > best_score:
                    best_score = neighbor_score
                    best_strategy = int(grid[nr][nc])
            next_grid[r][c] = best_strategy
    return next_grid


def apply_mutation(grid: Grid, rng: Random, mutation_rate: float) -> Grid:
    """Independently flip each cell with probability ``mutation_rate``."""
    _require_grid(grid)
    if mutation_rate < 0 or mutation_rate > 1:
        raise ValueError("mutation_rate must be in [0, 1]")
    next_grid = _copy_grid(grid)
    if mutation_rate == 0:
        return next_grid
    height, width = len(next_grid), len(next_grid[0])
    for r in range(height):
        for c in range(width):
            if rng.random() < mutation_rate:
                next_grid[r][c] = COOPERATE if int(next_grid[r][c]) == DEFECT else DEFECT
    return next_grid


def step(grid: Grid, rng: Random | None = None, mutation_rate: float = 0.0) -> Grid:
    """One generation: score → imitate best → optional mutation.

    ``step`` always returns a new grid; the input is not mutated.
    """
    updated = adopt_best(grid, score_cells(grid))
    if mutation_rate == 0 or rng is None:
        return updated
    return apply_mutation(updated, rng, mutation_rate)


def simulate(
    height: int,
    width: int,
    generations: int,
    seed: int,
    *,
    mutation_rate: float = 0.0,
    cooperate_p: float = 0.5,
    initial_grid: Grid | None = None,
) -> SimulationResult:
    """Run ``generations`` updates from a seeded random (or provided) grid.

    ``cooperation_rates[0]`` is the initial lattice; subsequent entries are
    the rate after each generation.
    """
    if generations < 0:
        raise ValueError("generations must be non-negative")
    rng = Random(seed)
    grid = (
        _copy_grid(initial_grid)
        if initial_grid is not None
        else random_grid(height, width, seed=seed, cooperate_p=cooperate_p)
    )
    rates = [cooperation_rate(grid)]
    for _ in range(generations):
        grid = step(grid, rng=rng, mutation_rate=mutation_rate)
        rates.append(cooperation_rate(grid))
    frozen = tuple(tuple(int(cell) for cell in row) for row in grid)
    return SimulationResult(
        grid=frozen,
        cooperation_rates=tuple(rates),
        final_cooperation_rate=rates[-1],
        generations=generations,
        seed=seed,
    )
