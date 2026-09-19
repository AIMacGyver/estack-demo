"""Pure spatial and trajectory metrics for Spatial IPD experiments."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean

from spatial_ipd.engine import Grid, score_cells
from spatial_ipd.neighborhood import moore_neighbors
from spatial_ipd.payoffs import COOPERATE, DEFECT


@dataclass(frozen=True)
class SpatialMetrics:
    """Structure and payoff summaries for one final lattice."""

    cooperator_clusters: int
    largest_cooperator_cluster: int
    frontier_cells: int
    cooperator_frontier_cells: int
    mean_cooperator_payoff: float | None
    mean_defector_payoff: float | None


def _shape(grid: Grid) -> tuple[int, int]:
    if not grid or not grid[0]:
        raise ValueError("grid must be non-empty")
    width = len(grid[0])
    if any(len(row) != width for row in grid):
        raise ValueError("grid rows must be rectangular")
    return len(grid), width


def _distinct_neighbors(row: int, col: int, height: int, width: int) -> set[tuple[int, int]]:
    neighbors = set(moore_neighbors(row, col, height, width))
    neighbors.discard((row, col))
    return neighbors


def _cooperator_components(grid: Grid) -> list[set[tuple[int, int]]]:
    height, width = _shape(grid)
    remaining = {(row, col) for row in range(height) for col in range(width) if int(grid[row][col]) == COOPERATE}
    components = []
    while remaining:
        root = remaining.pop()
        component = {root}
        pending = [root]
        while pending:
            row, col = pending.pop()
            connected = _distinct_neighbors(row, col, height, width) & remaining
            remaining.difference_update(connected)
            component.update(connected)
            pending.extend(connected)
        components.append(component)
    return components


def analyze_grid(grid: Grid) -> SpatialMetrics:
    """Measure final cooperator clusters, strategy frontier, and payoffs."""
    height, width = _shape(grid)
    components = _cooperator_components(grid)
    frontier = set()
    cooperator_frontier = set()
    for row in range(height):
        for col in range(width):
            strategy = int(grid[row][col])
            if any(int(grid[nr][nc]) != strategy for nr, nc in _distinct_neighbors(row, col, height, width)):
                frontier.add((row, col))
                if strategy == COOPERATE:
                    cooperator_frontier.add((row, col))

    scores = score_cells(grid)
    cooperator_scores = [
        float(scores[row][col]) for row in range(height) for col in range(width) if int(grid[row][col]) == COOPERATE
    ]
    defector_scores = [
        float(scores[row][col]) for row in range(height) for col in range(width) if int(grid[row][col]) == DEFECT
    ]
    return SpatialMetrics(
        cooperator_clusters=len(components),
        largest_cooperator_cluster=max((len(component) for component in components), default=0),
        frontier_cells=len(frontier),
        cooperator_frontier_cells=len(cooperator_frontier),
        mean_cooperator_payoff=fmean(cooperator_scores) if cooperator_scores else None,
        mean_defector_payoff=fmean(defector_scores) if defector_scores else None,
    )


def cooperation_persistence(rates: tuple[float, ...] | list[float]) -> float:
    """Fraction of recorded generations with non-zero cooperation."""
    if not rates:
        raise ValueError("cooperation rates must not be empty")
    return sum(rate > 0.0 for rate in rates) / len(rates)
