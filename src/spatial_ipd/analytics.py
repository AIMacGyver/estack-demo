"""Pure spatial and trajectory metrics for Spatial IPD experiments."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from spatial_ipd.engine import Grid, score_cells
from spatial_ipd.neighborhood import MOORE_OFFSETS
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


def _distinct_neighbors(
    row: int,
    col: int,
    height: int,
    width: int,
) -> Iterator[tuple[int, int]]:
    if height >= 3 and width >= 3:
        for dr, dc in MOORE_OFFSETS:
            yield (row + dr) % height, (col + dc) % width
        return
    seen = set()
    for dr, dc in MOORE_OFFSETS:
        neighbor = ((row + dr) % height, (col + dc) % width)
        if neighbor != (row, col) and neighbor not in seen:
            seen.add(neighbor)
            yield neighbor


def _cooperator_component_sizes(grid: Grid) -> list[int]:
    height, width = _shape(grid)
    remaining = {(row, col) for row in range(height) for col in range(width) if int(grid[row][col]) == COOPERATE}
    sizes = []
    while remaining:
        root = remaining.pop()
        size = 1
        pending = [root]
        while pending:
            row, col = pending.pop()
            for neighbor in _distinct_neighbors(row, col, height, width):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    pending.append(neighbor)
                    size += 1
        sizes.append(size)
    return sizes


def analyze_grid(grid: Grid) -> SpatialMetrics:
    """Measure final cooperator clusters, strategy frontier, and payoffs."""
    height, width = _shape(grid)
    component_sizes = _cooperator_component_sizes(grid)
    scores = score_cells(grid)
    frontier_cells = 0
    cooperator_frontier_cells = 0
    cooperator_payoff = 0
    cooperator_count = 0
    defector_payoff = 0
    defector_count = 0
    for row in range(height):
        for col in range(width):
            strategy = int(grid[row][col])
            if any(int(grid[nr][nc]) != strategy for nr, nc in _distinct_neighbors(row, col, height, width)):
                frontier_cells += 1
                if strategy == COOPERATE:
                    cooperator_frontier_cells += 1
            if strategy == COOPERATE:
                cooperator_payoff += scores[row][col]
                cooperator_count += 1
            elif strategy == DEFECT:
                defector_payoff += scores[row][col]
                defector_count += 1
    return SpatialMetrics(
        cooperator_clusters=len(component_sizes),
        largest_cooperator_cluster=max(component_sizes, default=0),
        frontier_cells=frontier_cells,
        cooperator_frontier_cells=cooperator_frontier_cells,
        mean_cooperator_payoff=cooperator_payoff / cooperator_count if cooperator_count else None,
        mean_defector_payoff=defector_payoff / defector_count if defector_count else None,
    )


def cooperation_persistence(rates: tuple[float, ...] | list[float]) -> float:
    """Fraction of recorded generations with non-zero cooperation."""
    if not rates:
        raise ValueError("cooperation rates must not be empty")
    return sum(rate > 0.0 for rate in rates) / len(rates)
