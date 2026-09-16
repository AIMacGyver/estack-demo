"""Moore neighborhood on a toroidal 2-D lattice.

Edges wrap: the grid is a torus. Every cell therefore has exactly eight
neighbors, including corners and 1×N strips.
"""

from __future__ import annotations

# Clockwise from NW. Order is part of the documented tie-break rule.
MOORE_OFFSETS: tuple[tuple[int, int], ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
)


def moore_neighbors(row: int, col: int, height: int, width: int) -> list[tuple[int, int]]:
    """Return the eight Moore neighbors of ``(row, col)`` with toroidal wrap."""
    if height < 1 or width < 1:
        raise ValueError("grid dimensions must be positive")
    return [((row + dr) % height, (col + dc) % width) for dr, dc in MOORE_OFFSETS]
