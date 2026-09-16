"""Optional Pygame window for the existing Spatial IPD engine.

This module does not reimplement payoffs, neighborhoods, or the update
rule. It only paints ``random_grid`` / ``step`` output. Pygame is imported
when the window loop starts so color and CLI helpers stay testable without
a display or the pygame package.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from random import Random

from spatial_ipd.engine import cooperation_rate, random_grid, step
from spatial_ipd.payoffs import COOPERATE

# Blue = Cooperate, red = Defect. RGB tuples, no pygame required.
COLOR_COOPERATE = (40, 90, 220)
COLOR_DEFECT = (200, 40, 40)

DEFAULT_HEIGHT = 32
DEFAULT_WIDTH = 32
DEFAULT_SEED = 42
DEFAULT_MUTATION_RATE = 0.01
DEFAULT_CELL_SIZE = 12
DEFAULT_FPS = 8


@dataclass(frozen=True)
class ViewerConfig:
    """CLI / launch settings for the viewer window."""

    height: int = DEFAULT_HEIGHT
    width: int = DEFAULT_WIDTH
    seed: int = DEFAULT_SEED
    mutation_rate: float = DEFAULT_MUTATION_RATE
    cell_size: int = DEFAULT_CELL_SIZE
    fps: int = DEFAULT_FPS


def cell_color(strategy: int) -> tuple[int, int, int]:
    """Map a cell strategy to an RGB triple."""
    return COLOR_COOPERATE if int(strategy) == COOPERATE else COLOR_DEFECT


def color_frame(grid: list[list[int]]) -> list[list[tuple[int, int, int]]]:
    """Return an RGB frame buffer matching the lattice (no pygame)."""
    return [[cell_color(cell) for cell in row] for row in grid]


def validate_config(config: ViewerConfig) -> ViewerConfig:
    """Reject settings that cannot drive a grid or a window."""
    if config.height < 1 or config.width < 1:
        raise ValueError("height and width must be positive")
    if config.cell_size < 1:
        raise ValueError("cell-size must be positive")
    if config.fps < 1:
        raise ValueError("fps must be positive")
    if config.mutation_rate < 0 or config.mutation_rate > 1:
        raise ValueError("mutation-rate must be in [0, 1]")
    return config


def parse_args(argv: list[str] | None = None) -> ViewerConfig:
    """Parse viewer CLI flags. ``argv`` is injected by tests."""
    parser = argparse.ArgumentParser(
        prog="spatial_ipd.viewer",
        description="Visualize the Spatial IPD lattice (C=blue, D=red).",
    )
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=DEFAULT_MUTATION_RATE,
        dest="mutation_rate",
    )
    parser.add_argument("--cell-size", type=int, default=DEFAULT_CELL_SIZE, dest="cell_size")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    ns = parser.parse_args(argv)
    return validate_config(
        ViewerConfig(
            height=ns.height,
            width=ns.width,
            seed=ns.seed,
            mutation_rate=ns.mutation_rate,
            cell_size=ns.cell_size,
            fps=ns.fps,
        )
    )


def new_run(height: int, width: int, seed: int) -> tuple[list[list[int]], Random]:
    """Seeded initial lattice plus the RNG used for later mutation.

    Matches ``simulate``: the grid is drawn from ``random_grid(..., seed=seed)``
    and mutation uses a separate ``Random(seed)``.
    """
    return random_grid(height, width, seed=seed), Random(seed)


def window_caption(generation: int, grid: list[list[int]]) -> str:
    """Status line for the window title."""
    rate = cooperation_rate(grid)
    return f"Spatial IPD  gen={generation}  C={rate:.3f}"


def import_pygame():
    """Load pygame only when a window is actually requested."""
    try:
        import pygame
    except ImportError as exc:
        raise ImportError(
            'Pygame is required for the viewer. Install with: pip install -e ".[viewer]"'
        ) from exc
    return pygame


def _draw_grid(pygame, screen, grid: list[list[int]], cell_size: int) -> None:
    """Paint one pixel per cell, then scale up to ``cell_size``."""
    height = len(grid)
    width = len(grid[0])
    surface = pygame.Surface((width, height))
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            surface.set_at((c, r), cell_color(cell))
    scaled = pygame.transform.scale(surface, (width * cell_size, height * cell_size))
    screen.blit(scaled, (0, 0))


def run(config: ViewerConfig) -> None:
    """Open a window and step the existing engine until quit.

    Requires a local display. Cloud / headless VMs will not show a window.
    Keys: space pause, r reset (same seed), q or Esc quit.
    """
    config = validate_config(config)
    pygame = import_pygame()
    pygame.init()
    try:
        screen = pygame.display.set_mode(
            (config.width * config.cell_size, config.height * config.cell_size)
        )
        clock = pygame.time.Clock()
        grid, rng = new_run(config.height, config.width, config.seed)
        generation = 0
        paused = False
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_r:
                        grid, rng = new_run(config.height, config.width, config.seed)
                        generation = 0
            pygame.display.set_caption(window_caption(generation, grid))
            _draw_grid(pygame, screen, grid, config.cell_size)
            pygame.display.flip()
            clock.tick(config.fps)
            if not paused:
                grid = step(grid, rng=rng, mutation_rate=config.mutation_rate)
                generation += 1
    finally:
        pygame.quit()


def main(argv: list[str] | None = None) -> None:
    """CLI entry: ``python -m spatial_ipd.viewer``."""
    try:
        config = parse_args(argv)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    run(config)


if __name__ == "__main__":
    main()
