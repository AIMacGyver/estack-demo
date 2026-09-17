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
from spatial_ipd.think import SEAT_FRONTIER, SEAT_MODES, think_after_step

# Blue = Cooperate, red = Defect. RGB tuples, no pygame required.
COLOR_COOPERATE = (40, 90, 220)
COLOR_DEFECT = (200, 40, 40)
COLOR_THINKER_OUTLINE = (240, 200, 40)


@dataclass(frozen=True)
class ViewerConfig:
    """CLI / launch settings for the viewer window."""

    height: int = 32
    width: int = 32
    seed: int = 42
    mutation_rate: float = 0.01
    cell_size: int = 12
    fps: int = 8
    think_every: int = 0
    thinkers: int = 4
    seat_mode: str = SEAT_FRONTIER


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
    if config.think_every < 0:
        raise ValueError("think-every must be non-negative")
    if config.thinkers < 0:
        raise ValueError("thinkers must be non-negative")
    if config.seat_mode not in SEAT_MODES:
        raise ValueError(f"seats must be one of {', '.join(SEAT_MODES)}")
    return config


def parse_args(argv: list[str] | None = None) -> ViewerConfig:
    """Parse viewer CLI flags. ``argv`` is injected by tests."""
    parser = argparse.ArgumentParser(
        prog="spatial_ipd.viewer",
        description="Visualize the Spatial IPD lattice (C=blue, D=red).",
    )
    defaults = ViewerConfig()
    parser.add_argument("--height", type=int, default=defaults.height)
    parser.add_argument("--width", type=int, default=defaults.width)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=defaults.mutation_rate,
        dest="mutation_rate",
    )
    parser.add_argument("--cell-size", type=int, default=defaults.cell_size, dest="cell_size")
    parser.add_argument("--fps", type=int, default=defaults.fps)
    parser.add_argument("--think-every", type=int, default=defaults.think_every, dest="think_every")
    parser.add_argument("--thinkers", type=int, default=defaults.thinkers)
    parser.add_argument(
        "--seats",
        choices=SEAT_MODES,
        default=defaults.seat_mode,
        dest="seat_mode",
        help="random = old uniform seats; frontier = just-changed then C/D edges.",
    )
    ns = parser.parse_args(argv)
    return validate_config(
        ViewerConfig(
            height=ns.height,
            width=ns.width,
            seed=ns.seed,
            mutation_rate=ns.mutation_rate,
            cell_size=ns.cell_size,
            fps=ns.fps,
            think_every=ns.think_every,
            thinkers=ns.thinkers,
            seat_mode=ns.seat_mode,
        )
    )


def new_run(height: int, width: int, seed: int) -> tuple[list[list[int]], Random]:
    """Seeded initial lattice plus the RNG used for later mutation.

    Matches ``simulate``: the grid is drawn from ``random_grid(..., seed=seed)``
    and mutation uses a separate ``Random(seed)``.
    """
    return random_grid(height, width, seed=seed), Random(seed)


def window_caption(
    generation: int,
    grid: list[list[int]],
    *,
    thinker_count: int = 0,
) -> str:
    """Status line for the window title."""
    rate = cooperation_rate(grid)
    extra = f"  thinkers={thinker_count}" if thinker_count else ""
    return f"Spatial IPD  gen={generation}  C={rate:.3f}{extra}"


def import_pygame():
    """Load pygame only when a window is actually requested."""
    try:
        import pygame
    except ImportError as exc:
        raise ImportError('Pygame is required for the viewer. Install with: pip install -e ".[viewer]"') from exc
    return pygame


def _draw_grid(
    pygame,
    screen,
    grid: list[list[int]],
    cell_size: int,
    thinker_seats: set[tuple[int, int]] | None = None,
) -> None:
    """Paint the same RGB frame ``color_frame`` builds as cell-sized squares."""
    seats = thinker_seats or set()
    for r, row in enumerate(color_frame(grid)):
        for c, color in enumerate(row):
            rect = (c * cell_size, r * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, color, rect)
            if (r, c) in seats:
                pygame.draw.rect(screen, COLOR_THINKER_OUTLINE, rect, max(1, cell_size // 6))


def run(config: ViewerConfig) -> None:
    """Open a window and step the existing engine until quit.

    Requires a local display. Cloud / headless VMs will not show a window.
    Keys: space pause, r reset (same seed), q or Esc quit.
    """
    config = validate_config(config)
    if config.think_every:
        from spatial_ipd.label import load_dotenv

        load_dotenv()
    pygame = import_pygame()
    pygame.init()
    try:
        screen = pygame.display.set_mode((config.width * config.cell_size, config.height * config.cell_size))
        clock = pygame.time.Clock()
        grid, rng = new_run(config.height, config.width, config.seed)
        generation = 0
        paused = False
        running = True
        seats: set[tuple[int, int]] = set()
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
                        seats = set()
            pygame.display.set_caption(window_caption(generation, grid, thinker_count=len(seats)))
            _draw_grid(pygame, screen, grid, config.cell_size, seats)
            pygame.display.flip()
            clock.tick(config.fps)
            if not paused:
                before = [row[:] for row in grid]
                grid = step(grid, rng=rng, mutation_rate=config.mutation_rate)
                generation += 1
                if config.think_every and generation % config.think_every == 0:
                    grid, stats = think_after_step(
                        before,
                        grid,
                        generation=generation,
                        seed=config.seed,
                        thinker_count=config.thinkers,
                        seat_mode=config.seat_mode,
                    )
                    seats = set(stats.last_seats)
                else:
                    seats = set()
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
