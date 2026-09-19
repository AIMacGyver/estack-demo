"""Optional Pygame window for the existing Spatial IPD engine.

This module does not reimplement payoffs, neighborhoods, or the update
rule. It only paints ``random_grid`` / ``step`` output. Pygame is imported
when the window loop starts so color and CLI helpers stay testable without
a display or the pygame package.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from random import Random

from spatial_ipd.analytics import analyze_grid
from spatial_ipd.engine import cooperation_rate, random_grid, score_cells, step
from spatial_ipd.local_llm import (
    DEFAULT_LOCAL_LLM_ENDPOINT,
    DEFAULT_LOCAL_LLM_TIMEOUT,
    LOCAL_LLM_REASONING_EFFORTS,
    LocalLLMThinkerClient,
)
from spatial_ipd.neighborhood import moore_neighbors
from spatial_ipd.payoffs import COOPERATE
from spatial_ipd.think import (
    BACKEND_JEV,
    BACKEND_LOCAL,
    BACKEND_RANDOM,
    BACKENDS,
    SEAT_FRONTIER,
    SEAT_MODES,
    RandomThinkerClient,
    ThinkerDecision,
    think_after_step,
)

# Blue = Cooperate, red = Defect. RGB tuples, no pygame required.
COLOR_COOPERATE = (40, 90, 220)
COLOR_DEFECT = (200, 40, 40)
COLOR_THINKER_OUTLINE = (240, 200, 40)
COLOR_HOLD_OUTLINE = (40, 220, 100)
COLOR_SELECTED_OUTLINE = (80, 230, 240)
COLOR_HISTORY_BACKGROUND = (24, 24, 28)
COLOR_COOPERATION_LINE = (80, 150, 255)
COLOR_CLUSTER_LINE = (80, 220, 130)


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
    backend: str = BACKEND_JEV
    local_model: str | None = None
    local_endpoint: str = DEFAULT_LOCAL_LLM_ENDPOINT
    local_timeout: float = DEFAULT_LOCAL_LLM_TIMEOUT
    local_reasoning_effort: str | None = None
    history_height: int = 80


@dataclass
class LabState:
    """Mutable display state kept separate from Pygame event handling."""

    grid: list[list[int]]
    rng: Random
    generation: int
    paused: bool
    selected: tuple[int, int] | None
    decisions: tuple[ThinkerDecision, ...]
    cooperation_history: list[float]
    cluster_history: list[float]


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
    if config.backend not in BACKENDS:
        raise ValueError(f"backend must be one of {', '.join(BACKENDS)}")
    if config.backend == BACKEND_LOCAL and not config.local_model:
        raise ValueError("local-model is required when backend is local")
    if config.local_timeout <= 0:
        raise ValueError("local-timeout must be positive")
    if config.local_reasoning_effort is not None and config.local_reasoning_effort not in LOCAL_LLM_REASONING_EFFORTS:
        raise ValueError(f"local-reasoning-effort must be one of {LOCAL_LLM_REASONING_EFFORTS}")
    if config.history_height < 1:
        raise ValueError("history-height must be positive")
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
    parser.add_argument("--backend", choices=BACKENDS, default=defaults.backend)
    parser.add_argument("--local-model")
    parser.add_argument("--local-endpoint", default=defaults.local_endpoint)
    parser.add_argument("--local-timeout", type=float, default=defaults.local_timeout)
    parser.add_argument("--local-reasoning-effort", choices=LOCAL_LLM_REASONING_EFFORTS)
    parser.add_argument("--history-height", type=int, default=defaults.history_height)
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
            backend=ns.backend,
            local_model=ns.local_model,
            local_endpoint=ns.local_endpoint,
            local_timeout=ns.local_timeout,
            local_reasoning_effort=ns.local_reasoning_effort,
            history_height=ns.history_height,
        )
    )


def new_run(height: int, width: int, seed: int) -> tuple[list[list[int]], Random]:
    """Seeded initial lattice plus the RNG used for later mutation.

    Matches ``simulate``: the grid is drawn from ``random_grid(..., seed=seed)``
    and mutation uses a separate ``Random(seed)``.
    """
    return random_grid(height, width, seed=seed), Random(seed)


def new_lab_state(config: ViewerConfig) -> LabState:
    """Create a seeded viewer state with initial history metrics."""
    grid, rng = new_run(config.height, config.width, config.seed)
    metrics = analyze_grid(grid)
    return LabState(
        grid=grid,
        rng=rng,
        generation=0,
        paused=False,
        selected=None,
        decisions=(),
        cooperation_history=[cooperation_rate(grid)],
        cluster_history=[float(metrics.largest_cooperator_cluster)],
    )


def viewer_client(config: ViewerConfig) -> object | None:
    """Build the configured thinker client without importing Pygame."""
    if config.backend == BACKEND_RANDOM:
        return RandomThinkerClient(config.seed)
    if config.backend == BACKEND_LOCAL:
        return LocalLLMThinkerClient(
            config.local_model or "",
            endpoint=config.local_endpoint,
            api_key=os.environ.get("LOCAL_LLM_API_KEY"),
            timeout=config.local_timeout,
            reasoning_effort=config.local_reasoning_effort,
        )
    return None


def advance_lab_state(
    state: LabState,
    config: ViewerConfig,
    *,
    client: object | None = None,
) -> LabState:
    """Advance one engine generation and append inspectable lab history."""
    before = [row[:] for row in state.grid]
    grid = step(state.grid, rng=state.rng, mutation_rate=config.mutation_rate)
    generation = state.generation + 1
    decisions: tuple[ThinkerDecision, ...] = ()
    if config.think_every and generation % config.think_every == 0:
        grid, stats = think_after_step(
            before,
            grid,
            generation=generation,
            seed=config.seed,
            thinker_count=config.thinkers,
            seat_mode=config.seat_mode,
            client=client,
        )
        decisions = stats.last_decisions
    state.grid = grid
    state.generation = generation
    state.decisions = decisions
    state.cooperation_history.append(cooperation_rate(grid))
    state.cluster_history.append(float(analyze_grid(grid).largest_cooperator_cluster))
    return state


def cell_facts(grid: list[list[int]], row: int, col: int) -> dict[str, int | bool]:
    """Return code-owned facts for one selected lattice cell."""
    height, width = len(grid), len(grid[0])
    if not 0 <= row < height or not 0 <= col < width:
        raise ValueError("selected cell is outside the grid")
    strategy = int(grid[row][col])
    neighbors = set(moore_neighbors(row, col, height, width))
    neighbors.discard((row, col))
    different = sum(int(grid[nr][nc]) != strategy for nr, nc in neighbors)
    cooperative = sum(int(grid[nr][nc]) == COOPERATE for nr, nc in neighbors)
    return {
        "row": row,
        "col": col,
        "strategy": strategy,
        "payoff": int(score_cells(grid)[row][col]),
        "cooperative_neighbors": cooperative,
        "different_neighbors": different,
        "frontier": different > 0,
    }


def decision_overlays(
    decisions: Sequence[ThinkerDecision],
) -> dict[tuple[int, int], tuple[int, int, int]]:
    """Map thinker decisions to display colors."""
    return {
        (decision.row, decision.col): (
            COLOR_HOLD_OUTLINE if decision.applied and decision.act == "hold" else COLOR_THINKER_OUTLINE
        )
        for decision in decisions
    }


def history_points(
    values: Sequence[float],
    *,
    width: int,
    height: int,
    maximum: float,
) -> list[tuple[int, int]]:
    """Scale a numeric history into plot points without Pygame."""
    if width < 1 or height < 1 or maximum <= 0:
        raise ValueError("history plot dimensions and maximum must be positive")
    if not values:
        return []
    denominator = max(1, len(values) - 1)
    return [
        (
            round(index * (width - 1) / denominator),
            round((height - 1) * (1.0 - min(max(value / maximum, 0.0), 1.0))),
        )
        for index, value in enumerate(values)
    ]


def window_caption(
    generation: int,
    grid: list[list[int]],
    *,
    thinker_count: int = 0,
    backend: str | None = None,
    selected: tuple[int, int] | None = None,
) -> str:
    """Status line for the window title."""
    rate = cooperation_rate(grid)
    extra = f"  thinkers={thinker_count}" if thinker_count else ""
    backend_text = f"  backend={backend}" if backend else ""
    selected_text = ""
    if selected is not None:
        facts = cell_facts(grid, *selected)
        selected_text = f"  cell=({facts['row']},{facts['col']}) strategy={facts['strategy']} payoff={facts['payoff']}"
    return f"Spatial IPD  gen={generation}  C={rate:.3f}{extra}{backend_text}{selected_text}"


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
    overlays: Mapping[tuple[int, int], tuple[int, int, int]] | None = None,
    selected: tuple[int, int] | None = None,
) -> None:
    """Paint the same RGB frame ``color_frame`` builds as cell-sized squares."""
    outlines = overlays or {}
    for r, row in enumerate(color_frame(grid)):
        for c, color in enumerate(row):
            rect = (c * cell_size, r * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, color, rect)
            if (r, c) in outlines:
                pygame.draw.rect(screen, outlines[(r, c)], rect, max(1, cell_size // 6))
            if (r, c) == selected:
                pygame.draw.rect(screen, COLOR_SELECTED_OUTLINE, rect, max(1, cell_size // 5))


def _draw_history(
    pygame,
    screen,
    *,
    top: int,
    width: int,
    height: int,
    cooperation_history: Sequence[float],
    cluster_history: Sequence[float],
    cells: int,
) -> None:
    """Draw compact cooperation and largest-cluster history lines."""
    pygame.draw.rect(screen, COLOR_HISTORY_BACKGROUND, (0, top, width, height))
    cooperation = history_points(cooperation_history, width=width, height=height, maximum=1.0)
    clusters = history_points(cluster_history, width=width, height=height, maximum=float(cells))
    if len(cooperation) > 1:
        pygame.draw.lines(
            screen,
            COLOR_COOPERATION_LINE,
            False,
            [(x, top + y) for x, y in cooperation],
            2,
        )
    if len(clusters) > 1:
        pygame.draw.lines(
            screen,
            COLOR_CLUSTER_LINE,
            False,
            [(x, top + y) for x, y in clusters],
            2,
        )


def run(config: ViewerConfig) -> None:
    """Open a window and step the existing engine until quit.

    Requires a local display. Cloud / headless VMs will not show a window.
    Keys: space pause, n single-step, r reset, mouse select, q or Esc quit.
    """
    config = validate_config(config)
    if config.think_every and config.backend in (BACKEND_JEV, BACKEND_LOCAL):
        from spatial_ipd.label import load_dotenv

        load_dotenv()
    client = viewer_client(config)
    pygame = import_pygame()
    pygame.init()
    try:
        grid_width = config.width * config.cell_size
        grid_height = config.height * config.cell_size
        screen = pygame.display.set_mode((grid_width, grid_height + config.history_height))
        clock = pygame.time.Clock()
        state = new_lab_state(config)
        running = True
        while running:
            single_step = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key == pygame.K_SPACE:
                        state.paused = not state.paused
                    elif event.key == pygame.K_n:
                        single_step = True
                    elif event.key == pygame.K_r:
                        state = new_lab_state(config)
                    elif event.key == pygame.K_c:
                        state.selected = None
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    x, y = event.pos
                    if y < grid_height:
                        state.selected = (y // config.cell_size, x // config.cell_size)
            pygame.display.set_caption(
                window_caption(
                    state.generation,
                    state.grid,
                    thinker_count=len(state.decisions),
                    backend=config.backend if config.think_every else None,
                    selected=state.selected,
                )
            )
            _draw_grid(
                pygame,
                screen,
                state.grid,
                config.cell_size,
                decision_overlays(state.decisions),
                state.selected,
            )
            _draw_history(
                pygame,
                screen,
                top=grid_height,
                width=grid_width,
                height=config.history_height,
                cooperation_history=state.cooperation_history,
                cluster_history=state.cluster_history,
                cells=config.height * config.width,
            )
            pygame.display.flip()
            clock.tick(config.fps)
            if not state.paused or single_step:
                advance_lab_state(state, config, client=client)
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
