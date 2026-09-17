"""Opt-in dual-process step: most cells imitate; a few ask Jev."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass, field
from random import Random

from spatial_ipd.engine import (
    Grid,
    SimulationResult,
    cooperation_rate,
    random_grid,
    simulate,
    step,
)
from spatial_ipd.judgments import (
    ACT_OVERRIDE_CONFIDENCE,
    WORTH_THINKING_THRESHOLD,
    typesafe_thinker_questions,
)
from spatial_ipd.label import load_dotenv
from spatial_ipd.payoffs import COOPERATE, DEFECT


def _copy_grid(grid: Grid) -> Grid:
    return [row[:] for row in grid]


@dataclass(frozen=True)
class ThinkerDecision:
    row: int
    col: int
    act: str
    worth_thinking: float
    act_confidence: float
    applied: bool
    before: int
    after_imitate: int
    after_think: int


@dataclass
class ThinkerStats:
    calls: int = 0
    holds: int = 0
    flips: int = 0
    imitates: int = 0
    last_seats: tuple[tuple[int, int], ...] = ()
    last_decisions: tuple[ThinkerDecision, ...] = field(default_factory=tuple)


def patch3(grid: Grid, row: int, col: int) -> list[list[int]]:
    """3x3 neighborhood centered on ``(row, col)`` with toroidal wrap."""
    height, width = len(grid), len(grid[0])
    return [
        [int(grid[(row + dr) % height][(col + dc) % width]) for dc in (-1, 0, 1)]
        for dr in (-1, 0, 1)
    ]


def thinker_seats(
    height: int,
    width: int,
    count: int,
    *,
    seed: int,
    generation: int,
) -> tuple[tuple[int, int], ...]:
    """Deterministic unique seats for this generation."""
    if count < 1:
        return ()
    cells = height * width
    take = min(count, cells)
    rng = Random((seed + 1) * 10007 + generation)
    picked = rng.sample(range(cells), take)
    return tuple((idx // width, idx % width) for idx in picked)


def resolve_strategy(before: int, after_imitate: int, act: str) -> int:
    """Map a Jev act onto a cell strategy. Unknown acts keep imitation."""
    if act == "hold":
        return int(before)
    if act == "flip":
        return COOPERATE if int(after_imitate) == DEFECT else DEFECT
    return int(after_imitate)


def should_apply(worth_thinking: float, act_confidence: float) -> bool:
    return (
        worth_thinking >= WORTH_THINKING_THRESHOLD
        and act_confidence >= ACT_OVERRIDE_CONFIDENCE
    )


def state_for_thinkers(
    before: Grid,
    after: Grid,
    seats: Sequence[tuple[int, int]],
    generation: int,
) -> dict:
    thinkers = []
    for row, col in seats:
        thinkers.append(
            {
                "row": row,
                "col": col,
                "before": int(before[row][col]),
                "after_imitate": int(after[row][col]),
                "patch_after": patch3(after, row, col),
            }
        )
    return {"generation": generation, "thinkers": thinkers}


def decisions_from_response(
    seats: Sequence[tuple[int, int]],
    before: Grid,
    after: Grid,
    response: object,
) -> list[ThinkerDecision]:
    out: list[ThinkerDecision] = []
    for i, (row, col) in enumerate(seats):
        worth = float(response.nouls[f"worth_thinking_{i}"].noul)
        choice = response.choices[f"act_{i}"]
        act = str(choice.choice)
        conf = float(choice.confidence)
        applied = should_apply(worth, conf) and act in {"hold", "flip"}
        pre = int(before[row][col])
        mid = int(after[row][col])
        nxt = resolve_strategy(pre, mid, act) if applied else mid
        out.append(
            ThinkerDecision(
                row=row,
                col=col,
                act=act,
                worth_thinking=worth,
                act_confidence=conf,
                applied=applied,
                before=pre,
                after_imitate=mid,
                after_think=nxt,
            )
        )
    return out


def apply_decisions(grid: Grid, decisions: Sequence[ThinkerDecision]) -> Grid:
    nxt = _copy_grid(grid)
    for item in decisions:
        nxt[item.row][item.col] = item.after_think
    return nxt


def think_after_step(
    before: Grid,
    after: Grid,
    *,
    generation: int,
    seed: int,
    thinker_count: int,
    client: object | None = None,
    questions: dict | None = None,
    stats: ThinkerStats | None = None,
) -> tuple[Grid, ThinkerStats]:
    """Maybe override a few cells after a normal ``step``."""
    tally = stats if stats is not None else ThinkerStats()
    height, width = len(after), len(after[0])
    seats = thinker_seats(
        height, width, thinker_count, seed=seed, generation=generation
    )
    tally.last_seats = seats
    if not seats:
        tally.last_decisions = ()
        return _copy_grid(after), tally

    state = state_for_thinkers(before, after, seats, generation)
    qs = typesafe_thinker_questions(len(seats)) if questions is None else questions
    if client is None:
        try:
            from typesafe_sdk import TypeSafeClient
        except ImportError as exc:
            raise ImportError(
                'TypeSafe is optional. Install with: python3 -m pip install -e ".[typesafe]"'
            ) from exc
        with TypeSafeClient() as opened:
            response = opened.system_one(state=state, questions=qs)
    else:
        response = client.system_one(state=state, questions=qs)

    tally.calls += 1
    decisions = decisions_from_response(seats, before, after, response)
    tally.last_decisions = tuple(decisions)
    for item in decisions:
        if item.applied and item.act == "hold":
            tally.holds += 1
        elif item.applied and item.act == "flip":
            tally.flips += 1
        else:
            tally.imitates += 1
    return apply_decisions(after, decisions), tally


def simulate_with_thinkers(
    height: int,
    width: int,
    generations: int,
    seed: int,
    *,
    mutation_rate: float = 0.0,
    cooperate_p: float = 0.5,
    think_every: int = 5,
    thinker_count: int = 4,
    client: object | None = None,
    questions: dict | None = None,
) -> tuple[SimulationResult, ThinkerStats]:
    """Like ``simulate``, plus optional Jev overrides every ``think_every`` gens.

    ``think_every=0`` is exactly ``simulate`` (no API calls).
    """
    if think_every < 0:
        raise ValueError("think_every must be non-negative")
    if think_every == 0:
        return (
            simulate(
                height,
                width,
                generations,
                seed=seed,
                mutation_rate=mutation_rate,
                cooperate_p=cooperate_p,
            ),
            ThinkerStats(),
        )

    rng = Random(seed)
    grid = random_grid(height, width, seed=seed, cooperate_p=cooperate_p)
    rates = [cooperation_rate(grid)]
    stats = ThinkerStats()
    for generation in range(1, generations + 1):
        before = _copy_grid(grid)
        grid = step(grid, rng=rng, mutation_rate=mutation_rate)
        if generation % think_every == 0:
            grid, stats = think_after_step(
                before,
                grid,
                generation=generation,
                seed=seed,
                thinker_count=thinker_count,
                client=client,
                questions=questions,
                stats=stats,
            )
        rates.append(cooperation_rate(grid))
    frozen = tuple(tuple(int(cell) for cell in row) for row in grid)
    result = SimulationResult(
        grid=frozen,
        cooperation_rates=tuple(rates),
        final_cooperation_rate=rates[-1],
        generations=generations,
        seed=seed,
    )
    return result, stats


def format_think_summary(result: SimulationResult, stats: ThinkerStats) -> str:
    return (
        f"final_cooperation_rate={result.final_cooperation_rate} "
        f"generations={result.generations} seed={result.seed} "
        f"think_calls={stats.calls} holds={stats.holds} "
        f"flips={stats.flips} imitates={stats.imitates}"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m spatial_ipd.think",
        description=(
            "Run Spatial IPD with a few Jev thinker cells (optional extra; "
            "needs TYPESAFE_API_KEY unless --think-every 0)."
        ),
    )
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--generations", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mutation-rate", type=float, default=0.0)
    parser.add_argument("--think-every", type=int, default=5)
    parser.add_argument("--thinkers", type=int, default=4)
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    *,
    client: object | None = None,
    questions: dict | None = None,
) -> int:
    load_dotenv()
    args = parse_args(argv)
    result, stats = simulate_with_thinkers(
        args.height,
        args.width,
        args.generations,
        seed=args.seed,
        mutation_rate=args.mutation_rate,
        think_every=args.think_every,
        thinker_count=args.thinkers,
        client=client,
        questions=questions,
    )
    print(format_think_summary(result, stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
