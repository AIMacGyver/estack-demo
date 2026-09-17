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
    """One Jev decision for a thinker seat after imitation."""

    generation: int
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
    """Counters and the last thinker's seats for a run."""

    calls: int = 0
    holds: int = 0
    flips: int = 0
    imitates: int = 0
    last_seats: tuple[tuple[int, int], ...] = ()
    last_decisions: tuple[ThinkerDecision, ...] = field(default_factory=tuple)
    all_decisions: list[ThinkerDecision] = field(default_factory=list)


def patch3(grid: Grid, row: int, col: int) -> list[list[int]]:
    """3x3 neighborhood centered on ``(row, col)`` with toroidal wrap."""
    height, width = len(grid), len(grid[0])
    return [[int(grid[(row + dr) % height][(col + dc) % width]) for dc in (-1, 0, 1)] for dr in (-1, 0, 1)]


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
    """Return True when Jev is sure enough to override imitation."""
    return worth_thinking >= WORTH_THINKING_THRESHOLD and act_confidence >= ACT_OVERRIDE_CONFIDENCE


def state_for_thinkers(
    before: Grid,
    after: Grid,
    seats: Sequence[tuple[int, int]],
    generation: int,
) -> dict:
    """Build the named JSON state for one TypeSafe thinker call."""
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
    *,
    generation: int,
) -> list[ThinkerDecision]:
    """Map a TypeSafe (or test double) response onto per-seat decisions."""
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
                generation=generation,
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
    """Return a new grid with applied thinker overrides."""
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
    seats = thinker_seats(height, width, thinker_count, seed=seed, generation=generation)
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
            raise ImportError('TypeSafe is optional. Install with: python3 -m pip install -e ".[typesafe]"') from exc
        with TypeSafeClient() as opened:
            response = opened.system_one(state=state, questions=qs)
    else:
        response = client.system_one(state=state, questions=qs)

    tally.calls += 1
    decisions = decisions_from_response(seats, before, after, response, generation=generation)
    tally.last_decisions = tuple(decisions)
    tally.all_decisions.extend(decisions)
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
    """One-line CLI summary after a thinker run."""
    return (
        f"final_cooperation_rate={result.final_cooperation_rate} "
        f"generations={result.generations} seed={result.seed} "
        f"think_calls={stats.calls} holds={stats.holds} "
        f"flips={stats.flips} imitates={stats.imitates}"
    )


def format_compare_summary(plain: SimulationResult, think: SimulationResult, stats: ThinkerStats) -> str:
    """Plain simulate() vs thinker run, so you can see if Jev moved the needle."""
    delta = think.final_cooperation_rate - plain.final_cooperation_rate
    return (
        f"plain_final={plain.final_cooperation_rate} "
        f"think_final={think.final_cooperation_rate} "
        f"delta={delta} think_calls={stats.calls} "
        f"holds={stats.holds} flips={stats.flips} imitates={stats.imitates}"
    )


def format_decision_line(item: ThinkerDecision) -> str:
    """One inspectable line for a single thinker seat."""
    applied = "yes" if item.applied else "no"
    return (
        f"gen={item.generation} row={item.row} col={item.col} "
        f"act={item.act} worth={item.worth_thinking} conf={item.act_confidence} "
        f"applied={applied} before={item.before} after_imitate={item.after_imitate} "
        f"after_think={item.after_think}"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse thinker CLI flags."""
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
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Also run plain simulate() and print both final cooperation rates.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print one line per thinker seat (act, worth, confidence, applied).",
    )
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    *,
    client: object | None = None,
    questions: dict | None = None,
) -> int:
    """Run Spatial IPD with optional Jev thinker overrides."""
    load_dotenv()
    args = parse_args(argv)
    kwargs = dict(
        height=args.height,
        width=args.width,
        generations=args.generations,
        seed=args.seed,
        mutation_rate=args.mutation_rate,
    )
    result, stats = simulate_with_thinkers(
        **kwargs,
        think_every=args.think_every,
        thinker_count=args.thinkers,
        client=client,
        questions=questions,
    )
    if args.compare:
        plain = simulate(**kwargs)
        print(format_compare_summary(plain, result, stats))
    else:
        print(format_think_summary(result, stats))
    if args.verbose:
        for item in stats.all_decisions:
            print(format_decision_line(item))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
