"""Opt-in dual-process step: most cells imitate; a few ask Jev."""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from random import Random
from types import SimpleNamespace

from spatial_ipd.engine import (
    Grid,
    SimulationResult,
    cooperation_rate,
    random_grid,
    score_cells,
    simulate,
    step,
)
from spatial_ipd.judgments import (
    RESIST_YES_THRESHOLD,
    WORTH_THINKING_THRESHOLD,
    typesafe_thinker_questions,
)
from spatial_ipd.label import load_dotenv
from spatial_ipd.local_llm import (
    DEFAULT_LOCAL_LLM_ENDPOINT,
    DEFAULT_LOCAL_LLM_TIMEOUT,
    LOCAL_LLM_REASONING_EFFORTS,
    LocalLLMThinkerClient,
)
from spatial_ipd.neighborhood import moore_neighbors
from spatial_ipd.payoffs import COOPERATE, DEFECT
from spatial_ipd.replay import (
    DecisionRecord,
    ReplayThinkerClient,
    capture_decision_record,
    write_decision_records,
)

SEAT_RANDOM = "random"
SEAT_FRONTIER = "frontier"
SEAT_MODES = (SEAT_RANDOM, SEAT_FRONTIER)
BACKEND_JEV = "jev"
BACKEND_LOCAL = "local"
BACKEND_RANDOM = "random"
BACKENDS = (BACKEND_JEV, BACKEND_LOCAL, BACKEND_RANDOM)


def _copy_grid(grid: Grid) -> Grid:
    return [row[:] for row in grid]


@dataclass(frozen=True)
class ThinkerDecision:
    """One backend decision for a thinker seat after imitation."""

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
    focal_score: int = 0
    best_neighbor_score: int = 0
    best_neighbor_strategy: int = 0
    worth_confidence: float | None = None
    resist_confidence: float | None = None
    confidence_kind: str = "jev_noul"


@dataclass(frozen=True)
class BackendAudit:
    """Raw backend output retained with its generation and confidence semantics."""

    generation: int
    confidence_kind: str
    raw_output: str


@dataclass(frozen=True)
class StickySeat:
    """A held cell that resists imitation until ``until_generation``."""

    row: int
    col: int
    strategy: int
    until_generation: int


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
    backend_audits: list[BackendAudit] = field(default_factory=list)
    decision_records: list[DecisionRecord] = field(default_factory=list)


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


def is_strategy_frontier(grid: Grid, row: int, col: int) -> bool:
    """True if a Moore neighbor plays a different strategy than this cell."""
    height, width = len(grid), len(grid[0])
    focal = int(grid[row][col])
    return any(int(grid[nr][nc]) != focal for nr, nc in moore_neighbors(row, col, height, width))


def frontier_pools(
    before: Grid,
    after: Grid,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    """Changed cells, then C/D edges, then the rest (each list unique)."""
    height, width = len(after), len(after[0])
    changed: list[tuple[int, int]] = []
    edge: list[tuple[int, int]] = []
    rest: list[tuple[int, int]] = []
    for row in range(height):
        for col in range(width):
            seat = (row, col)
            if int(before[row][col]) != int(after[row][col]):
                changed.append(seat)
            elif is_strategy_frontier(after, row, col):
                edge.append(seat)
            else:
                rest.append(seat)
    return changed, edge, rest


def frontier_seats(
    before: Grid,
    after: Grid,
    count: int,
    *,
    seed: int,
    generation: int,
) -> tuple[tuple[int, int], ...]:
    """Deterministic seats: prefer just-changed cells, then C/D frontier."""
    if count < 1:
        return ()
    rng = Random((seed + 1) * 10007 + generation)
    picked: list[tuple[int, int]] = []
    for pool in frontier_pools(before, after):
        order = list(pool)
        rng.shuffle(order)
        for seat in order:
            if len(picked) >= count:
                return tuple(picked)
            picked.append(seat)
    return tuple(picked[:count])


def choose_seats(
    height: int,
    width: int,
    count: int,
    *,
    seed: int,
    generation: int,
    mode: str = SEAT_FRONTIER,
    before: Grid | None = None,
    after: Grid | None = None,
) -> tuple[tuple[int, int], ...]:
    """Pick thinker seats using ``mode`` (frontier needs before/after grids)."""
    if mode == SEAT_FRONTIER and before is not None and after is not None:
        return frontier_seats(before, after, count, seed=seed, generation=generation)
    return thinker_seats(height, width, count, seed=seed, generation=generation)


def resolve_strategy(before: int, after_imitate: int, act: str) -> int:
    """Map a Jev act onto a cell strategy. Unknown acts keep imitation."""
    if act == "hold":
        return int(before)
    if act == "flip":
        return COOPERATE if int(after_imitate) == DEFECT else DEFECT
    return int(after_imitate)


def should_resist(worth_thinking: float, resist: float, before: int, after_imitate: int) -> bool:
    """Hold only on a C→D copy when worth and resist nouls clear the gate."""
    return (
        is_cooperate_to_defect(before, after_imitate)
        and worth_thinking >= WORTH_THINKING_THRESHOLD
        and resist >= RESIST_YES_THRESHOLD
    )


def is_cooperate_to_defect(before: int, after_imitate: int) -> bool:
    """True when imitation just turned a cooperator into a defector."""
    return int(before) == COOPERATE and int(after_imitate) == DEFECT


class RandomThinkerClient:
    """Seeded Uniform[0, 1] stand-in for worth/resist. No TypeSafe import."""

    uses_sdk_questions = False

    def __init__(self, seed: int):
        """Bind a ``random.Random`` stream to ``seed``."""
        self._rng = Random(seed)

    def system_one(self, state: object, questions: Mapping[str, object]) -> SimpleNamespace:
        """Draw one noul per worth/resist key; ignore unused fragility keys."""
        del state
        nouls = {}
        scores = {}
        for key in questions:
            if key.startswith("worth_thinking_") or key.startswith("resist_"):
                nouls[key] = SimpleNamespace(noul=self._rng.random())
            elif key.startswith("cluster_fragility_"):
                scores[key] = SimpleNamespace(score=self._rng.uniform(0, 2), confidence=0.5)
        return SimpleNamespace(
            nouls=nouls,
            scores=scores,
            choices={},
            confidence_kind="random_draw",
        )


def thinker_question_ids(count: int, resist_indices: Sequence[int]) -> dict[str, None]:
    """Question keys only, so a random backend does not need typesafe_sdk."""
    if count < 1:
        return {}
    ask_resist = {int(i) for i in resist_indices}
    keys: dict[str, None] = {}
    for i in range(count):
        keys[f"worth_thinking_{i}"] = None
        if i in ask_resist:
            keys[f"resist_{i}"] = None
        keys[f"cluster_fragility_{i}"] = None
    return keys


def resist_indices_for_seats(
    seats: Sequence[tuple[int, int]],
    before: Grid,
    after: Grid,
) -> tuple[int, ...]:
    """Thinker indexes that are real C→D copies (the only seats that get resist)."""
    return tuple(i for i, (row, col) in enumerate(seats) if is_cooperate_to_defect(before[row][col], after[row][col]))


def winning_imitate(grid: Grid, scores: list[list[int]], row: int, col: int) -> tuple[int, int]:
    """Return (best_score, best_strategy) using the same rule as ``adopt_best``."""
    height, width = len(grid), len(grid[0])
    best_score = scores[row][col]
    best_strategy = int(grid[row][col])
    for nr, nc in moore_neighbors(row, col, height, width):
        neighbor_score = scores[nr][nc]
        if neighbor_score > best_score:
            best_score = neighbor_score
            best_strategy = int(grid[nr][nc])
    return int(best_score), int(best_strategy)


def state_for_thinkers(
    before: Grid,
    after: Grid,
    seats: Sequence[tuple[int, int]],
    generation: int,
) -> dict:
    """Build the named JSON state for one TypeSafe thinker call."""
    scores = score_cells(before)
    thinkers = []
    for row, col in seats:
        best_score, best_strategy = winning_imitate(before, scores, row, col)
        thinkers.append(
            {
                "row": row,
                "col": col,
                "before": int(before[row][col]),
                "after_imitate": int(after[row][col]),
                "patch_after": patch3(after, row, col),
                "focal_score": int(scores[row][col]),
                "best_neighbor_score": best_score,
                "best_neighbor_strategy": best_strategy,
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
    scores = score_cells(before)
    confidence_kind = str(getattr(response, "confidence_kind", "jev_noul"))
    out: list[ThinkerDecision] = []
    for i, (row, col) in enumerate(seats):
        worth_answer = response.nouls[f"worth_thinking_{i}"]
        worth = float(worth_answer.noul)
        worth_confidence = (
            float(getattr(worth_answer, "confidence", worth)) if confidence_kind == "llm_self_report" else worth
        )
        resist_key = f"resist_{i}"
        resist_answer = response.nouls.get(resist_key)
        resist = float(resist_answer.noul) if resist_answer is not None else 0.0
        resist_confidence = (
            float(getattr(resist_answer, "confidence", resist))
            if resist_answer is not None and confidence_kind == "llm_self_report"
            else (resist if resist_answer is not None else None)
        )
        pre = int(before[row][col])
        mid = int(after[row][col])
        applied = should_resist(worth, resist, pre, mid)
        act = "hold" if applied else "imitate"
        nxt = resolve_strategy(pre, mid, act) if applied else mid
        best_score, best_strategy = winning_imitate(before, scores, row, col)
        out.append(
            ThinkerDecision(
                generation=generation,
                row=row,
                col=col,
                act=act,
                worth_thinking=worth,
                act_confidence=resist,
                applied=applied,
                before=pre,
                after_imitate=mid,
                after_think=nxt,
                focal_score=int(scores[row][col]),
                best_neighbor_score=best_score,
                best_neighbor_strategy=best_strategy,
                worth_confidence=worth_confidence,
                resist_confidence=resist_confidence,
                confidence_kind=confidence_kind,
            )
        )
    return out


def apply_decisions(grid: Grid, decisions: Sequence[ThinkerDecision]) -> Grid:
    """Return a new grid with applied thinker overrides."""
    nxt = _copy_grid(grid)
    for item in decisions:
        nxt[item.row][item.col] = item.after_think
    return nxt


def register_holds(
    book: tuple[StickySeat, ...],
    decisions: Sequence[ThinkerDecision],
    generation: int,
    sticky: int,
) -> tuple[StickySeat, ...]:
    """Remember applied holds so later ``step`` calls can restore them."""
    if sticky < 1:
        return book
    by_cell = {(seat.row, seat.col): seat for seat in book}
    for item in decisions:
        if item.applied and item.act == "hold":
            by_cell[(item.row, item.col)] = StickySeat(
                row=item.row,
                col=item.col,
                strategy=item.after_think,
                until_generation=generation + sticky,
            )
    return tuple(by_cell.values())


def apply_sticky(
    grid: Grid,
    book: tuple[StickySeat, ...],
    generation: int,
) -> tuple[Grid, tuple[StickySeat, ...]]:
    """Restore still-live holds after imitation; drop expired seats."""
    nxt = _copy_grid(grid)
    kept: list[StickySeat] = []
    for seat in book:
        if generation <= seat.until_generation:
            nxt[seat.row][seat.col] = seat.strategy
            kept.append(seat)
    return nxt, tuple(kept)


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
    seat_mode: str = SEAT_FRONTIER,
) -> tuple[Grid, ThinkerStats]:
    """Maybe override a few cells after a normal ``step``."""
    tally = stats if stats is not None else ThinkerStats()
    height, width = len(after), len(after[0])
    seats = choose_seats(
        height,
        width,
        thinker_count,
        seed=seed,
        generation=generation,
        mode=seat_mode,
        before=before,
        after=after,
    )
    tally.last_seats = seats
    if not seats:
        tally.last_decisions = ()
        return _copy_grid(after), tally

    state = state_for_thinkers(before, after, seats, generation)
    resist_ix = resist_indices_for_seats(seats, before, after)
    if questions is None:
        if client is not None and not getattr(client, "uses_sdk_questions", True):
            qs = thinker_question_ids(len(seats), resist_ix)
        else:
            qs = typesafe_thinker_questions(len(seats), resist_indices=resist_ix)
    else:
        qs = questions
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
    tally.decision_records.append(capture_decision_record(state, qs, response))
    raw_output = getattr(response, "raw_output", None)
    if isinstance(raw_output, str):
        tally.backend_audits.append(
            BackendAudit(
                generation=generation,
                confidence_kind=str(getattr(response, "confidence_kind", "unknown")),
                raw_output=raw_output,
            )
        )
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


def should_think(generation: int, generations: int, think_every: int, think_last: int) -> bool:
    """True when this generation gets a Jev pass (every N, plus the last M)."""
    if think_every < 1 or generation < 1:
        return False
    if generation % think_every == 0:
        return True
    return think_last > 0 and generation > generations - think_last


def simulate_with_thinkers(
    height: int,
    width: int,
    generations: int,
    seed: int,
    *,
    mutation_rate: float = 0.0,
    cooperate_p: float = 0.5,
    think_every: int = 5,
    think_last: int = 5,
    sticky: int = 5,
    thinker_count: int = 4,
    client: object | None = None,
    questions: dict | None = None,
    seat_mode: str = SEAT_FRONTIER,
) -> tuple[SimulationResult, ThinkerStats]:
    """Like ``simulate``, plus optional Jev overrides every ``think_every`` gens.

    ``think_every=0`` is exactly ``simulate`` (no API calls).
    ``think_last`` also thinks on every generation in the last M (deduped).
    ``sticky`` keeps an applied hold for that many later generations.
    """
    if think_every < 0:
        raise ValueError("think_every must be non-negative")
    if think_last < 0:
        raise ValueError("think_last must be non-negative")
    if sticky < 0:
        raise ValueError("sticky must be non-negative")
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
    book: tuple[StickySeat, ...] = ()
    for generation in range(1, generations + 1):
        before = _copy_grid(grid)
        grid = step(grid, rng=rng, mutation_rate=mutation_rate)
        grid, book = apply_sticky(grid, book, generation)
        if should_think(generation, generations, think_every, think_last):
            grid, stats = think_after_step(
                before,
                grid,
                generation=generation,
                seed=seed,
                thinker_count=thinker_count,
                client=client,
                questions=questions,
                stats=stats,
                seat_mode=seat_mode,
            )
            book = register_holds(book, stats.last_decisions, generation, sticky)
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
    worth_confidence = item.worth_thinking if item.worth_confidence is None else item.worth_confidence
    resist = (
        f"resist={item.act_confidence} "
        f"resist_confidence={item.act_confidence if item.resist_confidence is None else item.resist_confidence}"
        if is_cooperate_to_defect(item.before, item.after_imitate)
        else "resist=n/a resist_confidence=n/a"
    )
    return (
        f"gen={item.generation} row={item.row} col={item.col} "
        f"act={item.act} confidence_kind={item.confidence_kind} "
        f"worth={item.worth_thinking} worth_confidence={worth_confidence} {resist} "
        f"focal_score={item.focal_score} best_neighbor_score={item.best_neighbor_score} "
        f"best_neighbor_strategy={item.best_neighbor_strategy} "
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
    parser.add_argument(
        "--think-last",
        type=int,
        default=5,
        help="Also think every generation in the last N (Jev pick). 0 = only --think-every.",
    )
    parser.add_argument(
        "--sticky",
        type=int,
        default=5,
        help="Keep an applied hold for N later gens (Jev pick). 0 = one-shot hold.",
    )
    parser.add_argument("--thinkers", type=int, default=4)
    parser.add_argument(
        "--seats",
        choices=SEAT_MODES,
        default=SEAT_FRONTIER,
        help="random = old uniform seats; frontier = just-changed then C/D edges (Jev pick).",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Also run plain simulate() and print both final cooperation rates.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print one line per thinker seat (act, worth, resist noul, imitate scores, applied).",
    )
    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        default=BACKEND_JEV,
        help=("jev = TypeSafe; local = OpenAI-compatible chat completions; random = seeded Uniform[0,1] control."),
    )
    parser.add_argument(
        "--local-model",
        help="Model name for --backend local (for example qwen3:8b).",
    )
    parser.add_argument(
        "--local-endpoint",
        default=DEFAULT_LOCAL_LLM_ENDPOINT,
        help=f"Full OpenAI-compatible chat-completions URL (default: {DEFAULT_LOCAL_LLM_ENDPOINT}).",
    )
    parser.add_argument(
        "--local-timeout",
        type=float,
        default=DEFAULT_LOCAL_LLM_TIMEOUT,
        help=f"Local endpoint timeout in seconds (default: {DEFAULT_LOCAL_LLM_TIMEOUT:g}).",
    )
    parser.add_argument(
        "--local-reasoning-effort",
        choices=LOCAL_LLM_REASONING_EFFORTS,
        help="Optional OpenAI reasoning control; use 'none' to disable thinking models.",
    )
    decisions = parser.add_mutually_exclusive_group()
    decisions.add_argument(
        "--record-decisions",
        metavar="PATH",
        help="Write each successful thinker call as versioned JSONL.",
    )
    decisions.add_argument(
        "--replay-decisions",
        metavar="PATH",
        help="Replay versioned JSONL without calling a live thinker backend.",
    )
    args = parser.parse_args(argv)
    if args.backend == BACKEND_LOCAL and not args.local_model and not args.replay_decisions:
        parser.error("--local-model is required when --backend local")
    return args


def main(
    argv: Sequence[str] | None = None,
    *,
    client: object | None = None,
    questions: dict | None = None,
) -> int:
    """Run Spatial IPD with optional Jev thinker overrides."""
    load_dotenv()
    args = parse_args(argv)
    replay_client = None
    if args.replay_decisions:
        if client is not None:
            raise ValueError("--replay-decisions cannot be combined with an injected client")
        replay_client = ReplayThinkerClient.from_path(args.replay_decisions)
        client = replay_client
    elif client is None and args.backend == BACKEND_RANDOM:
        client = RandomThinkerClient(seed=args.seed)
    elif client is None and args.backend == BACKEND_LOCAL:
        client = LocalLLMThinkerClient(
            args.local_model,
            endpoint=args.local_endpoint,
            api_key=os.environ.get("LOCAL_LLM_API_KEY"),
            timeout=args.local_timeout,
            reasoning_effort=args.local_reasoning_effort,
        )
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
        think_last=args.think_last,
        sticky=args.sticky,
        thinker_count=args.thinkers,
        client=client,
        questions=questions,
        seat_mode=args.seats,
    )
    if replay_client is not None:
        replay_client.assert_exhausted()
    if args.record_decisions:
        write_decision_records(args.record_decisions, stats.decision_records)
    if args.compare:
        plain = simulate(**kwargs)
        print(format_compare_summary(plain, result, stats))
    else:
        print(format_think_summary(result, stats))
    if args.verbose:
        for item in stats.all_decisions:
            print(format_decision_line(item))
    if args.record_decisions:
        print(f"decision_records_written={len(stats.decision_records)} path={args.record_decisions}")
    elif args.replay_decisions:
        print(f"decision_records_replayed={replay_client.consumed} path={args.replay_decisions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
