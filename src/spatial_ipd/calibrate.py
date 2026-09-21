"""Counterfactual outcome labels and calibration metrics for thinker probes."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from random import Random
from statistics import fmean

from spatial_ipd.engine import Grid, adopt_best, cooperation_rate, random_grid, score_cells, step
from spatial_ipd.judgments import RESIST_YES_THRESHOLD
from spatial_ipd.label import load_dotenv
from spatial_ipd.local_llm import (
    DEFAULT_LOCAL_LLM_ENDPOINT,
    DEFAULT_LOCAL_LLM_TIMEOUT,
    LOCAL_LLM_REASONING_EFFORTS,
    LocalLLMThinkerClient,
)
from spatial_ipd.neighborhood import moore_neighbors
from spatial_ipd.payoffs import COOPERATE
from spatial_ipd.probes import SCENARIOS, ProbeScenario, run_probe, transition_name
from spatial_ipd.think import (
    BACKEND_JEV,
    BACKEND_LOCAL,
    BACKEND_RANDOM,
    BACKENDS,
    RandomThinkerClient,
)

TIE_EPSILON = 1e-12
LOG_EPSILON = 1e-15


@dataclass(frozen=True)
class CounterfactualOutcome:
    """Future cooperation under imitate and one-cell hold interventions."""

    horizon: int
    seed: int
    mutation_rate: float
    imitate_rates: tuple[float, ...]
    hold_rates: tuple[float, ...]
    imitate_mean: float
    hold_mean: float
    mean_delta: float
    imitate_final: float
    hold_final: float
    hold_better: bool | None


def _future_rates(
    grid: Grid,
    *,
    horizon: int,
    seed: int,
    mutation_rate: float,
) -> tuple[float, ...]:
    rng = Random(seed)
    current = [row[:] for row in grid]
    rates = []
    for _ in range(horizon):
        current = step(current, rng=rng, mutation_rate=mutation_rate)
        rates.append(cooperation_rate(current))
    return tuple(rates)


def cooperating_component(grid: Grid, seat: tuple[int, int]) -> set[tuple[int, int]]:
    """Return the Moore-connected cooperating component that contains ``seat``."""
    height = len(grid)
    width = len(grid[0])
    row, col = seat
    if int(grid[row][col]) != COOPERATE:
        raise ValueError("component seat must be cooperating")
    found: set[tuple[int, int]] = set()
    pending = [seat]
    while pending:
        current = pending.pop()
        if current in found:
            continue
        current_row, current_col = current
        if int(grid[current_row][current_col]) != COOPERATE:
            continue
        found.add(current)
        for neighbor in moore_neighbors(current_row, current_col, height, width):
            if neighbor not in found:
                pending.append(neighbor)
    return found


def restore_component(before: Grid, after: Grid, seat: tuple[int, int]) -> Grid:
    """Set the pre-imitation cooperating component back to C on the post-imitation grid."""
    restored = [row[:] for row in after]
    for row, col in cooperating_component(before, seat):
        restored[row][col] = COOPERATE
    return restored


def evaluate_counterfactual(
    scenario: ProbeScenario,
    *,
    horizon: int,
    seed: int,
    mutation_rate: float = 0.0,
    intervention: str = "cell",
) -> CounterfactualOutcome:
    """Label one C→D event by deterministic future cooperation after intervention."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if not 0.0 <= mutation_rate <= 1.0:
        raise ValueError("mutation_rate must be in [0, 1]")
    if intervention not in {"cell", "component"}:
        raise ValueError("intervention must be cell or component")
    before = scenario.before_grid()
    after = scenario.after_grid()
    row, col = scenario.seat
    transition = transition_name(before[row][col], after[row][col])
    if transition != "C->D":
        raise ValueError(f"scenario {scenario.id!r} is {transition}; calibration requires C->D")

    imitate = [line[:] for line in after]
    if intervention == "cell":
        hold = [line[:] for line in after]
        hold[row][col] = COOPERATE
    else:
        hold = restore_component(before, after, (row, col))
    imitate_rates = _future_rates(
        imitate,
        horizon=horizon,
        seed=seed,
        mutation_rate=mutation_rate,
    )
    hold_rates = _future_rates(
        hold,
        horizon=horizon,
        seed=seed,
        mutation_rate=mutation_rate,
    )
    imitate_mean = fmean(imitate_rates)
    hold_mean = fmean(hold_rates)
    delta = hold_mean - imitate_mean
    label = None if abs(delta) <= TIE_EPSILON else delta > 0.0
    return CounterfactualOutcome(
        horizon=horizon,
        seed=seed,
        mutation_rate=mutation_rate,
        imitate_rates=imitate_rates,
        hold_rates=hold_rates,
        imitate_mean=imitate_mean,
        hold_mean=hold_mean,
        mean_delta=delta,
        imitate_final=imitate_rates[-1],
        hold_final=hold_rates[-1],
        hold_better=label,
    )


def count_hold_yield(
    *,
    seeds: Sequence[int],
    height: int = 8,
    width: int = 8,
    generations: int = 10,
    horizon: int = 3,
    mutation_rate: float = 0.0,
    intervention: str = "cell",
) -> dict[str, int]:
    """Count resolved labels for imitation C→D events on seeded lattices."""
    if not seeds:
        raise ValueError("seeds must not be empty")
    if mutation_rate != 0.0:
        raise ValueError("yield counts are defined for mutation_rate 0")
    counts = {"events": 0, "positive": 0, "negative": 0, "tie": 0}
    for seed in seeds:
        grid = random_grid(height, width, seed=int(seed))
        for generation in range(generations):
            after = adopt_best(grid, score_cells(grid))
            for row, before_row in enumerate(grid):
                for col, cell in enumerate(before_row):
                    if int(cell) != COOPERATE or int(after[row][col]) == COOPERATE:
                        continue
                    scenario = ProbeScenario(
                        id=f"yield-{seed}-{generation}-{row}-{col}",
                        description="mined C-to-D imitation event",
                        before=tuple(tuple(int(value) for value in line) for line in grid),
                        seat=(row, col),
                    )
                    outcome = evaluate_counterfactual(
                        scenario,
                        horizon=horizon,
                        seed=int(seed) + generation,
                        mutation_rate=0.0,
                        intervention=intervention,
                    )
                    counts["events"] += 1
                    if outcome.hold_better is True:
                        counts["positive"] += 1
                    elif outcome.hold_better is False:
                        counts["negative"] += 1
                    else:
                        counts["tie"] += 1
            grid = after
    return counts


def resist_true_probability(probe: Mapping[str, object]) -> float:
    """Orient a backend signal as P(resist=true) without changing its action."""
    signal = probe.get("resist_signal")
    if isinstance(signal, bool) or not isinstance(signal, (int, float)):
        raise ValueError("probe resist_signal must be numeric")
    signal_value = float(signal)
    kind = probe.get("confidence_kind")
    if kind == "llm_self_report":
        confidence = probe.get("resist_confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError("local LLM resist_confidence must be numeric")
        confidence_value = float(confidence)
        if not 0.0 <= confidence_value <= 1.0:
            raise ValueError("local LLM resist_confidence must be in [0, 1]")
        if signal_value not in (0.0, 1.0):
            raise ValueError("local LLM resist_signal must be a boolean mapped to 0 or 1")
        return confidence_value if signal_value == 1.0 else 1.0 - confidence_value
    if not 0.0 <= signal_value <= 1.0:
        raise ValueError("probe resist_signal must be in [0, 1]")
    return signal_value


def brier_score(probabilities: Sequence[float], labels: Sequence[bool]) -> float:
    """Mean squared probability error for binary outcomes."""
    if len(probabilities) != len(labels) or not probabilities:
        raise ValueError("probabilities and labels must have the same non-zero length")
    if any(not 0.0 <= probability <= 1.0 for probability in probabilities):
        raise ValueError("probabilities must be in [0, 1]")
    return fmean((probability - float(label)) ** 2 for probability, label in zip(probabilities, labels, strict=True))


def binary_log_loss(probabilities: Sequence[float], labels: Sequence[bool]) -> float:
    """Mean clipped negative log-likelihood for binary outcomes."""
    if len(probabilities) != len(labels) or not probabilities:
        raise ValueError("probabilities and labels must have the same non-zero length")
    losses = []
    for probability, label in zip(probabilities, labels, strict=True):
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probabilities must be in [0, 1]")
        clipped = min(1.0 - LOG_EPSILON, max(LOG_EPSILON, probability))
        losses.append(-math.log(clipped if label else 1.0 - clipped))
    return fmean(losses)


def calibration_summary(events: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Score only resolved labels and report ties separately."""
    resolved = [event for event in events if isinstance(event.get("hold_better"), bool)]
    probabilities = [float(event["resist_probability"]) for event in resolved]
    labels = [bool(event["hold_better"]) for event in resolved]
    return {
        "record_type": "summary",
        "events": len(events),
        "resolved": len(resolved),
        "ties": len(events) - len(resolved),
        "brier_score": brier_score(probabilities, labels) if resolved else None,
        "log_loss": binary_log_loss(probabilities, labels) if resolved else None,
        "calibration_claim_supported": False,
        "note": "This tiny deterministic motif set is diagnostic evidence, not a calibration claim.",
    }


def run_calibration(
    client: object,
    *,
    horizon: int,
    seed: int,
    mutation_rate: float = 0.0,
    scenarios: Sequence[ProbeScenario] = SCENARIOS,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Collect backend probabilities and code-owned outcomes for C→D motifs."""
    events = []
    for scenario in scenarios:
        before = scenario.before_grid()
        after = scenario.after_grid()
        row, col = scenario.seat
        if transition_name(before[row][col], after[row][col]) != "C->D":
            continue
        probe = run_probe(scenario, client)
        outcome = evaluate_counterfactual(
            scenario,
            horizon=horizon,
            seed=seed,
            mutation_rate=mutation_rate,
        )
        events.append(
            {
                "record_type": "event",
                "scenario": scenario.id,
                "confidence_kind": probe["confidence_kind"],
                "resist_signal": probe["resist_signal"],
                "resist_confidence": probe["resist_confidence"],
                "resist_probability": resist_true_probability(probe),
                "resist_decision": float(probe["resist_signal"]) >= RESIST_YES_THRESHOLD,
                "applied": probe["applied"],
                **asdict(outcome),
            }
        )
    return events, calibration_summary(events)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse calibration CLI flags."""
    parser = argparse.ArgumentParser(
        prog="python -m spatial_ipd.calibrate",
        description="Compare thinker resist probabilities with deterministic motif outcomes.",
    )
    parser.add_argument("--backend", choices=BACKENDS, default=BACKEND_RANDOM)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--mutation-rate", type=float, default=0.0)
    parser.add_argument("--local-model", help="Model name for --backend local.")
    parser.add_argument("--local-endpoint", default=DEFAULT_LOCAL_LLM_ENDPOINT)
    parser.add_argument("--local-timeout", type=float, default=DEFAULT_LOCAL_LLM_TIMEOUT)
    parser.add_argument("--local-reasoning-effort", choices=LOCAL_LLM_REASONING_EFFORTS)
    args = parser.parse_args(argv)
    if args.horizon < 1:
        parser.error("--horizon must be positive")
    if not 0.0 <= args.mutation_rate <= 1.0:
        parser.error("--mutation-rate must be in [0, 1]")
    if args.backend == BACKEND_LOCAL and not args.local_model:
        parser.error("--local-model is required when --backend local")
    return args


def _local_client(args: argparse.Namespace) -> LocalLLMThinkerClient:
    return LocalLLMThinkerClient(
        args.local_model,
        endpoint=args.local_endpoint,
        api_key=os.environ.get("LOCAL_LLM_API_KEY"),
        timeout=args.local_timeout,
        reasoning_effort=args.local_reasoning_effort,
    )


def _print_calibration(client: object, args: argparse.Namespace) -> None:
    events, summary = run_calibration(
        client,
        horizon=args.horizon,
        seed=args.seed,
        mutation_rate=args.mutation_rate,
    )
    for record in (*events, summary):
        print(json.dumps(record, sort_keys=True, separators=(",", ":")))


def main(argv: Sequence[str] | None = None, *, client: object | None = None) -> int:
    """Run counterfactual calibration diagnostics for one backend."""
    load_dotenv()
    args = parse_args(argv)
    if client is not None:
        _print_calibration(client, args)
        return 0
    if args.backend == BACKEND_RANDOM:
        _print_calibration(RandomThinkerClient(args.seed), args)
        return 0
    if args.backend == BACKEND_LOCAL:
        _print_calibration(_local_client(args), args)
        return 0
    if args.backend == BACKEND_JEV:
        try:
            from typesafe_sdk import TypeSafeClient
        except ImportError as exc:
            raise ImportError('TypeSafe is optional. Install with: python3 -m pip install -e ".[typesafe]"') from exc
        with TypeSafeClient() as opened:
            _print_calibration(opened, args)
        return 0
    raise AssertionError(f"Unhandled backend: {args.backend}")


if __name__ == "__main__":
    raise SystemExit(main())
