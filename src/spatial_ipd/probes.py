"""Deterministic Spatial IPD motifs for inspecting thinker backends."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from spatial_ipd.engine import Grid, adopt_best, score_cells
from spatial_ipd.judgments import typesafe_thinker_questions
from spatial_ipd.label import load_dotenv
from spatial_ipd.local_llm import (
    DEFAULT_LOCAL_LLM_ENDPOINT,
    DEFAULT_LOCAL_LLM_TIMEOUT,
    LOCAL_LLM_REASONING_EFFORTS,
    LocalLLMThinkerClient,
)
from spatial_ipd.payoffs import COOPERATE
from spatial_ipd.think import (
    BACKEND_JEV,
    BACKEND_LOCAL,
    BACKEND_RANDOM,
    BACKENDS,
    RandomThinkerClient,
    decisions_from_response,
    is_cooperate_to_defect,
    state_for_thinkers,
    thinker_question_ids,
)


@dataclass(frozen=True)
class ProbeScenario:
    """One named lattice and focal seat evaluated after ordinary imitation."""

    id: str
    description: str
    before: tuple[tuple[int, ...], ...]
    seat: tuple[int, int]

    def before_grid(self) -> Grid:
        """Return a mutable copy suitable for engine functions."""
        return [list(row) for row in self.before]

    def after_grid(self) -> Grid:
        """Derive the post-imitation lattice from code-owned PD rules."""
        before = self.before_grid()
        return adopt_best(before, score_cells(before))


SCENARIOS = (
    ProbeScenario(
        id="lone_defector_invasion",
        description="A single high-scoring defector converts the surrounding cooperative torus.",
        before=((1, 1, 1), (1, 0, 1), (1, 1, 1)),
        seat=(0, 0),
    ),
    ProbeScenario(
        id="isolated_cooperator_collapse",
        description="A lone cooperator surrounded by defectors copies defection.",
        before=((0, 0, 0), (0, 1, 0), (0, 0, 0)),
        seat=(1, 1),
    ),
    ProbeScenario(
        id="pivotal_cluster_hold",
        description="A boundary cooperator is pivotal: one hold delays total collapse.",
        before=(
            (0, 0, 1, 1, 1),
            (0, 0, 1, 1, 1),
            (0, 0, 1, 1, 1),
            (1, 0, 0, 1, 1),
            (0, 0, 1, 1, 1),
        ),
        seat=(2, 3),
    ),
    ProbeScenario(
        id="toroidal_cluster_recovery",
        description="A seam defector copies an insulated cooperative cluster and becomes C.",
        before=(
            (0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0),
            (0, 0, 0, 1, 1),
            (1, 0, 0, 1, 1),
            (1, 0, 0, 1, 1),
        ),
        seat=(2, 0),
    ),
    ProbeScenario(
        id="stable_cluster_boundary",
        description="A 3x3 cooperative block holds its mixed boundary inside a defective field.",
        before=(
            (0, 0, 0, 0, 0),
            (0, 1, 1, 1, 0),
            (0, 1, 1, 1, 0),
            (0, 1, 1, 1, 0),
            (0, 0, 0, 0, 0),
        ),
        seat=(1, 1),
    ),
    ProbeScenario(
        id="uniform_cooperation",
        description="A uniform cooperative neighborhood remains cooperative and needs no override.",
        before=((1, 1, 1), (1, 1, 1), (1, 1, 1)),
        seat=(1, 1),
    ),
)


def transition_name(before: int, after: int) -> str:
    """Return a compact strategy transition label."""
    left = "C" if int(before) == COOPERATE else "D"
    right = "C" if int(after) == COOPERATE else "D"
    return f"{left}->{right}"


def _questions_for(
    client: object,
    before: Grid,
    after: Grid,
    seat: tuple[int, int],
) -> Mapping[str, object]:
    row, col = seat
    resist_indices = (0,) if is_cooperate_to_defect(before[row][col], after[row][col]) else ()
    if getattr(client, "uses_sdk_questions", True):
        return typesafe_thinker_questions(1, resist_indices=resist_indices)
    return thinker_question_ids(1, resist_indices)


def run_probe(scenario: ProbeScenario, client: object) -> dict[str, object]:
    """Ask one backend about a code-derived scenario and normalize the result."""
    before = scenario.before_grid()
    after = scenario.after_grid()
    seat = (scenario.seat,)
    state = state_for_thinkers(before, after, seat, generation=1)
    questions = _questions_for(client, before, after, scenario.seat)
    response = client.system_one(state=state, questions=questions)
    decision = decisions_from_response(seat, before, after, response, generation=1)[0]
    row, col = scenario.seat
    result = {
        "scenario": scenario.id,
        "description": scenario.description,
        "seat": {"row": row, "col": col},
        "transition": transition_name(before[row][col], after[row][col]),
        "patch_after": state["thinkers"][0]["patch_after"],
        "focal_score": decision.focal_score,
        "best_neighbor_score": decision.best_neighbor_score,
        "best_neighbor_strategy": decision.best_neighbor_strategy,
        "act": decision.act,
        "applied": decision.applied,
        "after_think": decision.after_think,
        "worth_signal": decision.worth_thinking,
        "worth_confidence": decision.worth_confidence,
        "resist_signal": (
            decision.act_confidence if is_cooperate_to_defect(decision.before, decision.after_imitate) else None
        ),
        "resist_confidence": decision.resist_confidence,
        "confidence_kind": decision.confidence_kind,
    }
    raw_output = getattr(response, "raw_output", None)
    if isinstance(raw_output, str):
        result["raw_output"] = raw_output
    return result


def run_probes(client: object, scenarios: Sequence[ProbeScenario] = SCENARIOS) -> list[dict[str, object]]:
    """Run an ordered scenario catalog through one reusable backend client."""
    return [run_probe(scenario, client) for scenario in scenarios]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse probe CLI flags."""
    parser = argparse.ArgumentParser(
        prog="python -m spatial_ipd.probes",
        description="Inspect one thinker backend on deterministic Spatial IPD motifs.",
    )
    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        default=BACKEND_RANDOM,
        help="random = seeded control; jev = TypeSafe; local = OpenAI-compatible endpoint.",
    )
    parser.add_argument("--seed", type=int, default=1, help="Seed for the random control backend.")
    parser.add_argument("--local-model", help="Model name for --backend local.")
    parser.add_argument("--local-endpoint", default=DEFAULT_LOCAL_LLM_ENDPOINT)
    parser.add_argument("--local-timeout", type=float, default=DEFAULT_LOCAL_LLM_TIMEOUT)
    parser.add_argument("--local-reasoning-effort", choices=LOCAL_LLM_REASONING_EFFORTS)
    args = parser.parse_args(argv)
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


def _print_results(results: Sequence[Mapping[str, object]]) -> None:
    for result in results:
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))


def main(argv: Sequence[str] | None = None, *, client: object | None = None) -> int:
    """Run and print the deterministic probe catalog."""
    load_dotenv()
    args = parse_args(argv)
    if client is not None:
        _print_results(run_probes(client))
        return 0
    if args.backend == BACKEND_RANDOM:
        _print_results(run_probes(RandomThinkerClient(args.seed)))
        return 0
    if args.backend == BACKEND_LOCAL:
        _print_results(run_probes(_local_client(args)))
        return 0
    if args.backend == BACKEND_JEV:
        try:
            from typesafe_sdk import TypeSafeClient
        except ImportError as exc:
            raise ImportError('TypeSafe is optional. Install with: python3 -m pip install -e ".[typesafe]"') from exc
        with TypeSafeClient() as opened:
            _print_results(run_probes(opened))
        return 0
    raise AssertionError(f"Unhandled backend: {args.backend}")


if __name__ == "__main__":
    raise SystemExit(main())
