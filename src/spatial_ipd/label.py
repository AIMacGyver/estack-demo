"""Optional TypeSafe labels for a seeded Spatial IPD cooperation-rate series."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from spatial_ipd.engine import SimulationResult, simulate
from spatial_ipd.judgments import (
    COOPERATION_SURVIVED_YES_THRESHOLD,
    typesafe_questions,
)


def load_dotenv(path: Path | None = None) -> Path | None:
    """Load KEY=VALUE pairs from `.env` without overwriting existing env vars."""
    candidate = Path(path) if path is not None else Path.cwd() / ".env"
    if not candidate.is_file():
        return None
    for raw in candidate.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value
    return candidate


@dataclass(frozen=True)
class SeriesLabel:
    """Typed judgments about a finished simulation (not a new PD rule)."""

    regime: str
    regime_confidence: float
    demo_worthy: float
    demo_worthy_confidence: float
    cooperation_survived: float
    cooperation_survived_yes: bool


def state_from_result(
    result: SimulationResult,
    *,
    height: int,
    width: int,
    mutation_rate: float,
) -> dict:
    """Named facts for TypeSafe. Rates are already computed by `simulate`."""
    rates = list(result.cooperation_rates)
    return {
        "height": height,
        "width": width,
        "generations": result.generations,
        "seed": result.seed,
        "mutation_rate": mutation_rate,
        "initial_cooperation_rate": rates[0],
        "final_cooperation_rate": result.final_cooperation_rate,
        "min_cooperation_rate": min(rates),
        "max_cooperation_rate": max(rates),
        "cooperation_rates": rates,
    }


def label_from_response(response: object) -> SeriesLabel:
    """Map a TypeSafe (or test double) response onto SeriesLabel."""
    regime = response.choices["regime"]
    demo = response.scores["demo_worthy"]
    survived = response.nouls["cooperation_survived"]
    noul = float(survived.noul)
    return SeriesLabel(
        regime=str(regime.choice),
        regime_confidence=float(regime.confidence),
        demo_worthy=float(demo.score),
        demo_worthy_confidence=float(demo.confidence),
        cooperation_survived=noul,
        cooperation_survived_yes=noul >= COOPERATION_SURVIVED_YES_THRESHOLD,
    )


def label_series(
    state: dict,
    *,
    client: object | None = None,
    questions: dict | None = None,
) -> SeriesLabel:
    """Ask the three series questions in one System One call."""
    qs = typesafe_questions() if questions is None else questions
    if client is None:
        try:
            from typesafe_sdk import TypeSafeClient
        except ImportError as exc:
            raise ImportError('TypeSafe is optional. Install with: python3 -m pip install -e ".[typesafe]"') from exc
        with TypeSafeClient() as opened:
            return label_from_response(opened.system_one(state=state, questions=qs))
    return label_from_response(client.system_one(state=state, questions=qs))


def format_label(label: SeriesLabel) -> str:
    """One-line summary printed by the CLI."""
    survived = "yes" if label.cooperation_survived_yes else "no"
    return (
        f"regime={label.regime} regime_confidence={label.regime_confidence} "
        f"cooperation_survived={label.cooperation_survived} survived={survived} "
        f"demo_worthy={label.demo_worthy}"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse label CLI flags."""
    parser = argparse.ArgumentParser(
        prog="python -m spatial_ipd.label",
        description=(
            "Run a seeded Spatial IPD simulation, then label the cooperation-rate "
            "series with TypeSafe (optional extra; needs TYPESAFE_API_KEY)."
        ),
    )
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--generations", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mutation-rate", type=float, default=0.0)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the label as JSON instead of the one-line summary.",
    )
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    *,
    client: object | None = None,
    questions: dict | None = None,
) -> int:
    """Run a seeded simulation and print TypeSafe series labels."""
    load_dotenv()
    args = parse_args(argv)
    result = simulate(
        args.height,
        args.width,
        args.generations,
        seed=args.seed,
        mutation_rate=args.mutation_rate,
    )
    state = state_from_result(
        result,
        height=args.height,
        width=args.width,
        mutation_rate=args.mutation_rate,
    )
    label = label_series(state, client=client, questions=questions)
    if args.json:
        print(json.dumps(asdict(label), sort_keys=True))
    else:
        print(format_label(label))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
