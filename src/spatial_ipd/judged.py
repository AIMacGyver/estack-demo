"""One cooperate-or-defect judgment inside the active-agent arena."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from random import Random
from types import SimpleNamespace

from spatial_ipd.arena import Observation
from spatial_ipd.label import load_dotenv
from spatial_ipd.local_llm import (
    DEFAULT_LOCAL_LLM_ENDPOINT,
    DEFAULT_LOCAL_LLM_TIMEOUT,
    LOCAL_LLM_REASONING_EFFORTS,
    LocalCooperateClient,
)
from spatial_ipd.payoffs import COOPERATE, DEFECT
from spatial_ipd.population import (
    normalize_manifest,
    run_population,
    write_csv,
    write_jsonl,
)
from spatial_ipd.think import BACKEND_JEV, BACKEND_LOCAL, BACKEND_RANDOM, BACKENDS

JUDGED_POLICY = "judged"
COOPERATE_THRESHOLD = 0.5
COOPERATE_INSTRUCTIONS = (
    "Should this agent cooperate on this round of a repeated Prisoner's Dilemma? "
    "Use `round_index`, `own_history`, and `opponent_history`, where 1 is cooperate and 0 is defect. "
    "Use `opponent_reputation` and `opponent_warning_rate` when they are present. "
    "A null signal means that information is absent."
)
COOPERATE_CRITERIA = {
    "true": "Cooperating is the better action given the history and any signals.",
    "false": "Defecting is the better action given the history and any signals.",
}


class JudgedPolicy:
    """Cooperate when a backend's cooperate Noul is at least 0.5."""

    name = JUDGED_POLICY

    def __init__(self, client: object, records: list[dict[str, object]] | None = None):
        """Bind one backend client and an optional decision log."""
        self.client = client
        self.records = records

    def choose(self, observation: Observation) -> int:
        """Ask once, then apply the code-owned 0.5 gate."""
        state = {
            "round_index": observation.round_index,
            "own_history": list(observation.own_history),
            "opponent_history": list(observation.opponent_history),
            "opponent_reputation": observation.opponent_reputation,
            "opponent_warning_rate": observation.opponent_warning_rate,
        }
        response = self.client.system_one(state, {"cooperate": None})
        decision = response.nouls["cooperate"]
        noul = float(decision.noul)
        if not 0.0 <= noul <= 1.0:
            raise ValueError("cooperate noul must be in [0, 1]")
        action = COOPERATE if noul >= COOPERATE_THRESHOLD else DEFECT
        if self.records is not None:
            confidence = getattr(decision, "confidence", noul)
            self.records.append(
                {
                    "round_index": observation.round_index,
                    "own_history": list(observation.own_history),
                    "opponent_history": list(observation.opponent_history),
                    "opponent_reputation": observation.opponent_reputation,
                    "opponent_warning_rate": observation.opponent_warning_rate,
                    "noul": noul,
                    "confidence": None if confidence is None else float(confidence),
                    "confidence_kind": str(getattr(response, "confidence_kind", "noul")),
                    "action": action,
                }
            )
        return action


class RandomCooperateClient:
    """Seeded Uniform[0, 1] cooperate probability. No TypeSafe import."""

    uses_sdk_questions = False

    def __init__(self, seed: int):
        """Bind one random stream."""
        self._rng = Random(seed)

    def system_one(self, state: object, questions: Mapping[str, object]) -> SimpleNamespace:
        """Draw one cooperate Noul. The draw is both the probability and the confidence."""
        del state, questions
        noul = self._rng.random()
        return SimpleNamespace(
            nouls={"cooperate": SimpleNamespace(noul=noul, confidence=noul)},
            scores={},
            choices={},
            confidence_kind="random_draw",
        )


class ScriptedCooperateClient:
    """Return one fixed Noul. Tests and offline stand-ins use this."""

    uses_sdk_questions = False

    def __init__(self, noul: float, *, confidence_kind: str = "scripted"):
        """Bind a probability in [0, 1]."""
        if not 0.0 <= noul <= 1.0:
            raise ValueError("scripted noul must be in [0, 1]")
        self.noul = noul
        self.confidence_kind = confidence_kind

    def system_one(self, state: object, questions: Mapping[str, object]) -> SimpleNamespace:
        """Repeat the bound Noul."""
        del state, questions
        return SimpleNamespace(
            nouls={"cooperate": SimpleNamespace(noul=self.noul, confidence=self.noul)},
            scores={},
            choices={},
            confidence_kind=self.confidence_kind,
        )


class JevCooperateClient:
    """Ask TypeSafe whether to cooperate. The SDK stays optional."""

    uses_sdk_questions = True

    def __init__(self):
        """Open one TypeSafe client for the run."""
        try:
            from typesafe_sdk import Noul, TypeSafeClient
        except ImportError as exc:
            raise ImportError('TypeSafe is optional. Install with: python3 -m pip install -e ".[typesafe]"') from exc
        self._questions = {
            "cooperate": Noul(instructions=COOPERATE_INSTRUCTIONS, criteria=COOPERATE_CRITERIA),
        }
        self._context = TypeSafeClient()
        self._client = self._context.__enter__()

    def close(self) -> None:
        """Close the TypeSafe client."""
        self._context.__exit__(None, None, None)

    def system_one(self, state: object, questions: Mapping[str, object]) -> SimpleNamespace:
        """Forward one observation. Questions are the cooperate Noul, not the caller's placeholders."""
        del questions
        response = self._client.system_one(state=state, questions=self._questions)
        decision = response.nouls["cooperate"]
        return SimpleNamespace(
            nouls={"cooperate": SimpleNamespace(noul=float(decision.noul), confidence=float(decision.noul))},
            scores={},
            choices={},
            confidence_kind="jev_noul",
        )


def make_client(
    backend: str,
    seed: int,
    *,
    model: str | None = None,
    endpoint: str = DEFAULT_LOCAL_LLM_ENDPOINT,
    timeout: float = DEFAULT_LOCAL_LLM_TIMEOUT,
    reasoning_effort: str | None = None,
    transport: object | None = None,
) -> object:
    """Build the named cooperate client."""
    if backend == BACKEND_RANDOM:
        return RandomCooperateClient(seed)
    if backend == BACKEND_JEV:
        return JevCooperateClient()
    if backend == BACKEND_LOCAL:
        if not model:
            raise ValueError("local backend requires a model")
        return LocalCooperateClient(
            model,
            endpoint=endpoint,
            timeout=timeout,
            reasoning_effort=reasoning_effort,
            transport=transport,
        )
    raise ValueError(f"backend must be one of {BACKENDS}")


def run_judged(
    manifest: Mapping[str, object],
    client: object,
    records: list[dict[str, object]] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Run one fixed population whose judged seats share ``client``."""
    decisions = records if records is not None else []
    return run_population(
        normalize_manifest(manifest, extra_policies=(JUDGED_POLICY,)),
        factories={JUDGED_POLICY: lambda: JudgedPolicy(client, decisions)},
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse judged-population flags."""
    parser = argparse.ArgumentParser(
        prog="python -m spatial_ipd.judged",
        description="Run a fixed population where judged agents choose C or D from a backend.",
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--decisions")
    parser.add_argument("--backend", choices=BACKENDS, default=BACKEND_RANDOM)
    parser.add_argument("--local-model")
    parser.add_argument("--local-endpoint", default=DEFAULT_LOCAL_LLM_ENDPOINT)
    parser.add_argument("--local-timeout", type=float, default=DEFAULT_LOCAL_LLM_TIMEOUT)
    parser.add_argument("--local-reasoning-effort", choices=LOCAL_LLM_REASONING_EFFORTS)
    args = parser.parse_args(argv)
    if args.backend == BACKEND_LOCAL and not args.local_model:
        parser.error("--local-model is required for --backend local")
    return args


def main(argv: Sequence[str] | None = None, *, client: object | None = None) -> int:
    """Run one judged population and write its artifacts."""
    load_dotenv()
    args = parse_args(argv)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    owns_client = client is None
    if client is None:
        client = make_client(
            args.backend,
            int(manifest["seed"]),
            model=args.local_model,
            endpoint=args.local_endpoint,
            timeout=args.local_timeout,
            reasoning_effort=args.local_reasoning_effort,
        )
    records: list[dict[str, object]] = []
    try:
        events, summaries = run_judged(manifest, client, records)
    finally:
        if owns_client and hasattr(client, "close"):
            client.close()
    write_jsonl(args.jsonl, events, summaries)
    write_csv(args.csv, summaries)
    if args.decisions:
        Path(args.decisions).write_text(
            "".join(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n" for record in records),
            encoding="utf-8",
        )
    judged = next(summary for summary in summaries if summary["policy"] == JUDGED_POLICY)
    print(
        f"backend={args.backend} seed={manifest['seed']} "
        f"judged_cooperation={judged['cooperation_rate']} "
        f"judged_payoff={judged['payoff']} decisions={len(records)} "
        f"jsonl={args.jsonl} csv={args.csv}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
