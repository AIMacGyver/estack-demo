"""Offline matched-decision reports for local-model experiment ablations."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from statistics import fmean

from spatial_ipd.experiment import ExperimentError, read_result_records


class AblationError(ValueError):
    """Malformed or unmatched experiment evidence."""


def _backend_id(record: Mapping[str, object]) -> str:
    backend = record.get("backend")
    if not isinstance(backend, Mapping) or not isinstance(backend.get("id"), str):
        raise AblationError("experiment record is missing backend.id")
    return str(backend["id"])


def _decision_key(decision: Mapping[str, object]) -> tuple[int, int, int]:
    try:
        return (int(decision["generation"]), int(decision["row"]), int(decision["col"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise AblationError("decision is missing generation, row, or col") from exc


def _decisions(record: Mapping[str, object]) -> dict[tuple[int, int, int], Mapping[str, object]]:
    values = record.get("decisions")
    if not isinstance(values, list):
        raise AblationError("successful local record is missing decisions")
    decisions = {}
    for value in values:
        if not isinstance(value, Mapping):
            raise AblationError("decision entries must be objects")
        key = _decision_key(value)
        if key in decisions:
            raise AblationError(f"duplicate decision key {key}")
        decisions[key] = value
    return decisions


def _resist_probability(decision: Mapping[str, object]) -> float | None:
    confidence = decision.get("resist_confidence")
    if confidence is None:
        return None
    signal = decision.get("act_confidence")
    if not isinstance(signal, (int, float)) or not isinstance(confidence, (int, float)):
        raise AblationError("local resist signal/confidence must be numeric")
    signal_value = float(signal)
    confidence_value = float(confidence)
    if signal_value not in (0.0, 1.0) or not 0.0 <= confidence_value <= 1.0:
        raise AblationError("local resist signal must be 0/1 and confidence must be in [0, 1]")
    return confidence_value if signal_value == 1.0 else 1.0 - confidence_value


def _latest_by_backend_seed(
    records: Sequence[Mapping[str, object]],
) -> dict[tuple[str, int], Mapping[str, object]]:
    latest = {}
    for record in records:
        backend = record.get("backend")
        if not isinstance(backend, Mapping) or backend.get("kind") != "local":
            continue
        seed = record.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise AblationError("local experiment record is missing integer seed")
        latest[(_backend_id(record), seed)] = record
    return latest


def compare_local_ablations(
    records: Sequence[Mapping[str, object]],
    *,
    reference_id: str,
) -> list[dict[str, object]]:
    """Compare every local backend with a reference on identical seeds/seats."""
    latest = _latest_by_backend_seed(records)
    backend_ids = sorted({backend_id for backend_id, _seed in latest})
    if reference_id not in backend_ids:
        raise AblationError(f"reference backend {reference_id!r} was not found")
    candidates = [backend_id for backend_id in backend_ids if backend_id != reference_id]
    if not candidates:
        raise AblationError("at least two local backend configurations are required")

    reference_seeds = {seed for backend_id, seed in latest if backend_id == reference_id}
    reports = []
    for candidate_id in candidates:
        candidate_seeds = {seed for backend_id, seed in latest if backend_id == candidate_id}
        if candidate_seeds != reference_seeds:
            raise AblationError(
                f"seed mismatch for {candidate_id!r}: got {sorted(candidate_seeds)}, expected {sorted(reference_seeds)}"
            )
        reference_failures = 0
        candidate_failures = 0
        matched_seeds = 0
        matched_decisions = 0
        action_matches = 0
        resist_shifts = []
        worth_shifts = []
        candidate_latencies = []
        reference_latencies = []
        candidate_backend = None
        for seed in sorted(reference_seeds):
            reference = latest[(reference_id, seed)]
            candidate = latest[(candidate_id, seed)]
            candidate_backend = candidate["backend"]
            if reference.get("status") != "success":
                reference_failures += 1
            if candidate.get("status") != "success":
                candidate_failures += 1
            if reference.get("status") != "success" or candidate.get("status") != "success":
                continue
            matched_seeds += 1
            reference_decisions = _decisions(reference)
            candidate_decisions = _decisions(candidate)
            if set(candidate_decisions) != set(reference_decisions):
                raise AblationError(f"decision key mismatch for {candidate_id!r} seed {seed}")
            for key in sorted(reference_decisions):
                reference_decision = reference_decisions[key]
                candidate_decision = candidate_decisions[key]
                matched_decisions += 1
                action_matches += candidate_decision.get("act") == reference_decision.get("act")
                reference_probability = _resist_probability(reference_decision)
                candidate_probability = _resist_probability(candidate_decision)
                if reference_probability is not None and candidate_probability is not None:
                    resist_shifts.append(candidate_probability - reference_probability)
                reference_worth = reference_decision.get("worth_confidence")
                candidate_worth = candidate_decision.get("worth_confidence")
                if isinstance(reference_worth, (int, float)) and isinstance(candidate_worth, (int, float)):
                    worth_shifts.append(float(candidate_worth) - float(reference_worth))
            reference_latencies.append(float(reference.get("mean_backend_call_seconds", 0.0)))
            candidate_latencies.append(float(candidate.get("mean_backend_call_seconds", 0.0)))
        reports.append(
            {
                "reference_id": reference_id,
                "candidate_id": candidate_id,
                "candidate_backend": candidate_backend,
                "seeds": len(reference_seeds),
                "matched_seeds": matched_seeds,
                "matched_decisions": matched_decisions,
                "action_agreement": action_matches / matched_decisions if matched_decisions else None,
                "mean_resist_probability_shift": fmean(resist_shifts) if resist_shifts else None,
                "mean_worth_confidence_shift": fmean(worth_shifts) if worth_shifts else None,
                "reference_mean_call_seconds": fmean(reference_latencies) if reference_latencies else None,
                "candidate_mean_call_seconds": fmean(candidate_latencies) if candidate_latencies else None,
                "reference_failures": reference_failures,
                "candidate_failures": candidate_failures,
                "calibration_claim_supported": False,
            }
        )
    return reports


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse local-ablation report flags."""
    parser = argparse.ArgumentParser(
        prog="python -m spatial_ipd.ablate",
        description="Compare matched local thinker decisions from experiment JSONL.",
    )
    parser.add_argument("--jsonl", required=True, help="Experiment result JSONL.")
    parser.add_argument("--reference", required=True, help="Reference local backend id.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Print one JSON comparison per non-reference local backend."""
    args = parse_args(argv)
    try:
        records = read_result_records(args.jsonl)
    except ExperimentError as exc:
        raise AblationError(str(exc)) from exc
    for report in compare_local_ablations(records, reference_id=args.reference):
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
