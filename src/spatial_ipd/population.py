"""Deterministic mixed populations built on the active-agent arena."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from random import Random

from spatial_ipd.arena import POLICY_TYPES, MemoryWindowPolicy, play_match

POPULATION_MANIFEST_VERSION = 1
SUMMARY_FIELDS = (
    "memory_window",
    "reputation_enabled",
    "communication_enabled",
    "policy",
    "agents",
    "matches",
    "rounds",
    "payoff",
    "average_payoff_per_round",
    "cooperation_rate",
    "mutual_cooperation_rounds",
)


class PopulationError(ValueError):
    """A malformed population manifest or unsupported policy."""


def normalize_manifest(value: object) -> dict[str, object]:
    """Validate and normalize one mixed-population manifest."""
    if not isinstance(value, Mapping):
        raise PopulationError("manifest must be an object")
    if value.get("schema_version") != POPULATION_MANIFEST_VERSION:
        raise PopulationError(f"schema_version must be {POPULATION_MANIFEST_VERSION}")
    seed = value.get("seed")
    encounters = value.get("encounters")
    rounds = value.get("rounds_per_match")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise PopulationError("seed must be an integer")
    if isinstance(encounters, bool) or not isinstance(encounters, int) or encounters < 1:
        raise PopulationError("encounters must be a positive integer")
    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 1:
        raise PopulationError("rounds_per_match must be a positive integer")
    population = value.get("population")
    if not isinstance(population, Mapping) or not population:
        raise PopulationError("population must be a non-empty object")
    normalized_population = {}
    for policy, count in sorted(population.items()):
        if policy not in POLICY_TYPES:
            raise PopulationError(f"unknown policy {policy!r}")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise PopulationError(f"population count for {policy!r} must be positive")
        normalized_population[str(policy)] = count
    if sum(normalized_population.values()) < 2:
        raise PopulationError("population must contain at least two agents")
    memory_window = value.get("memory_window")
    if memory_window is not None and (
        isinstance(memory_window, bool) or not isinstance(memory_window, int) or memory_window < 1
    ):
        raise PopulationError("memory_window must be null or a positive integer")
    reputation_enabled = value.get("reputation_enabled", False)
    if not isinstance(reputation_enabled, bool):
        raise PopulationError("reputation_enabled must be boolean")
    communication_enabled = value.get("communication_enabled", False)
    if not isinstance(communication_enabled, bool):
        raise PopulationError("communication_enabled must be boolean")
    return {
        "schema_version": POPULATION_MANIFEST_VERSION,
        "seed": seed,
        "encounters": encounters,
        "rounds_per_match": rounds,
        "memory_window": memory_window,
        "reputation_enabled": reputation_enabled,
        "communication_enabled": communication_enabled,
        "population": normalized_population,
    }


def read_manifest(path: str | Path) -> dict[str, object]:
    """Read a population manifest from JSON."""
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PopulationError(f"manifest {source} is invalid JSON") from exc
    return normalize_manifest(value)


def _agents(population: Mapping[str, int]) -> list[tuple[str, str]]:
    return [(f"{policy}-{index:03d}", policy) for policy, count in sorted(population.items()) for index in range(count)]


def run_population(
    manifest: Mapping[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Run seeded pair schedules and aggregate evidence by policy."""
    normalized = normalize_manifest(manifest)
    rng = Random(int(normalized["seed"]))
    agents = _agents(normalized["population"])
    rounds = int(normalized["rounds_per_match"])
    memory_window = normalized["memory_window"]
    reputation_enabled = bool(normalized["reputation_enabled"])
    communication_enabled = bool(normalized["communication_enabled"])
    events = []
    reputations = {agent_id: [0, 0] for agent_id, _policy in agents}
    warnings = {agent_id: [0, 0] for agent_id, _policy in agents}
    totals = {
        policy: {
            "policy": policy,
            "agents": count,
            "matches": 0,
            "rounds": 0,
            "payoff": 0,
            "cooperative_actions": 0,
            "mutual_cooperation_rounds": 0,
        }
        for policy, count in normalized["population"].items()
    }
    for encounter in range(int(normalized["encounters"])):
        scheduled = list(agents)
        rng.shuffle(scheduled)
        for pair_index in range(0, len(scheduled) - 1, 2):
            (agent_a, policy_a), (agent_b, policy_b) = scheduled[pair_index : pair_index + 2]
            active_a = POLICY_TYPES[policy_a]()
            active_b = POLICY_TYPES[policy_b]()
            if memory_window is not None:
                active_a = MemoryWindowPolicy(active_a, int(memory_window))
                active_b = MemoryWindowPolicy(active_b, int(memory_window))
            reputation_a = (
                reputations[agent_b][0] / reputations[agent_b][1]
                if reputation_enabled and reputations[agent_b][1]
                else None
            )
            reputation_b = (
                reputations[agent_a][0] / reputations[agent_a][1]
                if reputation_enabled and reputations[agent_a][1]
                else None
            )
            warning_rate_a = (
                warnings[agent_b][0] / warnings[agent_b][1] if communication_enabled and warnings[agent_b][1] else None
            )
            warning_rate_b = (
                warnings[agent_a][0] / warnings[agent_a][1] if communication_enabled and warnings[agent_a][1] else None
            )
            result = play_match(
                active_a,
                active_b,
                rounds=rounds,
                opponent_reputation_a=reputation_a,
                opponent_reputation_b=reputation_b,
                opponent_warning_rate_a=warning_rate_a,
                opponent_warning_rate_b=warning_rate_b,
            )
            warning_about_a = result.cooperation_rate_a < 0.5
            warning_about_b = result.cooperation_rate_b < 0.5
            event = {
                "record_type": "encounter",
                "memory_window": memory_window,
                "reputation_enabled": reputation_enabled,
                "communication_enabled": communication_enabled,
                "opponent_reputation_a": reputation_a,
                "opponent_reputation_b": reputation_b,
                "opponent_warning_rate_a": warning_rate_a,
                "opponent_warning_rate_b": warning_rate_b,
                "warning_about_a": warning_about_a if communication_enabled else None,
                "warning_about_b": warning_about_b if communication_enabled else None,
                "encounter": encounter,
                "pair": pair_index // 2,
                "agent_a": agent_a,
                "agent_b": agent_b,
                **asdict(result),
            }
            events.append(event)
            reputations[agent_a][0] += sum(result.actions_a)
            reputations[agent_a][1] += rounds
            reputations[agent_b][0] += sum(result.actions_b)
            reputations[agent_b][1] += rounds
            if communication_enabled:
                warnings[agent_a][0] += warning_about_a
                warnings[agent_a][1] += 1
                warnings[agent_b][0] += warning_about_b
                warnings[agent_b][1] += 1
            for policy, payoff, actions in (
                (policy_a, result.payoff_a, result.actions_a),
                (policy_b, result.payoff_b, result.actions_b),
            ):
                total = totals[policy]
                total["matches"] += 1
                total["rounds"] += rounds
                total["payoff"] += payoff
                total["cooperative_actions"] += sum(actions)
                total["mutual_cooperation_rounds"] += result.mutual_cooperation_rounds
    summaries = []
    for policy in sorted(totals):
        total = totals[policy]
        played_rounds = int(total["rounds"])
        summaries.append(
            {
                "memory_window": memory_window,
                "reputation_enabled": reputation_enabled,
                "communication_enabled": communication_enabled,
                "policy": policy,
                "agents": total["agents"],
                "matches": total["matches"],
                "rounds": played_rounds,
                "payoff": total["payoff"],
                "average_payoff_per_round": total["payoff"] / played_rounds,
                "cooperation_rate": total["cooperative_actions"] / played_rounds,
                "mutual_cooperation_rounds": total["mutual_cooperation_rounds"],
            }
        )
    return events, summaries


def write_jsonl(
    path: str | Path,
    events: Sequence[Mapping[str, object]],
    summaries: Sequence[Mapping[str, object]],
) -> Path:
    """Write deterministic encounter and summary JSONL records."""
    destination = Path(path)
    records = [
        *events,
        *({"record_type": "policy_summary", **summary} for summary in summaries),
    ]
    destination.write_text(
        "".join(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )
    return destination


def write_csv(path: str | Path, summaries: Sequence[Mapping[str, object]]) -> Path:
    """Write compact per-policy aggregate CSV."""
    destination = Path(path)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summaries)
    return destination


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse mixed-population CLI flags."""
    parser = argparse.ArgumentParser(
        prog="python -m spatial_ipd.population",
        description="Run deterministic mixed canonical-policy populations.",
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--csv", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one population manifest and write evidence artifacts."""
    args = parse_args(argv)
    manifest = read_manifest(args.manifest)
    events, summaries = run_population(manifest)
    write_jsonl(args.jsonl, events, summaries)
    write_csv(args.csv, summaries)
    print(
        f"agents={sum(manifest['population'].values())} encounters={manifest['encounters']} "
        f"matches={len(events)} policies={len(summaries)} memory_window={manifest['memory_window']} "
        f"reputation_enabled={manifest['reputation_enabled']} "
        f"communication_enabled={manifest['communication_enabled']} "
        f"jsonl={args.jsonl} csv={args.csv}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
