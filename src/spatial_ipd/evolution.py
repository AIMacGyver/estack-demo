"""Deterministic payoff-proportional evolution over canonical populations."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from spatial_ipd.population import (
    POPULATION_MANIFEST_VERSION,
    run_population,
)
from spatial_ipd.population import (
    normalize_manifest as normalize_population_manifest,
)

EVOLUTION_MANIFEST_VERSION = 1
CSV_FIELDS = (
    "generation",
    "policy",
    "count",
    "payoff",
    "average_payoff_per_round",
    "cooperation_rate",
    "next_count",
)


class EvolutionError(ValueError):
    """A malformed evolution manifest or invalid fitness allocation."""


def normalize_manifest(value: object) -> dict[str, object]:
    """Validate one no-mutation evolutionary manifest."""
    if not isinstance(value, Mapping):
        raise EvolutionError("manifest must be an object")
    if value.get("schema_version") != EVOLUTION_MANIFEST_VERSION:
        raise EvolutionError(f"schema_version must be {EVOLUTION_MANIFEST_VERSION}")
    generations = value.get("generations")
    if isinstance(generations, bool) or not isinstance(generations, int) or generations < 1:
        raise EvolutionError("generations must be a positive integer")
    population_manifest = normalize_population_manifest(
        {
            "schema_version": POPULATION_MANIFEST_VERSION,
            "seed": value.get("seed"),
            "encounters": value.get("encounters_per_generation"),
            "rounds_per_match": value.get("rounds_per_match"),
            "memory_window": None,
            "reputation_enabled": False,
            "communication_enabled": False,
            "population": value.get("population"),
        }
    )
    population_size = sum(population_manifest["population"].values())
    if population_size % 2:
        raise EvolutionError("population size must be even so every encounter pairs all agents")
    return {
        "schema_version": EVOLUTION_MANIFEST_VERSION,
        "seed": population_manifest["seed"],
        "generations": generations,
        "encounters_per_generation": population_manifest["encounters"],
        "rounds_per_match": population_manifest["rounds_per_match"],
        "population": population_manifest["population"],
    }


def read_manifest(path: str | Path) -> dict[str, object]:
    """Read and validate an evolution manifest."""
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvolutionError(f"manifest {source} is invalid JSON") from exc
    return normalize_manifest(value)


def allocate_offspring(payoffs: Mapping[str, float], population_size: int) -> dict[str, int]:
    """Convert non-negative total payoff shares to stable integer counts."""
    if population_size < 1:
        raise EvolutionError("population_size must be positive")
    if not payoffs:
        raise EvolutionError("payoffs must not be empty")
    if any(not math.isfinite(value) or value < 0 for value in payoffs.values()):
        raise EvolutionError("payoffs must be finite and non-negative")
    total = sum(payoffs.values())
    if total <= 0:
        raise EvolutionError("total payoff must be positive")
    expected = {policy: value / total * population_size for policy, value in payoffs.items()}
    counts = {policy: math.floor(value) for policy, value in expected.items()}
    remaining = population_size - sum(counts.values())
    order = sorted(expected, key=lambda policy: (-(expected[policy] - counts[policy]), policy))
    for policy in order[:remaining]:
        counts[policy] += 1
    return {policy: count for policy, count in sorted(counts.items()) if count}


def run_evolution(manifest: Mapping[str, object]) -> list[dict[str, object]]:
    """Evaluate and reproduce policy populations for fixed generations."""
    normalized = normalize_manifest(manifest)
    population = dict(normalized["population"])
    population_size = sum(population.values())
    rows = []
    for generation in range(int(normalized["generations"])):
        population_manifest = normalize_population_manifest(
            {
                "schema_version": POPULATION_MANIFEST_VERSION,
                "seed": int(normalized["seed"]) + generation,
                "encounters": normalized["encounters_per_generation"],
                "rounds_per_match": normalized["rounds_per_match"],
                "memory_window": None,
                "reputation_enabled": False,
                "communication_enabled": False,
                "population": population,
            }
        )
        _events, summaries = run_population(population_manifest)
        payoffs = {str(summary["policy"]): float(summary["payoff"]) for summary in summaries}
        next_population = allocate_offspring(payoffs, population_size)
        for summary in summaries:
            policy = str(summary["policy"])
            rows.append(
                {
                    "record_type": "generation_policy",
                    "generation": generation,
                    "seed": population_manifest["seed"],
                    "policy": policy,
                    "count": population[policy],
                    "payoff": summary["payoff"],
                    "average_payoff_per_round": summary["average_payoff_per_round"],
                    "cooperation_rate": summary["cooperation_rate"],
                    "next_count": next_population.get(policy, 0),
                }
            )
        if sum(next_population.values()) != population_size:
            raise AssertionError("offspring allocation changed population size")
        population = next_population
    return rows


def write_jsonl(path: str | Path, rows: Sequence[Mapping[str, object]]) -> Path:
    """Write deterministic generation-policy records."""
    destination = Path(path)
    destination.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    return destination


def write_csv(path: str | Path, rows: Sequence[Mapping[str, object]]) -> Path:
    """Write compact frequency/payoff evolution CSV."""
    destination = Path(path)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in CSV_FIELDS})
    return destination


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse evolutionary population CLI flags."""
    parser = argparse.ArgumentParser(
        prog="python -m spatial_ipd.evolution",
        description="Run deterministic payoff-proportional policy evolution.",
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--csv", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one evolution manifest and write evidence artifacts."""
    args = parse_args(argv)
    manifest = read_manifest(args.manifest)
    rows = run_evolution(manifest)
    write_jsonl(args.jsonl, rows)
    write_csv(args.csv, rows)
    final_generation = int(manifest["generations"]) - 1
    final = {str(row["policy"]): int(row["next_count"]) for row in rows if row["generation"] == final_generation}
    print(
        f"generations={manifest['generations']} population={sum(final.values())} "
        f"final={json.dumps(final, sort_keys=True, separators=(',', ':'))} "
        f"jsonl={args.jsonl} csv={args.csv}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
