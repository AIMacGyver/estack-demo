"""Deterministic payoff-proportional evolution over canonical populations."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from random import Random
from statistics import fmean

from spatial_ipd.population import (
    POPULATION_MANIFEST_VERSION,
    run_population,
)
from spatial_ipd.population import (
    normalize_manifest as normalize_population_manifest,
)

EVOLUTION_MANIFEST_VERSION = 1
CSV_FIELDS = (
    "replicate_seed",
    "generation",
    "mutation_rate",
    "reputation_enabled",
    "policy",
    "count",
    "payoff",
    "average_payoff_per_round",
    "cooperation_rate",
    "selected_count",
    "mutations_in",
    "mutations_out",
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
    seeds_value = value.get("seeds")
    if seeds_value is None:
        seeds_value = [value.get("seed")]
    if not isinstance(seeds_value, list) or not seeds_value:
        raise EvolutionError("seeds must be a non-empty list")
    seeds = []
    for index, seed in enumerate(seeds_value):
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise EvolutionError(f"seeds[{index}] must be an integer")
        seeds.append(seed)
    if len(set(seeds)) != len(seeds):
        raise EvolutionError("seeds must be unique")
    mutation_rate = value.get("mutation_rate", 0.0)
    if (
        isinstance(mutation_rate, bool)
        or not isinstance(mutation_rate, (int, float))
        or not 0.0 <= float(mutation_rate) <= 1.0
    ):
        raise EvolutionError("mutation_rate must be in [0, 1]")
    reputation_enabled = value.get("reputation_enabled", False)
    if not isinstance(reputation_enabled, bool):
        raise EvolutionError("reputation_enabled must be boolean")
    population_manifest = normalize_population_manifest(
        {
            "schema_version": POPULATION_MANIFEST_VERSION,
            "seed": seeds[0],
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
        "seed": seeds[0],
        "seeds": seeds,
        "generations": generations,
        "encounters_per_generation": population_manifest["encounters"],
        "rounds_per_match": population_manifest["rounds_per_match"],
        "mutation_rate": float(mutation_rate),
        "reputation_enabled": reputation_enabled,
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


def mutate_offspring(
    selected: Mapping[str, int],
    *,
    roster: Sequence[str],
    mutation_rate: float,
    seed: int,
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    """Mutate selected offspring uniformly to a different roster policy."""
    if not 0.0 <= mutation_rate <= 1.0:
        raise EvolutionError("mutation_rate must be in [0, 1]")
    policies = tuple(sorted(roster))
    if not policies:
        raise EvolutionError("mutation roster must not be empty")
    counts = {policy: 0 for policy in policies}
    mutations_in = {policy: 0 for policy in policies}
    mutations_out = {policy: 0 for policy in policies}
    rng = Random(seed)
    for policy in policies:
        for _ in range(int(selected.get(policy, 0))):
            target = policy
            alternatives = tuple(candidate for candidate in policies if candidate != policy)
            if alternatives and rng.random() < mutation_rate:
                target = alternatives[rng.randrange(len(alternatives))]
                mutations_out[policy] += 1
                mutations_in[target] += 1
            counts[target] += 1
    return (
        {policy: count for policy, count in counts.items() if count},
        mutations_in,
        mutations_out,
    )


def run_evolution(manifest: Mapping[str, object]) -> list[dict[str, object]]:
    """Evaluate and reproduce policy populations for fixed generations."""
    normalized = normalize_manifest(manifest)
    rows = []
    roster = tuple(sorted(normalized["population"]))
    for replicate_seed in normalized["seeds"]:
        population = dict(normalized["population"])
        population_size = sum(population.values())
        for generation in range(int(normalized["generations"])):
            population_manifest = normalize_population_manifest(
                {
                    "schema_version": POPULATION_MANIFEST_VERSION,
                    "seed": int(replicate_seed) + generation,
                    "encounters": normalized["encounters_per_generation"],
                    "rounds_per_match": normalized["rounds_per_match"],
                    "memory_window": None,
                    "reputation_enabled": normalized["reputation_enabled"],
                    "communication_enabled": False,
                    "population": population,
                }
            )
            _events, summaries = run_population(population_manifest)
            payoffs = {str(summary["policy"]): float(summary["payoff"]) for summary in summaries}
            selected_population = allocate_offspring(payoffs, population_size)
            next_population, mutations_in, mutations_out = mutate_offspring(
                selected_population,
                roster=roster,
                mutation_rate=float(normalized["mutation_rate"]),
                seed=int(replicate_seed) * 1_000_003 + generation,
            )
            summaries_by_policy = {str(summary["policy"]): summary for summary in summaries}
            for policy in roster:
                summary = summaries_by_policy.get(policy)
                rows.append(
                    {
                        "record_type": "generation_policy",
                        "replicate_seed": replicate_seed,
                        "generation": generation,
                        "mutation_rate": normalized["mutation_rate"],
                        "reputation_enabled": normalized["reputation_enabled"],
                        "seed": population_manifest["seed"],
                        "policy": policy,
                        "count": population.get(policy, 0),
                        "payoff": summary["payoff"] if summary else 0,
                        "average_payoff_per_round": (summary["average_payoff_per_round"] if summary else None),
                        "cooperation_rate": summary["cooperation_rate"] if summary else None,
                        "selected_count": selected_population.get(policy, 0),
                        "mutations_in": mutations_in[policy],
                        "mutations_out": mutations_out[policy],
                        "next_count": next_population.get(policy, 0),
                    }
                )
            if sum(next_population.values()) != population_size:
                raise AssertionError("offspring allocation changed population size")
            population = next_population
    return rows


def summarize_final(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Summarize final policy counts across replicate seeds."""
    if not rows:
        raise EvolutionError("evolution rows must not be empty")
    final_generation = max(int(row["generation"]) for row in rows)
    counts: dict[str, list[int]] = {}
    for row in rows:
        if int(row["generation"]) == final_generation:
            counts.setdefault(str(row["policy"]), []).append(int(row["next_count"]))
    return [
        {
            "policy": policy,
            "replicates": len(values),
            "mean_final_count": fmean(values),
            "min_final_count": min(values),
            "max_final_count": max(values),
        }
        for policy, values in sorted(counts.items())
    ]


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
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
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
    final = summarize_final(rows)
    print(
        f"generations={manifest['generations']} replicates={len(manifest['seeds'])} "
        f"final={json.dumps(final, sort_keys=True, separators=(',', ':'))} "
        f"jsonl={args.jsonl} csv={args.csv}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
