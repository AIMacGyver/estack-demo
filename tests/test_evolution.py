"""Payoff-proportional deterministic population evolution."""

import json

import pytest

from spatial_ipd.engine import simulate
from spatial_ipd.evolution import (
    EVOLUTION_MANIFEST_VERSION,
    EvolutionError,
    allocate_offspring,
    main,
    mutate_offspring,
    normalize_manifest,
    run_evolution,
    summarize_final,
)


def _manifest(generations=4):
    return normalize_manifest(
        {
            "schema_version": EVOLUTION_MANIFEST_VERSION,
            "seed": 21,
            "generations": generations,
            "encounters_per_generation": 8,
            "rounds_per_match": 5,
            "population": {
                "always_cooperate": 2,
                "always_defect": 2,
                "pavlov": 2,
                "tit_for_tat": 2,
            },
        }
    )


def test_largest_remainder_allocation_is_stable_and_size_preserving():
    assert allocate_offspring({"a": 3.0, "b": 2.0, "c": 1.0}, 10) == {
        "a": 5,
        "b": 3,
        "c": 2,
    }
    assert allocate_offspring({"a": 1.0, "b": 1.0, "c": 1.0}, 10) == {
        "a": 4,
        "b": 3,
        "c": 3,
    }
    assert allocate_offspring({"a": 10.0, "b": 0.0}, 4) == {"a": 4}


def test_seeded_mutation_preserves_size_and_can_reintroduce_policies():
    unchanged, incoming, outgoing = mutate_offspring(
        {"a": 4},
        roster=("a", "b", "c"),
        mutation_rate=0.0,
        seed=1,
    )
    assert unchanged == {"a": 4}
    assert sum(incoming.values()) == sum(outgoing.values()) == 0

    mutated, incoming, outgoing = mutate_offspring(
        {"a": 100},
        roster=("a", "b", "c"),
        mutation_rate=0.5,
        seed=1,
    )
    assert sum(mutated.values()) == 100
    assert mutated["b"] > 0
    assert mutated["c"] > 0
    assert sum(incoming.values()) == sum(outgoing.values())


def test_evolution_is_deterministic_and_population_size_is_constant():
    first = run_evolution(_manifest())
    second = run_evolution(_manifest())
    assert first == second
    for generation in range(4):
        generation_rows = [row for row in first if row["generation"] == generation]
        assert sum(row["count"] for row in generation_rows) == 8
        assert sum(row["next_count"] for row in generation_rows) == 8


def test_generation_seeds_are_stable_and_incremented():
    rows = run_evolution(_manifest(generations=3))
    assert sorted({(row["generation"], row["seed"]) for row in rows}) == [
        (0, 21),
        (1, 22),
        (2, 23),
    ]


def test_multiple_base_seeds_produce_replicate_summaries():
    manifest = dict(_manifest(generations=3))
    manifest["seeds"] = [21, 31]
    rows = run_evolution(manifest)
    assert {row["replicate_seed"] for row in rows} == {21, 31}
    for replicate_seed in (21, 31):
        for generation in range(3):
            generation_rows = [
                row for row in rows if row["replicate_seed"] == replicate_seed and row["generation"] == generation
            ]
            assert sum(row["next_count"] for row in generation_rows) == 8
    summary = summarize_final(rows)
    assert all(item["replicates"] == 2 for item in summary)
    assert all(item["min_final_count"] <= item["mean_final_count"] <= item["max_final_count"] for item in summary)


def test_manifest_rejects_odd_population_and_bad_generation_count():
    value = {
        "schema_version": EVOLUTION_MANIFEST_VERSION,
        "seed": 1,
        "generations": 2,
        "encounters_per_generation": 2,
        "rounds_per_match": 2,
        "population": {"always_cooperate": 1, "always_defect": 2},
    }
    with pytest.raises(EvolutionError, match="even"):
        normalize_manifest(value)
    value["population"]["always_cooperate"] = 2
    value["generations"] = 0
    with pytest.raises(EvolutionError, match="generations"):
        normalize_manifest(value)
    value["generations"] = 2
    value["mutation_rate"] = 1.5
    with pytest.raises(EvolutionError, match="mutation_rate"):
        normalize_manifest(value)


def test_evolution_cli_writes_repeatable_artifacts(tmp_path, capsys):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
    first_jsonl = tmp_path / "first.jsonl"
    first_csv = tmp_path / "first.csv"
    second_jsonl = tmp_path / "second.jsonl"
    second_csv = tmp_path / "second.csv"
    args = ["--manifest", str(manifest_path)]
    assert main([*args, "--jsonl", str(first_jsonl), "--csv", str(first_csv)]) == 0
    assert main([*args, "--jsonl", str(second_jsonl), "--csv", str(second_csv)]) == 0
    assert first_jsonl.read_bytes() == second_jsonl.read_bytes()
    assert first_csv.read_bytes() == second_csv.read_bytes()
    assert "generations=4" in capsys.readouterr().out


def test_evolution_slice_preserves_spatial_golden():
    result = simulate(12, 12, 30, seed=20260316, mutation_rate=0.02)
    assert result.final_cooperation_rate == 2 / 144
