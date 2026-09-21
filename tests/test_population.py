"""Deterministic mixed-policy schedules and aggregate accounting."""

import json

import pytest

from spatial_ipd.engine import simulate
from spatial_ipd.population import (
    POPULATION_MANIFEST_VERSION,
    PopulationError,
    main,
    normalize_manifest,
    run_population,
    write_jsonl,
)


def _manifest(
    seed=7,
    memory_window=None,
    reputation_enabled=False,
    communication_enabled=False,
):
    return normalize_manifest(
        {
            "schema_version": POPULATION_MANIFEST_VERSION,
            "seed": seed,
            "encounters": 4,
            "rounds_per_match": 5,
            "memory_window": memory_window,
            "reputation_enabled": reputation_enabled,
            "communication_enabled": communication_enabled,
            "population": {
                "always_cooperate": 2,
                "always_defect": 2,
                "forgiving_tit_for_tat": 1,
                "tit_for_tat": 2,
            },
        }
    )


def test_population_run_is_deterministic_for_same_seed():
    first = run_population(_manifest())
    second = run_population(_manifest())
    assert first == second
    events, summaries = first
    assert len(events) == 12
    assert len(summaries) == 4


def test_different_seed_changes_pair_schedule():
    first_events, _ = run_population(_manifest(seed=7))
    second_events, _ = run_population(_manifest(seed=8))
    first_pairs = [(event["agent_a"], event["agent_b"]) for event in first_events]
    second_pairs = [(event["agent_a"], event["agent_b"]) for event in second_events]
    assert first_pairs != second_pairs


def test_memory_windows_keep_schedule_and_isolate_policy_effect():
    full_events, full_summaries = run_population(_manifest(memory_window=None))
    one_events, one_summaries = run_population(_manifest(memory_window=1))
    two_events, two_summaries = run_population(_manifest(memory_window=2))
    four_events, four_summaries = run_population(_manifest(memory_window=4))
    assert [(event["agent_a"], event["agent_b"]) for event in full_events] == [
        (event["agent_a"], event["agent_b"]) for event in one_events
    ]
    full_by_policy = {summary["policy"]: summary for summary in full_summaries}
    one_by_policy = {summary["policy"]: summary for summary in one_summaries}
    assert one_by_policy["forgiving_tit_for_tat"] != full_by_policy["forgiving_tit_for_tat"]
    assert (
        [{key: value for key, value in summary.items() if key != "memory_window"} for summary in two_summaries]
        == [{key: value for key, value in summary.items() if key != "memory_window"} for summary in four_summaries]
        == [{key: value for key, value in summary.items() if key != "memory_window"} for summary in full_summaries]
    )


def test_reputation_enabled_keeps_schedule_and_changes_guard_behavior():
    base = {
        "schema_version": POPULATION_MANIFEST_VERSION,
        "seed": 11,
        "encounters": 12,
        "rounds_per_match": 5,
        "population": {
            "always_cooperate": 2,
            "always_defect": 2,
            "reputation_guard": 2,
        },
    }
    disabled_events, disabled_summaries = run_population(normalize_manifest({**base, "reputation_enabled": False}))
    enabled_events, enabled_summaries = run_population(normalize_manifest({**base, "reputation_enabled": True}))
    assert [(event["agent_a"], event["agent_b"]) for event in disabled_events] == [
        (event["agent_a"], event["agent_b"]) for event in enabled_events
    ]
    assert all(event["opponent_reputation_a"] is None for event in disabled_events)
    assert any(event["opponent_reputation_a"] is not None for event in enabled_events)
    disabled_guard = next(summary for summary in disabled_summaries if summary["policy"] == "reputation_guard")
    enabled_guard = next(summary for summary in enabled_summaries if summary["policy"] == "reputation_guard")
    assert disabled_guard["cooperation_rate"] == 1.0
    assert enabled_guard["cooperation_rate"] < 1.0


def test_communication_enabled_keeps_schedule_and_changes_guard_behavior():
    base = {
        "schema_version": POPULATION_MANIFEST_VERSION,
        "seed": 13,
        "encounters": 12,
        "rounds_per_match": 5,
        "population": {
            "always_cooperate": 2,
            "always_defect": 2,
            "communication_guard": 2,
        },
    }
    disabled_events, disabled_summaries = run_population(normalize_manifest({**base, "communication_enabled": False}))
    enabled_events, enabled_summaries = run_population(normalize_manifest({**base, "communication_enabled": True}))
    assert [(event["agent_a"], event["agent_b"]) for event in disabled_events] == [
        (event["agent_a"], event["agent_b"]) for event in enabled_events
    ]
    assert all(event["warning_about_a"] is None for event in disabled_events)
    assert any(isinstance(event["warning_about_a"], bool) for event in enabled_events)
    assert any(event["opponent_warning_rate_a"] is not None for event in enabled_events)
    disabled_guard = next(summary for summary in disabled_summaries if summary["policy"] == "communication_guard")
    enabled_guard = next(summary for summary in enabled_summaries if summary["policy"] == "communication_guard")
    assert disabled_guard["cooperation_rate"] == 1.0
    assert enabled_guard["cooperation_rate"] < 1.0


def test_policy_aggregates_match_encounter_records():
    manifest = _manifest()
    events, summaries = run_population(manifest)
    summary_by_policy = {summary["policy"]: summary for summary in summaries}
    assert sum(summary["matches"] for summary in summaries) == 2 * len(events)
    assert sum(summary["rounds"] for summary in summaries) == 2 * len(events) * 5
    for policy, count in manifest["population"].items():
        assert summary_by_policy[policy]["agents"] == count
        assert 0.0 <= summary_by_policy[policy]["cooperation_rate"] <= 1.0
        assert summary_by_policy[policy]["rounds"] == summary_by_policy[policy]["matches"] * 5


def test_jsonl_is_byte_identical_for_same_evidence(tmp_path):
    events, summaries = run_population(_manifest())
    first = write_jsonl(tmp_path / "first.jsonl", events, summaries)
    second = write_jsonl(tmp_path / "second.jsonl", events, summaries)
    assert first.read_bytes() == second.read_bytes()


def test_manifest_rejects_unknown_policy_and_bad_counts():
    value = {
        "schema_version": POPULATION_MANIFEST_VERSION,
        "seed": 1,
        "encounters": 1,
        "rounds_per_match": 1,
        "population": {"unknown": 2},
    }
    with pytest.raises(PopulationError, match="unknown policy"):
        normalize_manifest(value)
    value["population"] = {"always_cooperate": 0}
    with pytest.raises(PopulationError, match="positive"):
        normalize_manifest(value)


def test_population_cli_writes_jsonl_and_csv(tmp_path, capsys):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
    jsonl_path = tmp_path / "population.jsonl"
    csv_path = tmp_path / "population.csv"
    assert (
        main(
            [
                "--manifest",
                str(manifest_path),
                "--jsonl",
                str(jsonl_path),
                "--csv",
                str(csv_path),
            ]
        )
        == 0
    )
    assert len(jsonl_path.read_text(encoding="utf-8").splitlines()) == 16
    assert csv_path.read_text(encoding="utf-8").count("\n") == 5
    assert "matches=12" in capsys.readouterr().out


def test_population_slice_preserves_spatial_golden():
    result = simulate(12, 12, 30, seed=20260316, mutation_rate=0.02)
    assert result.final_cooperation_rate == 2 / 144
