"""Manifest planning, resumable outputs, and fake-backed experiments."""

import csv
import json
from types import SimpleNamespace

import pytest

from spatial_ipd.experiment import (
    EXPERIMENT_MANIFEST_VERSION,
    ExperimentError,
    cooperation_auc,
    execute_run,
    normalize_manifest,
    planned_runs,
    read_result_records,
    run_manifest,
)


class FakeThinkerClient:
    """Return low worth/resist signals for every requested seat."""

    uses_sdk_questions = False

    def __init__(self, confidence_kind="fake"):
        self.confidence_kind = confidence_kind
        self.calls = 0

    def system_one(self, state, questions):
        del state
        self.calls += 1
        nouls = {}
        scores = {}
        for key in questions:
            if key.startswith(("worth_thinking_", "resist_")):
                nouls[key] = SimpleNamespace(noul=0.2)
            elif key.startswith("cluster_fragility_"):
                scores[key] = SimpleNamespace(score=1.0, confidence=1.0)
        return SimpleNamespace(
            nouls=nouls,
            scores=scores,
            choices={},
            confidence_kind=self.confidence_kind,
        )


class FailingThinkerClient:
    """Fail every backend call with a stable message."""

    uses_sdk_questions = False

    def system_one(self, state, questions):
        del state, questions
        raise TimeoutError("backend timed out")


def _manifest(backends=None, seeds=None):
    return normalize_manifest(
        {
            "schema_version": EXPERIMENT_MANIFEST_VERSION,
            "simulation": {
                "height": 4,
                "width": 4,
                "generations": 2,
                "mutation_rate": 0.02,
            },
            "thinker": {
                "think_every": 1,
                "think_last": 0,
                "sticky": 0,
                "thinkers": 1,
                "seats": "frontier",
            },
            "seeds": [1, 2] if seeds is None else seeds,
            "backends": (
                [
                    {"id": "plain", "kind": "baseline"},
                    {"id": "random", "kind": "random"},
                ]
                if backends is None
                else backends
            ),
        }
    )


def test_manifest_expands_stable_backend_by_seed_plans():
    manifest = _manifest()
    first = planned_runs(manifest)
    second = planned_runs(manifest)
    assert len(first) == 4
    assert [plan["run_id"] for plan in first] == [plan["run_id"] for plan in second]
    assert len({plan["run_id"] for plan in first}) == 4
    assert {(plan["backend"]["kind"], plan["seed"]) for plan in first} == {
        ("baseline", 1),
        ("baseline", 2),
        ("random", 1),
        ("random", 2),
    }


def test_manifest_rejects_duplicate_seeds_and_missing_local_model():
    with pytest.raises(ExperimentError, match="seeds must be unique"):
        _manifest(seeds=[1, 1])
    with pytest.raises(ExperimentError, match="model is required"):
        _manifest(backends=[{"id": "local", "kind": "local"}])


def test_cooperation_auc_is_normalized_trapezoid():
    assert cooperation_auc([0.5]) == 0.5
    assert cooperation_auc([0.0, 1.0]) == 0.5
    assert cooperation_auc([0.0, 0.5, 1.0]) == 0.5


def test_manifest_writes_jsonl_and_compact_csv(tmp_path):
    jsonl_path = tmp_path / "runs.jsonl"
    csv_path = tmp_path / "summary.csv"
    summary = run_manifest(_manifest(), jsonl_path=jsonl_path, csv_path=csv_path)
    records = read_result_records(jsonl_path)

    assert summary == {"planned": 4, "executed": 4, "skipped": 0, "succeeded": 4, "failed": 0}
    assert len(records) == 4
    assert all(record["status"] == "success" for record in records)
    assert all(len(record["cooperation_rates"]) == 3 for record in records)
    assert all(0.0 <= record["cooperation_auc"] <= 1.0 for record in records)

    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert {(row["backend_kind"], row["seed"]) for row in rows} == {
        ("baseline", "1"),
        ("baseline", "2"),
        ("random", "1"),
        ("random", "2"),
    }


def test_rerun_skips_successful_ids_without_duplicate_rows(tmp_path):
    jsonl_path = tmp_path / "runs.jsonl"
    csv_path = tmp_path / "summary.csv"
    manifest = _manifest()
    first = run_manifest(manifest, jsonl_path=jsonl_path, csv_path=csv_path)
    original_jsonl = jsonl_path.read_text(encoding="utf-8")
    second = run_manifest(manifest, jsonl_path=jsonl_path, csv_path=csv_path)

    assert first["executed"] == 4
    assert second == {"planned": 4, "executed": 0, "skipped": 4, "succeeded": 0, "failed": 0}
    assert jsonl_path.read_text(encoding="utf-8") == original_jsonl
    assert len(read_result_records(jsonl_path)) == 4


def test_fake_jev_and_local_backends_run_without_live_services(tmp_path):
    manifest = _manifest(
        seeds=[7],
        backends=[
            {"id": "jev", "kind": "jev"},
            {
                "id": "qwen",
                "kind": "local",
                "model": "qwen3:8b",
                "reasoning_effort": "none",
            },
        ],
    )
    jsonl_path = tmp_path / "runs.jsonl"
    summary = run_manifest(
        manifest,
        jsonl_path=jsonl_path,
        csv_path=tmp_path / "summary.csv",
        client_factories={
            "jev": lambda _backend, _seed: FakeThinkerClient("jev_noul"),
            "local": lambda _backend, _seed: FakeThinkerClient("llm_self_report"),
        },
    )
    records = read_result_records(jsonl_path)
    assert summary["succeeded"] == 2
    assert {record["backend"]["kind"] for record in records} == {"jev", "local"}
    assert all(record["think_calls"] == 2 for record in records)
    assert all(record["status"] == "success" for record in records)


def test_failure_is_isolated_and_retried_while_success_is_skipped(tmp_path):
    manifest = _manifest(
        seeds=[1],
        backends=[
            {"id": "plain", "kind": "baseline"},
            {"id": "jev", "kind": "jev"},
        ],
    )
    jsonl_path = tmp_path / "runs.jsonl"
    csv_path = tmp_path / "summary.csv"
    first = run_manifest(
        manifest,
        jsonl_path=jsonl_path,
        csv_path=csv_path,
        client_factories={"jev": lambda _backend, _seed: FailingThinkerClient()},
    )
    assert first == {"planned": 2, "executed": 2, "skipped": 0, "succeeded": 1, "failed": 1}
    first_records = read_result_records(jsonl_path)
    error = next(record for record in first_records if record["status"] == "error")
    assert error["error_type"] == "TimeoutError"
    assert error["error"] == "backend timed out"

    second = run_manifest(
        manifest,
        jsonl_path=jsonl_path,
        csv_path=csv_path,
        client_factories={"jev": lambda _backend, _seed: FakeThinkerClient("jev_noul")},
    )
    assert second == {"planned": 2, "executed": 1, "skipped": 1, "succeeded": 1, "failed": 0}
    assert len(read_result_records(jsonl_path)) == 3
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert all(row["status"] == "success" for row in rows)


def test_baseline_plan_preserves_locked_golden():
    manifest = normalize_manifest(
        {
            "schema_version": EXPERIMENT_MANIFEST_VERSION,
            "simulation": {
                "height": 12,
                "width": 12,
                "generations": 30,
                "mutation_rate": 0.02,
            },
            "seeds": [20260316],
            "backends": [{"id": "plain", "kind": "baseline"}],
        }
    )
    record = execute_run(planned_runs(manifest)[0])
    assert record["status"] == "success"
    assert record["final_cooperation_rate"] == 2 / 144


def test_result_jsonl_is_machine_readable(tmp_path):
    jsonl_path = tmp_path / "runs.jsonl"
    run_manifest(_manifest(seeds=[1]), jsonl_path=jsonl_path, csv_path=tmp_path / "summary.csv")
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        assert json.loads(line)["schema_version"] == 1
