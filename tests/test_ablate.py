"""Manifest-driven local configurations and matched offline comparisons."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from spatial_ipd.ablate import AblationError, compare_local_ablations
from spatial_ipd.experiment import (
    EXPERIMENT_MANIFEST_VERSION,
    normalize_manifest,
    read_result_records,
    run_manifest,
)


class FakeLocalAblationClient:
    """Return profile-specific confidence without changing the action."""

    uses_sdk_questions = False

    def __init__(self, profile):
        self.profile = profile

    def system_one(self, state, questions):
        del state
        worth_confidence = 0.7 if self.profile == "cluster_guard_v1" else 0.5
        resist_confidence = 0.6 if self.profile == "cluster_guard_v1" else 0.8
        nouls = {}
        scores = {}
        for key in questions:
            if key.startswith("worth_thinking_"):
                nouls[key] = SimpleNamespace(noul=0.0, confidence=worth_confidence)
            elif key.startswith("resist_"):
                nouls[key] = SimpleNamespace(noul=0.0, confidence=resist_confidence)
            elif key.startswith("cluster_fragility_"):
                scores[key] = SimpleNamespace(score=1.0, confidence=1.0)
        return SimpleNamespace(
            nouls=nouls,
            scores=scores,
            choices={},
            confidence_kind="llm_self_report",
        )


def _manifest():
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
            "seeds": [1, 2],
            "backends": [
                {
                    "id": "guard",
                    "kind": "local",
                    "model": "fake",
                    "prompt_profile": "cluster_guard_v1",
                    "temperature": 0.0,
                    "max_tokens": 128,
                },
                {
                    "id": "defer",
                    "kind": "local",
                    "model": "fake",
                    "prompt_profile": "score_defer_v1",
                    "temperature": 0.4,
                    "max_tokens": 64,
                },
            ],
        }
    )


def test_manifest_runs_fake_local_profiles_and_reports_matched_shifts(tmp_path):
    manifest = _manifest()
    jsonl_path = tmp_path / "runs.jsonl"
    summary = run_manifest(
        manifest,
        jsonl_path=jsonl_path,
        csv_path=tmp_path / "summary.csv",
        client_factories={"local": lambda backend, _seed: FakeLocalAblationClient(backend["prompt_profile"])},
    )
    records = read_result_records(jsonl_path)
    report = compare_local_ablations(records, reference_id="guard")[0]

    assert summary["succeeded"] == 4
    assert all(record["decisions"] for record in records)
    assert report["candidate_id"] == "defer"
    assert report["matched_seeds"] == 2
    assert report["matched_decisions"] == 4
    assert report["action_agreement"] == 1.0
    assert report["mean_resist_probability_shift"] == pytest.approx(-0.2)
    assert report["mean_worth_confidence_shift"] == pytest.approx(-0.2)
    assert report["candidate_backend"]["temperature"] == 0.4
    assert report["candidate_backend"]["max_tokens"] == 64
    assert report["calibration_claim_supported"] is False


def test_ablation_rejects_unmatched_seeds(tmp_path):
    manifest = _manifest()
    jsonl_path = tmp_path / "runs.jsonl"
    run_manifest(
        manifest,
        jsonl_path=jsonl_path,
        csv_path=tmp_path / "summary.csv",
        client_factories={"local": lambda backend, _seed: FakeLocalAblationClient(backend["prompt_profile"])},
    )
    records = read_result_records(jsonl_path)
    missing = [record for record in records if not (record["backend"]["id"] == "defer" and record["seed"] == 2)]
    with pytest.raises(AblationError, match="seed mismatch"):
        compare_local_ablations(missing, reference_id="guard")


def test_ablation_rejects_unmatched_decision_keys(tmp_path):
    manifest = _manifest()
    jsonl_path = tmp_path / "runs.jsonl"
    run_manifest(
        manifest,
        jsonl_path=jsonl_path,
        csv_path=tmp_path / "summary.csv",
        client_factories={"local": lambda backend, _seed: FakeLocalAblationClient(backend["prompt_profile"])},
    )
    records = deepcopy(read_result_records(jsonl_path))
    candidate = next(record for record in records if record["backend"]["id"] == "defer" and record["seed"] == 1)
    candidate["decisions"][0]["row"] = 99
    with pytest.raises(AblationError, match="decision key mismatch"):
        compare_local_ablations(records, reference_id="guard")
