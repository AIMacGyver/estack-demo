"""Forecasts against committed arena outcomes, not lattice ties."""

import json

import pytest

from spatial_ipd.engine import simulate
from spatial_ipd.forecast import (
    MINIMUM_EVENTS,
    LocalForecastClient,
    RandomForecastClient,
    committed_propositions,
    main,
    run_forecasts,
    score_forecasts,
)


def test_catalog_has_both_classes_and_hides_outcomes():
    items = committed_propositions()
    labels = [bool(item["label"]) for item in items]
    assert len(items) >= MINIMUM_EVENTS
    assert any(labels) and not all(labels)
    forbidden = {"next_count", "payoff", "cooperation_rate", "label"}
    for item in items:
        assert forbidden.isdisjoint(item["state"])


def test_random_forecast_is_deterministic_and_scored():
    items = committed_propositions()
    first = run_forecasts(RandomForecastClient(1), items, backend="random")
    second = run_forecasts(RandomForecastClient(1), items, backend="random")
    assert first == second
    summary = score_forecasts(first)
    assert summary["ties"] == 0
    assert summary["positives"] > 0
    assert summary["negatives"] > 0
    assert summary["calibration_claim_supported"] is True
    assert 0.0 <= summary["brier_score"] <= 1.0


def test_local_false_answer_orients_probability_below_one_half():
    def transport(endpoint, payload, headers, timeout):
        del endpoint, headers, timeout
        assert "final Always Defect" in json.loads(payload["messages"][1]["content"])["state"]["question"]
        return {"choices": [{"message": {"content": '{"answer": false, "confidence": 0.8}'}}]}

    client = LocalForecastClient("qwen3:8b", transport=transport)
    events = run_forecasts(
        client,
        [{"id": "one", "state": {"question": "Is the final Always Defect count below 10?"}, "label": False}],
        backend="local",
    )
    assert events[0]["probability"] == pytest.approx(0.2)
    assert events[0]["decision"] is False
    assert events[0]["confidence_kind"] == "llm_self_report"


def test_perfect_probabilities_score_zero_and_small_samples_cannot_claim():
    events = [
        {"probability": 1.0, "label": True},
        {"probability": 0.0, "label": False},
    ]
    summary = score_forecasts(events)
    assert summary["brier_score"] == 0.0
    assert summary["calibration_claim_supported"] is False


def test_forecast_cli_writes_repeatable_random_jsonl(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    assert main(["--backend", "random", "--seed", "1", "--jsonl", str(first)]) == 0
    assert main(["--backend", "random", "--seed", "1", "--jsonl", str(second)]) == 0
    assert first.read_bytes() == second.read_bytes()
    summary = json.loads(first.read_text(encoding="utf-8").splitlines()[-1])
    assert summary["backend"] == "random"
    assert summary["record_type"] == "summary"


def test_forecast_slice_preserves_spatial_golden():
    result = simulate(12, 12, 30, seed=20260316, mutation_rate=0.02)
    assert result.final_cooperation_rate == 2 / 144
