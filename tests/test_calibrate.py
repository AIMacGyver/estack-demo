"""Counterfactual labels, probability orientation, and calibration metrics."""

import json
import math
from types import SimpleNamespace

import pytest

import spatial_ipd.calibrate as calibrate_mod
from spatial_ipd.calibrate import (
    binary_log_loss,
    brier_score,
    calibration_summary,
    evaluate_counterfactual,
    main,
    resist_true_probability,
    run_calibration,
)
from spatial_ipd.engine import simulate
from spatial_ipd.probes import SCENARIOS


class FakeCalibrationClient:
    """Return a stable high resist probability for every C→D motif."""

    uses_sdk_questions = False

    def system_one(self, state, questions):
        del state
        nouls = {}
        scores = {}
        for key in questions:
            if key.startswith("worth_thinking_"):
                nouls[key] = SimpleNamespace(noul=0.9)
            elif key.startswith("resist_"):
                nouls[key] = SimpleNamespace(noul=0.8)
            elif key.startswith("cluster_fragility_"):
                scores[key] = SimpleNamespace(score=1.5, confidence=0.8)
        return SimpleNamespace(
            nouls=nouls,
            scores=scores,
            choices={},
            confidence_kind="jev_noul",
        )


def _scenario(name):
    return next(scenario for scenario in SCENARIOS if scenario.id == name)


def test_counterfactual_labels_pivotal_hold_better_and_collapse_tie():
    pivotal = evaluate_counterfactual(
        _scenario("pivotal_cluster_hold"),
        horizon=3,
        seed=1,
    )
    assert pivotal.hold_rates == (0.08, 0.0, 0.0)
    assert pivotal.imitate_rates == (0.0, 0.0, 0.0)
    assert pivotal.hold_better is True
    assert pivotal.mean_delta == pytest.approx(0.08 / 3)

    collapse = evaluate_counterfactual(
        _scenario("lone_defector_invasion"),
        horizon=3,
        seed=1,
    )
    assert collapse.hold_rates == collapse.imitate_rates == (0.0, 0.0, 0.0)
    assert collapse.hold_better is None


def test_counterfactual_branches_use_identical_rng_streams(monkeypatch):
    draws = []

    def draw_once(grid, rng, mutation_rate):
        del mutation_rate
        draws.append(rng.random())
        return [row[:] for row in grid]

    monkeypatch.setattr(calibrate_mod, "step", draw_once)
    evaluate_counterfactual(
        _scenario("pivotal_cluster_hold"),
        horizon=3,
        seed=17,
        mutation_rate=0.5,
    )
    assert draws[:3] == draws[3:]


def test_local_self_report_is_oriented_as_probability_of_true():
    assert (
        resist_true_probability(
            {
                "resist_signal": 1.0,
                "resist_confidence": 0.7,
                "confidence_kind": "llm_self_report",
            }
        )
        == 0.7
    )
    assert resist_true_probability(
        {
            "resist_signal": 0.0,
            "resist_confidence": 0.7,
            "confidence_kind": "llm_self_report",
        }
    ) == pytest.approx(0.3)
    assert (
        resist_true_probability(
            {
                "resist_signal": 0.82,
                "resist_confidence": 0.1,
                "confidence_kind": "jev_noul",
            }
        )
        == 0.82
    )


def test_brier_and_log_loss_use_resolved_binary_labels():
    probabilities = [0.8, 0.2]
    labels = [True, False]
    assert brier_score(probabilities, labels) == pytest.approx(0.04)
    assert binary_log_loss(probabilities, labels) == pytest.approx(-math.log(0.8))
    summary = calibration_summary(
        [
            {"resist_probability": 0.8, "hold_better": True},
            {"resist_probability": 0.2, "hold_better": False},
            {"resist_probability": 0.9, "hold_better": None},
        ]
    )
    assert summary["resolved"] == 2
    assert summary["ties"] == 1
    assert summary["brier_score"] == pytest.approx(0.04)
    assert summary["calibration_claim_supported"] is False


def test_fake_backend_calibration_scores_only_resolved_motifs():
    events, summary = run_calibration(
        FakeCalibrationClient(),
        horizon=3,
        seed=1,
    )
    assert [event["scenario"] for event in events] == [
        "lone_defector_invasion",
        "isolated_cooperator_collapse",
        "pivotal_cluster_hold",
    ]
    assert summary["events"] == 3
    assert summary["resolved"] == 1
    assert summary["ties"] == 2
    assert summary["brier_score"] == pytest.approx(0.04)
    assert all(event["resist_probability"] == 0.8 for event in events)


def test_random_cli_prints_events_and_summary(capsys):
    assert main(["--backend", "random", "--seed", "1", "--horizon", "3"]) == 0
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(records) == 4
    assert [record["record_type"] for record in records] == ["event", "event", "event", "summary"]
    assert records[-1]["resolved"] == 1
    assert records[-1]["ties"] == 2
    assert records[-1]["calibration_claim_supported"] is False


def test_calibration_slice_preserves_locked_golden():
    result = simulate(12, 12, 30, seed=20260316, mutation_rate=0.02)
    assert result.final_cooperation_rate == 2 / 144
