"""Reliability bins, selective coverage, and sample-size guardrails."""

import json

import pytest

from spatial_ipd.engine import simulate
from spatial_ipd.reliability import (
    ReliabilityError,
    main,
    read_calibration_events,
    reliability_report,
)


def _events():
    return [
        {
            "record_type": "event",
            "scenario": "true_high",
            "resist_probability": 0.9,
            "resist_decision": True,
            "hold_better": True,
        },
        {
            "record_type": "event",
            "scenario": "false_high",
            "resist_probability": 0.8,
            "resist_decision": True,
            "hold_better": False,
        },
        {
            "record_type": "event",
            "scenario": "false_low",
            "resist_probability": 0.2,
            "resist_decision": False,
            "hold_better": False,
        },
        {
            "record_type": "event",
            "scenario": "true_low",
            "resist_probability": 0.1,
            "resist_decision": False,
            "hold_better": True,
        },
        {
            "record_type": "event",
            "scenario": "tie",
            "resist_probability": 0.7,
            "resist_decision": True,
            "hold_better": None,
        },
    ]


def test_reliability_report_bins_ece_scores_and_ties():
    report = reliability_report(
        _events(),
        bins=2,
        confidence_threshold=0.8,
        minimum_events=4,
    )
    assert report["events"] == 5
    assert report["resolved"] == 4
    assert report["ties"] == 1
    assert report["positives"] == report["negatives"] == 2
    assert report["brier_score"] == pytest.approx(0.375)
    assert report["expected_calibration_error"] == pytest.approx(0.35)
    assert report["accuracy"] == 0.5
    assert report["coverage"] == 1.0
    assert report["selective_accuracy"] == 0.5
    assert report["action_probability_consistency"] == 1.0
    assert report["calibration_claim_supported"] is True
    assert report["reliability_bins"] == [
        {
            "lower": 0.0,
            "upper": 0.5,
            "count": 2,
            "mean_probability": pytest.approx(0.15),
            "observed_frequency": 0.5,
            "calibration_gap": pytest.approx(0.35),
        },
        {
            "lower": 0.5,
            "upper": 1.0,
            "count": 2,
            "mean_probability": pytest.approx(0.85),
            "observed_frequency": 0.5,
            "calibration_gap": pytest.approx(0.35),
        },
    ]


def test_minimum_sample_and_class_balance_gate_claims():
    too_small = reliability_report(_events(), minimum_events=5)
    assert too_small["resolved"] == 4
    assert too_small["calibration_claim_supported"] is False
    one_class = reliability_report(
        [
            {
                "resist_probability": 0.8,
                "resist_decision": True,
                "hold_better": True,
            }
        ],
        minimum_events=1,
    )
    assert one_class["calibration_claim_supported"] is False


def test_action_consistency_is_separate_from_correctness():
    events = _events()
    events[0]["resist_decision"] = False
    report = reliability_report(events, minimum_events=10)
    assert report["accuracy"] == 0.5
    assert report["action_probability_consistency"] == 0.75


def test_reliability_rejects_no_resolved_or_malformed_events():
    with pytest.raises(ReliabilityError, match="resolved"):
        reliability_report([{"resist_probability": 0.5, "resist_decision": True, "hold_better": None}])
    with pytest.raises(ReliabilityError, match="probability"):
        reliability_report([{"resist_probability": 2.0, "resist_decision": True, "hold_better": True}])
    with pytest.raises(ReliabilityError, match="resist_decision"):
        reliability_report([{"resist_probability": 0.8, "hold_better": True}])


def test_jsonl_reader_ignores_summary_and_cli_prints_report(tmp_path, capsys):
    path = tmp_path / "calibration.jsonl"
    records = [
        *_events(),
        {"record_type": "summary", "resolved": 4},
    ]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    assert len(read_calibration_events([path])) == 5
    assert (
        main(
            [
                "--jsonl",
                str(path),
                "--bins",
                "2",
                "--confidence-threshold",
                "0.8",
                "--minimum-events",
                "4",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["schema_version"] == 1
    assert report["resolved"] == 4
    assert report["calibration_claim_supported"] is True


def test_reliability_slice_preserves_locked_golden():
    result = simulate(12, 12, 30, seed=20260316, mutation_rate=0.02)
    assert result.final_cooperation_rate == 2 / 144
