"""Judged arena actions: one cooperate Noul, code-owned 0.5 gate."""

import json

import pytest

from spatial_ipd.arena import Observation
from spatial_ipd.engine import simulate
from spatial_ipd.judged import (
    COOPERATE_THRESHOLD,
    JUDGED_POLICY,
    JudgedPolicy,
    ScriptedCooperateClient,
    main,
    run_judged,
)
from spatial_ipd.local_llm import LocalCooperateClient
from spatial_ipd.payoffs import COOPERATE as C
from spatial_ipd.payoffs import DEFECT as D
from spatial_ipd.population import POPULATION_MANIFEST_VERSION


def _manifest():
    return {
        "schema_version": POPULATION_MANIFEST_VERSION,
        "seed": 20260921,
        "encounters": 8,
        "rounds_per_match": 4,
        "population": {
            "always_cooperate": 2,
            "always_defect": 2,
            "judged": 2,
            "tit_for_tat": 2,
        },
    }


def _observation():
    return Observation(round_index=1, own_history=(C,), opponent_history=(D,))


def test_gate_cooperates_at_one_half_and_defects_below():
    assert COOPERATE_THRESHOLD == 0.5
    hold = JudgedPolicy(ScriptedCooperateClient(0.5))
    drop = JudgedPolicy(ScriptedCooperateClient(0.49))
    assert hold.choose(_observation()) == C
    assert drop.choose(_observation()) == D


def test_local_boolean_maps_to_the_gate_and_keeps_confidence():
    def transport(endpoint, payload, headers, timeout):
        del endpoint, headers, timeout
        body = json.loads(payload["messages"][1]["content"])
        assert "opponent_history" in body["state"]
        return {"choices": [{"message": {"content": '{"cooperate": false, "confidence": 0.8}'}}]}

    records = []
    policy = JudgedPolicy(LocalCooperateClient("qwen3:8b", transport=transport), records)
    assert policy.choose(_observation()) == D
    assert records[0]["noul"] == 0.0
    assert records[0]["confidence"] == 0.8
    assert records[0]["confidence_kind"] == "llm_self_report"


def test_backends_share_pairs_and_random_is_deterministic():
    manifest = _manifest()
    high_events, high_summaries = run_judged(manifest, ScriptedCooperateClient(0.9))
    low_events, low_summaries = run_judged(manifest, ScriptedCooperateClient(0.1))
    first_events, first_summaries = run_judged(manifest, ScriptedCooperateClient(0.9))
    assert [(event["agent_a"], event["agent_b"]) for event in high_events] == [
        (event["agent_a"], event["agent_b"]) for event in low_events
    ]
    assert high_events[0]["actions_a"] != low_events[0]["actions_a"] or high_summaries != low_summaries
    judged_high = next(summary for summary in high_summaries if summary["policy"] == JUDGED_POLICY)
    judged_low = next(summary for summary in low_summaries if summary["policy"] == JUDGED_POLICY)
    assert judged_high["cooperation_rate"] > judged_low["cooperation_rate"]
    again_events, again_summaries = run_judged(manifest, ScriptedCooperateClient(0.9))
    assert first_events == again_events
    assert first_summaries == again_summaries


def test_random_population_repeats_and_cli_writes_decisions(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
    jsonl = tmp_path / "population.jsonl"
    csv_path = tmp_path / "summary.csv"
    decisions = tmp_path / "decisions.jsonl"
    args = [
        "--manifest",
        str(manifest_path),
        "--backend",
        "random",
        "--jsonl",
        str(jsonl),
        "--csv",
        str(csv_path),
        "--decisions",
        str(decisions),
    ]
    assert main(args) == 0
    assert (
        main(
            [
                "--manifest",
                str(manifest_path),
                "--backend",
                "random",
                "--jsonl",
                str(tmp_path / "second.jsonl"),
                "--csv",
                str(tmp_path / "second.csv"),
                "--decisions",
                str(tmp_path / "second-decisions.jsonl"),
            ]
        )
        == 0
    )
    assert jsonl.read_bytes() == (tmp_path / "second.jsonl").read_bytes()
    rows = [json.loads(line) for line in decisions.read_text(encoding="utf-8").splitlines()]
    assert rows
    assert all(row["confidence_kind"] == "random_draw" for row in rows)
    assert all(row["action"] == (C if row["noul"] >= 0.5 else D) for row in rows)


def test_judged_slice_preserves_spatial_golden():
    result = simulate(12, 12, 30, seed=20260316, mutation_rate=0.02)
    assert result.final_cooperation_rate == 2 / 144


def test_local_backend_requires_a_model():
    with pytest.raises(SystemExit):
        main(["--manifest", "missing", "--jsonl", "a", "--csv", "b", "--backend", "local"])
