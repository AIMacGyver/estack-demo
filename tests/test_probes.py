"""Deterministic motif transitions and fake-backed probe decisions."""

import json
from types import SimpleNamespace

from spatial_ipd.engine import adopt_best, score_cells
from spatial_ipd.probes import SCENARIOS, main, run_probe, run_probes, transition_name


class FakeProbeClient:
    """Return high worth/resist signals for every requested question."""

    uses_sdk_questions = False

    def __init__(self):
        self.calls = []

    def system_one(self, state, questions):
        self.calls.append((state, questions))
        nouls = {}
        scores = {}
        for key in questions:
            if key.startswith(("worth_thinking_", "resist_")):
                nouls[key] = SimpleNamespace(noul=0.9)
            elif key.startswith("cluster_fragility_"):
                scores[key] = SimpleNamespace(score=1.5, confidence=0.8)
        return SimpleNamespace(
            nouls=nouls,
            scores=scores,
            choices={},
            confidence_kind="fake_signal",
        )


def test_scenario_catalog_has_code_derived_named_transitions():
    transitions = {}
    for scenario in SCENARIOS:
        before = scenario.before_grid()
        expected_after = adopt_best(before, score_cells(before))
        after = scenario.after_grid()
        row, col = scenario.seat
        assert after == expected_after
        transitions[scenario.id] = transition_name(before[row][col], after[row][col])

    assert transitions == {
        "lone_defector_invasion": "C->D",
        "isolated_cooperator_collapse": "C->D",
        "toroidal_cluster_recovery": "D->C",
        "stable_cluster_boundary": "C->C",
        "uniform_cooperation": "C->C",
    }


def test_fake_backend_holds_only_real_c_to_d_scenarios():
    client = FakeProbeClient()
    results = run_probes(client)
    by_name = {result["scenario"]: result for result in results}

    for name in ("lone_defector_invasion", "isolated_cooperator_collapse"):
        assert by_name[name]["transition"] == "C->D"
        assert by_name[name]["act"] == "hold"
        assert by_name[name]["applied"] is True
        assert by_name[name]["resist_signal"] == 0.9

    for name in ("toroidal_cluster_recovery", "stable_cluster_boundary", "uniform_cooperation"):
        assert by_name[name]["act"] == "imitate"
        assert by_name[name]["applied"] is False
        assert by_name[name]["resist_signal"] is None
        assert by_name[name]["resist_confidence"] is None

    assert len(client.calls) == len(SCENARIOS)
    assert "resist_0" in client.calls[0][1]
    assert "resist_0" not in client.calls[2][1]


def test_probe_output_includes_state_scores_and_confidence_provenance():
    result = run_probe(SCENARIOS[0], FakeProbeClient())
    assert result["seat"] == {"row": 0, "col": 0}
    assert len(result["patch_after"]) == 3
    assert result["focal_score"] < result["best_neighbor_score"]
    assert result["best_neighbor_strategy"] == 0
    assert result["worth_signal"] == 0.9
    assert result["worth_confidence"] == 0.9
    assert result["confidence_kind"] == "fake_signal"


def test_random_cli_prints_one_json_object_per_scenario(capsys):
    assert main(["--backend", "random", "--seed", "1"]) == 0
    lines = capsys.readouterr().out.splitlines()
    records = [json.loads(line) for line in lines]
    assert len(records) == len(SCENARIOS)
    assert [record["scenario"] for record in records] == [scenario.id for scenario in SCENARIOS]
    assert all(record["confidence_kind"] == "random_draw" for record in records)
