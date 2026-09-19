"""Versioned thinker records and deterministic offline replay."""

import json
from types import SimpleNamespace

import pytest

from spatial_ipd.replay import (
    DECISION_RECORD_VERSION,
    ReplayError,
    ReplayThinkerClient,
    capture_decision_record,
    question_snapshot,
    read_decision_records,
    write_decision_records,
)
from spatial_ipd.think import RandomThinkerClient, main, simulate_with_thinkers


def _run_kwargs():
    return {
        "height": 4,
        "width": 4,
        "generations": 4,
        "seed": 1,
        "mutation_rate": 0.02,
        "think_every": 2,
        "think_last": 0,
        "sticky": 0,
        "thinker_count": 2,
    }


def test_decision_record_round_trip_replays_same_run(tmp_path):
    expected, expected_stats = simulate_with_thinkers(
        **_run_kwargs(),
        client=RandomThinkerClient(1),
    )
    path = write_decision_records(tmp_path / "decisions.jsonl", expected_stats.decision_records)
    records = read_decision_records(path)

    assert len(records) == expected_stats.calls == 2
    assert records[0].schema_version == DECISION_RECORD_VERSION
    assert records[0].state["generation"] == 2
    assert "worth_thinking_0" in records[0].questions
    assert records[0].response["confidence_kind"] == "random_draw"

    replay = ReplayThinkerClient(records)
    actual, actual_stats = simulate_with_thinkers(**_run_kwargs(), client=replay)
    replay.assert_exhausted()

    assert actual.cooperation_rates == expected.cooperation_rates
    assert actual.grid == expected.grid
    assert actual_stats.all_decisions == expected_stats.all_decisions


def test_question_snapshot_retains_available_semantics():
    question = SimpleNamespace(
        instructions="Should this cell hold?",
        criteria={"true": "Preserve a cluster.", "false": "Imitate the winner."},
    )
    snapshot = question_snapshot({"resist_0": question, "worth_thinking_0": None})
    assert snapshot["resist_0"] == {
        "type": "SimpleNamespace",
        "instructions": "Should this cell hold?",
        "criteria": {"true": "Preserve a cluster.", "false": "Imitate the winner."},
    }
    assert snapshot["worth_thinking_0"] is None


def test_replay_rejects_state_mismatch():
    response = SimpleNamespace(
        nouls={"worth_thinking_0": SimpleNamespace(noul=0.8)},
        scores={},
        choices={},
    )
    record = capture_decision_record(
        {"generation": 1, "thinkers": []},
        {"worth_thinking_0": None},
        response,
    )
    replay = ReplayThinkerClient([record])
    with pytest.raises(ReplayError, match="state mismatch at call 1"):
        replay.system_one(
            {"generation": 2, "thinkers": []},
            {"worth_thinking_0": None},
        )


def test_replay_rejects_question_mismatch_and_exhaustion():
    response = SimpleNamespace(nouls={}, scores={}, choices={})
    record = capture_decision_record({"generation": 1}, {"expected": None}, response)
    mismatch = ReplayThinkerClient([record])
    with pytest.raises(ReplayError, match="question IDs mismatch at call 1"):
        mismatch.system_one({"generation": 1}, {"other": None})

    unused = ReplayThinkerClient([record])
    with pytest.raises(ReplayError, match="1 unused record"):
        unused.assert_exhausted()

    consumed = ReplayThinkerClient([record])
    consumed.system_one({"generation": 1}, {"expected": None})
    consumed.assert_exhausted()
    with pytest.raises(ReplayError, match="exhausted after 1 calls"):
        consumed.system_one({"generation": 1}, {"expected": None})


def test_replay_rejects_unknown_schema_version(tmp_path):
    path = tmp_path / "future.jsonl"
    path.write_text(
        json.dumps(
            {
                "schema_version": DECISION_RECORD_VERSION + 1,
                "state": {},
                "questions": {},
                "response": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ReplayError, match="schema version"):
        read_decision_records(path)


def test_cli_records_then_replays_without_backend(capsys, tmp_path):
    path = tmp_path / "decisions.jsonl"
    common = [
        "--height",
        "4",
        "--width",
        "4",
        "--generations",
        "2",
        "--seed",
        "1",
        "--mutation-rate",
        "0.02",
        "--think-every",
        "1",
        "--think-last",
        "0",
        "--sticky",
        "0",
        "--thinkers",
        "1",
    ]
    assert main([*common, "--backend", "random", "--record-decisions", str(path)]) == 0
    recorded_output = capsys.readouterr().out
    assert path.is_file()
    assert "decision_records_written=2" in recorded_output

    assert main([*common, "--replay-decisions", str(path)]) == 0
    replayed_output = capsys.readouterr().out
    assert recorded_output.splitlines()[0] == replayed_output.splitlines()[0]
    assert "decision_records_replayed=2" in replayed_output
