"""Canonical active policies and deterministic repeated matches."""

import json

import pytest

from spatial_ipd.arena import (
    GUARD_POLICIES,
    AlwaysCooperate,
    AlwaysDefect,
    CommunicationGuard,
    ForgivingTitForTat,
    MemoryWindowPolicy,
    Observation,
    Pavlov,
    ReputationGuard,
    TitForTat,
    information_signals,
    main,
    play_match,
    rank_policies,
    round_robin,
    run_tournament,
)
from spatial_ipd.engine import simulate
from spatial_ipd.payoffs import COOPERATE as C
from spatial_ipd.payoffs import DEFECT as D


def test_always_cooperate_vs_always_defect():
    result = play_match(AlwaysCooperate(), AlwaysDefect(), rounds=3)
    assert result.actions_a == (C, C, C)
    assert result.actions_b == (D, D, D)
    assert result.payoff_a == 0
    assert result.payoff_b == 15
    assert result.cooperation_rate_a == 1.0
    assert result.cooperation_rate_b == 0.0


def test_tit_for_tat_retaliates_after_first_defection():
    result = play_match(TitForTat(), AlwaysDefect(), rounds=4)
    assert result.actions_a == (C, D, D, D)
    assert result.actions_b == (D, D, D, D)
    assert result.payoff_a == 3
    assert result.payoff_b == 8


def test_tit_for_tat_self_play_sustains_cooperation():
    result = play_match(TitForTat(), TitForTat(), rounds=5)
    assert result.actions_a == result.actions_b == (C, C, C, C, C)
    assert result.payoff_a == result.payoff_b == 15
    assert result.mutual_cooperation_rounds == 5


def test_pavlov_switches_after_sucker_and_punishment():
    result = play_match(Pavlov(), AlwaysDefect(), rounds=4)
    assert result.actions_a == (C, D, C, D)
    assert Pavlov().choose(Observation(round_index=2, own_history=(C, C), opponent_history=(C, C))) == C


def test_forgiving_tit_for_tat_waits_for_two_defections():
    result = play_match(ForgivingTitForTat(), AlwaysDefect(), rounds=4)
    assert result.actions_a == (C, C, D, D)


def test_memory_window_changes_only_visible_history():
    limited = play_match(
        MemoryWindowPolicy(ForgivingTitForTat(), 1),
        AlwaysDefect(),
        rounds=4,
    )
    assert limited.actions_a == (C, C, C, C)
    tft = play_match(
        MemoryWindowPolicy(TitForTat(), 1),
        AlwaysDefect(),
        rounds=4,
    )
    assert tft.actions_a == (C, D, D, D)


def test_reputation_guard_cooperates_without_signal_and_gates_known_opponents():
    policy = ReputationGuard(threshold=0.5)
    base = dict(round_index=0, own_history=(), opponent_history=())
    assert policy.choose(Observation(**base)) == C
    assert policy.choose(Observation(**base, opponent_reputation=0.5)) == C
    assert policy.choose(Observation(**base, opponent_reputation=0.49)) == D


def test_communication_guard_cooperates_without_reports_and_gates_warnings():
    policy = CommunicationGuard(threshold=0.5)
    base = dict(round_index=0, own_history=(), opponent_history=())
    assert policy.choose(Observation(**base)) == C
    assert policy.choose(Observation(**base, opponent_warning_rate=0.49)) == C
    assert policy.choose(Observation(**base, opponent_warning_rate=0.5)) == D


def test_round_robin_includes_self_play_and_every_pair():
    results = round_robin(rounds=2)
    assert len(results) == 15
    assert results[0].policy_a == results[0].policy_b == "always_cooperate"
    assert any(result.policy_a == "pavlov" and result.policy_b == "forgiving_tit_for_tat" for result in results)


def test_guard_roster_and_informed_replay_change_defector_matches():
    first = round_robin(rounds=4, policy_types=GUARD_POLICIES)
    assert len(first) == 28
    signals = information_signals(first)
    assert signals["always_defect"]["reputation"] < 0.5
    assert signals["always_defect"]["warning_rate"] == 1.0
    assert signals["always_cooperate"]["warning_rate"] == 0.0
    second = round_robin(rounds=4, policy_types=GUARD_POLICIES, signals=signals)
    uninformed = next(
        result for result in first if result.policy_a == "always_defect" and result.policy_b == "reputation_guard"
    )
    informed = next(
        result for result in second if result.policy_a == "always_defect" and result.policy_b == "reputation_guard"
    )
    assert uninformed.actions_b == (C, C, C, C)
    assert informed.actions_b == (D, D, D, D)
    assert run_tournament(rounds=4, roster="guards", information="on")["passes"][1]["matches"] == second


def test_tournament_rankings_are_deterministic_and_size_preserving():
    first = run_tournament(rounds=5, roster="guards", information="on")
    second = run_tournament(rounds=5, roster="guards", information="on")
    assert first == second
    informed = first["passes"][1]["rankings"]
    assert [row["rank"] for row in informed] == list(range(1, 8))
    assert {row["policy"] for row in informed} == {policy.name for policy in GUARD_POLICIES}
    assert rank_policies(first["passes"][0]["matches"], information="off", pass_name="uninformed")[0]["rank"] == 1


def test_policy_action_is_validated():
    class InvalidPolicy:
        name = "invalid"

        def choose(self, observation):
            del observation
            return 7

    with pytest.raises(ValueError, match="invalid action"):
        play_match(InvalidPolicy(), AlwaysCooperate(), rounds=1)


def test_arena_cli_writes_guard_tournament_artifacts(tmp_path, capsys):
    jsonl = tmp_path / "tournament.jsonl"
    csv_path = tmp_path / "tournament.csv"
    assert (
        main(
            [
                "--rounds",
                "3",
                "--roster",
                "guards",
                "--information",
                "on",
                "--jsonl",
                str(jsonl),
                "--csv",
                str(csv_path),
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out.strip())
    assert summary["matches"] == 56
    assert summary["passes"] == ["uninformed", "informed"]
    records = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines()]
    assert records[-1]["roster"] == "guards"
    assert csv_path.read_text(encoding="utf-8").splitlines()[0].startswith("information,pass_name,rank")


def test_arena_cli_prints_fifteen_matches_and_summary(capsys):
    assert main(["--rounds", "3"]) == 0
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(records) == 16
    assert all(record["record_type"] == "match" for record in records[:-1])
    assert records[-1] == {
        "record_type": "summary",
        "policies": 5,
        "matches": 15,
        "rounds_per_match": 3,
    }


def test_arena_does_not_change_locked_spatial_golden():
    result = simulate(12, 12, 30, seed=20260316, mutation_rate=0.02)
    assert result.final_cooperation_rate == 2 / 144
