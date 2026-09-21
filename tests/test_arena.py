"""Canonical active policies and deterministic repeated matches."""

import json

import pytest

from spatial_ipd.arena import (
    AlwaysCooperate,
    AlwaysDefect,
    ForgivingTitForTat,
    MemoryWindowPolicy,
    Observation,
    Pavlov,
    TitForTat,
    main,
    play_match,
    round_robin,
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


def test_round_robin_includes_self_play_and_every_pair():
    results = round_robin(rounds=2)
    assert len(results) == 15
    assert results[0].policy_a == results[0].policy_b == "always_cooperate"
    assert any(result.policy_a == "pavlov" and result.policy_b == "forgiving_tit_for_tat" for result in results)


def test_policy_action_is_validated():
    class InvalidPolicy:
        name = "invalid"

        def choose(self, observation):
            del observation
            return 7

    with pytest.raises(ValueError, match="invalid action"):
        play_match(InvalidPolicy(), AlwaysCooperate(), rounds=1)


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
