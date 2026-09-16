"""Payoff matrix: CC→3,3; DD→1,1; D vs C → D gets 5, C gets 0."""

from spatial_ipd import (
    COOPERATE,
    DEFECT,
    PUNISHMENT,
    REWARD,
    SUCKER,
    TEMPTATION,
    mutual_payoffs,
    payoff,
)


def test_cooperate_cooperate():
    assert mutual_payoffs(COOPERATE, COOPERATE) == (3, 3)
    assert payoff(COOPERATE, COOPERATE) == REWARD == 3


def test_defect_defect():
    assert mutual_payoffs(DEFECT, DEFECT) == (1, 1)
    assert payoff(DEFECT, DEFECT) == PUNISHMENT == 1


def test_defect_versus_cooperate():
    assert mutual_payoffs(DEFECT, COOPERATE) == (5, 0)
    assert payoff(DEFECT, COOPERATE) == TEMPTATION == 5
    assert payoff(COOPERATE, DEFECT) == SUCKER == 0


def test_cooperate_versus_defect_is_symmetric():
    assert mutual_payoffs(COOPERATE, DEFECT) == (0, 5)
