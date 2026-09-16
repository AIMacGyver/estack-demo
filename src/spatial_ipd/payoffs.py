"""Prisoner's Dilemma payoff matrix.

Canonical one-shot payoffs (Nowak & May, 1992 form with T=5, R=3, P=1, S=0):

    C vs C → 3, 3  (Reward)
    D vs D → 1, 1  (Punishment)
    D vs C → 5, 0  (Temptation / Sucker)
    C vs D → 0, 5
"""

COOPERATE = 1
DEFECT = 0

REWARD = 3
TEMPTATION = 5
PUNISHMENT = 1
SUCKER = 0

# payoff[focal][opponent] → score for the focal player
_PAYOFF = (
    (PUNISHMENT, TEMPTATION),  # focal Defect vs (D, C)
    (SUCKER, REWARD),  # focal Cooperate vs (D, C)
)


def payoff(focal: int, opponent: int) -> int:
    """Return the payoff earned by `focal` against `opponent`."""
    return _PAYOFF[int(focal)][int(opponent)]


def mutual_payoffs(a: int, b: int) -> tuple[int, int]:
    """Return (payoff_to_a, payoff_to_b) for a pairwise encounter."""
    return payoff(a, b), payoff(b, a)
