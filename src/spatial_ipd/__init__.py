"""Headless Spatial Iterated Prisoner's Dilemma engine."""

from spatial_ipd.engine import (
    SimulationResult,
    adopt_best,
    apply_mutation,
    cooperation_rate,
    random_grid,
    score_cells,
    simulate,
    step,
)
from spatial_ipd.neighborhood import MOORE_OFFSETS, moore_neighbors
from spatial_ipd.payoffs import (
    COOPERATE,
    DEFECT,
    PUNISHMENT,
    REWARD,
    SUCKER,
    TEMPTATION,
    mutual_payoffs,
    payoff,
)

__all__ = [
    "COOPERATE",
    "DEFECT",
    "MOORE_OFFSETS",
    "PUNISHMENT",
    "REWARD",
    "SUCKER",
    "TEMPTATION",
    "SimulationResult",
    "adopt_best",
    "apply_mutation",
    "cooperation_rate",
    "moore_neighbors",
    "mutual_payoffs",
    "payoff",
    "random_grid",
    "score_cells",
    "simulate",
    "step",
]
