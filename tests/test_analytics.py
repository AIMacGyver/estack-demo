"""Pure cluster, frontier, payoff, and persistence metrics."""

from spatial_ipd.analytics import analyze_grid, cooperation_persistence
from spatial_ipd.payoffs import COOPERATE as C
from spatial_ipd.payoffs import DEFECT as D


def test_uniform_cooperation_is_one_cluster_without_frontier():
    metrics = analyze_grid([[C, C, C], [C, C, C], [C, C, C]])
    assert metrics.cooperator_clusters == 1
    assert metrics.largest_cooperator_cluster == 9
    assert metrics.frontier_cells == 0
    assert metrics.cooperator_frontier_cells == 0
    assert metrics.mean_cooperator_payoff == 24.0
    assert metrics.mean_defector_payoff is None


def test_uniform_defection_has_no_cooperator_cluster():
    metrics = analyze_grid([[D, D], [D, D]])
    assert metrics.cooperator_clusters == 0
    assert metrics.largest_cooperator_cluster == 0
    assert metrics.frontier_cells == 0
    assert metrics.mean_cooperator_payoff is None
    assert metrics.mean_defector_payoff == 8.0


def test_toroidal_corner_cooperators_form_one_moore_cluster():
    grid = [
        [C, D, C],
        [D, D, D],
        [C, D, C],
    ]
    metrics = analyze_grid(grid)
    assert metrics.cooperator_clusters == 1
    assert metrics.largest_cooperator_cluster == 4
    assert metrics.frontier_cells == 9
    assert metrics.cooperator_frontier_cells == 4
    assert metrics.mean_cooperator_payoff is not None
    assert metrics.mean_defector_payoff is not None


def test_cooperation_persistence_counts_nonzero_recorded_rates():
    assert cooperation_persistence([0.5]) == 1.0
    assert cooperation_persistence([0.5, 0.0, 0.25, 0.0]) == 0.5
