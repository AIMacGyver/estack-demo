"""Representative microbenchmarks; excluded from the default pytest path."""

from random import Random

from spatial_ipd.analytics import analyze_grid
from spatial_ipd.engine import random_grid, simulate, step


def test_simulate_medium_lattice(benchmark):
    """Benchmark the primary deterministic simulation boundary."""
    result = benchmark(
        simulate,
        64,
        64,
        30,
        seed=20260316,
        mutation_rate=0.02,
    )
    assert len(result.cooperation_rates) == 31


def test_step_medium_lattice(benchmark):
    """Benchmark one score/imitate/mutate generation."""
    grid = random_grid(64, 64, seed=20260316)
    result = benchmark(step, grid, rng=Random(20260316), mutation_rate=0.02)
    assert len(result) == 64


def test_analyze_medium_lattice(benchmark):
    """Benchmark final-grid spatial analytics."""
    grid = random_grid(64, 64, seed=20260316)
    result = benchmark(analyze_grid, grid)
    assert result.cooperator_clusters >= 1
