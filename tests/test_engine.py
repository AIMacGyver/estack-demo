"""Scoring, imitation, mutation, and seeded determinism."""

from itertools import product
from random import Random

from spatial_ipd import (
    COOPERATE,
    DEFECT,
    adopt_best,
    apply_mutation,
    cooperation_rate,
    moore_neighbors,
    payoff,
    random_grid,
    score_cells,
    simulate,
    step,
)

C, D = COOPERATE, DEFECT

SEEDED_HEIGHT = 12
SEEDED_WIDTH = 12
SEEDED_GENERATIONS = 30
SEEDED_SEED = 20260316
SEEDED_MUTATION_RATE = 0.02
SEEDED_CELLS = SEEDED_HEIGHT * SEEDED_WIDTH
# Cooperator counts after each generation (index 0 = initial lattice). Locked
# from one seeded run of simulate(...) with the constants above.
SEEDED_COOPERATOR_COUNTS = (
    68,
    1,
    4,
    4,
    3,
    4,
    3,
    4,
    4,
    2,
    3,
    4,
    7,
    2,
    5,
    4,
    1,
    2,
    4,
    4,
    1,
    2,
    1,
    0,
    2,
    1,
    3,
    6,
    8,
    4,
    2,
)
SEEDED_RATES = tuple(count / SEEDED_CELLS for count in SEEDED_COOPERATOR_COUNTS)
SEEDED_FINAL_COOPERATION_RATE = SEEDED_RATES[-1]


def test_score_known_3x3_interior_defector():
    # Neighbors of center D: six C and two D → 6*5 + 2*1 = 32.
    grid = [
        [C, C, D],
        [C, D, D],
        [C, C, C],
    ]
    scores = score_cells(grid)
    assert scores[1][1] == 32


def test_score_all_cooperators_each_earn_eight_rewards():
    grid = [[C, C], [C, C]]
    scores = score_cells(grid)
    assert scores == [[24, 24], [24, 24]]  # 8 neighbors × 3


def test_score_all_defectors_each_earn_eight_punishments():
    grid = [[D, D], [D, D]]
    scores = score_cells(grid)
    assert scores == [[8, 8], [8, 8]]  # 8 neighbors × 1


def test_score_cells_matches_public_neighborhood_and_payoff_exhaustively():
    for height, width in ((1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 3)):
        for cells in product((D, C), repeat=height * width):
            grid = [list(cells[row * width : (row + 1) * width]) for row in range(height)]
            expected = [
                [
                    sum(payoff(grid[row][col], grid[nr][nc]) for nr, nc in moore_neighbors(row, col, height, width))
                    for col in range(width)
                ]
                for row in range(height)
            ]
            assert score_cells(grid) == expected


def test_lone_defector_among_cooperators_tempts_and_is_imitated():
    # Classic Nowak–May motif on a torus: a single D in a sea of C.
    grid = [
        [C, C, C],
        [C, D, C],
        [C, C, C],
    ]
    scores = score_cells(grid)
    # Center D vs 8 C: 40. Each C plays 1 D + 7 C (with wrap, some C meet D
    # more than once). After imitation the D's higher score should spread
    # to its neighbors.
    assert scores[1][1] == 40
    nxt = adopt_best(grid, scores)
    assert nxt[1][1] == D
    # Every neighbor of the defector sees that 40 and copies D.
    for r, c in ((0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)):
        assert nxt[r][c] == D


def test_tie_prefers_focal_cell():
    # Equal scores: self is considered first, so a neighbor never replaces
    # the focal strategy on a tie.
    grid = [[C, D], [D, C]]
    scores = [[9, 9], [9, 9]]
    nxt = adopt_best(grid, scores)
    assert nxt == grid


def test_strictly_higher_neighbor_is_copied():
    # Two cells: construct scores so the left C copies the right D.
    grid = [[C, D]]
    scores = [[10, 11]]
    nxt = adopt_best(grid, scores)
    assert nxt == [[D, D]]


def test_step_does_not_mutate_input():
    grid = [[C, D], [D, C]]
    snapshot = [row[:] for row in grid]
    step(grid)
    assert grid == snapshot


def test_mutation_flips_with_seeded_rng():
    grid = [[C, C], [C, C]]
    mutated = apply_mutation(grid, Random(0), mutation_rate=1.0)
    assert mutated == [[D, D], [D, D]]
    assert grid == [[C, C], [C, C]]


def test_zero_mutation_is_identity():
    grid = [[C, D], [D, C]]
    assert apply_mutation(grid, Random(1), mutation_rate=0.0) == grid


def test_cooperation_rate_empty_and_full():
    assert cooperation_rate([[C, C], [C, C]]) == 1.0
    assert cooperation_rate([[D, D], [D, D]]) == 0.0
    assert cooperation_rate([[C, D], [D, C]]) == 0.5


def test_random_grid_is_deterministic():
    a = random_grid(8, 8, seed=99, cooperate_p=0.4)
    b = random_grid(8, 8, seed=99, cooperate_p=0.4)
    assert a == b
    assert a != random_grid(8, 8, seed=100, cooperate_p=0.4)


def test_simulate_is_deterministic_across_calls():
    kwargs = dict(
        height=SEEDED_HEIGHT,
        width=SEEDED_WIDTH,
        generations=SEEDED_GENERATIONS,
        seed=SEEDED_SEED,
        mutation_rate=SEEDED_MUTATION_RATE,
    )
    first = simulate(**kwargs)
    second = simulate(**kwargs)
    assert first.grid == second.grid
    assert first.cooperation_rates == second.cooperation_rates
    assert first.final_cooperation_rate == second.final_cooperation_rate


def test_seeded_run_locks_cooperation_rate():
    result = simulate(
        SEEDED_HEIGHT,
        SEEDED_WIDTH,
        SEEDED_GENERATIONS,
        seed=SEEDED_SEED,
        mutation_rate=SEEDED_MUTATION_RATE,
    )
    # Exact lock: 144-cell grid ⇒ rates are k/144.
    assert result.final_cooperation_rate == SEEDED_FINAL_COOPERATION_RATE
    assert result.cooperation_rates == SEEDED_RATES
    assert abs(result.final_cooperation_rate * 144 - round(result.final_cooperation_rate * 144)) < 1e-12
