"""Moore neighborhood with toroidal wrap."""

from spatial_ipd import MOORE_OFFSETS, moore_neighbors


def test_interior_cell_has_eight_unique_neighbors():
    neighbors = moore_neighbors(2, 2, height=5, width=5)
    assert len(neighbors) == 8
    assert len(set(neighbors)) == 8
    assert (2, 2) not in neighbors
    expected = {
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 1),
        (2, 3),
        (3, 1),
        (3, 2),
        (3, 3),
    }
    assert set(neighbors) == expected


def test_corner_wraps_toroidally():
    # (0, 0) on a 4×5 torus.
    neighbors = set(moore_neighbors(0, 0, height=4, width=5))
    assert neighbors == {
        (3, 4),
        (3, 0),
        (3, 1),
        (0, 4),
        (0, 1),
        (1, 4),
        (1, 0),
        (1, 1),
    }


def test_edge_wraps_only_off_grid_axis():
    neighbors = set(moore_neighbors(0, 2, height=4, width=5))
    assert neighbors == {
        (3, 1),
        (3, 2),
        (3, 3),
        (0, 1),
        (0, 3),
        (1, 1),
        (1, 2),
        (1, 3),
    }


def test_every_cell_has_exactly_eight_neighbors():
    height, width = 3, 3
    for r in range(height):
        for c in range(width):
            n = moore_neighbors(r, c, height, width)
            assert len(n) == 8


def test_moore_offset_count_and_no_self():
    assert len(MOORE_OFFSETS) == 8
    assert (0, 0) not in MOORE_OFFSETS
