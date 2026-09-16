"""Display-free tests for viewer helpers. No pygame window is opened."""

import sys

import pytest

from spatial_ipd import COOPERATE, DEFECT, random_grid, step
from spatial_ipd.viewer import (
    COLOR_COOPERATE,
    COLOR_DEFECT,
    ViewerConfig,
    cell_color,
    color_frame,
    new_run,
    parse_args,
    validate_config,
    window_caption,
)


def test_cell_color_maps_c_blue_and_d_red():
    assert cell_color(COOPERATE) == COLOR_COOPERATE
    assert cell_color(DEFECT) == COLOR_DEFECT
    assert cell_color(1) == COLOR_COOPERATE
    assert cell_color(0) == COLOR_DEFECT


def test_color_frame_matches_grid_shape_and_strategies():
    grid = [
        [COOPERATE, DEFECT],
        [DEFECT, COOPERATE],
    ]
    frame = color_frame(grid)
    assert frame == [
        [COLOR_COOPERATE, COLOR_DEFECT],
        [COLOR_DEFECT, COLOR_COOPERATE],
    ]


def test_parse_args_defaults():
    config = parse_args([])
    assert config == ViewerConfig()


def test_parse_args_overrides():
    config = parse_args(
        [
            "--height",
            "10",
            "--width",
            "8",
            "--seed",
            "7",
            "--mutation-rate",
            "0.25",
            "--cell-size",
            "4",
            "--fps",
            "3",
        ]
    )
    assert config == ViewerConfig(
        height=10,
        width=8,
        seed=7,
        mutation_rate=0.25,
        cell_size=4,
        fps=3,
    )


def test_validate_config_rejects_bad_values():
    with pytest.raises(ValueError):
        validate_config(ViewerConfig(height=0))
    with pytest.raises(ValueError):
        validate_config(ViewerConfig(width=0))
    with pytest.raises(ValueError):
        validate_config(ViewerConfig(cell_size=0))
    with pytest.raises(ValueError):
        validate_config(ViewerConfig(fps=0))
    with pytest.raises(ValueError):
        validate_config(ViewerConfig(mutation_rate=-0.1))
    with pytest.raises(ValueError):
        validate_config(ViewerConfig(mutation_rate=1.1))


def test_new_run_uses_engine_random_grid_and_does_not_reimplement_pd():
    grid, rng = new_run(6, 5, seed=99)
    assert grid == random_grid(6, 5, seed=99)
    # First engine step with the same RNG must match a direct step() call.
    expected_rng = type(rng)(99)
    expected = step(grid, rng=expected_rng, mutation_rate=0.2)
    got = step(grid, rng=rng, mutation_rate=0.2)
    assert got == expected


def test_window_caption_includes_generation_and_rate():
    grid = [[COOPERATE, DEFECT], [DEFECT, DEFECT]]
    caption = window_caption(4, grid)
    assert "gen=4" in caption
    assert "C=0.250" in caption


def test_importing_viewer_helpers_does_not_load_pygame():
    sys.modules.pop("pygame", None)
    # Re-import helpers after ensuring pygame is absent.
    from spatial_ipd.viewer import cell_color as color_again

    assert color_again(COOPERATE) == COLOR_COOPERATE
    assert "pygame" not in sys.modules


def test_import_pygame_raises_helpful_error_when_missing(monkeypatch):
    import builtins

    from spatial_ipd.viewer import import_pygame

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pygame" or name.startswith("pygame."):
            raise ImportError("No module named pygame")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r'pip install -e "\.\[viewer\]"'):
        import_pygame()
