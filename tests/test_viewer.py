"""Display-free tests for viewer helpers. No pygame window is opened."""

import sys

import pytest

from spatial_ipd import COOPERATE, DEFECT, random_grid, step
from spatial_ipd.think import RandomThinkerClient, ThinkerDecision
from spatial_ipd.viewer import (
    COLOR_COOPERATE,
    COLOR_DEFECT,
    COLOR_HOLD_OUTLINE,
    COLOR_THINKER_OUTLINE,
    ViewerConfig,
    advance_lab_state,
    cell_color,
    cell_facts,
    color_frame,
    decision_overlays,
    history_points,
    new_lab_state,
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
            "--think-every",
            "5",
            "--thinkers",
            "2",
            "--seats",
            "random",
        ]
    )
    assert config == ViewerConfig(
        height=10,
        width=8,
        seed=7,
        mutation_rate=0.25,
        cell_size=4,
        fps=3,
        think_every=5,
        thinkers=2,
        seat_mode="random",
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
    with pytest.raises(ValueError):
        validate_config(ViewerConfig(think_every=-1))
    with pytest.raises(ValueError):
        validate_config(ViewerConfig(seat_mode="diagonal"))
    with pytest.raises(ValueError, match="local-model"):
        validate_config(ViewerConfig(backend="local"))
    with pytest.raises(ValueError, match="history-height"):
        validate_config(ViewerConfig(history_height=0))


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
    assert "thinkers=" not in caption
    assert "thinkers=2" in window_caption(4, grid, thinker_count=2)
    selected = window_caption(4, grid, backend="random", selected=(0, 0))
    assert "backend=random" in selected
    assert "cell=(0,0)" in selected
    assert "payoff=" in selected


def test_lab_state_advances_with_backend_decisions_and_histories():
    config = ViewerConfig(
        height=4,
        width=4,
        seed=1,
        mutation_rate=0.02,
        think_every=1,
        thinkers=2,
        backend="random",
    )
    state = new_lab_state(config)
    assert state.generation == 0
    assert len(state.cooperation_history) == len(state.cluster_history) == 1
    advance_lab_state(state, config, client=RandomThinkerClient(1))
    assert state.generation == 1
    assert len(state.decisions) == 2
    assert len(state.cooperation_history) == len(state.cluster_history) == 2


def test_cell_facts_use_engine_scores_and_frontier():
    grid = [[COOPERATE, DEFECT], [DEFECT, DEFECT]]
    facts = cell_facts(grid, 0, 0)
    assert facts["strategy"] == COOPERATE
    assert facts["frontier"] is True
    assert facts["different_neighbors"] > 0
    assert facts["payoff"] >= 0
    with pytest.raises(ValueError, match="outside"):
        cell_facts(grid, 4, 4)


def test_decision_overlays_distinguish_applied_hold():
    base = dict(
        generation=1,
        worth_thinking=0.9,
        act_confidence=0.9,
        before=COOPERATE,
        after_imitate=DEFECT,
        focal_score=10,
        best_neighbor_score=20,
        best_neighbor_strategy=DEFECT,
    )
    hold = ThinkerDecision(
        row=0,
        col=0,
        act="hold",
        applied=True,
        after_think=COOPERATE,
        **base,
    )
    imitate = ThinkerDecision(
        row=1,
        col=1,
        act="imitate",
        applied=False,
        after_think=DEFECT,
        **base,
    )
    assert decision_overlays((hold, imitate)) == {
        (0, 0): COLOR_HOLD_OUTLINE,
        (1, 1): COLOR_THINKER_OUTLINE,
    }


def test_history_points_scale_and_clip_without_pygame():
    assert history_points([0.0, 0.5, 1.0], width=11, height=11, maximum=1.0) == [
        (0, 10),
        (5, 5),
        (10, 0),
    ]
    assert history_points([20.0], width=5, height=5, maximum=10.0) == [(0, 0)]


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
