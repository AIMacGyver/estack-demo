"""Dual-process thinkers: seats, patches, apply policy. Fake Jev client."""

from types import SimpleNamespace

import pytest

import spatial_ipd.think as think_mod
from spatial_ipd.engine import adopt_best, score_cells, simulate
from spatial_ipd.judgments import (
    RESIST_YES_THRESHOLD,
    THINKER_RESIST_CRITERIA,
    THINKER_RESIST_INSTRUCTIONS,
    WORTH_THINKING_THRESHOLD,
)
from spatial_ipd.payoffs import COOPERATE as C
from spatial_ipd.payoffs import DEFECT as D
from spatial_ipd.think import (
    SEAT_FRONTIER,
    SEAT_RANDOM,
    StickySeat,
    ThinkerDecision,
    apply_decisions,
    apply_sticky,
    choose_seats,
    decisions_from_response,
    format_compare_summary,
    format_decision_line,
    format_think_summary,
    frontier_pools,
    frontier_seats,
    main,
    patch3,
    register_holds,
    resolve_strategy,
    is_cooperate_to_defect,
    resist_indices_for_seats,
    should_resist,
    should_think,
    simulate_with_thinkers,
    state_for_thinkers,
    think_after_step,
    thinker_seats,
    winning_imitate,
)


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def system_one(self, state, questions):
        self.calls.append((state, questions))
        return self.response


def _response(items: list[tuple[str, float, float]]):
    """items: (unused_act_label, worth, resist_noul) per thinker index."""
    nouls = {}
    scores = {}
    for i, (_act, worth, resist) in enumerate(items):
        nouls[f"worth_thinking_{i}"] = SimpleNamespace(noul=worth)
        nouls[f"resist_{i}"] = SimpleNamespace(noul=resist)
        scores[f"cluster_fragility_{i}"] = SimpleNamespace(score=1.0, confidence=0.5)
    return SimpleNamespace(nouls=nouls, choices={}, scores=scores)


def test_patch3_wraps_toroidally():
    grid = [
        [C, D, C],
        [D, C, D],
        [C, D, C],
    ]
    center = patch3(grid, 0, 0)
    assert center[1][1] == C
    assert center[0][0] == grid[-1][-1]


def test_thinker_seats_are_deterministic_and_unique():
    a = thinker_seats(12, 12, 4, seed=20260316, generation=5)
    b = thinker_seats(12, 12, 4, seed=20260316, generation=5)
    assert a == b
    assert len(a) == 4
    assert len(set(a)) == 4
    assert thinker_seats(12, 12, 4, seed=20260316, generation=10) != a


def test_frontier_pools_changed_then_edge_then_rest():
    before = [[C, C, C], [C, C, C], [C, C, C]]
    after = [[C, C, C], [C, D, C], [C, C, C]]
    changed, edge, rest = frontier_pools(before, after)
    assert changed == [(1, 1)]
    assert (1, 1) not in edge
    assert (1, 1) not in rest
    # 3x3 torus: every other cell is a Moore neighbor of the lone D.
    assert set(edge) == {(r, c) for r in range(3) for c in range(3) if (r, c) != (1, 1)}
    assert rest == []


def test_frontier_seats_prefer_changed_cells():
    before = [[C, C], [C, C]]
    after = [[D, C], [C, C]]
    seats = frontier_seats(before, after, 1, seed=1, generation=1)
    assert seats == ((0, 0),)


def test_frontier_seats_are_deterministic_and_unique():
    before = [[C, D, C], [D, C, D], [C, D, C]]
    after = [[D, D, C], [D, C, D], [C, D, C]]
    a = frontier_seats(before, after, 4, seed=20260316, generation=5)
    b = frontier_seats(before, after, 4, seed=20260316, generation=5)
    assert a == b
    assert len(a) == 4
    assert len(set(a)) == 4
    assert frontier_seats(before, after, 4, seed=20260316, generation=10) != a


def test_choose_seats_random_matches_thinker_seats():
    expected = thinker_seats(12, 12, 4, seed=20260316, generation=5)
    got = choose_seats(12, 12, 4, seed=20260316, generation=5, mode=SEAT_RANDOM)
    assert got == expected
    assert choose_seats(12, 12, 4, seed=20260316, generation=5, mode=SEAT_FRONTIER) == expected


def test_think_after_step_frontier_prefers_changed_seat():
    before = [[C, C, C], [C, C, C], [C, C, C]]
    after = [[C, C, C], [C, D, C], [C, C, C]]
    client = FakeClient(_response([("imitate", 0.2, 0.8)]))
    _, stats = think_after_step(
        before,
        after,
        generation=5,
        seed=20260316,
        thinker_count=1,
        client=client,
        questions={"placeholder": object()},
        seat_mode=SEAT_FRONTIER,
    )
    assert stats.last_seats == ((1, 1),)
    _, random_stats = think_after_step(
        before,
        after,
        generation=5,
        seed=20260316,
        thinker_count=1,
        client=client,
        questions={"placeholder": object()},
        seat_mode=SEAT_RANDOM,
    )
    assert random_stats.last_seats == thinker_seats(3, 3, 1, seed=20260316, generation=5)


def test_resolve_and_should_resist_policy():
    assert resolve_strategy(C, D, "hold") == C
    assert resolve_strategy(C, D, "imitate") == D
    assert resolve_strategy(C, D, "flip") == C
    assert resolve_strategy(D, D, "flip") == C
    assert should_resist(0.9, 0.7, C, D) is True
    assert should_resist(0.9, 0.4, C, D) is False
    assert should_resist(0.08, 0.9, C, D) is False
    assert should_resist(0.9, 0.9, D, C) is False
    assert WORTH_THINKING_THRESHOLD == 0.6
    assert RESIST_YES_THRESHOLD == 0.6


def test_hold_restores_before_when_jev_is_sure():
    before = [[C, C, C], [C, C, C], [C, C, C]]
    after = [[C, C, C], [C, D, C], [C, C, C]]
    seats = ((1, 1),)
    response = _response([("hold", 0.96, 0.9)])
    decisions = decisions_from_response(seats, before, after, response, generation=1)
    assert decisions[0].applied is True
    assert decisions[0].after_think == C
    nxt = apply_decisions(after, decisions)
    assert nxt[1][1] == C
    assert after[1][1] == D


def test_low_resist_noul_does_not_override():
    before = [[C]]
    after = [[D]]
    decisions = decisions_from_response(((0, 0),), before, after, _response([("hold", 0.88, 0.23)]), generation=2)
    assert decisions[0].applied is False
    assert decisions[0].after_think == D


def test_d_to_c_cannot_resist():
    before = [[D]]
    after = [[C]]
    decisions = decisions_from_response(((0, 0),), before, after, _response([("hold", 0.99, 0.99)]), generation=2)
    assert decisions[0].applied is False
    assert decisions[0].after_think == C


def test_resist_indices_only_c_to_d():
    before = [[C, D], [C, C]]
    after = [[D, C], [C, D]]
    seats = ((0, 0), (0, 1), (1, 0), (1, 1))
    assert is_cooperate_to_defect(C, D) is True
    assert is_cooperate_to_defect(D, C) is False
    assert resist_indices_for_seats(seats, before, after) == (0, 3)


def test_missing_resist_noul_is_not_asked():
    before = [[D]]
    after = [[C]]
    response = _response([("imitate", 0.9, 0.9)])
    del response.nouls["resist_0"]
    decisions = decisions_from_response(((0, 0),), before, after, response, generation=2)
    assert decisions[0].applied is False
    assert "resist=n/a" in format_decision_line(decisions[0])


def test_think_every_zero_matches_engine_golden():
    kwargs = dict(height=12, width=12, generations=30, seed=20260316, mutation_rate=0.02)
    plain = simulate(**kwargs)
    with_off, stats = simulate_with_thinkers(**kwargs, think_every=0, think_last=5, sticky=5)
    assert with_off.cooperation_rates == plain.cooperation_rates
    assert with_off.final_cooperation_rate == 2 / 144
    assert stats.calls == 0


def test_register_and_apply_sticky():
    decision = ThinkerDecision(
        generation=10,
        row=0,
        col=0,
        act="hold",
        worth_thinking=0.9,
        act_confidence=0.4,
        applied=True,
        before=C,
        after_imitate=D,
        after_think=C,
    )
    book = register_holds((), (decision,), generation=10, sticky=5)
    assert book == (StickySeat(row=0, col=0, strategy=C, until_generation=15),)
    restored, kept = apply_sticky([[D]], book, generation=12)
    assert restored == [[C]]
    assert kept == book
    expired, empty = apply_sticky([[D]], book, generation=16)
    assert expired == [[D]]
    assert empty == ()
    assert register_holds((), (decision,), generation=10, sticky=0) == ()


def test_sticky_hold_survives_later_forced_defect(monkeypatch):
    def always_defect(grid, rng=None, mutation_rate=0.0):
        return [[D for _ in row] for row in grid]

    monkeypatch.setattr(think_mod, "step", always_defect)
    monkeypatch.setattr(
        think_mod,
        "should_think",
        lambda generation, generations, think_every, think_last: generation == 1,
    )
    client = FakeClient(_response([("hold", 0.96, 0.9)]))
    questions = {"placeholder": object()}
    kwargs = dict(
        height=1,
        width=1,
        generations=3,
        seed=1,
        cooperate_p=1.0,
        think_every=5,
        think_last=0,
        thinker_count=1,
        client=client,
        questions=questions,
    )
    sticky, _ = simulate_with_thinkers(**kwargs, sticky=2)
    oneshot, _ = simulate_with_thinkers(**kwargs, sticky=0)
    assert sticky.grid[0][0] == C
    assert oneshot.grid[0][0] == D


def test_should_think_last_n_without_double_counting():
    assert should_think(5, 30, think_every=5, think_last=0) is True
    assert should_think(26, 30, think_every=5, think_last=0) is False
    assert should_think(26, 30, think_every=5, think_last=5) is True
    assert should_think(30, 30, think_every=5, think_last=5) is True
    assert should_think(25, 30, think_every=5, think_last=5) is True
    gens = [g for g in range(1, 31) if should_think(g, 30, 5, 5)]
    assert gens == [5, 10, 15, 20, 25, 26, 27, 28, 29, 30]


def test_think_last_zero_keeps_periodic_schedule():
    client = FakeClient(_response([("imitate", 0.2, 0.8)]))
    questions = {"placeholder": object()}
    _, stats = simulate_with_thinkers(
        3,
        3,
        8,
        seed=1,
        think_every=4,
        think_last=0,
        thinker_count=1,
        client=client,
        questions=questions,
    )
    assert stats.calls == 2
    assert [item.generation for item in stats.all_decisions] == [4, 8]


def test_think_last_adds_late_generations():
    client = FakeClient(_response([("imitate", 0.2, 0.8)]))
    questions = {"placeholder": object()}
    _, stats = simulate_with_thinkers(
        3,
        3,
        8,
        seed=1,
        think_every=4,
        think_last=3,
        thinker_count=1,
        client=client,
        questions=questions,
    )
    assert [item.generation for item in stats.all_decisions] == [4, 6, 7, 8]
    assert stats.calls == 4


def test_think_after_step_uses_injected_client():
    before = [[C, C], [C, C]]
    after = [[C, D], [D, C]]
    client = FakeClient(_response([("hold", 0.96, 0.4), ("imitate", 0.2, 0.8)]))
    questions = {"placeholder": object()}
    nxt, stats = think_after_step(
        before,
        after,
        generation=5,
        seed=1,
        thinker_count=2,
        client=client,
        questions=questions,
    )
    assert stats.calls == 1
    assert client.calls[0][1] is questions
    state = client.calls[0][0]
    assert state["generation"] == 5
    assert len(state["thinkers"]) == 2
    seat = state["thinkers"][0]
    assert seat["patch_after"][1][1] == after[seat["row"]][seat["col"]]


def test_cli_think_every_zero(capsys):
    code = main(
        [
            "--height",
            "12",
            "--width",
            "12",
            "--generations",
            "30",
            "--seed",
            "20260316",
            "--mutation-rate",
            "0.02",
            "--think-every",
            "0",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "think_calls=0" in out
    assert "final_cooperation_rate=0.013888888888888888" in out
    result, stats = simulate_with_thinkers(12, 12, 30, seed=20260316, mutation_rate=0.02, think_every=0)
    assert format_think_summary(result, stats) in out


def test_compare_and_verbose_cli(capsys):
    client = FakeClient(_response([("hold", 0.96, 0.9)]))
    questions = {"placeholder": object()}
    code = main(
        [
            "--height",
            "2",
            "--width",
            "2",
            "--generations",
            "2",
            "--seed",
            "1",
            "--think-every",
            "2",
            "--thinkers",
            "1",
            "--seats",
            "frontier",
            "--think-last",
            "0",
            "--compare",
            "--verbose",
        ],
        client=client,
        questions=questions,
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "plain_final=" in out
    assert "think_final=" in out
    assert "delta=" in out
    assert "gen=2" in out
    assert "act=" in out
    assert client.calls


def test_cli_seats_random_uses_uniform_seats():
    client = FakeClient(_response([("imitate", 0.2, 0.8)]))
    questions = {"placeholder": object()}
    code = main(
        [
            "--height",
            "3",
            "--width",
            "3",
            "--generations",
            "1",
            "--seed",
            "20260316",
            "--think-every",
            "1",
            "--thinkers",
            "1",
            "--seats",
            "random",
            "--think-last",
            "0",
        ],
        client=client,
        questions=questions,
    )
    assert code == 0
    expected = thinker_seats(3, 3, 1, seed=20260316, generation=1)
    assert client.calls
    assert client.calls[0][0]["thinkers"][0]["row"] == expected[0][0]
    assert client.calls[0][0]["thinkers"][0]["col"] == expected[0][1]


def test_format_compare_zero_thinkers_matches_golden():
    kwargs = dict(height=12, width=12, generations=30, seed=20260316, mutation_rate=0.02)
    plain = simulate(**kwargs)
    think, stats = simulate_with_thinkers(**kwargs, think_every=0)
    line = format_compare_summary(plain, think, stats)
    assert "delta=0.0" in line or "delta=0" in line
    assert think.final_cooperation_rate == 2 / 144


def test_format_decision_line_includes_act():
    line = format_decision_line(
        ThinkerDecision(
            generation=5,
            row=1,
            col=2,
            act="hold",
            worth_thinking=0.96,
            act_confidence=0.4,
            applied=True,
            before=1,
            after_imitate=0,
            after_think=1,
            focal_score=21,
            best_neighbor_score=40,
            best_neighbor_strategy=D,
        )
    )
    assert line.startswith("gen=5 row=1 col=2 act=hold")
    assert "applied=yes" in line
    assert "resist=0.4" in line
    assert "resist=n/a" not in line
    assert "conf=" not in line
    assert "focal_score=21" in line
    assert "best_neighbor_score=40" in line
    assert "best_neighbor_strategy=0" in line


def test_decisions_copy_imitate_scores_onto_the_line():
    before = [
        [C, C, C],
        [C, D, C],
        [C, C, C],
    ]
    after = adopt_best(before, score_cells(before))
    decisions = decisions_from_response(
        ((0, 0),),
        before,
        after,
        _response([("imitate", 0.9, 0.3)]),
        generation=4,
    )
    item = decisions[0]
    scores = score_cells(before)
    best_score, best_strategy = winning_imitate(before, scores, 0, 0)
    assert item.focal_score == scores[0][0]
    assert item.best_neighbor_score == best_score
    assert item.best_neighbor_strategy == best_strategy
    line = format_decision_line(item)
    assert f"focal_score={item.focal_score}" in line
    assert f"best_neighbor_score={item.best_neighbor_score}" in line
    assert f"resist={item.act_confidence}" in line


def test_state_for_thinkers_lists_before_and_after():
    before = [[C, D], [D, C]]
    after = [[D, D], [D, C]]
    state = state_for_thinkers(before, after, ((0, 0),), generation=3)
    assert state["thinkers"][0]["before"] == C
    assert state["thinkers"][0]["after_imitate"] == D
    assert state["generation"] == 3


def test_state_for_thinkers_includes_imitate_scores():
    before = [
        [C, C, C],
        [C, D, C],
        [C, C, C],
    ]
    after = adopt_best(before, score_cells(before))
    state = state_for_thinkers(before, after, ((0, 0),), generation=4)
    seat = state["thinkers"][0]
    scores = score_cells(before)
    best_score, best_strategy = winning_imitate(before, scores, 0, 0)
    assert seat["focal_score"] == scores[0][0]
    assert seat["best_neighbor_score"] == best_score
    assert seat["best_neighbor_strategy"] == best_strategy
    assert seat["focal_score"] < seat["best_neighbor_score"]
    assert seat["best_neighbor_strategy"] == D
    assert after[0][0] == D


def test_resist_question_asks_cluster_hold_despite_score_gap():
    text = THINKER_RESIST_INSTRUCTIONS.format(i=0)
    assert "`thinkers[0].focal_score`" in text
    assert "`thinkers[0].best_neighbor_score`" in text
    assert "score-justified" in text
    assert "cooperating group" in text
    assert "keep" in THINKER_RESIST_CRITERIA["true"]
    assert "payoff" in THINKER_RESIST_CRITERIA["true"]
    assert "score-justified" in THINKER_RESIST_CRITERIA["false"]


def test_winning_imitate_matches_adopt_best():
    before = [
        [C, C, C],
        [C, D, C],
        [C, C, C],
    ]
    scores = score_cells(before)
    nxt = adopt_best(before, scores)
    for row in range(3):
        for col in range(3):
            _best_score, best_strategy = winning_imitate(before, scores, row, col)
            assert best_strategy == nxt[row][col]
    tied = [[C, D], [D, C]]
    even = [[9, 9], [9, 9]]
    assert winning_imitate(tied, even, 0, 0) == (9, C)


def test_thinker_questions_omit_resist_unless_c_to_d():
    pytest.importorskip("typesafe_sdk")
    from spatial_ipd.judgments import typesafe_thinker_questions

    qs = typesafe_thinker_questions(2, resist_indices=(0,))
    assert "resist_0" in qs
    assert "resist_1" not in qs
    assert "worth_thinking_1" in qs
    assert "cluster_fragility_1" in qs
