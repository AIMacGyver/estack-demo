"""Dual-process thinkers: seats, patches, apply policy. Fake Jev client."""

from types import SimpleNamespace

from spatial_ipd.engine import simulate
from spatial_ipd.judgments import (
    ACT_OVERRIDE_CONFIDENCE,
    WORTH_THINKING_THRESHOLD,
)
from spatial_ipd.payoffs import COOPERATE as C
from spatial_ipd.payoffs import DEFECT as D
from spatial_ipd.think import (
    apply_decisions,
    decisions_from_response,
    format_think_summary,
    main,
    patch3,
    resolve_strategy,
    should_apply,
    simulate_with_thinkers,
    state_for_thinkers,
    think_after_step,
    thinker_seats,
)


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def system_one(self, state, questions):
        self.calls.append((state, questions))
        return self.response


def _response(items: list[tuple[str, float, float]]):
    """items: (act, worth, conf) per thinker index."""
    nouls = {}
    choices = {}
    scores = {}
    for i, (act, worth, conf) in enumerate(items):
        nouls[f"worth_thinking_{i}"] = SimpleNamespace(noul=worth)
        choices[f"act_{i}"] = SimpleNamespace(choice=act, confidence=conf)
        scores[f"cluster_fragility_{i}"] = SimpleNamespace(score=1.0, confidence=0.5)
    return SimpleNamespace(nouls=nouls, choices=choices, scores=scores)


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


def test_resolve_and_should_apply_policy():
    assert resolve_strategy(C, D, "hold") == C
    assert resolve_strategy(C, D, "imitate") == D
    assert resolve_strategy(C, D, "flip") == C
    assert resolve_strategy(D, D, "flip") == C
    assert should_apply(0.9, 0.4) is True
    assert should_apply(0.08, 0.8) is False
    assert should_apply(0.9, 0.1) is False
    assert WORTH_THINKING_THRESHOLD == 0.6
    assert ACT_OVERRIDE_CONFIDENCE == 0.3


def test_hold_restores_before_when_jev_is_sure():
    before = [[C, C, C], [C, C, C], [C, C, C]]
    after = [[C, C, C], [C, D, C], [C, C, C]]
    seats = ((1, 1),)
    response = _response([("hold", 0.96, 0.4)])
    decisions = decisions_from_response(seats, before, after, response)
    assert decisions[0].applied is True
    assert decisions[0].after_think == C
    nxt = apply_decisions(after, decisions)
    assert nxt[1][1] == C
    assert after[1][1] == D


def test_low_confidence_does_not_override():
    before = [[C]]
    after = [[D]]
    decisions = decisions_from_response(((0, 0),), before, after, _response([("hold", 0.88, 0.23)]))
    assert decisions[0].applied is False
    assert decisions[0].after_think == D


def test_think_every_zero_matches_engine_golden():
    kwargs = dict(height=12, width=12, generations=30, seed=20260316, mutation_rate=0.02)
    plain = simulate(**kwargs)
    with_off, stats = simulate_with_thinkers(**kwargs, think_every=0)
    assert with_off.cooperation_rates == plain.cooperation_rates
    assert with_off.final_cooperation_rate == 2 / 144
    assert stats.calls == 0


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


def test_state_for_thinkers_lists_before_and_after():
    before = [[C, D], [D, C]]
    after = [[D, D], [D, C]]
    state = state_for_thinkers(before, after, ((0, 0),), generation=3)
    assert state["thinkers"][0]["before"] == C
    assert state["thinkers"][0]["after_imitate"] == D
    assert state["generation"] == 3
