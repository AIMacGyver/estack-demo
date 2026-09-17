"""TypeSafe questions and thresholds.

Review this file when changing labels or thinker policy. The engine never
calls TypeSafe; `label.py` and `think.py` send these after code has numbers.
"""

from __future__ import annotations

from collections.abc import Sequence

# Noul above this is treated as "cooperation still meaningfully present".
# Golden 12x12 seed 20260316 scored 0.17; all-C no-mutation scored 0.99.
COOPERATION_SURVIVED_YES_THRESHOLD = 0.5

REGIME_INSTRUCTIONS = (
    "Classify the cooperation-rate time series in `cooperation_rates` "
    "(index 0 is the initial lattice; later values are after each generation). "
    "Use `initial_cooperation_rate`, `final_cooperation_rate`, "
    "`min_cooperation_rate`, and `max_cooperation_rate` as supporting facts. "
    "Pick the single best regime."
)

REGIME_CRITERIA = {
    "collapse": (
        "Cooperation is initially substantial, then falls to near zero "
        "and does not recover to a large share of the lattice."
    ),
    "flicker": (
        "Cooperation stays near zero through most of the run, with only "
        "small short-lived pops up and down (typical of mutation noise)."
    ),
    "persist": (
        "Cooperation remains a large share of the lattice (well above a few leftover cells) for most generations."
    ),
    "other": "The series does not fit collapse, flicker, or persist.",
}

DEMO_WORTHY_INSTRUCTIONS = (
    "How interesting is this cooperation-rate series as a 30-second "
    "Spatial IPD demo? Judge the shape of `cooperation_rates`, not the "
    "scientific correctness of the model."
)

DEMO_WORTHY_LEVELS = [
    "Flat or featureless: almost no change, or noise around a single value.",
    "One clear event (for example a crash or a recovery) then little else.",
    "Several distinct phases or a visually striking contrast between start and end.",
]

SURVIVED_INSTRUCTIONS = (
    "At the end of the run, is cooperation still meaningfully present "
    "on the lattice given `final_cooperation_rate` and the late values "
    "of `cooperation_rates`? Near-zero leftover cells from mutation "
    "do not count as meaningful survival."
)

SURVIVED_CRITERIA = {
    "true": "A noticeable cooperating population remains at the end.",
    "false": "Cooperation is gone or only a tiny residue remains.",
}

# Dual-process thinkers. Bake-off (jev-1.13.0): mixed 3x3 worth~0.9;
# uniform all-C/all-D worth~0.08. A C cell that just copied an invading D
# chose hold. Live runs: 3-way act Choice often said hold at conf 0.21–0.28
# (split mass). Resist is a Noul (P(yes) gate, not Choice confidence).
# After scores-in-state, "was the copy wrong?" sat at 0.38–0.55. The Noul
# now asks whether to keep a C cluster even though the neighbor won on payoff.
WORTH_THINKING_THRESHOLD = 0.6
RESIST_YES_THRESHOLD = 0.6

THINKER_WORTH_INSTRUCTIONS = (
    "Using `thinkers[{i}].patch_after`, `thinkers[{i}].after_imitate`, "
    "and the imitate scores `thinkers[{i}].focal_score` / "
    "`thinkers[{i}].best_neighbor_score`, "
    "is this a mixed neighborhood where overriding imitation could matter? "
    "Uniform all-C or all-D is not worth overriding."
)

THINKER_WORTH_CRITERIA = {
    "true": "C and D both appear, or `thinkers[{i}].before` differs from `thinkers[{i}].after_imitate`.",
    "false": "The patch is uniform and the cell did not just change.",
}

THINKER_RESIST_INSTRUCTIONS = (
    "This seat was Cooperate (`thinkers[{i}].before` = 1) and imitation just "
    "made it Defect (`thinkers[{i}].after_imitate` = 0). "
    "`thinkers[{i}].patch_after` is the 3x3 after imitation (center is this cell). "
    "`thinkers[{i}].focal_score` lost to `thinkers[{i}].best_neighbor_score` "
    "(winner strategy `thinkers[{i}].best_neighbor_strategy`). "
    "That copy is score-justified by imitate-the-best. Even so, should this "
    "cell stay C to keep a cooperating group in the patch?"
)

THINKER_RESIST_CRITERIA = {
    "true": (
        "A C group in the patch is worth keeping even though a neighbor "
        "won imitate-the-best on payoff."
    ),
    "false": "The score-justified copy should stand; do not hold this cell at C.",
}

THINKER_FRAGILITY_INSTRUCTIONS = "How fragile is cooperation in `thinkers[{i}].patch_after` for the next generation?"

THINKER_FRAGILITY_LEVELS = [
    "Cooperation is absent, or a solid C block with no adjacent D.",
    "C and D both present; the C group could shrink or hold.",
    "A C group is being invaded and is likely to shrink a lot next.",
]


def typesafe_questions():
    """Build SDK question objects. Imports typesafe_sdk only when called."""
    try:
        from typesafe_sdk import Choice, Noul, Score
    except ImportError as exc:
        raise ImportError('TypeSafe is optional. Install with: python3 -m pip install -e ".[typesafe]"') from exc
    return {
        "regime": Choice(
            instructions=REGIME_INSTRUCTIONS,
            criteria=REGIME_CRITERIA,
        ),
        "demo_worthy": Score(
            instructions=DEMO_WORTHY_INSTRUCTIONS,
            criteria=DEMO_WORTHY_LEVELS,
        ),
        "cooperation_survived": Noul(
            instructions=SURVIVED_INSTRUCTIONS,
            criteria=SURVIVED_CRITERIA,
        ),
    }


def typesafe_thinker_questions(count: int, resist_indices: Sequence[int] | None = None) -> dict:
    """Worth and fragility for every seat; resist Noul only for C→D seats."""
    if count < 1:
        return {}
    try:
        from typesafe_sdk import Noul, Score
    except ImportError as exc:
        raise ImportError('TypeSafe is optional. Install with: python3 -m pip install -e ".[typesafe]"') from exc
    ask_resist = set(range(count) if resist_indices is None else (int(i) for i in resist_indices))
    questions = {}
    for i in range(count):
        questions[f"worth_thinking_{i}"] = Noul(
            instructions=THINKER_WORTH_INSTRUCTIONS.format(i=i),
            criteria={
                "true": THINKER_WORTH_CRITERIA["true"].format(i=i),
                "false": THINKER_WORTH_CRITERIA["false"],
            },
        )
        if i in ask_resist:
            questions[f"resist_{i}"] = Noul(
                instructions=THINKER_RESIST_INSTRUCTIONS.format(i=i),
                criteria=THINKER_RESIST_CRITERIA,
            )
        questions[f"cluster_fragility_{i}"] = Score(
            instructions=THINKER_FRAGILITY_INSTRUCTIONS.format(i=i),
            criteria=THINKER_FRAGILITY_LEVELS,
        )
    return questions
