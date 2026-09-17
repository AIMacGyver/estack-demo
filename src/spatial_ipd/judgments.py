"""TypeSafe questions and thresholds for a cooperation-rate series.

Review this file when changing labels. The engine never calls TypeSafe;
`label.py` sends these questions after `simulate` has already produced numbers.
"""

from __future__ import annotations

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
        "Cooperation remains a large share of the lattice (well above a few "
        "leftover cells) for most generations."
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


def typesafe_questions():
    """Build SDK question objects. Imports typesafe_sdk only when called."""
    try:
        from typesafe_sdk import Choice, Noul, Score
    except ImportError as exc:
        raise ImportError(
            'TypeSafe is optional. Install with: python3 -m pip install -e ".[typesafe]"'
        ) from exc
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
