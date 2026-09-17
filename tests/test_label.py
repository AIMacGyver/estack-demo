"""Optional TypeSafe series labels: state shape, thresholds, fake client."""

import os
import sys
from types import SimpleNamespace

from spatial_ipd.engine import simulate
from spatial_ipd.judgments import (
    COOPERATION_SURVIVED_YES_THRESHOLD,
    DEMO_WORTHY_LEVELS,
    REGIME_CRITERIA,
)
from spatial_ipd.label import (
    format_label,
    label_from_response,
    label_series,
    load_dotenv,
    main,
    state_from_result,
)

SEEDED = dict(height=12, width=12, generations=30, seed=20260316, mutation_rate=0.02)
PLACEHOLDER_QUESTIONS = {
    "regime": object(),
    "demo_worthy": object(),
    "cooperation_survived": object(),
}


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def system_one(self, state, questions):
        self.calls.append((state, questions))
        return self.response


def _response(*, regime="collapse", regime_conf=0.86, demo=1.64, demo_conf=0.46, survived=0.17):
    return SimpleNamespace(
        choices={
            "regime": SimpleNamespace(
                choice=regime,
                confidence=regime_conf,
                probabilities={regime: 0.9},
            )
        },
        scores={"demo_worthy": SimpleNamespace(score=demo, confidence=demo_conf)},
        nouls={"cooperation_survived": SimpleNamespace(noul=survived)},
    )


def test_load_dotenv_sets_missing_keys_only(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("TYPESAFE_API_KEY=from-file\nALREADY=file-value\n", encoding="utf-8")
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setenv("ALREADY", "from-process")
    assert load_dotenv(env_file) == env_file
    assert os.environ["TYPESAFE_API_KEY"] == "from-file"
    assert os.environ["ALREADY"] == "from-process"


def test_importing_engine_does_not_load_typesafe_sdk():
    sys.modules.pop("typesafe_sdk", None)
    for name in ("spatial_ipd", "spatial_ipd.engine", "spatial_ipd.export"):
        sys.modules.pop(name, None)
    import spatial_ipd.engine  # noqa: F401
    import spatial_ipd.export  # noqa: F401

    assert "typesafe_sdk" not in sys.modules


def test_state_from_result_matches_engine_golden():
    result = simulate(**SEEDED)
    state = state_from_result(result, height=12, width=12, mutation_rate=0.02)
    assert state["seed"] == 20260316
    assert state["generations"] == 30
    assert state["final_cooperation_rate"] == 2 / 144
    assert state["cooperation_rates"] == list(result.cooperation_rates)
    assert state["initial_cooperation_rate"] == result.cooperation_rates[0]
    assert state["min_cooperation_rate"] == min(result.cooperation_rates)
    assert state["max_cooperation_rate"] == max(result.cooperation_rates)


def test_label_from_response_applies_survived_threshold():
    low = label_from_response(_response(survived=0.17))
    assert low.cooperation_survived_yes is False
    assert low.regime == "collapse"
    high = label_from_response(_response(regime="persist", survived=0.99))
    assert high.cooperation_survived_yes is True
    assert 0.17 < COOPERATION_SURVIVED_YES_THRESHOLD < 0.99


def test_label_series_uses_injected_client():
    result = simulate(**SEEDED)
    state = state_from_result(result, height=12, width=12, mutation_rate=0.02)
    client = FakeClient(_response())
    label = label_series(state, client=client, questions=PLACEHOLDER_QUESTIONS)
    assert len(client.calls) == 1
    sent_state, questions = client.calls[0]
    assert sent_state == state
    assert questions is PLACEHOLDER_QUESTIONS
    assert set(REGIME_CRITERIA) == {"collapse", "flicker", "persist", "other"}
    assert len(DEMO_WORTHY_LEVELS) == 3
    assert format_label(label) == (
        "regime=collapse regime_confidence=0.86 "
        "cooperation_survived=0.17 survived=no demo_worthy=1.64"
    )


def test_cli_prints_one_line_with_fake_client(capsys):
    client = FakeClient(_response())
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
        ],
        client=client,
        questions=PLACEHOLDER_QUESTIONS,
    )
    assert code == 0
    captured = capsys.readouterr()
    assert captured.out == (
        "regime=collapse regime_confidence=0.86 "
        "cooperation_survived=0.17 survived=no demo_worthy=1.64\n"
    )
    assert captured.err == ""
    assert client.calls
