"""Per-call backend timing, usage, size, and failure classification."""

from types import SimpleNamespace

import pytest

from spatial_ipd.local_llm import LocalLLMError
from spatial_ipd.telemetry import classify_backend_error, telemetry_summary
from spatial_ipd.think import think_after_step


class SuccessfulClient:
    """Return one complete response with synthetic usage metadata."""

    uses_sdk_questions = False

    def system_one(self, state, questions):
        del state, questions
        return SimpleNamespace(
            nouls={
                "worth_thinking_0": SimpleNamespace(noul=0.9),
                "resist_0": SimpleNamespace(noul=0.8),
            },
            scores={"cluster_fragility_0": SimpleNamespace(score=1.0, confidence=0.5)},
            choices={},
            confidence_kind="jev_noul",
            raw_response_bytes=321,
            backend_usage={"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
        )


class MissingWorthClient:
    """Return a malformed response that fails policy normalization."""

    uses_sdk_questions = False

    def system_one(self, state, questions):
        del state, questions
        return SimpleNamespace(
            nouls={"resist_0": SimpleNamespace(noul=0.8)},
            scores={},
            choices={},
            raw_response_bytes=20,
        )


class TimeoutClient:
    """Raise a transport timeout."""

    uses_sdk_questions = False

    def system_one(self, state, questions):
        del state, questions
        raise TimeoutError("slow backend")


def _clock(*values):
    stream = iter(values)
    return lambda: next(stream)


def test_successful_call_captures_elapsed_usage_and_raw_size():
    _grid, stats = think_after_step(
        [[1]],
        [[0]],
        generation=4,
        seed=1,
        thinker_count=1,
        client=SuccessfulClient(),
        clock=_clock(10.0, 10.25),
    )
    assert len(stats.backend_telemetry) == 1
    telemetry = stats.backend_telemetry[0]
    assert telemetry.generation == 4
    assert telemetry.elapsed_seconds == 0.25
    assert telemetry.status == "success"
    assert telemetry.raw_response_bytes == 321
    assert telemetry.input_tokens == 12
    assert telemetry.output_tokens == 3
    assert telemetry.total_tokens == 15
    assert telemetry_summary(stats.backend_telemetry) == {
        "backend_calls": 1,
        "backend_call_seconds": 0.25,
        "mean_backend_call_seconds": 0.25,
        "backend_call_failures": 0,
        "raw_response_bytes": 321,
        "input_tokens": 12,
        "output_tokens": 3,
        "total_tokens": 15,
    }


def test_schema_failure_is_classified_and_attached():
    with pytest.raises(KeyError) as caught:
        think_after_step(
            [[1]],
            [[0]],
            generation=2,
            seed=1,
            thinker_count=1,
            client=MissingWorthClient(),
            clock=_clock(1.0, 1.1),
        )
    telemetry = caught.value.spatial_ipd_backend_telemetry
    assert telemetry.status == "error"
    assert telemetry.error_category == "schema"
    assert telemetry.raw_response_bytes == 20


def test_timeout_is_classified_and_attached_without_swallowing():
    with pytest.raises(TimeoutError, match="slow backend") as caught:
        think_after_step(
            [[1]],
            [[0]],
            generation=3,
            seed=1,
            thinker_count=1,
            client=TimeoutClient(),
            clock=_clock(5.0, 5.75),
        )
    telemetry = caught.value.spatial_ipd_backend_telemetry
    assert telemetry.elapsed_seconds == 0.75
    assert telemetry.error_category == "timeout"
    assert telemetry.error_type == "TimeoutError"


def test_local_llm_error_distinguishes_schema_from_transport():
    assert classify_backend_error(LocalLLMError("message contained malformed decision JSON")) == "schema"
    assert classify_backend_error(LocalLLMError("Could not reach local LLM endpoint")) == "transport"
