"""Backend call telemetry with no decision-policy side effects."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class BackendCallTelemetry:
    """Operational evidence for one thinker backend call."""

    generation: int
    elapsed_seconds: float
    status: str
    error_category: str | None = None
    error_type: str | None = None
    raw_response_bytes: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


def classify_backend_error(error: BaseException) -> str:
    """Classify a backend exception without changing or swallowing it."""
    if isinstance(error, TimeoutError):
        return "timeout"
    if isinstance(error, (ConnectionError, OSError)):
        return "transport"
    name = type(error).__name__
    message = str(error).lower()
    if name == "LocalLLMError":
        if any(word in message for word in ("json", "response", "decision", "choices", "content")):
            return "schema"
        return "transport"
    if isinstance(error, (KeyError, TypeError, ValueError, AttributeError)):
        return "schema"
    return "backend"


def _usage_value(usage: object, names: Sequence[str]) -> int | None:
    for name in names:
        if isinstance(usage, Mapping) and name in usage:
            value = usage[name]
        elif hasattr(usage, name):
            value = getattr(usage, name)
        else:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        return int(value)
    return None


def telemetry_for_response(
    response: object,
    *,
    generation: int,
    elapsed_seconds: float,
    error: BaseException | None = None,
) -> BackendCallTelemetry:
    """Capture available response size and token usage."""
    usage = getattr(response, "backend_usage", getattr(response, "usage", None))
    raw_bytes = getattr(response, "raw_response_bytes", None)
    if not isinstance(raw_bytes, int):
        raw_output = getattr(response, "raw_output", None)
        raw_bytes = len(raw_output.encode("utf-8")) if isinstance(raw_output, str) else None
    return BackendCallTelemetry(
        generation=generation,
        elapsed_seconds=elapsed_seconds,
        status="error" if error is not None else "success",
        error_category=classify_backend_error(error) if error is not None else None,
        error_type=type(error).__name__ if error is not None else None,
        raw_response_bytes=raw_bytes,
        input_tokens=_usage_value(usage, ("input_tokens", "prompt_tokens")),
        output_tokens=_usage_value(usage, ("output_tokens", "completion_tokens")),
        total_tokens=_usage_value(usage, ("total_tokens",)),
    )


def telemetry_for_error(
    error: BaseException,
    *,
    generation: int,
    elapsed_seconds: float,
) -> BackendCallTelemetry:
    """Capture a failed call that produced no response."""
    return BackendCallTelemetry(
        generation=generation,
        elapsed_seconds=elapsed_seconds,
        status="error",
        error_category=classify_backend_error(error),
        error_type=type(error).__name__,
    )


def attach_error_telemetry(error: BaseException, telemetry: BackendCallTelemetry) -> None:
    """Attach telemetry for callers that isolate and serialize failures."""
    try:
        error.spatial_ipd_backend_telemetry = telemetry
    except (AttributeError, TypeError):
        return


def telemetry_summary(records: Sequence[BackendCallTelemetry]) -> dict[str, int | float]:
    """Aggregate successful call telemetry for experiment output."""
    calls = len(records)
    total_seconds = sum(record.elapsed_seconds for record in records)
    return {
        "backend_calls": calls,
        "backend_call_seconds": total_seconds,
        "mean_backend_call_seconds": total_seconds / calls if calls else 0.0,
        "backend_call_failures": sum(record.status == "error" for record in records),
        "raw_response_bytes": sum(record.raw_response_bytes or 0 for record in records),
        "input_tokens": sum(record.input_tokens or 0 for record in records),
        "output_tokens": sum(record.output_tokens or 0 for record in records),
        "total_tokens": sum(record.total_tokens or 0 for record in records),
    }
