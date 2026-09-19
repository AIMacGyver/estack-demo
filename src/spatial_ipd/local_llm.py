"""OpenAI-compatible local LLM adapter for Spatial IPD thinkers."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from types import SimpleNamespace
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_LOCAL_LLM_ENDPOINT = "http://localhost:11434/v1/chat/completions"
DEFAULT_LOCAL_LLM_TIMEOUT = 30.0
LOCAL_LLM_REASONING_EFFORTS = ("none", "low", "medium", "high")

JsonObject = Mapping[str, Any]
Transport = Callable[[str, JsonObject, Mapping[str, str], float], object]


class LocalLLMError(RuntimeError):
    """A clear configuration, transport, or response error from a local LLM."""


def _post_json(
    endpoint: str,
    payload: JsonObject,
    headers: Mapping[str, str],
    timeout: float,
) -> object:
    """POST JSON with urllib and decode one JSON response."""
    request = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=dict(headers),
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - endpoint is explicit CLI configuration
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        suffix = f": {detail[:300]}" if detail else ""
        raise LocalLLMError(f"Local LLM request failed with HTTP {exc.code}{suffix}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        raise LocalLLMError(f"Could not reach local LLM endpoint {endpoint}: {reason}") from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LocalLLMError("Local LLM endpoint returned invalid JSON") from exc


def _bounded_number(value: object, *, name: str, low: float, high: float) -> float:
    """Convert a model value to a finite float inside a closed interval."""
    if isinstance(value, bool):
        raise LocalLLMError(f"Local LLM decision field {name!r} must be a number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise LocalLLMError(f"Local LLM decision field {name!r} must be a number") from exc
    if not math.isfinite(number):
        raise LocalLLMError(f"Local LLM decision field {name!r} must be finite")
    if not low <= number <= high:
        raise LocalLLMError(f"Local LLM decision field {name!r} must be between {low:g} and {high:g}")
    return number


def _boolean_decision(value: object, *, name: str) -> bool:
    """Require a JSON boolean rather than treating a probability as an action."""
    if not isinstance(value, bool):
        raise LocalLLMError(f"Local LLM decision field {name!r} must be true or false")
    return value


def _content_json(text: str) -> JsonObject:
    """Extract one JSON object from plain content or a fenced/reasoning response."""
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise LocalLLMError("Local LLM message did not contain a JSON object")
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise LocalLLMError("Local LLM message contained malformed decision JSON") from exc
    if not isinstance(value, dict):
        raise LocalLLMError("Local LLM decision JSON must be an object")
    return value


def _message_content(response: object) -> str:
    """Read ``choices[0].message.content`` from an OpenAI-compatible response."""
    if not isinstance(response, Mapping):
        raise LocalLLMError("Local LLM response must be a JSON object")
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LocalLLMError("Local LLM response is missing choices[0]")
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise LocalLLMError("Local LLM response choices[0] must be an object")
    message = choice.get("message")
    if not isinstance(message, Mapping):
        raise LocalLLMError("Local LLM response is missing choices[0].message")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise LocalLLMError("Local LLM response has empty choices[0].message.content")
    return content


def _normalized_response(
    decision_json: JsonObject,
    *,
    count: int,
    question_ids: set[str],
    raw_output: str,
) -> SimpleNamespace:
    """Map booleans to code-owned gates while retaining reported confidence."""
    decisions = decision_json.get("decisions")
    if not isinstance(decisions, list):
        raise LocalLLMError("Local LLM decision JSON must contain a decisions list")
    if len(decisions) != count:
        raise LocalLLMError(f"Local LLM returned {len(decisions)} decisions; expected {count}")

    by_index: dict[int, Mapping[str, object]] = {}
    for item in decisions:
        if not isinstance(item, Mapping):
            raise LocalLLMError("Each local LLM decision must be an object")
        index = item.get("index")
        if isinstance(index, bool) or not isinstance(index, int):
            raise LocalLLMError("Each local LLM decision index must be an integer")
        if index in by_index:
            raise LocalLLMError(f"Local LLM returned duplicate decision index {index}")
        by_index[index] = item
    expected = set(range(count))
    if set(by_index) != expected:
        raise LocalLLMError(f"Local LLM decision indexes must be {sorted(expected)}")

    nouls: dict[str, SimpleNamespace] = {}
    scores: dict[str, SimpleNamespace] = {}
    for index in range(count):
        item = by_index[index]
        worth_key = f"worth_thinking_{index}"
        if worth_key in question_ids:
            worth = _boolean_decision(item.get("worth_thinking"), name=worth_key)
            worth_confidence = _bounded_number(
                item.get("worth_confidence"),
                name=f"worth_confidence_{index}",
                low=0.0,
                high=1.0,
            )
            nouls[worth_key] = SimpleNamespace(noul=float(worth), confidence=worth_confidence)
        resist_key = f"resist_{index}"
        if resist_key in question_ids:
            resist = _boolean_decision(item.get("resist"), name=resist_key)
            resist_confidence = _bounded_number(
                item.get("resist_confidence"),
                name=f"resist_confidence_{index}",
                low=0.0,
                high=1.0,
            )
            nouls[resist_key] = SimpleNamespace(noul=float(resist), confidence=resist_confidence)
        fragility_key = f"cluster_fragility_{index}"
        if fragility_key in question_ids:
            fragility = _bounded_number(
                item.get("cluster_fragility"),
                name=fragility_key,
                low=0.0,
                high=2.0,
            )
            scores[fragility_key] = SimpleNamespace(score=fragility, confidence=1.0)
    return SimpleNamespace(
        nouls=nouls,
        scores=scores,
        choices={},
        confidence_kind="llm_self_report",
        raw_output=raw_output,
    )


class LocalLLMThinkerClient:
    """Use a model behind an OpenAI-compatible chat-completions endpoint."""

    uses_sdk_questions = False

    def __init__(
        self,
        model: str,
        *,
        endpoint: str = DEFAULT_LOCAL_LLM_ENDPOINT,
        api_key: str | None = None,
        timeout: float = DEFAULT_LOCAL_LLM_TIMEOUT,
        reasoning_effort: str | None = None,
        transport: Transport | None = None,
    ):
        """Bind local model configuration and an injectable JSON transport."""
        if not model.strip():
            raise ValueError("local model must not be empty")
        if not endpoint.strip():
            raise ValueError("local endpoint must not be empty")
        if timeout <= 0:
            raise ValueError("local timeout must be positive")
        if reasoning_effort is not None and reasoning_effort not in LOCAL_LLM_REASONING_EFFORTS:
            raise ValueError(f"local reasoning effort must be one of {LOCAL_LLM_REASONING_EFFORTS}")
        self.model = model
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout = timeout
        self.reasoning_effort = reasoning_effort
        self._transport = transport or _post_json

    def system_one(self, state: object, questions: Mapping[str, object]) -> SimpleNamespace:
        """Request and normalize one bounded decision for each thinker seat."""
        if not isinstance(state, Mapping):
            raise LocalLLMError("Local LLM thinker state must be an object")
        thinkers = state.get("thinkers")
        if not isinstance(thinkers, list):
            raise LocalLLMError("Local LLM thinker state must contain a thinkers list")
        question_ids = set(questions)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You decide whether Spatial IPD thinker cells should resist imitation. "
                        'Return only JSON shaped as {"decisions":[{"index":0,'
                        '"worth_thinking":true,"worth_confidence":0.7,'
                        '"resist":false,"resist_confidence":0.8,"cluster_fragility":1}]}. '
                        "Return exactly one item per thinker, with indexes starting at 0. "
                        "worth_thinking and resist must be JSON booleans, never probabilities. "
                        "Their confidence fields are your self-reported confidence from 0 to 1; confidence "
                        "does not choose the action. Use null for resist and resist_confidence when a resist "
                        "question id is absent. "
                        "cluster_fragility is from 0 (stable/absent cooperation) to 2 (very fragile). "
                        "A mixed patch or changed strategy is worth thinking about. Resist means a C-to-D "
                        "cell should stay C to preserve a cooperating group despite the higher-scoring neighbor."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "state": state,
                            "requested_question_ids": sorted(question_ids),
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                },
            ],
            "stream": False,
            "temperature": 0,
        }
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            response = self._transport(self.endpoint, payload, headers, self.timeout)
        except LocalLLMError:
            raise
        except Exception as exc:
            raise LocalLLMError(f"Local LLM request to {self.endpoint} failed: {exc}") from exc
        content = _message_content(response)
        decision_json = _content_json(content)
        normalized = _normalized_response(
            decision_json,
            count=len(thinkers),
            question_ids=question_ids,
            raw_output=content,
        )
        if isinstance(response, Mapping):
            normalized.backend_usage = response.get("usage")
            normalized.raw_response_bytes = len(
                json.dumps(response, sort_keys=True, separators=(",", ":")).encode("utf-8")
            )
        return normalized
