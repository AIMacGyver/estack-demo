"""Local OpenAI-compatible thinker adapter and CLI wiring."""

from types import SimpleNamespace

import pytest

import spatial_ipd.think as think_mod
from spatial_ipd.local_llm import LocalLLMError, LocalLLMThinkerClient


def _state(count=2):
    return {
        "generation": 5,
        "thinkers": [
            {
                "row": index,
                "col": 0,
                "before": 1,
                "after_imitate": 0 if index == 0 else 1,
                "patch_after": [[1, 1, 0], [1, 0, 0], [1, 1, 0]],
                "focal_score": 18,
                "best_neighbor_score": 25,
                "best_neighbor_strategy": 0,
            }
            for index in range(count)
        ],
    }


def _question_ids():
    return {
        "worth_thinking_0": None,
        "resist_0": None,
        "cluster_fragility_0": None,
        "worth_thinking_1": None,
        "cluster_fragility_1": None,
    }


def test_local_client_posts_boolean_decisions_and_separate_confidence():
    calls = []

    def transport(endpoint, payload, headers, timeout):
        calls.append((endpoint, payload, headers, timeout))
        return {
            "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
            "choices": [
                {
                    "message": {
                        "content": (
                            "```json\n"
                            '{"decisions":['
                            '{"index":1,"worth_thinking":false,"worth_confidence":0.91,'
                            '"resist":true,"resist_confidence":0.99,"cluster_fragility":2},'
                            '{"index":0,"worth_thinking":true,"worth_confidence":0.62,'
                            '"resist":true,"resist_confidence":0.73,"cluster_fragility":1.25}'
                            "]}\n```"
                        )
                    }
                }
            ],
        }

    client = LocalLLMThinkerClient(
        "qwen3:8b",
        endpoint="http://model.test/v1/chat/completions",
        api_key="secret",
        timeout=4.5,
        reasoning_effort="none",
        prompt_profile="score_defer_v1",
        temperature=0.4,
        max_tokens=256,
        transport=transport,
    )
    response = client.system_one(_state(), _question_ids())

    assert len(calls) == 1
    endpoint, payload, headers, timeout = calls[0]
    assert endpoint == "http://model.test/v1/chat/completions"
    assert payload["model"] == "qwen3:8b"
    assert payload["stream"] is False
    assert payload["temperature"] == 0.4
    assert payload["reasoning_effort"] == "none"
    assert payload["max_tokens"] == 256
    assert "score_defer_v1" in payload["messages"][0]["content"]
    assert payload["messages"][0]["role"] == "system"
    assert '"generation":5' in payload["messages"][1]["content"]
    assert headers["Authorization"] == "Bearer secret"
    assert timeout == 4.5
    assert response.nouls["worth_thinking_0"].noul == 1.0
    assert response.nouls["worth_thinking_0"].confidence == 0.62
    assert response.nouls["resist_0"].noul == 1.0
    assert response.nouls["resist_0"].confidence == 0.73
    assert response.nouls["worth_thinking_1"].noul == 0.0
    assert response.nouls["worth_thinking_1"].confidence == 0.91
    assert "resist_1" not in response.nouls
    assert response.scores["cluster_fragility_0"].score == 1.25
    assert response.scores["cluster_fragility_1"].score == 2.0
    assert response.confidence_kind == "llm_self_report"
    assert '"worth_thinking":true' in response.raw_output
    assert response.backend_usage == {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28}
    assert response.raw_response_bytes > len(response.raw_output)


@pytest.mark.parametrize(
    ("transport_response", "message"),
    [
        ({}, "missing choices"),
        ({"choices": [{"message": {"content": "not JSON"}}]}, "did not contain a JSON object"),
        (
            {"choices": [{"message": {"content": '{"decisions":[]}'}}]},
            "returned 0 decisions; expected 2",
        ),
        (
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"decisions":['
                                '{"index":0,"worth_thinking":"high","worth_confidence":0.9,'
                                '"resist":true,"resist_confidence":0.5,"cluster_fragility":1},'
                                '{"index":1,"worth_thinking":false,"worth_confidence":0.5,'
                                '"resist":null,"resist_confidence":null,"cluster_fragility":1}'
                                "]}"
                            )
                        }
                    }
                ]
            },
            "worth_thinking_0",
        ),
        (
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"decisions":['
                                '{"index":0,"worth_thinking":true,"worth_confidence":1.2,'
                                '"resist":true,"resist_confidence":0.5,"cluster_fragility":1},'
                                '{"index":1,"worth_thinking":false,"worth_confidence":0.5,'
                                '"resist":null,"resist_confidence":null,"cluster_fragility":1}'
                                "]}"
                            )
                        }
                    }
                ]
            },
            "between 0 and 1",
        ),
    ],
)
def test_local_client_reports_malformed_responses(transport_response, message):
    client = LocalLLMThinkerClient("model", transport=lambda *_args: transport_response)
    with pytest.raises(LocalLLMError, match=message):
        client.system_one(_state(), _question_ids())


def test_local_client_wraps_transport_failure_with_endpoint():
    def fail(*_args):
        raise OSError("connection refused")

    endpoint = "http://localhost:9999/v1/chat/completions"
    client = LocalLLMThinkerClient("model", endpoint=endpoint, transport=fail)
    with pytest.raises(LocalLLMError, match=f"{endpoint} failed: connection refused"):
        client.system_one(_state(), _question_ids())


@pytest.mark.parametrize(
    ("worth", "worth_confidence", "resist", "resist_confidence", "applied"),
    [
        (True, 0.01, True, 0.02, True),
        (False, 0.99, True, 0.99, False),
        (True, 0.99, False, 0.99, False),
    ],
)
def test_boolean_decisions_control_action_not_reported_confidence(
    worth,
    worth_confidence,
    resist,
    resist_confidence,
    applied,
):
    content = (
        '{"decisions":[{"index":0,'
        f'"worth_thinking":{str(worth).lower()},"worth_confidence":{worth_confidence},'
        f'"resist":{str(resist).lower()},"resist_confidence":{resist_confidence},'
        '"cluster_fragility":1}]}'
    )
    client = LocalLLMThinkerClient(
        "model",
        transport=lambda *_args: {"choices": [{"message": {"content": content}}]},
    )
    after, stats = think_mod.think_after_step(
        [[1]],
        [[0]],
        generation=1,
        seed=1,
        thinker_count=1,
        client=client,
    )

    decision = stats.last_decisions[0]
    assert decision.applied is applied
    assert after == [[1 if applied else 0]]
    assert decision.worth_confidence == worth_confidence
    assert decision.resist_confidence == resist_confidence
    assert decision.confidence_kind == "llm_self_report"
    assert stats.backend_audits[0].raw_output == content
    assert "confidence_kind=llm_self_report" in think_mod.format_decision_line(decision)


def test_local_backend_requires_model(capsys):
    with pytest.raises(SystemExit):
        think_mod.parse_args(
            [
                "--height",
                "2",
                "--width",
                "2",
                "--generations",
                "1",
                "--seed",
                "1",
                "--backend",
                "local",
            ]
        )
    assert "--local-model is required when --backend local" in capsys.readouterr().err


def test_local_backend_cli_builds_configured_client(monkeypatch, capsys):
    created = {}

    class FakeLocalClient:
        uses_sdk_questions = False

        def __init__(self, model, *, endpoint, api_key, timeout, reasoning_effort):
            created.update(
                model=model,
                endpoint=endpoint,
                api_key=api_key,
                timeout=timeout,
                reasoning_effort=reasoning_effort,
            )

        def system_one(self, state, questions):
            del state
            nouls = {}
            scores = {}
            for key in questions:
                if key.startswith(("worth_thinking_", "resist_")):
                    nouls[key] = SimpleNamespace(noul=0.2)
                elif key.startswith("cluster_fragility_"):
                    scores[key] = SimpleNamespace(score=1.0, confidence=1.0)
            return SimpleNamespace(nouls=nouls, scores=scores, choices={})

    monkeypatch.setattr(think_mod, "LocalLLMThinkerClient", FakeLocalClient)
    monkeypatch.setenv("LOCAL_LLM_API_KEY", "from-env")
    code = think_mod.main(
        [
            "--height",
            "2",
            "--width",
            "2",
            "--generations",
            "1",
            "--seed",
            "1",
            "--think-every",
            "1",
            "--think-last",
            "0",
            "--thinkers",
            "1",
            "--backend",
            "local",
            "--local-model",
            "qwen3:8b",
            "--local-endpoint",
            "http://model.test/v1/chat/completions",
            "--local-timeout",
            "7",
            "--local-reasoning-effort",
            "none",
        ]
    )

    assert code == 0
    assert created == {
        "model": "qwen3:8b",
        "endpoint": "http://model.test/v1/chat/completions",
        "api_key": "from-env",
        "timeout": 7.0,
        "reasoning_effort": "none",
    }
    assert "think_calls=1" in capsys.readouterr().out
