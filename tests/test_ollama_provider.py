import json

import httpx
import pytest

from app.providers.errors import LlmRequestError, LlmTruncatedResponseError
from app.providers.ollama import OllamaProvider


MESSAGES = [{"role": "system", "content": "Системная инструкция"},
            {"role": "user", "content": "Вопрос"}]


def provider_with(handler, *, num_ctx=32768):
    return OllamaProvider(model="qwen-local", num_ctx=num_ctx,
                          client=httpx.Client(transport=httpx.MockTransport(handler)))


def answer(content="Ответ", reason="stop"):
    return httpx.Response(200, json={"message": {"content": content}, "done": True, "done_reason": reason})


@pytest.mark.parametrize("method,structured", [("generate", False), ("generate_json", True), ("complete", True)])
def test_native_requests_cover_every_application_contract(method, structured):
    calls = []
    def handler(request):
        assert str(request.url) == "http://127.0.0.1:11434/api/chat"
        calls.append(json.loads(request.content))
        return answer('{"ok":true}' if structured else "Ответ")
    provider = provider_with(handler)
    if method == "complete":
        result = provider.complete(system_prompt=MESSAGES[0]["content"], user_prompt="Вопрос",
                                   response_format={"type": "json_object"}, max_tokens=700)
    else:
        result = getattr(provider, method)(messages=MESSAGES, max_tokens=700)
    assert result
    payload = calls[0]
    assert payload["messages"] == MESSAGES
    assert payload["model"] == "qwen-local"
    assert payload["think"] is False and payload["stream"] is False
    assert payload["options"]["num_ctx"] == 32768
    assert payload["options"]["num_predict"] == 700
    assert (payload.get("format") == "json") is structured
    assert "reasoning_effort" not in payload and "thinking" not in payload


def test_truncated_text_continues_with_context():
    calls = []
    def handler(request):
        calls.append(json.loads(request.content))
        return answer("Начало " if len(calls) == 1 else "конец", "length" if len(calls) == 1 else "stop")
    assert provider_with(handler).generate(messages=MESSAGES) == "Начало конец"
    assert calls[1]["messages"][:2] == MESSAGES
    assert calls[1]["messages"][-2] == {"role": "assistant", "content": "Начало "}


def test_complete_forwards_profile_schema_to_native_format():
    from app.orchestration.profile_interviewer import ProfileInterviewOutput
    schema = ProfileInterviewOutput.model_json_schema()
    calls = []
    def handler(request):
        calls.append(json.loads(request.content))
        return answer('{"description":null}')
    provider_with(handler).complete(
        system_prompt="Верни профиль как JSON", user_prompt="Краткие ответы",
        response_format={"type": "json_schema", "json_schema": {"name": "profile", "schema": schema}},
    )
    assert calls[0]["format"] == schema
    assert "description" in calls[0]["format"]["required"]
    assert schema["properties"]["description"]["anyOf"][0]["minLength"] == 1


def test_truncated_json_is_rejected_without_concatenating():
    calls = []
    def handler(request):
        calls.append(request)
        return answer('{"ok":', "length")
    with pytest.raises(LlmTruncatedResponseError, match="Ollama"):
        provider_with(handler).generate_json(messages=MESSAGES)
    assert len(calls) == 1


@pytest.mark.parametrize("response,match", [
    (httpx.Response(404), "не найдена"),
    (httpx.Response(500), "ошибкой"),
    (httpx.Response(200, json={"message": {"content": ""}, "done": True}), "непустой"),
    (httpx.Response(200, json={"message": {"content": "Ответ"}, "done": False}), "завершённый"),
    (httpx.Response(200, json={}), "ошибкой"),
])
def test_failures_are_actionable_and_never_return_partial_answers(response, match):
    with pytest.raises(LlmRequestError, match=match):
        provider_with(lambda request: response).generate(messages=MESSAGES)


@pytest.mark.parametrize("error,match", [(httpx.ConnectError("secret"), "ollama serve"),
                                         (httpx.ReadTimeout("secret"), "не успела")])
def test_transport_errors_do_not_expose_upstream_details(error, match):
    def handler(request):
        raise error
    with pytest.raises(LlmRequestError, match=match) as caught:
        provider_with(handler).generate(messages=MESSAGES)
    assert "secret" not in str(caught.value)


def test_context_budget_keeps_system_and_current_question_with_answer_reserve():
    calls = []
    def handler(request):
        calls.append(json.loads(request.content))
        return answer()
    history = [MESSAGES[0], {"role": "user", "content": "Старая история" * 5000},
               {"role": "assistant", "content": "Последний ответ"}, MESSAGES[-1]]
    provider_with(handler, num_ctx=4096).generate(messages=history, max_tokens=1000)
    assert calls[0]["messages"] == [MESSAGES[0], history[-2], MESSAGES[-1]]


def test_oversized_required_context_is_rejected_before_network():
    def handler(request):
        pytest.fail("Oversized input must not reach the model")
    with pytest.raises(LlmRequestError, match="Контекст не помещается"):
        provider_with(handler, num_ctx=4096).generate(messages=[
            MESSAGES[0], {"role": "user", "content": "Я" * 4000},
        ])


def test_compact_context_grows_for_required_memory_without_losing_evidence():
    calls = []
    def handler(request):
        calls.append(json.loads(request.content))
        return answer('{"ok":true}')
    messages = [{"role": "system", "content": "Evidence and required memory. " * 350}, MESSAGES[-1]]
    provider = OllamaProvider(model="local", num_ctx=8192, max_num_ctx=32768,
                             temperature=.2, seed=17,
                             client=httpx.Client(transport=httpx.MockTransport(handler)))

    provider.generate_json(messages=messages, max_tokens=1000)

    assert calls[0]["messages"] == messages
    assert calls[0]["options"] == {"num_ctx": 16384, "num_predict": 1000, "temperature": .2, "seed": 17}
    assert provider.request_metrics[0]["options"] == calls[0]["options"]


def test_compact_context_trims_old_history_instead_of_growing_for_it():
    calls = []
    def handler(request):
        calls.append(json.loads(request.content))
        return answer()
    provider = OllamaProvider(model="local", num_ctx=8192, max_num_ctx=32768,
                             client=httpx.Client(transport=httpx.MockTransport(handler)))
    provider.generate(messages=[MESSAGES[0], {"role": "user", "content": "old " * 10000}, MESSAGES[-1]], max_tokens=1000)
    assert calls[0]["options"]["num_ctx"] == 8192
    assert calls[0]["messages"] == MESSAGES


def test_compact_context_has_a_hard_upper_bound():
    def handler(request):
        pytest.fail("Required context overflow must not reach the model")
    provider = OllamaProvider(model="local", num_ctx=8192, max_num_ctx=16384,
                             client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(LlmRequestError, match="Контекст не помещается"):
        provider.generate_json(messages=[{"role": "system", "content": "x" * 17000}, MESSAGES[-1]], max_tokens=1000)
