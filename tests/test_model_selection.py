from concurrent.futures import ThreadPoolExecutor
import json
import sqlite3
from threading import Event

from fastapi.testclient import TestClient
import httpx
import pytest

from app import main
from app.agents.agent import Agent
from app.providers.deepseek import DeepSeekProvider
from app.providers.errors import LlmRequestError
from app.providers.registry import ModelRegistry
from app.schemas import TaskActionRequest
from app.services.chat_sessions import ChatSessionService
from app.storage.chat_sessions import SQLiteChatSessionRepository


class TextModel:
    def __init__(self, answer="Ответ"):
        self.answer = answer
        self.calls = []

    def generate(self, *, messages, max_tokens=2000):
        self.calls.append(messages)
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


class Registry(ModelRegistry):
    def __init__(self, local, cloud):
        super().__init__()
        self.models = {"ollama": local, "deepseek": cloud}
        self.selections = []

    def build(self, selection, **kwargs):
        self.selections.append(selection)
        return self.models[selection.provider]


def service(tmp_path, registry):
    return ChatSessionService(SQLiteChatSessionRepository(tmp_path / "chat.sqlite3"), model_registry=registry)


def test_switching_preserves_history_and_persists_per_chat(tmp_path):
    local, cloud = TextModel("Локальный ответ"), TextModel("Облачный ответ")
    chat = service(tmp_path, Registry(local, cloud))
    first, other = chat.create(), chat.create()
    assert first.provider == "ollama"
    chat.send(first.id, "Вопрос один")
    chat.set_provider(first.id, "deepseek")
    result = chat.send(first.id, "Вопрос два")
    assert result.assistant_message.provider == "deepseek"
    assert result.assistant_message.model == "deepseek-v4-flash"
    assert cloud.calls[0][-3:] == [
        {"role": "user", "content": "Вопрос один"},
        {"role": "assistant", "content": "Локальный ответ"},
        {"role": "user", "content": "Вопрос два"},
    ]
    restarted = service(tmp_path, Registry(local, cloud))
    assert restarted.get(first.id).provider == "deepseek"
    assert restarted.get(other.id).provider == "ollama"
    assert restarted.get(first.id).messages[1].provider == "ollama"


def test_queued_provider_overrides_current_preference_without_changing_it(tmp_path):
    local, cloud = TextModel(), TextModel()
    chat = service(tmp_path, Registry(local, cloud))
    session = chat.create()
    chat.set_provider(session.id, "deepseek")
    result = chat.send(session.id, "Сохранённое сообщение из очереди", "ollama")
    assert result.assistant_message.provider == "ollama"
    assert result.session.provider == "deepseek"
    assert local.calls and not cloud.calls


def test_deepseek_uses_compatible_json_mode_for_schema_requests():
    request = DeepSeekProvider()._build_request(
        system_prompt="Return JSON", user_prompt="Profile", thinking_type="disabled",
        response_format={"type": "json_schema", "json_schema": {"name": "profile", "schema": {"type": "object"}}},
    )
    assert request["response_format"] == {"type": "json_object"}


def test_retry_after_restart_uses_original_provider_and_model(tmp_path, monkeypatch):
    cloud = TextModel()
    original = service(tmp_path, Registry(TextModel(LlmRequestError("temporary")), cloud))
    session = original.create()
    with pytest.raises(LlmRequestError):
        original.send(session.id, "Вопрос")
    pending = original.get(session.id).messages[-1]
    original.set_provider(session.id, "deepseek")
    monkeypatch.setenv("OLLAMA_MODEL", "different-after-restart")
    registry = Registry(TextModel("Ответ после повтора"), cloud)
    restarted = service(tmp_path, registry)
    result = restarted.retry(session.id, pending.id)
    assert result.user_message.id == pending.id
    assert result.assistant_message.model == pending.model == "qwen3.5:9b-q4_K_M"
    assert result.session.provider == "deepseek"
    assert all(selection.model == pending.model for selection in registry.selections)
    assert not cloud.calls
    assert len(restarted.get(session.id).messages) == 2


def test_switch_during_generation_does_not_reroute_the_active_turn(tmp_path):
    entered, release = Event(), Event()
    class BlockingModel(TextModel):
        def generate(self, **kwargs):
            entered.set()
            assert release.wait(5)
            return super().generate(**kwargs)
    local, cloud = BlockingModel(), TextModel()
    chat = service(tmp_path, Registry(local, cloud))
    session = chat.create()
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(chat.send, session.id, "Вопрос")
        assert entered.wait(5)
        try:
            chat.set_provider(session.id, "deepseek")
        finally:
            release.set()
        result = pending.result(timeout=5)
    assert result.assistant_message.provider == "ollama"
    assert result.session.provider == "deepseek"
    assert not cloud.calls


def test_migration_keeps_legacy_chats_and_does_not_invent_response_models(tmp_path):
    path = tmp_path / "legacy.sqlite3"
    original = ChatSessionService(SQLiteChatSessionRepository(path), Agent(TextModel()))
    session = original.create()
    original.send(session.id, "Старая история")
    with sqlite3.connect(path) as connection:
        connection.execute("ALTER TABLE chat_sessions DROP COLUMN provider")
        connection.execute("ALTER TABLE chat_messages DROP COLUMN provider")
        connection.execute("ALTER TABLE chat_messages DROP COLUMN model")
    migrated = SQLiteChatSessionRepository(path).get(session.id)
    assert migrated.provider == "deepseek"
    assert len(migrated.messages) == 2
    assert all(message.model is None for message in migrated.messages)


@pytest.fixture
def local_api(tmp_path, monkeypatch):
    for key in ("DEEPSEEK_API_KEY", "DEEPSEEK_MODEL", "OLLAMA_MODEL", "LLM_DEFAULT_PROVIDER"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("CHAT_DB_PATH", str(tmp_path / "api.sqlite3"))
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    monkeypatch.setenv("OLLAMA_NUM_CTX", "32768")
    repositories = [main.get_chat_repository, main.get_memory_repository,
                    main.get_profile_repository, main.get_invariant_repository]
    for repository in repositories:
        repository.cache_clear()
    calls = []
    def handler(request):
        assert request.url.host == "127.0.0.1", "Local mode attempted an external request"
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen3.5:9b-q4_K_M"}]})
        payload = json.loads(request.content)
        calls.append(payload)
        prompt = payload["messages"][0]["content"]
        if "Interpret explicit working-memory" in prompt:
            content = '{"goal_action":"keep","goal":"","changes":[],"clarification":""}'
        elif "You extract memory" in prompt:
            content = '{"memories":[{"layer":"long_term","category":"preference","content":"Любит краткие ответы"}]}'
        elif "You update a user profile" in prompt:
            content = json.dumps({"name": "Тест", "description": None, "language": "ru", "tone": "neutral",
                                  "detail_level": "brief", "response_format": "plain", "constraints": []})
        elif '"summary"' in prompt and '"criteria"' in prompt:
            content = '{"summary":"План","plan":["Написать приветствие"],"criteria":["Есть приветствие"]}'
        elif '"repair_steps"' in prompt:
            content = '{"passed":true,"report":"Приветствие готово","repair_steps":[]}'
        else:
            content = "Приветствие готово"
        return httpx.Response(200, json={"message": {"content": content}, "done": True, "done_reason": "stop"})
    client_type = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: client_type(transport=httpx.MockTransport(handler), **kwargs))
    def forbidden_cloud(*args, **kwargs):
        pytest.fail("Local mode reached DeepSeek")
    monkeypatch.setattr(DeepSeekProvider, "_get_client", forbidden_cloud)
    try:
        with TestClient(main.app) as client:
            settings = client.get("/api/invariants").json()
            settings.update(emoji_enabled=False, uppercase_enabled=False, sentence_limit_enabled=False)
            assert client.put("/api/invariants", json=settings).status_code == 200
            yield client, calls
    finally:
        for repository in repositories:
            repository.cache_clear()


def test_whole_application_works_without_cloud_key(local_api):
    client, calls = local_api
    assert client.get("/api/health").json()["deepseek_configured"] is False
    catalog = client.get("/api/models").json()
    assert catalog["default_provider"] == "ollama"
    assert catalog["providers"][0]["available"] is True
    assert catalog["providers"][1]["available"] is False
    profile = client.post("/api/profiles/auto").json()
    session = client.post("/api/chat/sessions", json={"profile_id": profile["id"]}).json()
    url = f"/api/chat/sessions/{session['id']}"
    for content in ["Напиши приветствие", "Меня зовут Тест", "Кратко", "Нейтрально"]:
        result = client.post(url + "/messages", json={"content": content})
        assert result.status_code == 200, result.text
    assert client.get("/api/profiles").json()[-1]["onboarding_complete"] is True
    assert client.get(url).json()["messages"][-1]["provider"] == "ollama"
    assert client.get(f"/api/memory?session_id={session['id']}").json()["long_term"]
    assert client.post(url + "/messages", json={"content": "Спасибо"}).status_code == 200
    assert client.post(url + "/task", json={"task": "Написать приветствие"}).status_code == 201
    for action in ["generate_plan", "approve", "execute_step", "validate"]:
        task = client.get(url).json()["task"]
        result = client.post(url + "/task/actions", json={"action": action, "revision": task["revision"]})
        assert result.status_code == 200, result.text
    session = result.json()
    assert session["task"]["state"] == "done"
    assert session["messages"][-1]["provider"] == "ollama"
    assert calls and all(call["model"] == "qwen3.5:9b-q4_K_M" for call in calls)
    assert any(call.get("format") == "json" for call in calls)
    assert any(call["options"]["num_predict"] == 8000 for call in calls)


def test_api_validates_choices_and_saves_preference(local_api):
    client, _ = local_api
    session = client.post("/api/chat/sessions", json={}).json()
    url = f"/api/chat/sessions/{session['id']}"
    assert client.put(url + "/model", json={"provider": "unknown"}).status_code == 422
    assert client.put(url + "/model", json={"provider": "deepseek", "url": "http://example.com"}).status_code == 422
    assert client.put(url + "/model", json={"provider": "deepseek"}).json()["provider"] == "deepseek"
    assert client.get(url).json()["provider"] == "deepseek"
    assert client.put("/api/chat/sessions/absent/model", json={"provider": "ollama"}).status_code == 404
