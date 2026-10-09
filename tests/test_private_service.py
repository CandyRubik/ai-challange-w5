import asyncio
import json

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.providers.errors import LlmConfigurationError, LlmRequestError
from app.providers.registry import ModelRegistry


def test_private_service_rejects_cloud_and_unapproved_models(monkeypatch):
    monkeypatch.setenv("PRIVATE_SERVICE", "1")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3.5:9b-q4_K_M")
    registry = ModelRegistry()
    assert registry.resolve("ollama").model == "qwen3.5:9b-q4_K_M"
    with pytest.raises(LlmConfigurationError, match="приватном"):
        registry.resolve("deepseek")
    with pytest.raises(LlmConfigurationError, match="модель"):
        registry.resolve("ollama", "qwen3.5:9b-q8_0")


def test_private_limits_override_saved_profiles_and_disable_continuations(monkeypatch):
    monkeypatch.setenv("PRIVATE_SERVICE", "1")
    calls = []
    def handler(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "Часть ответа"},
                                         "done": True, "done_reason": "length"})
    client_type = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: client_type(
        transport=httpx.MockTransport(handler), **kwargs))
    registry = ModelRegistry()
    model = registry.build(registry.resolve("ollama"), num_ctx=32768, max_num_ctx=32768)
    with pytest.raises(LlmRequestError, match="лимит"):
        model.generate(messages=[{"role": "user", "content": "Вопрос"}], max_tokens=3000)
    assert len(calls) == 1
    assert calls[0]["options"] == {"num_ctx": 8192, "num_predict": 1000}


def test_http_limits_reject_large_bodies_and_excess_generations():
    from app.private_service import PrivateServiceMiddleware, ServiceLimits
    app = FastAPI()
    @app.post("/api/chat/sessions/test/messages")
    def generate():
        return {"answer": "ok"}
    app.add_middleware(PrivateServiceMiddleware, limits=ServiceLimits(
        enabled=True, max_body_bytes=100, generations_per_minute=2))
    with TestClient(app) as client:
        path = "/api/chat/sessions/test/messages"
        assert client.post(path, content=b"x" * 101).status_code == 413
        assert client.post(path, json={"content": "one"}).status_code == 200
        assert client.post(path, json={"content": "two"}).status_code == 200
        rejected = client.post(path, json={"content": "three"})
        assert rejected.status_code == 429
        assert 1 <= int(rejected.headers["Retry-After"]) <= 60


def test_concurrent_http_requests_have_one_generator_and_bounded_queue():
    from app.private_service import PrivateServiceMiddleware, ServiceLimits
    async def scenario():
        app = FastAPI()
        entered, release = asyncio.Event(), asyncio.Event()
        active, peak = 0, 0
        @app.post("/api/chat/sessions/test/messages")
        async def generate():
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            entered.set()
            await release.wait()
            active -= 1
            return {"answer": "ok"}
        @app.get("/api/health")
        def health():
            return {"status": "ok"}
        app.add_middleware(PrivateServiceMiddleware, limits=ServiceLimits(enabled=True, max_queue=1))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            first = asyncio.create_task(client.post("/api/chat/sessions/test/messages", json={}))
            await entered.wait()
            second = asyncio.create_task(client.post("/api/chat/sessions/test/messages", json={}))
            await asyncio.sleep(0)
            try:
                overflow = await asyncio.wait_for(client.post("/api/chat/sessions/test/messages", json={}), 1)
                assert overflow.status_code == 503
                assert "Retry-After" in overflow.headers
                assert (await client.get("/api/health")).status_code == 200
            finally:
                release.set()
                results = await asyncio.gather(first, second)
            assert [result.status_code for result in results] == [200, 200]
            assert peak == 1
    asyncio.run(scenario())


def test_oversized_context_is_a_client_error_without_model_request(tmp_path, monkeypatch):
    from app import main
    monkeypatch.setenv("PRIVATE_SERVICE", "1")
    monkeypatch.setenv("CHAT_DB_PATH", str(tmp_path / "private.sqlite3"))
    repositories = (main.get_chat_repository, main.get_memory_repository,
                    main.get_profile_repository, main.get_invariant_repository)
    for repository in repositories:
        repository.cache_clear()
    def forbidden(request):
        pytest.fail("Oversized context reached Ollama")
    client_type = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: client_type(
        transport=httpx.MockTransport(forbidden), **kwargs))
    try:
        with TestClient(main.app) as client:
            settings = client.get("/api/invariants").json()
            settings.update(emoji_enabled=False, uppercase_enabled=False, sentence_limit_enabled=False)
            assert client.put("/api/invariants", json=settings).status_code == 200
            profile = client.post("/api/profiles", json={"name": "Тест"}).json()
            session = client.post("/api/chat/sessions", json={"profile_id": profile["id"]}).json()
            response = client.post(f"/api/chat/sessions/{session['id']}/messages",
                                   json={"content": "а" * 5000})
            assert response.status_code == 422
            assert "Контекст" in response.json()["detail"]
    finally:
        for repository in repositories:
            repository.cache_clear()


def test_disconnected_client_does_not_release_a_running_model_slot():
    from threading import Event
    from app.private_service import PrivateServiceMiddleware, ServiceLimits
    async def scenario():
        app = FastAPI()
        entered, release = Event(), Event()
        @app.post("/api/chat/sessions/test/messages")
        def generate():
            entered.set()
            assert release.wait(5)
            return {"answer": "ok"}
        app.add_middleware(PrivateServiceMiddleware, limits=ServiceLimits(enabled=True, max_queue=0))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            first = asyncio.create_task(client.post("/api/chat/sessions/test/messages", json={}))
            assert await asyncio.to_thread(entered.wait, 3)
            first.cancel()
            await asyncio.sleep(0.02)
            first.cancel()
            await asyncio.sleep(0)
            try:
                response = await asyncio.wait_for(client.post("/api/chat/sessions/test/messages", json={}), 1)
                assert response.status_code == 503
            finally:
                release.set()
                with pytest.raises(asyncio.CancelledError):
                    await first
                await asyncio.sleep(0.03)
    asyncio.run(scenario())


@pytest.mark.parametrize("action", [[], {}])
def test_malformed_task_actions_reach_http_validation(action):
    from app.private_service import PrivateServiceMiddleware, ServiceLimits
    from app.schemas import TaskActionRequest
    app = FastAPI()
    @app.post("/api/chat/sessions/test/task/actions")
    def task(request: TaskActionRequest):
        return {"ok": True}
    app.add_middleware(PrivateServiceMiddleware, limits=ServiceLimits(enabled=True))
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/api/chat/sessions/test/task/actions", json={"action": action, "revision": 0})
        assert response.status_code == 422


def test_private_rag_catalog_reports_effective_limits_and_allowed_model(tmp_path, monkeypatch):
    from app import main
    monkeypatch.setenv("PRIVATE_SERVICE", "1")
    monkeypatch.setenv("CHAT_DB_PATH", str(tmp_path / "rag.sqlite3"))
    main.get_rag_chat_service.cache_clear()
    client_type = httpx.Client
    def handler(request):
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": [{"name": "qwen3.5:9b-q4_K_M"}]})
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: client_type(
        transport=httpx.MockTransport(handler), **kwargs))
    try:
        with TestClient(main.app, raise_server_exceptions=False) as client:
            response = client.get("/api/rag-chat/configurations")
            assert response.status_code == 200
            configurations = response.json()["configurations"]
            assert {item["id"] for item in configurations} == {"baseline", "compact", "optimized"}
            for item in configurations:
                assert item["profile"]["num_ctx"] <= 8192
                assert item["profile"]["max_num_ctx"] <= 8192
                assert item["profile"]["max_tokens"] <= 1000
    finally:
        main.get_rag_chat_service.cache_clear()
