from __future__ import annotations

import json
from pathlib import Path
import sqlite3

from fastapi.testclient import TestClient
import pytest

from app.agents.agent import Agent, AgentOutputError
from app.main import app, get_chat_session_service, get_memory_service
from app.memory.service import MemoryService, SQLiteMemoryRepository
from app.memory.updates import MemoryUpdateConflict, WorkingMemoryInterpreter
from app.schemas import MemoryCreateRequest
from app.services.chat_sessions import ChatSessionService
from app.storage.chat_sessions import ChatTurnConflict, SQLiteChatSessionRepository


class JsonModel:
    def __init__(self, *updates: dict) -> None:
        self.updates = list(updates)
        self.calls: list[dict] = []

    def generate_json(self, *, messages, max_tokens=2000):
        self.calls.append(json.loads(messages[-1]["content"]))
        return json.dumps(self.updates.pop(0) if self.updates else {}, ensure_ascii=False)


class AnswerModel:
    def __init__(self, *answers) -> None:
        self.answers = list(answers)
        self.calls: list[list[dict]] = []

    def generate(self, *, messages, max_tokens=2000):
        self.calls.append(messages)
        answer = self.answers.pop(0) if self.answers else "Готово."
        if isinstance(answer, Exception):
            raise answer
        return answer


def create_services(tmp_path: Path, json_model=None, answers=None):
    database = tmp_path / "chat.sqlite3"
    sessions = SQLiteChatSessionRepository(database)
    memories = SQLiteMemoryRepository(database)
    interpreter = WorkingMemoryInterpreter(json_model or JsonModel())
    service = ChatSessionService(sessions, Agent(answers or AnswerModel()), memories,
                                memory_interpreter=interpreter)
    return service, memories, sessions


def constraint(value: str) -> dict:
    return {"changes": [{"group": "constraint", "action": "set", "key": "length", "value": value}]}


def test_failed_answer_survives_restart_and_retry_interprets_only_once(tmp_path):
    decisions = JsonModel({"goal_action": "set", "goal": "Написать письмо", **constraint("коротко")})
    answers = AnswerModel(AgentOutputError("temporary"))
    service, memories, sessions = create_services(tmp_path, decisions, answers)
    chat = service.create()
    with pytest.raises(AgentOutputError):
        service.send(chat.id, "Моя цель — написать письмо. Отвечай коротко.")
    stored = service.get(chat.id)
    assert len(stored.messages) == 1
    assert stored.messages[0].status == "failed"
    assert memories.list_working(chat.id) == []
    with pytest.raises(ChatTurnConflict):
        service.send(chat.id, "Новый вопрос")

    restarted_model = JsonModel()
    restarted, memories, _ = create_services(tmp_path, restarted_model, AnswerModel("Письмо готово."))
    result = restarted.retry(chat.id, stored.messages[0].id)
    assert result.user_message.id == stored.messages[0].id
    assert result.user_message.status == "done"
    assert len(restarted.get(chat.id).messages) == 2
    assert restarted_model.calls == []
    assert len(decisions.calls) == 1
    facts = memories.list_working(chat.id)
    assert [fact.category for fact in facts] == ["goal", "constraint"]
    assert all(fact.source_message_id == result.user_message.id for fact in facts)
    assert all(fact.source_session_id == chat.id for fact in facts)
    with pytest.raises(ChatTurnConflict):
        restarted.retry(chat.id, result.user_message.id)


def test_correction_replaces_value_and_points_to_correcting_message(tmp_path):
    service, memories, _ = create_services(tmp_path, JsonModel(constraint("коротко"), constraint("подробно")))
    chat = service.create()
    first = service.send(chat.id, "Отвечай коротко")
    corrected = service.send(chat.id, "Нет, теперь подробно")
    facts = memories.list_working(chat.id)
    assert len(facts) == 1
    assert facts[0].content == "length: подробно"
    assert facts[0].source_message_id == corrected.user_message.id
    assert facts[0].source_message_id != first.user_message.id


def test_new_goal_clears_working_scope_and_old_dialogue_but_preserves_long_term(tmp_path):
    answers = AnswerModel()
    service, memories, _ = create_services(tmp_path, JsonModel(
        {"goal_action": "set", "goal": "Старая цель", **constraint("коротко")},
        {"goal_action": "set", "goal": "Новая цель"}, {},
    ), answers)
    chat = service.create()
    memories.add(layer="long_term", category="preference", content="Русский язык", session_id=None)
    service.send(chat.id, "Старая цель: короткая статья")
    service.send(chat.id, "Новая цель: подготовить интервью")
    service.send(chat.id, "Продолжай")
    assert [(fact.category, fact.content) for fact in memories.list_working(chat.id)] == [("goal", "Новая цель")]
    assert len(memories.list_long_term()) == 1
    for call in answers.calls[1:]:
        assert "Старая цель" not in json.dumps(call, ensure_ascii=False)


def test_ambiguous_removal_asks_clarification_without_generation_or_changes(tmp_path):
    answers = AnswerModel()
    service, memories, _ = create_services(tmp_path, JsonModel(
        constraint("коротко"), {"clarification": "Какое ограничение убрать?"},
    ), answers)
    chat = service.create()
    service.send(chat.id, "Отвечай коротко")
    before = memories.list_working(chat.id)
    reply = service.send(chat.id, "Убери это")
    assert reply.assistant_message.content == "Какое ограничение убрать?"
    assert memories.list_working(chat.id) == before
    assert len(answers.calls) == 1


def test_clear_goal_and_unambiguous_removal(tmp_path):
    service, memories, _ = create_services(tmp_path, JsonModel(
        {"goal_action": "set", "goal": "Письмо", **constraint("коротко")},
        {"changes": [{"group": "constraint", "action": "remove", "key": "length"}]},
        {"goal_action": "clear"},
    ))
    chat = service.create()
    service.send(chat.id, "Письмо, коротко")
    service.send(chat.id, "Убери ограничение длины")
    assert [fact.category for fact in memories.list_working(chat.id)] == ["goal"]
    service.send(chat.id, "Отменяю эту цель")
    assert memories.list_working(chat.id) == []


def test_memory_update_and_answer_commit_roll_back_together(tmp_path, monkeypatch):
    service, memories, _ = create_services(tmp_path, JsonModel(constraint("коротко")))
    chat = service.create()
    original = memories.commit_update
    def broken_commit(*args):
        original(*args)
        raise sqlite3.OperationalError("simulated storage failure")
    monkeypatch.setattr(memories, "commit_update", broken_commit)
    with pytest.raises(sqlite3.OperationalError):
        service.send(chat.id, "Отвечай коротко")
    assert memories.list_working(chat.id) == []
    assert len(service.get(chat.id).messages) == 1
    monkeypatch.setattr(memories, "commit_update", original)
    service.retry(chat.id, service.get(chat.id).messages[-1].id)
    assert len(memories.list_working(chat.id)) == 1


def test_task_memory_survives_history_window_and_isolated_from_new_chat(tmp_path):
    answers = AnswerModel()
    service, memories, sessions = create_services(tmp_path, JsonModel(
        {"goal_action": "set", "goal": "Подготовить интервью"}, {}, {},
    ), answers)
    chat = service.create()
    service.send(chat.id, "Моя цель — подготовить интервью")
    for index in range(45):
        sessions.append_exchange(chat.id, f"Уточнение {index}", "Принято")
    service.send(chat.id, "Какова цель?")
    assert "Подготовить интервью" in answers.calls[-1][0]["content"]
    other = service.create()
    service.send(other.id, "Какова цель?")
    assert memories.list_working(other.id) == []
    assert "Подготовить интервью" not in answers.calls[-1][0]["content"]


def test_api_recovery_and_source_links_for_manual_memory(tmp_path):
    service, memories, sessions = create_services(tmp_path, answers=AnswerModel(AgentOutputError("temporary"), "Готово"))
    app.dependency_overrides[get_chat_session_service] = lambda: service
    app.dependency_overrides[get_memory_service] = lambda: MemoryService(memories, sessions)
    try:
        client = TestClient(app)
        chat = client.post("/api/chat/sessions", json={}).json()
        url = f"/api/chat/sessions/{chat['id']}"
        assert client.post(url + "/messages", json={"content": "Привет"}).status_code == 502
        message = client.get(url).json()["messages"][-1]
        assert message["status"] == "failed"
        assert client.post(url + "/messages", json={"content": "Дальше"}).status_code == 409
        assert client.post(url + "/messages/other/retry").status_code == 409
        retry_url = url + f"/messages/{message['id']}/retry"
        assert client.post(retry_url).status_code == 200
        assert client.post(retry_url).status_code == 409
        command = client.post("/api/memory", json={
            "layer": "working", "category": "goal", "content": "Подготовить письмо",
            "session_id": chat["id"], "source_session_id": chat["id"], "source_text": "/goal Подготовить письмо",
        })
        assert command.status_code == 201
        source = command.json()
        assert source["source_session_id"] == chat["id"]
        assert source["source_message_id"] == client.get(url).json()["messages"][-1]["id"]
    finally:
        app.dependency_overrides.clear()


@pytest.mark.parametrize("update", [
    {"goal_action": "set"},
    {"clarification": "Что убрать?", "changes": constraint("коротко")["changes"]},
    {"changes": [{"group": "constraint", "action": "set", "key": "length"}]},
])
def test_invalid_memory_interpretation_does_not_break_answer_or_mutate_memory(tmp_path, update):
    service, memories, _ = create_services(tmp_path, JsonModel(update))
    chat = service.create()
    assert service.send(chat.id, "Вопрос").assistant_message.content == "Готово."
    assert memories.list_working(chat.id) == []


def test_regular_chat_does_not_invoke_document_retrieval(tmp_path, monkeypatch):
    def unexpected():
        raise AssertionError("Regular chat attempted document retrieval")
    monkeypatch.setattr("app.main.get_document_index", unexpected)
    service, _, _ = create_services(tmp_path)
    session = service.create()
    assert service.send(session.id, "Обычный вопрос").assistant_message.content == "Готово."


def test_memory_edit_during_generation_requires_retry_from_new_snapshot(tmp_path):
    service, memories, _ = create_services(tmp_path, JsonModel(constraint("коротко")))
    chat = service.create()
    class EditingModel(AnswerModel):
        def generate(self, **kwargs):
            memories.add(layer="working", category="decision", content="Ручное решение", session_id=chat.id)
            return "Готово."
    service._agent = Agent(EditingModel())
    with pytest.raises(MemoryUpdateConflict):
        service.send(chat.id, "Отвечай коротко")
    assert [fact.content for fact in memories.list_working(chat.id)] == ["Ручное решение"]
    assert service.get(chat.id).messages[-1].status == "failed"
    service._agent = Agent(AnswerModel())
    service.retry(chat.id, service.get(chat.id).messages[-1].id)
    assert [fact.content for fact in memories.list_working(chat.id)] == ["Ручное решение", "length: коротко"]


def test_answer_rejected_by_invariants_does_not_commit_memory(tmp_path):
    from app.invariants import SQLiteInvariantRepository, InvariantUpdateRequest
    service, memories, _ = create_services(tmp_path, JsonModel(constraint("коротко")), AnswerModel("Первое. Второе."))
    rules = SQLiteInvariantRepository(tmp_path / "chat.sqlite3")
    rules.update(InvariantUpdateRequest(revision=rules.get().revision, emoji_enabled=False,
        uppercase_enabled=False, sentence_limit_enabled=True, max_sentences=1))
    service._invariants = rules
    chat = service.create()
    result = service.send(chat.id, "Отвечай коротко")
    assert result.assistant_message.refusal
    assert result.user_message.status == "done"
    assert memories.list_working(chat.id) == []


def test_invalid_json_schema_is_retried_with_validation_feedback(tmp_path):
    model = JsonModel({"changes": [{"group": "goal", "action": "set", "key": "goal", "value": "Новая цель"}]},
                      {"goal_action": "set", "goal": "Новая цель"})
    service, memories, _ = create_services(tmp_path, model)
    chat = service.create()
    service.send(chat.id, "Меняю цель")
    assert len(model.calls) == 2
    assert "validation_feedback" in model.calls[1]
    assert memories.list_working(chat.id)[0].content == "Новая цель"


def test_active_goal_remains_in_prompt_with_many_recent_facts(tmp_path):
    answers = AnswerModel()
    service, memories, _ = create_services(tmp_path, JsonModel({}), answers)
    chat = service.create()
    memories.add(layer="working", category="goal", content="Устойчивая цель", session_id=chat.id)
    for index in range(65):
        memories.add(layer="working", category="constraint", content=f"Ограничение {index}", session_id=chat.id)
    service.send(chat.id, "Что делаем?")
    assert "Устойчивая цель" in answers.calls[-1][0]["content"]
    assert any(fact.category == "goal" for fact in memories.list_working(chat.id))
