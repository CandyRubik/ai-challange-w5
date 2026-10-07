"""User-visible behavior of the dedicated RAG chat."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.agents.agent import Agent
from app.indexing.rag import DocumentRag, RetrievalSettings
from app.indexing.store import SearchHit
from app.main import app, get_rag_chat_service
from app.rag_chat.models import RagTaskState
from app.rag_chat.service import RagChatService
from app.rag_chat.state import TurnInterpreter
from app.rag_chat.store import SQLiteRagChatRepository


def hit(text: str = "Executor manages submitted tasks.") -> SearchHit:
    return SearchHit({
        "chunk_id": "structure-0001",
        "source": "https://example.test/book.pdf",
        "title": "Test book",
        "section": "6.2 Executor",
        "page_start": 6,
        "page_end": 6,
        "text": text,
    }, 0.9)


class Index:
    def __init__(self) -> None:
        self.searches: list[str] = []

    def ensure_built(self) -> dict:
        return {"version": "test"}

    def search(self, query: str, strategy: str, k: int) -> list[SearchHit]:
        self.searches.append(query)
        return [hit()]


class Reranker:
    def score(self, question: str, hits: list[SearchHit]) -> list[float]:
        return [0.9 for _ in hits]


class InterpretationModel:
    def generate_json(self, *, messages: list[dict], max_tokens: int = 2000, schema: dict | None = None) -> str:
        return json.dumps({
            "kind": "question",
            "question_scope": "document",
            "search_question": "What does Executor do?",
            "goal_action": "keep",
            "goal": "",
            "changes": [],
            "clarification": "",
        })


class QueueInterpretationModel:
    def __init__(self, *decisions: dict) -> None:
        self.decisions = list(decisions)
        self.calls = 0

    def generate_json(self, *, messages: list[dict], max_tokens: int = 2000, schema: dict | None = None) -> str:
        self.calls += 1
        return json.dumps(self.decisions.pop(0))


def decision(
    *, kind: str = "question", scope: str = "document",
    goal_action: str = "keep", goal: str = "", changes: list[dict] | None = None,
    clarification: str = "",
) -> dict:
    return {
        "kind": kind, "question_scope": scope,
        "search_question": "What does Executor do?" if kind == "question" else "",
        "goal_action": goal_action, "goal": goal,
        "changes": changes or [], "clarification": clarification,
    }


class AnswerModel:
    def generate_json(self, *, messages: list[dict], max_tokens: int = 2000, schema: dict | None = None) -> str:
        return json.dumps({"status": "answered", "claims": [{
            "text": "Executor выполняет отправленные задачи.",
            "evidence": [{
                "chunk_id": "structure-0001",
                "quote": "Executor manages submitted tasks.",
            }],
        }]})


def test_question_searches_and_persists_verified_source(tmp_path: Path) -> None:
    database = tmp_path / "rag.sqlite3"
    index = Index()
    rag = DocumentRag(
        index, InterpretationModel(), reranker=Reranker(),
        settings=RetrievalSettings(rewrite=False, candidate_k=1, final_k=1),
    )
    chat = RagChatService(
        SQLiteRagChatRepository(database), TurnInterpreter(InterpretationModel()),
        rag, Agent(AnswerModel()),
    )
    session = chat.create()

    turn = chat.send(session.id, "Что делает Executor?")

    assert index.searches == ["What does Executor do?"]
    assert turn.status == "done"
    assert turn.grounding_status == "answered"
    assert turn.sources == [{
        "kind": "document", "chunk_id": "structure-0001",
        "source": "https://example.test/book.pdf", "title": "Test book",
        "section": "6.2 Executor", "page_start": 6, "page_end": 6,
    }]
    assert turn.citations == [{
        "chunk_id": "structure-0001", "quote": "Executor manages submitted tasks.",
    }]
    assert SQLiteRagChatRepository(database).get(session.id).turns[0] == turn


def test_goal_survives_short_history_and_correction_replaces_constraint(tmp_path: Path) -> None:
    database = tmp_path / "rag.sqlite3"
    interpreter_model = QueueInterpretationModel(
        decision(kind="statement", goal_action="set", goal="Объяснить Executor новичку"),
        decision(kind="statement", changes=[{
            "group": "constraint", "action": "set", "key": "length", "value": "Кратко",
        }]),
        decision(kind="statement", changes=[{
            "group": "constraint", "action": "set", "key": "length", "value": "Подробно",
        }]),
        *[decision() for _ in range(10)],
        decision(scope="goal"),
    )
    index = Index()
    rag = DocumentRag(index, InterpretationModel(), reranker=Reranker(),
                      settings=RetrievalSettings(rewrite=False, candidate_k=1, final_k=1))
    chat = RagChatService(
        SQLiteRagChatRepository(database), TurnInterpreter(interpreter_model),
        rag, Agent(AnswerModel()), history_turns=2,
    )
    session = chat.create()

    goal_turn = chat.send(session.id, "Моя цель — объяснить Executor новичку.")
    chat.send(session.id, "Пиши кратко.")
    correction = chat.send(session.id, "Нет, пиши подробно.")
    for _ in range(10):
        chat.send(session.id, "Что делает Executor?")
    reminder = chat.send(session.id, "Какую цель мы зафиксировали?")

    persisted = SQLiteRagChatRepository(database).get(session.id)
    assert persisted.state.goal.value == "Объяснить Executor новичку"
    assert persisted.state.goal.source_message_id == goal_turn.id
    assert [(item.key, item.value, item.source_message_id)
            for item in persisted.state.constraints] == [("length", "Подробно", correction.id)]
    assert reminder.answer == "Цель: Объяснить Executor новичку"
    assert reminder.sources == [{
        "kind": "message", "message_id": goal_turn.id,
        "title": "Сообщение пользователя · goal",
    }]
    assert len(index.searches) == 11  # ten document questions plus the memory question


def test_failed_answer_can_retry_without_reapplying_state(tmp_path: Path) -> None:
    class FailingOnceAnswerModel(AnswerModel):
        def __init__(self) -> None:
            self.calls = 0

        def generate_json(self, *, messages: list[dict], max_tokens: int = 2000, schema: dict | None = None) -> str:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary provider error")
            return super().generate_json(messages=messages, max_tokens=max_tokens)

    interpreter_model = QueueInterpretationModel(decision(
        goal_action="set", goal="Объяснить Executor новичку",
    ))
    rag = DocumentRag(
        Index(), InterpretationModel(), reranker=Reranker(),
        settings=RetrievalSettings(rewrite=False, candidate_k=1, final_k=1),
    )
    chat = RagChatService(
        SQLiteRagChatRepository(tmp_path / "rag.sqlite3"),
        TurnInterpreter(interpreter_model), rag, Agent(FailingOnceAnswerModel()),
    )
    session = chat.create()

    failed = chat.send(session.id, "Хочу объяснить Executor новичку. Что это?")
    assert failed.status == "failed"
    assert chat.get(session.id).state.revision == 1
    assert len(chat.get(session.id).turns) == 1
    try:
        chat.send(session.id, "Следующий вопрос")
    except ValueError as error:
        assert "Повторите" in str(error)
    else:
        raise AssertionError("A new turn must not make failed retry inaccessible")

    completed = chat.retry(session.id, failed.id)
    assert completed.status == "done"
    assert completed.sources[0]["kind"] == "document"
    assert chat.get(session.id).state.revision == 1
    assert len(chat.get(session.id).turns) == 1
    assert interpreter_model.calls == 1


def test_new_chat_has_no_other_chats_goal_and_ambiguous_change_is_ignored(tmp_path: Path) -> None:
    interpreter_model = QueueInterpretationModel(
        decision(kind="statement", goal_action="set", goal="Изучить Executor"),
        decision(kind="clarification_needed", clarification="Что именно больше не нужно?"),
    )
    rag = DocumentRag(
        Index(), InterpretationModel(), reranker=Reranker(),
        settings=RetrievalSettings(rewrite=False, candidate_k=1, final_k=1),
    )
    chat = RagChatService(
        SQLiteRagChatRepository(tmp_path / "rag.sqlite3"),
        TurnInterpreter(interpreter_model), rag, Agent(AnswerModel()),
    )
    first = chat.create()
    chat.send(first.id, "Моя цель — изучить Executor.")
    second = chat.create()
    clarification = chat.send(first.id, "Это больше не нужно.")

    assert chat.get(second.id).state.goal is None
    assert chat.get(first.id).state.goal.value == "Изучить Executor"
    assert clarification.answer == "Что именно больше не нужно?"
    assert chat.get(first.id).state.revision == 1


def test_dedicated_rag_chat_screen_and_api(tmp_path: Path) -> None:
    rag = DocumentRag(
        Index(), InterpretationModel(), reranker=Reranker(),
        settings=RetrievalSettings(rewrite=False, candidate_k=1, final_k=1),
    )
    chat = RagChatService(
        SQLiteRagChatRepository(tmp_path / "rag.sqlite3"),
        TurnInterpreter(InterpretationModel()), rag, Agent(AnswerModel()),
    )
    app.dependency_overrides[get_rag_chat_service] = lambda: chat
    try:
        client = TestClient(app)
        page = client.get("/rag-chat/")
        assert page.status_code == 200
        assert "RAG-чат" in page.text

        created = client.post("/api/rag-chat/sessions").json()
        session_id = created["id"]
        turn = client.post(
            f"/api/rag-chat/sessions/{session_id}/turns",
            json={"content": "Что делает Executor?"},
        ).json()
        assert turn["sources"][0]["kind"] == "document"
        assert client.get(f"/api/rag-chat/sessions/{session_id}").json()["turns"] == [turn]
        assert len(client.get("/api/rag-chat/sessions").json()) == 1
    finally:
        app.dependency_overrides.clear()


def test_no_relevant_excerpt_abstains_and_still_displays_source_status(tmp_path: Path) -> None:
    class EmptyIndex(Index):
        def search(self, query: str, strategy: str, k: int) -> list[SearchHit]:
            self.searches.append(query)
            return []

    class NoAnswerModel:
        def generate_json(self, *, messages: list[dict], max_tokens: int = 2000, schema: dict | None = None) -> str:
            raise AssertionError("No answer should be generated without excerpts")

    index = EmptyIndex()
    chat = RagChatService(
        SQLiteRagChatRepository(tmp_path / "rag.sqlite3"),
        TurnInterpreter(InterpretationModel()),
        DocumentRag(index, InterpretationModel(), reranker=Reranker(),
                    settings=RetrievalSettings(rewrite=False, candidate_k=1, final_k=1)),
        Agent(NoAnswerModel()),
    )
    session = chat.create()

    turn = chat.send(session.id, "Что делает Executor?")

    assert turn.status == "done"
    assert turn.grounding_status == "insufficient_context"
    assert turn.sources == turn.citations == []
    assert "Не знаю" in turn.answer
    assert index.searches == ["What does Executor do?"]


def test_explicit_new_goal_clears_previous_task_scope(tmp_path: Path) -> None:
    interpreter_model = QueueInterpretationModel(
        decision(kind="statement", goal_action="set", goal="Объяснить Executor",
                 changes=[{"group": "constraint", "action": "set",
                           "key": "audience", "value": "Новички"}]),
        decision(kind="statement", goal_action="set", goal="Изучить Timer"),
    )
    chat = RagChatService(
        SQLiteRagChatRepository(tmp_path / "rag.sqlite3"),
        TurnInterpreter(interpreter_model),
        DocumentRag(Index(), InterpretationModel(), reranker=Reranker(),
                    settings=RetrievalSettings(rewrite=False, candidate_k=1, final_k=1)),
        Agent(AnswerModel()),
    )
    session = chat.create()
    chat.send(session.id, "Хочу объяснить Executor новичкам.")
    changed = chat.send(session.id, "Теперь вместо этого изучаю Timer.")

    assert chat.get(session.id).state.goal.value == "Изучить Timer"
    assert chat.get(session.id).state.goal.source_message_id == changed.id
    assert chat.get(session.id).state.constraints == []


def test_statement_acknowledges_only_actual_state_change(tmp_path: Path) -> None:
    interpreter_model = QueueInterpretationModel(
        decision(kind="statement", goal_action="set", goal="Объяснить Executor"),
        decision(kind="statement"),
    )
    chat = RagChatService(
        SQLiteRagChatRepository(tmp_path / "rag.sqlite3"),
        TurnInterpreter(interpreter_model),
        DocumentRag(Index(), InterpretationModel(), reranker=Reranker(),
                    settings=RetrievalSettings(rewrite=False, candidate_k=1, final_k=1)),
        Agent(AnswerModel()),
    )
    session = chat.create()
    chat.send(session.id, "Моя цель — объяснить Executor.")

    acknowledgement = chat.send(session.id, "Понял.")

    assert acknowledgement.answer == "Принято."


def test_comparison_uses_separate_retrieval_when_joint_terms_filter_out(tmp_path: Path) -> None:
    class SplitIndex(Index):
        def search(self, query: str, strategy: str, k: int) -> list[SearchHit]:
            self.searches.append(query)
            if query == "FutureRenderer CompletionService comparison":
                return []
            if query in {"FutureRenderer", "CompletionService"}:
                return [hit("Executor manages submitted tasks. FutureRenderer CompletionService")]
            return []

    class ComparisonModel(InterpretationModel):
        def generate_json(self, *, messages: list[dict], max_tokens: int = 2000, schema: dict | None = None) -> str:
            payload = decision()
            payload["search_question"] = "FutureRenderer CompletionService comparison"
            return json.dumps(payload)

    index = SplitIndex()
    chat = RagChatService(
        SQLiteRagChatRepository(tmp_path / "rag.sqlite3"),
        TurnInterpreter(ComparisonModel()),
        DocumentRag(index, InterpretationModel(), reranker=Reranker(),
                    settings=RetrievalSettings(rewrite=False, candidate_k=1, final_k=1)),
        Agent(AnswerModel()),
    )

    turn = chat.send(chat.create().id, "Сравни FutureRenderer и CompletionService")

    assert turn.grounding_status == "answered"
    assert index.searches == ["FutureRenderer CompletionService comparison",
                              "FutureRenderer", "CompletionService"]


def test_mixed_question_identifies_unsupported_topic(tmp_path: Path) -> None:
    class MixedIndex(Index):
        def search(self, query: str, strategy: str, k: int) -> list[SearchHit]:
            self.searches.append(query)
            return [hit()] if query == "Executor" else []

    class MixedModel(InterpretationModel):
        def generate_json(self, *, messages: list[dict], max_tokens: int = 2000, schema: dict | None = None) -> str:
            payload = decision()
            payload["search_question"] = "Executor Tokio comparison"
            return json.dumps(payload)

    chat = RagChatService(
        SQLiteRagChatRepository(tmp_path / "rag.sqlite3"),
        TurnInterpreter(MixedModel()),
        DocumentRag(MixedIndex(), InterpretationModel(), reranker=Reranker(),
                    settings=RetrievalSettings(rewrite=False, candidate_k=1, final_k=1)),
        Agent(AnswerModel()),
    )

    turn = chat.send(chat.create().id, "Что книга говорит об Executor и Tokio?")

    assert turn.grounding_status == "answered"
    assert turn.sources[0]["kind"] == "document"
    assert "Tokio" in turn.answer
    assert "нет подтверждения" in turn.answer


def test_question_without_model_search_query_uses_user_question() -> None:
    payload = decision(scope="state")
    payload["search_question"] = ""
    interpreter = TurnInterpreter(QueueInterpretationModel(payload))

    result = interpreter.interpret("Какова моя цель?", RagTaskState(), [])

    assert result.kind == "question"
    assert result.question_scope == "state"
    assert result.search_question == "Какова моя цель?"


def test_cost_question_expands_search_after_unsupported_first_excerpts(tmp_path: Path) -> None:
    class CostIndex(Index):
        def search(self, query: str, strategy: str, k: int) -> list[SearchHit]:
            self.searches.append(query)
            if "resource management" in query:
                result = hit("Creating threads has resource overhead.")
                result.metadata["chunk_id"] = "structure-0002"
                return [result]
            return [hit("ThreadPerTaskWebServer creates a thread for each task.")]

    class CostInterpreter(InterpretationModel):
        def generate_json(self, *, messages: list[dict], max_tokens: int = 2000, schema: dict | None = None) -> str:
            payload = decision()
            payload["search_question"] = "What are costs in ThreadPerTaskWebServer?"
            return json.dumps(payload)

    class CostAnswerModel:
        def generate_json(self, *, messages: list[dict], max_tokens: int = 2000, schema: dict | None = None) -> str:
            if "structure-0002" not in messages[0]["content"]:
                return json.dumps({"status": "unknown", "claims": []})
            return json.dumps({"status": "answered", "claims": [{
                "text": "Создание потоков расходует ресурсы.",
                "evidence": [{"chunk_id": "structure-0002",
                              "quote": "Creating threads has resource overhead."}],
            }]})

    index = CostIndex()
    chat = RagChatService(
        SQLiteRagChatRepository(tmp_path / "rag.sqlite3"),
        TurnInterpreter(CostInterpreter()),
        DocumentRag(index, InterpretationModel(), reranker=Reranker(),
                    settings=RetrievalSettings(rewrite=False, candidate_k=1, final_k=1)),
        Agent(CostAnswerModel()),
    )

    turn = chat.send(chat.create().id, "Каковы издержки ThreadPerTaskWebServer?")

    assert turn.grounding_status == "answered"
    assert turn.sources[0]["chunk_id"] == "structure-0002"
    assert len(index.searches) == 2
    assert "resource management" in index.searches[1]


def test_document_answer_uses_standalone_question_without_prior_assistant_text(
    tmp_path: Path,
) -> None:
    class RecordingAnswerModel(AnswerModel):
        def __init__(self) -> None:
            self.messages: list[dict] = []

        def generate_json(self, *, messages: list[dict], max_tokens: int = 2000, schema: dict | None = None) -> str:
            self.messages = messages
            return super().generate_json(messages=messages, max_tokens=max_tokens)

    model = RecordingAnswerModel()
    chat = RagChatService(
        SQLiteRagChatRepository(tmp_path / "rag.sqlite3"),
        TurnInterpreter(QueueInterpretationModel(
            decision(kind="statement", goal_action="set", goal="Изучить Executor"),
            decision(),
        )),
        DocumentRag(Index(), InterpretationModel(), reranker=Reranker(),
                    settings=RetrievalSettings(rewrite=False, candidate_k=1, final_k=1)),
        Agent(model),
    )
    session = chat.create()
    chat.send(session.id, "Хочу изучить Executor.")

    answered = chat.send(session.id, "Как он работает?")

    assert answered.grounding_status == "answered"
    assert [message["role"] for message in model.messages] == ["system", "user"]
    assert json.loads(model.messages[1]["content"]) == {
        "user_question": "Как он работает?",
        "standalone_search_question": "What does Executor do?",
    }
    assert "Изучить Executor" in model.messages[0]["content"]


def test_invalid_quote_retry_receives_validation_feedback(tmp_path: Path) -> None:
    class RepairingAnswerModel:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def generate_json(self, *, messages: list[dict], max_tokens: int = 2000, schema: dict | None = None) -> str:
            self.calls.append(messages[0]["content"])
            quote = ("Invented excerpt without source support."
                     if len(self.calls) < 4 else "Executor manages submitted tasks.")
            return json.dumps({"status": "answered", "claims": [{
                "text": "Executor выполняет задачи.",
                "evidence": [{"chunk_id": "structure-0001", "quote": quote}],
            }]})

    model = RepairingAnswerModel()
    chat = RagChatService(
        SQLiteRagChatRepository(tmp_path / "rag.sqlite3"),
        TurnInterpreter(InterpretationModel()),
        DocumentRag(Index(), InterpretationModel(), reranker=Reranker(),
                    settings=RetrievalSettings(rewrite=False, candidate_k=1, final_k=1)),
        Agent(model),
    )

    turn = chat.send(chat.create().id, "Что делает Executor?")

    assert turn.grounding_status == "answered"
    assert len(model.calls) == 4
    assert "Цитата отсутствует" in model.calls[1]
