import json
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import pytest

from app.agents.agent import Agent
from app.indexing.rag import DocumentRag, RetrievalSettings
from app.indexing.store import DocumentIndex, DocumentIndexError, MODEL_NAME
from app.indexing.rerank import LocalCrossEncoderReranker
from app.providers.errors import LlmRequestError
from app.providers.registry import ModelRegistry
from app.rag_chat.service import RagChatService
from app.rag_chat.state import TurnInterpreter
from app.rag_chat.store import SQLiteRagChatRepository, RagTurnConflict
from app.rag_chat.citation_options import prepare_citations, resolve_citations
from app.indexing.grounding import GroundingError, validate_grounded_answer
from tests.test_rag_chat import Index, Reranker, InterpretationModel, AnswerModel


class Model:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    def generate_json(self, *, messages, max_tokens=2000, schema=None):
        self.calls.append(messages)
        if "You interpret ONE" in messages[0]["content"]:
            return InterpretationModel().generate_json(messages=messages)
        if self.fail:
            raise LlmRequestError("temporary local failure")
        return AnswerModel().generate_json(messages=messages)


class Registry(ModelRegistry):
    def __init__(self, local, cloud):
        super().__init__()
        self.models = {"ollama": local, "deepseek": cloud}
        self.selections = []

    def build(self, selection, **kwargs):
        self.selections.append(selection)
        return self.models[selection.provider]


def service(path, registry):
    model = registry.models["ollama"]
    return RagChatService(
        SQLiteRagChatRepository(path), TurnInterpreter(model),
        DocumentRag(Index(), model, reranker=Reranker(),
                    settings=RetrievalSettings(rewrite=False, candidate_k=1, final_k=1)),
        Agent(model), model_registry=registry,
    )


def test_local_turn_uses_no_cloud_and_records_provenance(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    local, cloud = Model(), Model()
    registry = Registry(local, cloud)
    chat = service(tmp_path / "rag.sqlite3", registry)
    session = chat.create()
    turn = chat.send(session.id, "Что делает Executor?")
    assert turn.status == "done" and turn.grounding_status == "answered"
    assert turn.provider == "ollama" and turn.model == "qwen3.5:9b-q4_K_M"
    assert len(local.calls) == 2 and not cloud.calls
    assert turn.metrics["generation_attempts"] == 1
    assert turn.metrics["total_seconds"] >= turn.metrics["generation_seconds"]


def test_retry_after_restart_keeps_original_model_and_provider(tmp_path, monkeypatch):
    path = tmp_path / "rag.sqlite3"
    cloud = Model()
    chat = service(path, Registry(Model(fail=True), cloud))
    session = chat.create()
    failed = chat.send(session.id, "Что делает Executor?")
    assert failed.status == "failed"
    chat.set_provider(session.id, "deepseek")
    monkeypatch.setenv("OLLAMA_MODEL", "another-local-model")
    registry = Registry(Model(), cloud)
    restarted = service(path, registry)
    completed = restarted.retry(session.id, failed.id)
    assert completed.status == "done" and not cloud.calls
    assert registry.selections[0].model == "qwen3.5:9b-q4_K_M"
    assert restarted.get(session.id).provider == "deepseek"
    assert len(registry.models["ollama"].calls) == 1  # Saved interpretation is reused.


def test_explicit_turn_selection_preserves_chat_preference(tmp_path):
    registry = Registry(Model(), Model())
    chat = service(tmp_path / "rag.sqlite3", registry)
    session = chat.create("deepseek")
    local = chat.send(session.id, "Что делает Executor?", "ollama")
    cloud = chat.send(session.id, "Что делает Executor?")
    assert local.provider == "ollama" and cloud.provider == "deepseek"
    assert chat.get(session.id).provider == "deepseek"


def test_repository_prevents_two_unfinished_turns(tmp_path):
    first = SQLiteRagChatRepository(tmp_path / "rag.sqlite3")
    second = SQLiteRagChatRepository(first.database_path)
    session = first.create()
    first.start_turn(session.id, "Первый")
    with pytest.raises(RagTurnConflict):
        second.start_turn(session.id, "Второй")


def test_embedding_and_reranker_never_download_missing_weights(tmp_path, monkeypatch):
    calls = []
    def missing(*args, **kwargs):
        calls.append(kwargs)
        raise OSError("missing cache")
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=missing, CrossEncoder=missing))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(set_num_threads=lambda n: None, nn=SimpleNamespace(Sigmoid=lambda: None)))
    with pytest.raises(DocumentIndexError, match="Локальные веса"):
        DocumentIndex(tmp_path)._model()
    with pytest.raises(DocumentIndexError, match="Локальные веса"):
        LocalCrossEncoderReranker().score("Executor", Index().search("", "structure", 1))
    assert len(calls) == 2 and all(c["local_files_only"] is True for c in calls)


def test_index_rejects_a_different_embedding_model(tmp_path):
    (tmp_path / "active.json").write_text(json.dumps({"model": "different-e5", "version": "old"}))
    with pytest.raises(DocumentIndexError, match="другой моделью"):
        DocumentIndex(tmp_path).status()


def test_model_switch_during_answer_affects_only_next_turn(tmp_path):
    started, release = Event(), Event()
    class Blocking(Model):
        def generate_json(self, **kwargs):
            if "DOCUMENT_EXCERPTS" in kwargs["messages"][0]["content"]:
                started.set()
                assert release.wait(5)
            return super().generate_json(**kwargs)
    registry = Registry(Blocking(), Model())
    chat = service(tmp_path / "rag.sqlite3", registry)
    session = chat.create()
    with ThreadPoolExecutor() as pool:
        future = pool.submit(chat.send, session.id, "Что делает Executor?")
        assert started.wait(5)
        chat.set_provider(session.id, "deepseek")
        release.set()
        completed = future.result()
    assert completed.provider == "ollama" and not registry.models["deepseek"].calls
    assert chat.get(session.id).provider == "deepseek"


def test_cloud_failure_does_not_fall_back_to_local(tmp_path):
    registry = Registry(Model(), Model(fail=True))
    chat = service(tmp_path / "rag.sqlite3", registry)
    session = chat.create("deepseek")
    failed = chat.send(session.id, "Что делает Executor?")
    assert failed.status == "failed" and failed.provider == "deepseek"
    assert not registry.models["ollama"].calls


def test_legacy_database_keeps_cloud_provenance(tmp_path):
    path = tmp_path / "rag.sqlite3"
    original = SQLiteRagChatRepository(path)
    session = original.create()
    original.start_turn(session.id, "Старый вопрос")
    with sqlite3.connect(path) as connection:
        connection.execute("ALTER TABLE rag_chat_sessions DROP COLUMN provider")
        for column in ("provider", "model", "metrics_json"):
            connection.execute(f"ALTER TABLE rag_chat_turns DROP COLUMN {column}")
    migrated = SQLiteRagChatRepository(path).get(session.id)
    assert migrated.provider == migrated.turns[0].provider == "deepseek"
    assert migrated.turns[0].model is None and migrated.turns[0].metrics == {}


def test_quote_selection_attaches_exact_pdf_text():
    hits = Index().search("Executor", "structure", 1)
    prompt, schema, lookup = prepare_citations(hits)
    key = next(iter(lookup))
    assert key in schema["$defs"]["CitedClaim"]["properties"]["evidence_ids"]["items"]["enum"]
    raw = json.dumps({"status": "answered", "claims": [{"text": "Executor выполняет задачи.", "evidence_ids": [key]}]})
    answer = validate_grounded_answer(resolve_citations(raw, lookup), hits)
    assert answer.citations[0]["quote"] == hits[0].metadata["text"]
    with pytest.raises(GroundingError):
        resolve_citations(raw.replace(key, "invented-quote"), lookup)


def test_old_draft_cannot_bypass_exact_quote_validation():
    hits = Index().search("", "structure", 1)
    _, _, lookup = prepare_citations(hits)
    raw = json.dumps({"status": "answered", "claims": [{"text": "Выдумка", "evidence": [{
        "chunk_id": hits[0].metadata["chunk_id"], "quote": "Invented sentence not present in the book.",
    }]}]})
    with pytest.raises(GroundingError):
        validate_grounded_answer(resolve_citations(raw, lookup), hits)
