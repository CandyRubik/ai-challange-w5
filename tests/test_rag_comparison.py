"""Live profile selection and durable paired generation through the RAG API."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app.agents.agent import Agent
from app.indexing.rag import DocumentRag, RetrievalSettings
from app.main import app, get_rag_chat_service
from app.providers.errors import LlmRequestError
from app.providers.ollama import OllamaProvider
from app.providers.registry import ModelRegistry
from app.rag_chat.profiles import PROFILES
from app.rag_chat.service import RagChatService
from app.rag_chat.state import TurnInterpreter
from app.rag_chat.store import SQLiteRagChatRepository
from tests.test_rag_chat import Index, Reranker, InterpretationModel, AnswerModel


class NativeRegistry(ModelRegistry):
    def __init__(self, fail_candidate=False, on_reference=None):
        super().__init__()
        self.calls = []
        self.fail_candidate = fail_candidate
        self.on_reference = on_reference
        self.context = 0

    def build(self, selection, **kwargs):
        def handler(request):
            payload = json.loads(request.content)
            self.calls.append(payload)
            self.context = payload['options']['num_ctx']
            if payload['options']['num_predict'] == 700:
                content = InterpretationModel().generate_json(messages=[])
            else:
                if payload['options']['num_predict'] == 3000 and self.on_reference:
                    self.on_reference()
                if self.fail_candidate and payload['options']['num_predict'] == 1000:
                    raise httpx.ConnectError('temporary')
                content = AnswerModel().generate_json(messages=[])
            return httpx.Response(200, json={'message': {'content': content}, 'done': True,
                                            'done_reason': 'stop', 'eval_count': 20,
                                            'eval_duration': 1_000_000_000, 'load_duration': 10_000_000})
        kwargs.pop('thinking_enabled', None)
        return OllamaProvider(model=selection.model, client=httpx.Client(transport=httpx.MockTransport(handler)),
                              **kwargs)

    def installed_local_models(self):
        return {name: {'details': {'quantization_level': quant}} for name, quant in (
            ('qwen3.5:9b-q4_K_M', 'Q4_K_M'), ('qwen3.5:9b-q8_0', 'Q8_0'))}

    def loaded_local_model(self, model):
        return {'size_vram': self.context * 1000, 'context_length': self.context,
                'details': {'quantization_level': 'Q8_0' if 'q8_0' in model else 'Q4_K_M'}}


def service(tmp_path, registry, index=None):
    index = index or Index()
    return RagChatService(SQLiteRagChatRepository(tmp_path / 'compare.sqlite3'),
                          TurnInterpreter(InterpretationModel()),
                          DocumentRag(index, InterpretationModel(), reranker=Reranker(),
                                      settings=RetrievalSettings(rewrite=False, candidate_k=1, final_k=1)),
                          Agent(AnswerModel()), model_registry=registry, generation_profile=PROFILES['optimized'])


def test_pair_reuses_interpretation_evidence_and_memory(tmp_path):
    registry = NativeRegistry()
    index = Index()
    chat = service(tmp_path, registry, index)
    turn = chat.send(chat.create().id, 'Что делает Executor?', compare_with='baseline')

    assert turn.status == 'done'
    assert turn.metrics['generation_attempts'] == 2
    assert len(turn.metrics['ollama_requests']) == 3
    assert index.searches == ['What does Executor do?']
    assert [c['options']['num_predict'] for c in registry.calls] == [700, 3000, 1000]
    reference, candidate = (turn.metrics['comparison'][key] for key in ('reference', 'candidate'))
    assert reference['context_sha256'] == candidate['context_sha256']
    assert reference['profile']['num_ctx'] == 32768
    assert candidate['profile']['num_ctx'] == 8192
    assert reference['metrics']['tokens_per_second'] == 20
    assert reference['metrics']['model_allocation']['size_vram'] > candidate['metrics']['model_allocation']['size_vram']
    prompts = [c['messages'][0]['content'].split('DOCUMENT_EXCERPTS:\n')[-1] for c in registry.calls[1:]]
    assert prompts[0] == prompts[1]
    assert registry.calls[1]['messages'][1] == registry.calls[2]['messages'][1]
    assert chat.get(turn.session_id).turns[0].metrics['comparison'] == turn.metrics['comparison']


def test_switch_during_pair_affects_only_following_turn(tmp_path):
    registry = NativeRegistry()
    chat = service(tmp_path, registry)
    session = chat.create()
    registry.on_reference = lambda: chat.set_configuration(session.id, 'q8')
    turn = chat.send(session.id, 'Что делает Executor?', compare_with='baseline')
    assert turn.model == 'qwen3.5:9b-q4_K_M'
    assert turn.metrics['comparison']['candidate']['configuration_id'] == 'optimized'
    assert chat.get(session.id).configuration == 'q8'
    following = chat.send(session.id, 'Что делает Executor?')
    assert following.model == 'qwen3.5:9b-q8_0'
    assert following.metrics['configuration_id'] == 'q8'


def test_failed_pair_retries_only_missing_leg_on_frozen_context(tmp_path, monkeypatch):
    original = service(tmp_path, NativeRegistry(fail_candidate=True))
    session = original.create()
    failed = original.send(session.id, 'Что делает Executor?', compare_with='baseline')
    assert failed.status == 'failed'
    assert set(failed.metrics['comparison']) == {'reference'}
    digest = failed.metrics['comparison_context']['sha256']
    original.set_configuration(session.id, 'q8')
    monkeypatch.setenv('OLLAMA_MODEL', 'different-after-restart')

    class NoRetrieval(Index):
        def search(self, *args, **kwargs):
            pytest.fail('Retry must use durable evidence')
    registry = NativeRegistry()
    restarted = service(tmp_path, registry, NoRetrieval())
    result = restarted.retry(session.id, failed.id)
    assert result.status == 'done'
    assert result.metrics['comparison_reused'] == ['reference']
    assert result.metrics['generation_attempts'] == 1
    assert len(result.metrics['ollama_requests']) == 1
    assert len(registry.calls) == 1
    assert registry.calls[0]['model'] == 'qwen3.5:9b-q4_K_M'
    assert registry.calls[0]['options']['num_predict'] == 1000
    assert result.metrics['comparison_context']['sha256'] == digest
    assert result.metrics['comparison']['reference'] == failed.metrics['comparison']['reference']


def test_no_hits_reports_generation_skipped_for_both_variants(tmp_path):
    class EmptyIndex(Index):
        def search(self, *args, **kwargs):
            return []
    registry = NativeRegistry()
    chat = service(tmp_path, registry, EmptyIndex())
    turn = chat.send(chat.create().id, 'Что делает Executor?', compare_with='baseline')
    assert turn.status == 'done'
    assert len(registry.calls) == 1
    for result in turn.metrics['comparison'].values():
        assert result['answer']['status'] == 'insufficient_context'
        assert result['metrics']['generation_attempts'] == 0
        assert result['metrics']['model_allocation'] is None


def test_api_configuration_catalog_persistence_and_validation(tmp_path):
    chat = service(tmp_path, NativeRegistry())
    app.dependency_overrides[get_rag_chat_service] = lambda: chat
    try:
        with TestClient(app) as client:
            catalog = client.get('/api/rag-chat/configurations').json()
            assert catalog['default'] == 'optimized'
            assert len(catalog['configurations']) == 4
            assert all(c['available'] for c in catalog['configurations'])
            session = client.post('/api/rag-chat/sessions', json={'configuration': 'baseline'}).json()
            url = '/api/rag-chat/sessions/' + session['id']
            assert session['configuration'] == 'baseline'
            assert client.put(url + '/configuration', json={'configuration': 'q8'}).json()['configuration'] == 'q8'
            assert client.get(url).json()['configuration'] == 'q8'
            assert client.put(url + '/configuration', json={'configuration': 'evil'}).status_code == 422
            assert client.post(url + '/turns', json={'content': 'Executor', 'compare_with': 'evil'}).status_code == 422
            result = client.post(url + '/turns', json={'content': 'Executor', 'compare_with': 'baseline'}).json()
            assert result['model'] == 'qwen3.5:9b-q8_0'
            assert result['metrics']['comparison']['reference']['profile']['num_ctx'] == 32768
            assert client.get(url).json()['turns'][0] == result
    finally:
        app.dependency_overrides.clear()


def test_cloud_comparison_is_rejected_before_turn_creation(tmp_path):
    chat = service(tmp_path, NativeRegistry())
    session = chat.create('deepseek')
    with pytest.raises(ValueError, match='локальной'):
        chat.send(session.id, 'Executor', compare_with='baseline')
    assert chat.get(session.id).turns == []


def test_allocation_from_another_context_is_not_attributed_to_answer():
    from app.rag_chat.comparison import generation_statistics
    class Registry:
        def loaded_local_model(self, model):
            return {'context_length': 32768, 'size_vram': 7_000_000_000}
    result = generation_statistics([{'options': {'num_ctx': 8192}, 'eval_count': 20,
                                     'eval_duration': 1_000_000_000}], Registry(), 'local')
    assert result['tokens_per_second'] == 20
    assert result['model_allocation'] is None


def test_comparison_request_for_statement_preserves_normal_task_memory(tmp_path):
    registry = NativeRegistry()
    chat = service(tmp_path, registry)
    session = chat.create()
    from tests.test_rag_chat import decision
    payload = decision(kind='statement', goal_action='set', goal='Изучить Executor')
    native_build = registry.build
    def build(selection, **kwargs):
        model = native_build(selection, **kwargs)
        class StatementModel:
            def generate_json(self, **call):
                return json.dumps(payload)
        return StatementModel()
    registry.build = build
    result = chat.send(session.id, 'Хочу изучить Executor', compare_with='baseline')
    assert result.status == 'done' and result.kind == 'statement'
    assert result.metrics['comparison_note']
    assert chat.get(session.id).state.goal.value == 'Изучить Executor'
    assert not result.metrics.get('comparison')
