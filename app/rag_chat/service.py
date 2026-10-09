"""One user turn through task memory, retrieval, grounding, and durable history."""

from __future__ import annotations

import logging
from dataclasses import asdict
import re
from threading import Lock
from time import monotonic

from ..agents.agent import Agent
from ..indexing.grounding import GroundedAnswer
from ..indexing.rag import DocumentRag
from ..indexing.store import DocumentIndexError, SearchHit
from ..memory.context import MemoryContext
from ..providers.errors import LlmConfigurationError, LlmRequestError
from ..providers.registry import ModelRegistry
from .generation import GROUNDING_SYSTEM_PROMPT, generate_grounded_answer
from .models import RagSession, RagSessionSummary, RagTaskState, RagTurn, StateFact
from .profiles import (RagGenerationProfile, PROFILES, CONFIGURATION_LABELS,
                       configuration_profile, configuration_model)
from .comparison import freeze_context, run_comparison, generation_statistics
from .state import InterpretationError, TurnDecision, TurnInterpreter, apply_decision
from .store import RagTurnConflict, SQLiteRagChatRepository


logger = logging.getLogger(__name__)


class RagChatService:
    """The dedicated chat's public interface; one session is serialized per turn."""

    def __init__(
        self, repository: SQLiteRagChatRepository, interpreter: TurnInterpreter,
        rag: DocumentRag, agent: Agent, *, history_turns: int = 4,
        model_registry: ModelRegistry | None = None,
        generation_profile: RagGenerationProfile | None = None,
    ) -> None:
        self._repository = repository
        self._interpreter = interpreter
        self._rag = rag
        self._agent = agent
        self._history_turns = history_turns
        self._models = model_registry
        self._provider = None
        self._generation_profile = generation_profile
        self._runtime_registry = None
        self._locks: dict[str, Lock] = {}
        self._locks_guard = Lock()

    def _lock(self, session_id: str) -> Lock:
        with self._locks_guard:
            return self._locks.setdefault(session_id, Lock())

    def create(self, provider: str | None = None, configuration: str | None = None) -> RagSession:
        provider = provider or (self._models.default_provider if self._models else "ollama")
        if self._models:
            self._models.resolve(provider)
        if configuration:
            configuration_profile(configuration)
        return self._repository.create(provider, configuration)

    def _default_configuration(self):
        return self._generation_profile.name if self._generation_profile else "baseline"

    def configuration_catalog(self) -> dict:
        installed = self._models.installed_local_models() if self._models else {}
        items = []
        for name, label in CONFIGURATION_LABELS.items():
            profile = configuration_profile(name)
            model = configuration_model(self._models, name).model if self._models else ""
            items.append({"id": name, "label": label, "model": model, "profile": asdict(profile),
                          "available": model in installed,
                          "quantization": installed.get(model, {}).get("details", {}).get("quantization_level")})
        return {"default": self._default_configuration(), "configurations": items}

    def set_configuration(self, session_id: str, configuration: str) -> RagSession:
        configuration_profile(configuration)
        if self._models is None:
            raise ValueError("Для выбора конфигурации нужен реестр моделей")
        self._repository.set_configuration(session_id, configuration)
        return self.get(session_id)

    def set_provider(self, session_id: str, provider: str) -> RagSession:
        if self._models:
            self._models.resolve(provider)
        self._repository.set_provider(session_id, provider)
        return self.get(session_id)

    def _for_turn(self, turn: RagTurn) -> "RagChatService":
        if self._models is None:
            return self
        selection = self._models.resolve(turn.provider, turn.model)
        # Legacy pending turns retain the original baseline configuration.
        saved_profile = turn.metrics.get("generation_profile")
        profile = RagGenerationProfile(**saved_profile) if saved_profile else PROFILES["baseline"]
        if selection.provider == "ollama" and saved_profile:
            model = self._models.build(
                selection, thinking_enabled=False, num_ctx=profile.num_ctx,
                max_num_ctx=profile.max_num_ctx, temperature=profile.temperature,
            )
        else:
            model = self._models.build(selection, thinking_enabled=False)
        runtime = RagChatService(
            self._repository, TurnInterpreter(model),
            DocumentRag(self._rag.index, model, reranker=self._rag.reranker, settings=self._rag.settings),
            Agent(model, system_prompt=GROUNDING_SYSTEM_PROMPT, max_tokens=profile.max_tokens),
            history_turns=self._history_turns,
            generation_profile=profile,
        )
        runtime._provider = model
        runtime._runtime_registry = self._models
        return runtime

    def list(self) -> list[RagSessionSummary]:
        return self._repository.list()

    def get(self, session_id: str) -> RagSession:
        return self._repository.get(session_id)

    def delete(self, session_id: str) -> None:
        with self._lock(session_id):
            self._repository.delete(session_id)

    def send(self, session_id: str, content: str, provider: str | None = None, *,
             configuration: str | None = None, compare_with: str | None = None) -> RagTurn:
        if not content.strip() or len(content) > 12_000:
            raise ValueError("Сообщение должно содержать от 1 до 12000 символов")
        with self._lock(session_id):
            session = self._repository.get(session_id)
            if session.turns and session.turns[-1].status != "done":
                raise RagTurnConflict("Повторите последний ответ перед новым сообщением")
            selected = self._models.resolve(provider or session.provider) if self._models else None
            config = configuration or session.configuration or self._default_configuration()
            metrics = {}
            if selected and selected.provider == "ollama":
                profile = configuration_profile(config)
                selected = configuration_model(self._models, config)
                metrics = {"generation_profile": asdict(profile), "configuration_id": config}
            elif configuration or compare_with:
                raise ValueError("Конфигурации и сравнение доступны для локальной модели")
            if compare_with:
                if compare_with == config:
                    raise ValueError("Выберите разные конфигурации для сравнения")
                reference = configuration_model(self._models, compare_with)
                metrics["comparison_request"] = {
                    "reference": {"configuration_id": compare_with, "label": CONFIGURATION_LABELS[compare_with],
                                  "model": reference.model, "profile": asdict(configuration_profile(compare_with))},
                    "candidate": {"configuration_id": config, "label": CONFIGURATION_LABELS[config],
                                  "model": selected.model, "profile": asdict(profile)},
                }
            turn = self._repository.start_turn(
                session_id, content, provider=selected.provider if selected else session.provider,
                model=selected.model if selected else None,
                metrics=metrics,
            )
            return self._process(session, turn)

    def retry(self, session_id: str, turn_id: str) -> RagTurn:
        with self._lock(session_id):
            session = self._repository.get(session_id)
            turn = next((item for item in session.turns if item.id == turn_id), None)
            if turn is None:
                raise RagTurnConflict("Сообщение не принадлежит этому RAG-чату")
            if turn != session.turns[-1] or turn.status not in {"pending", "failed"}:
                raise RagTurnConflict("Повторить можно только последнее сообщение без ответа")
            return self._process(session, turn)

    def _process(self, session: RagSession, turn: RagTurn) -> RagTurn:
        started = monotonic()
        metrics = {"interpretation_seconds": 0, "retrieval_seconds": 0,
                   "generation_seconds": 0, "generation_attempts": 0}
        for key in ("generation_profile", "configuration_id", "comparison_request", "comparison_context", "comparison"):
            if key in turn.metrics:
                metrics[key] = turn.metrics[key]
        runtime = None
        def snapshot() -> dict:
            metrics["total_seconds"] = monotonic() - started
            metrics["ollama_requests"] = list(getattr(runtime._provider, "request_metrics", [])) if runtime else []
            for leg, result in metrics.get("comparison", {}).items():
                if leg not in metrics.get("comparison_reused", []):
                    metrics["ollama_requests"] += result["metrics"]["ollama_requests"]
            return metrics
        try:
            runtime = self._for_turn(turn)
            decision = self._repository.decision(turn.id)
            if decision is None:
                interpreted = monotonic()
                decision = runtime._interpreter.interpret(
                    turn.content, session.state, self._recent_payload(session),
                )
                metrics["interpretation_seconds"] = monotonic() - interpreted
                state = apply_decision(session.state, decision, turn.id)
                turn = self._repository.save_decision(
                    turn.id, decision, state, session.state.revision,
                )
            else:
                state = session.state
            answer, sources, citations, grounding_status = runtime._answer(
                session, turn, decision, state, metrics,
            )
            return self._repository.finish_turn(
                turn.id, answer, sources=sources, citations=citations,
                grounding_status=grounding_status, metrics=snapshot(),
            )
        except Exception as error:
            logger.warning("RAG chat turn failed: %s", type(error).__name__, exc_info=True)
            return self._repository.fail_turn(turn.id, self._public_error(error), metrics=snapshot())

    @staticmethod
    def _public_error(error: Exception) -> str:
        if isinstance(error, (InterpretationError, DocumentIndexError, LlmConfigurationError, LlmRequestError)):
            return str(error)
        return "Не удалось ответить. Повторите попытку."

    def _recent_payload(self, session: RagSession) -> list[dict]:
        return [
            {"user": turn.content, "assistant": turn.answer}
            for turn in session.turns if turn.status == "done"
        ][-self._history_turns:]

    @staticmethod
    def _memory(state: RagTaskState) -> MemoryContext:
        entries = []
        if state.goal is not None:
            entries.append({"category": "goal", "content": state.goal.value})
        for category, facts in (
            ("constraint", state.constraints),
            ("term", state.terms),
            ("clarification", state.clarifications),
        ):
            entries.extend(
                {"category": category, "content": f"{fact.key}: {fact.value}"}
                for fact in facts
            )
        return MemoryContext(working=tuple(entries))

    @staticmethod
    def _message_source(fact: StateFact) -> dict:
        return {
            "kind": "message", "message_id": fact.source_message_id,
            "title": f"Сообщение пользователя · {fact.key}",
        }

    def _state_answer(
        self, decision: TurnDecision, state: RagTaskState,
    ) -> tuple[str, list[dict]]:
        if decision.question_scope == "goal":
            if state.goal is None:
                return "Цель пока не зафиксирована.", []
            return f"Цель: {state.goal.value}", [self._message_source(state.goal)]
        facts = ([state.goal] if state.goal is not None else []) + [
            *state.constraints, *state.terms, *state.clarifications,
        ]
        if not facts:
            return "Договорённости пока не зафиксированы.", []
        lines = [f"{fact.key}: {fact.value}" for fact in facts]
        sources = list({
            fact.source_message_id: self._message_source(fact) for fact in facts
        }.values())
        return "\n".join(lines), sources

    def _document_answer(
        self, turn: RagTurn, decision: TurnDecision, state: RagTaskState,
        hits: list[SearchHit], metrics: dict | None = None,
    ) -> GroundedAnswer:
        if metrics is not None and metrics.get("comparison_request"):
            if "comparison_context" not in metrics:
                metrics["comparison_context"] = freeze_context(
                    turn.content, decision.search_question, hits, self._memory(state),
                )
            results = metrics.setdefault("comparison", {})
            reused = set(results)
            metrics["comparison_reused"] = sorted(reused)
            def save():
                fresh = [result for leg, result in results.items() if leg not in reused]
                metrics["generation_attempts"] = sum(r["metrics"]["generation_attempts"] for r in fresh)
                metrics["generation_seconds"] = sum(r["elapsed_seconds"] for r in fresh)
                self._repository.save_metrics(turn.id, metrics)
            save()
            return run_comparison(self._runtime_registry, metrics["comparison_request"],
                                  metrics["comparison_context"], results, save)
        offset = len(getattr(self._provider, "request_metrics", []))
        answer = generate_grounded_answer(
            self._agent, turn.content, decision.search_question, hits,
            memory=self._memory(state), metrics=metrics,
            prompt_version=self._generation_profile.prompt_version if self._generation_profile else "baseline",
        )
        if metrics is not None:
            requests = list(getattr(self._provider, "request_metrics", []))[offset:]
            metrics["generation_statistics"] = generation_statistics(
                requests, self._runtime_registry, turn.model,
            )
        return answer

    @staticmethod
    def _named_terms(query: str) -> tuple[str, ...]:
        # Prefix a space so a leading identifier is not treated as an ordinary
        # sentence-initial word by the shared retrieval helper.
        return tuple(term for term in DocumentRag._required_terms(" " + query)
                     if len(term) > 2 and term.casefold() not in {
                         "compare", "describe", "summarize", "discuss", "define",
                     })

    def retrieve_context(self, decision: TurnDecision) -> list[SearchHit]:
        hits = self._rag.retrieve(
            decision.search_question, [], "structure", force=True,
        )
        if hits or decision.question_scope != "document":
            return hits
        terms = self._named_terms(decision.search_question)
        if len(terms) < 2:
            return hits
        # The existing lexical filter requires every named identifier in one
        # chunk. A comparison may need passages from separate sections.
        collected: dict[str, SearchHit] = {}
        for term in terms[:3]:
            for hit in self._rag.retrieve(term, [], "structure", force=True):
                collected.setdefault(hit.metadata["chunk_id"], hit)
        return list(collected.values())[:self._rag.settings.final_k * 2]

    def _expand_cost_hits(
        self, decision: TurnDecision, hits: list[SearchHit],
    ) -> list[SearchHit]:
        if not hits or not re.search(
            r"\b(costs?|drawbacks?|overhead|limitations?)\b",
            decision.search_question, flags=re.I,
        ):
            return hits
        broad_query = decision.search_question
        for term in self._named_terms(broad_query):
            broad_query = re.sub(rf"\b{re.escape(term)}\b", "", broad_query, flags=re.I)
        broad_query = broad_query.strip(" ?.!")
        if len(broad_query) < 12:
            return hits
        related = self._rag.retrieve(
            broad_query + " drawbacks overhead resource management",
            [], "structure", force=True,
        )
        unique = {hit.metadata["chunk_id"]: hit for hit in [*hits, *related]}
        return list(unique.values())[:self._rag.settings.final_k * 2]

    def _answer(
        self, session: RagSession, turn: RagTurn, decision: TurnDecision,
        state: RagTaskState, metrics: dict | None = None,
    ) -> tuple[str, list[dict], list[dict], str]:
        metrics = metrics if metrics is not None else {}
        if metrics.get("comparison_request") and (decision.kind != "question" or decision.question_scope != "document"):
            metrics["comparison_note"] = "Сравнение доступно для вопросов по книге; это сообщение обработано как обычный ход."
        if decision.kind == "clarification_needed":
            return decision.clarification, [self._message_source(StateFact(
                key="уточнение", value=turn.content, source_message_id=turn.id,
            ))], [], "clarification_needed"
        if decision.kind == "statement":
            content = (
                "Уточнение сохранено."
                if state.revision > session.state.revision else "Принято."
            )
            return content, [self._message_source(StateFact(
                key="уточнение", value=turn.content, source_message_id=turn.id,
            ))], [], "statement"

        # Every question searches the indexed corpus, including questions about
        # the user's own agreements. Only relevant evidence appears in the answer.
        metrics = metrics if metrics is not None else {}
        started = monotonic()
        frozen = metrics.get("comparison_context")
        hits = [SearchHit(h["metadata"], h["score"]) for h in frozen["hits"]] if frozen else self.retrieve_context(decision)
        metrics["retrieval_seconds"] = monotonic() - started
        metrics["selected_chunk_ids"] = [hit.metadata["chunk_id"] for hit in hits]
        if decision.question_scope in {"goal", "state"}:
            content, sources = self._state_answer(decision, state)
            return content, sources, [], "state"
        grounded = self._document_answer(turn, decision, state, hits, metrics)
        if grounded.status == "insufficient_context" and not metrics.get("comparison_request"):
            started = monotonic()
            expanded = self._expand_cost_hits(decision, hits)
            metrics["retrieval_seconds"] += monotonic() - started
            if len(expanded) > len(hits):
                retry_answer = self._document_answer(turn, decision, state, expanded, metrics)
                if retry_answer.status == "answered":
                    grounded = retry_answer
                    hits = expanded
                    metrics["selected_chunk_ids"] = [hit.metadata["chunk_id"] for hit in hits]
        content = grounded.content
        if grounded.status == "answered":
            available = " ".join(
                hit.metadata["section"] + " " + hit.metadata["text"]
                for hit in hits
            ).casefold()
            missing = [term for term in self._named_terms(decision.search_question)
                       if term.casefold() not in available]
            if missing:
                content += (
                    "\nПо темам " + ", ".join(missing)
                    + " в найденных фрагментах нет подтверждения."
                )
        return (
            content,
            [{"kind": "document", **source} for source in grounded.sources],
            grounded.citations,
            grounded.status,
        )
