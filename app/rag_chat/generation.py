"""Shared generation path for the chat and paired model evaluation."""

import json
from time import monotonic

from ..agents.agent import Agent, AgentOutputError
from ..indexing.grounding import GroundedAnswer, GroundingError, validate_grounded_answer
from ..indexing.store import SearchHit
from ..memory.context import MemoryContext
from .citation_options import prepare_citations, resolve_citations


GROUNDING_SYSTEM_PROMPT = (
    "You format answers to questions about retrieved document excerpts. "
    "Return only one valid JSON object matching the requested schema. "
    "Never emit prose outside the JSON object."
)
MAX_GROUNDING_ATTEMPTS = 4


def generate_grounded_answer(
    agent: Agent, question: str, search_question: str, hits: list[SearchHit], *,
    memory: MemoryContext | None = None, metrics: dict | None = None,
) -> GroundedAnswer:
    metrics = metrics if metrics is not None else {}
    if not hits:
        return GroundedAnswer.unknown()
    payload = json.dumps({
        "user_question": question, "standalone_search_question": search_question,
    }, ensure_ascii=False)
    feedback = ""
    prompt, schema, lookup = prepare_citations(hits)
    if not lookup:
        return GroundedAnswer.unknown()
    for attempt in range(MAX_GROUNDING_ATTEMPTS):
        metrics["generation_attempts"] = metrics.get("generation_attempts", 0) + 1
        started = monotonic()
        try:
            raw = agent.respond_json(
                [], payload, memory=memory, schema=schema,
                document_context=prompt + (
                    "\nPrevious output failed validation: " + feedback[:200] if feedback else ""
                ),
            )
            return validate_grounded_answer(resolve_citations(raw, lookup), hits)
        except (AgentOutputError, GroundingError) as error:
            feedback = str(error)
            metrics.setdefault("validation_errors", []).append(feedback)
        finally:
            metrics["generation_seconds"] = metrics.get("generation_seconds", 0) + monotonic() - started
    return GroundedAnswer.failed()
