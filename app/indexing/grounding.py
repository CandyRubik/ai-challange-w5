"""Validate cited RAG claims against the exact excerpts shown to the model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import re

from pydantic import Field, ValidationError, model_validator

from ..schemas import StrictModel
from .store import SearchHit


MAX_EXCERPT_CHARS = 1800
UNKNOWN_ANSWER = "Не знаю по найденным фрагментам; уточните раздел или термин."
UNVERIFIED_ANSWER = (
    "Не могу подтвердить ответ цитатами из найденных фрагментов; уточните вопрос."
)


class GroundingError(ValueError):
    """The generated claims cannot be tied to the retrieved excerpts."""


class DraftEvidence(StrictModel):
    chunk_id: str = Field(min_length=1, max_length=80)
    quote: str = Field(min_length=12, max_length=500)


class DraftClaim(StrictModel):
    text: str = Field(min_length=1, max_length=600)
    evidence: list[DraftEvidence] = Field(min_length=1, max_length=3)


class GroundedDraft(StrictModel):
    status: Literal["answered", "unknown"]
    claims: list[DraftClaim] = Field(max_length=5)

    @model_validator(mode="after")
    def consistent_status(self) -> "GroundedDraft":
        if (self.status == "answered") != bool(self.claims):
            raise ValueError("Ответу нужны тезисы, а отказу — пустой список")
        return self


@dataclass(frozen=True)
class GroundedAnswer:
    content: str
    sources: list[dict]
    citations: list[dict]
    status: Literal["answered", "insufficient_context", "grounding_failed"]

    @classmethod
    def unknown(cls) -> "GroundedAnswer":
        return cls(UNKNOWN_ANSWER, [], [], "insufficient_context")

    @classmethod
    def failed(cls) -> "GroundedAnswer":
        return cls(UNVERIFIED_ANSWER, [], [], "grounding_failed")


def _canonical_quote(quote: str, excerpt: str) -> str | None:
    # PDF extraction inserts layout whitespace. Preserve the source's actual text.
    pattern = r"\s+".join(re.escape(part) for part in quote.split())
    match = re.search(pattern, excerpt)
    return match.group(0) if match else None


def validate_grounded_answer(raw: str, hits: list[SearchHit]) -> GroundedAnswer:
    try:
        draft = GroundedDraft.model_validate_json(raw)
    except ValidationError as error:
        raise GroundingError("Неверная структура ответа") from error
    if draft.status == "unknown":
        return GroundedAnswer.unknown()

    selected = {hit.metadata["chunk_id"]: hit for hit in hits}
    sources: list[dict] = []
    citations: list[dict] = []
    source_ids: set[str] = set()
    citation_keys: set[tuple[str, str]] = set()
    sentences: list[str] = []
    for claim in draft.claims:
        if re.search(r"\[DOC:", claim.text, flags=re.I):
            raise GroundingError("Тезис содержит неподтверждённую ссылку")
        claim_ids: list[str] = []
        for evidence in claim.evidence:
            hit = selected.get(evidence.chunk_id)
            if hit is None:
                raise GroundingError("Цитата ссылается на невыбранный чанк")
            excerpt = hit.metadata["text"][:MAX_EXCERPT_CHARS]
            quote = _canonical_quote(evidence.quote, excerpt)
            if quote is None:
                raise GroundingError("Цитата отсутствует в переданном модели фрагменте")
            if evidence.chunk_id not in claim_ids:
                claim_ids.append(evidence.chunk_id)
            if evidence.chunk_id not in source_ids:
                source_ids.add(evidence.chunk_id)
                sources.append({
                    "chunk_id": evidence.chunk_id,
                    "source": hit.metadata["source"],
                    "title": hit.metadata["title"],
                    "section": hit.metadata["section"],
                    "page_start": hit.metadata["page_start"],
                    "page_end": hit.metadata["page_end"],
                })
            key = (evidence.chunk_id, quote)
            if key not in citation_keys:
                citation_keys.add(key)
                citations.append({"chunk_id": evidence.chunk_id, "quote": quote})
        sentences.append(
            claim.text + " " + " ".join(f"[DOC:{chunk_id}]" for chunk_id in claim_ids)
        )
    return GroundedAnswer("\n".join(sentences), sources, citations, "answered")
