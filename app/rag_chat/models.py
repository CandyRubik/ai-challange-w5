from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from ..schemas import StrictModel


class StateFact(StrictModel):
    key: str
    value: str
    source_message_id: str


class RagTaskState(StrictModel):
    goal: StateFact | None = None
    constraints: list[StateFact] = Field(default_factory=list)
    terms: list[StateFact] = Field(default_factory=list)
    clarifications: list[StateFact] = Field(default_factory=list)
    revision: int = 0


class RagTurn(StrictModel):
    id: str
    session_id: str
    position: int
    content: str
    answer: str = ""
    status: Literal["pending", "done", "failed"] = "pending"
    kind: Literal["question", "statement", "clarification_needed"] | None = None
    question_scope: Literal["document", "goal", "state"] | None = None
    search_question: str = ""
    sources: list[dict] = Field(default_factory=list)
    citations: list[dict] = Field(default_factory=list)
    grounding_status: str | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime
    provider: Literal["ollama", "deepseek"] = "deepseek"
    model: str | None = None
    metrics: dict = Field(default_factory=dict)


class RagSessionSummary(StrictModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    provider: Literal["ollama", "deepseek"] = "deepseek"


class RagSession(RagSessionSummary):
    state: RagTaskState = Field(default_factory=RagTaskState)
    turns: list[RagTurn] = Field(default_factory=list)


class RagSendRequest(StrictModel):
    content: str = Field(min_length=1, max_length=12_000)
    provider: Literal["ollama", "deepseek"] | None = None


class RagCreateRequest(StrictModel):
    provider: Literal["ollama", "deepseek"] | None = None
