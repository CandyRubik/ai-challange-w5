"""Interpret explicit user updates and maintain one active task state."""

from __future__ import annotations

import json
from typing import Literal, Protocol

from pydantic import Field, ValidationError

from ..schemas import StrictModel
from .models import RagTaskState, StateFact


class InterpretationError(ValueError):
    pass


class JsonModel(Protocol):
    def generate_json(self, *, messages: list[dict], max_tokens: int = 2000, schema: dict | None = None) -> str: ...


class StateChange(StrictModel):
    group: Literal["constraint", "term", "clarification"]
    action: Literal["set", "remove"]
    key: str = Field(min_length=1, max_length=100)
    value: str = Field(default="", max_length=1000)


class TurnDecision(StrictModel):
    kind: Literal["question", "statement", "clarification_needed"]
    question_scope: Literal["document", "goal", "state"] = "document"
    search_question: str = Field(default="", max_length=500)
    goal_action: Literal["keep", "set", "clear"] = "keep"
    goal: str = Field(default="", max_length=1000)
    changes: list[StateChange] = Field(default_factory=list, max_length=10)
    clarification: str = Field(default="", max_length=500)


class TurnInterpreter:
    """One JSON call turns the latest user message into a bounded state patch."""

    system_prompt = """You interpret ONE user turn in a Russian RAG chat about the indexed sample of
Java Concurrency in Practice, chapter 6. Return one JSON object with exactly:
kind, question_scope, search_question, goal_action, goal, changes, clarification.
kind is question, statement, or clarification_needed. A user request for an
explanation is a question even without a question mark. A pure clarification or
constraint is a statement. If the user tries to work on a second simultaneous
goal in this chat, or says to cancel 'this' when the referent is unclear, use
clarification_needed and ask one short clarifying question in clarification;
make no state changes in that case.

question_scope is document for questions about the book, goal for questions
about the user's active goal, and state for questions about other agreements.
For EVERY question produce a standalone, concise English search_question for
the indexed chapter, preserving named technical terms. The search still runs
for questions about agreements, but retrieved book text is not evidence for
what the user agreed to. For statements set search_question to an empty string.

Only record goal, constraints, term meanings, or clarifications EXPLICITLY
provided by the current user message. Do not infer them from assistant turns,
book text, or a question. goal_action is keep, set, or clear; goal is nonempty
only for set. Setting a different active goal replaces the previous task scope.
changes is a list of {group,action,key,value}; group is constraint, term, or
clarification; action is set or remove. Use short stable keys, for example
'length', 'Timer', or 'audience'. Set a value only when explicitly given by the
user. A clear correction replaces the earlier value with the same key. Use
remove only when the target is unambiguous. Never store unsupported claims
about the book as facts about the book. Do not store secrets.

Return JSON only. Example for 'Моя цель — объяснить Executor новичку. Что это?':
{"kind":"question","question_scope":"document","search_question":"What is Executor in chapter 6?","goal_action":"set","goal":"Объяснить Executor новичку","changes":[],"clarification":""}
"""

    def __init__(self, model: JsonModel) -> None:
        self._model = model

    def interpret(
        self, message: str, state: RagTaskState, recent: list[dict],
    ) -> TurnDecision:
        payload = {
            "current_user_message": message,
            "current_task_state": state.model_dump(),
            "recent_conversation": recent[-4:],
        }
        instruction = self.system_prompt + """
Separate the goal from answer-style constraints. Never include a sentence like
'Отвечай кратко' or 'Пиши подробно' in the goal value; record it in changes as a
constraint with key 'length'. Reuse an existing constraint key for a correction.
A goal declaration is a statement, unless there is a separate question or a
request to explain something now. Do not treat 'Моя цель — объяснить...' alone
as a request for an immediate explanation.
For search_question, do not add the book title, chapter title, or unrelated
capitalized words. Use only technical concepts needed for the actual question.

Example input: 'Моя цель — объяснить Executor новичку. Отвечай кратко.'
{"kind":"statement","question_scope":"document","search_question":"","goal_action":"set","goal":"Объяснить Executor новичку","changes":[{"group":"constraint","action":"set","key":"length","value":"Кратко"}],"clarification":""}
Example input: 'Нет, отвечай подробно.'
{"kind":"statement","question_scope":"document","search_question":"","goal_action":"keep","goal":"","changes":[{"group":"constraint","action":"set","key":"length","value":"Подробно"}],"clarification":""}
Questions about saved goals or constraints are QUESTIONS, never statements.
They must not add clarifications or change memory. Read only the current user
message to decide whether this turn is a question; prior statements do not
make the current question a statement.
Example input: 'Какую цель мы зафиксировали?'
{"kind":"question","question_scope":"goal","search_question":"Executor explanation goal","goal_action":"keep","goal":"","changes":[],"clarification":""}
Example input: 'Какие ограничения мы согласовали?'
{"kind":"question","question_scope":"state","search_question":"Executor explanation constraints","goal_action":"keep","goal":"","changes":[],"clarification":""}
Example input: 'Почему обработчик ThreadPerTaskWebServer должен быть потокобезопасным?'
{"kind":"question","question_scope":"document","search_question":"Why must the ThreadPerTaskWebServer request handler be thread-safe?","goal_action":"keep","goal":"","changes":[],"clarification":""}
"""
        try:
            raw = self._model.generate_json(messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ], max_tokens=700, schema=TurnDecision.model_json_schema())
            decision = TurnDecision.model_validate_json(raw)
        except (ValidationError, ValueError, TypeError) as error:
            raise InterpretationError("Не удалось разобрать сообщение") from error
        if decision.kind == "question" and not decision.search_question.strip():
            decision = decision.model_copy(update={"search_question": message.strip()[:500]})
        if decision.kind == "clarification_needed" and (
            not decision.clarification.strip()
            or decision.goal_action != "keep"
            or decision.changes
        ):
            raise InterpretationError("Неясное уточнение должно оставить память неизменной")
        if decision.goal_action == "set" and not decision.goal.strip():
            raise InterpretationError("Новая цель пуста")
        return decision


def apply_decision(
    current: RagTaskState, decision: TurnDecision, message_id: str,
) -> RagTaskState:
    """Return the next state; changed values point to the user turn that set them."""
    if decision.kind == "clarification_needed":
        return current
    goal = current.goal
    groups = {
        "constraint": {item.key.casefold(): item for item in current.constraints},
        "term": {item.key.casefold(): item for item in current.terms},
        "clarification": {item.key.casefold(): item for item in current.clarifications},
    }
    if decision.goal_action == "clear":
        goal = None
        groups = {name: {} for name in groups}
    elif decision.goal_action == "set" and (
        goal is None or goal.value != decision.goal.strip()
    ):
        goal = StateFact(key="goal", value=decision.goal.strip(), source_message_id=message_id)
        if current.goal is not None:
            groups = {name: {} for name in groups}
    for change in decision.changes:
        key = change.key.strip().casefold()
        if change.action == "remove":
            groups[change.group].pop(key, None)
        elif change.value.strip():
            groups[change.group][key] = StateFact(
                key=change.key.strip(), value=change.value.strip(),
                source_message_id=message_id,
            )
    next_state = RagTaskState(
        goal=goal,
        constraints=list(groups["constraint"].values())[:20],
        terms=list(groups["term"].values())[:20],
        clarifications=list(groups["clarification"].values())[:20],
        revision=current.revision,
    )
    if next_state.model_dump(exclude={"revision"}) != current.model_dump(exclude={"revision"}):
        next_state.revision += 1
    return next_state
