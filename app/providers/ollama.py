from __future__ import annotations

from collections.abc import Sequence
import logging
from typing import Any

import httpx

from ..agents.agent import AgentMessage
from .deepseek import CONTINUATION_PROMPT, MAX_RESPONSE_SEGMENTS
from .errors import LlmConfigurationError, LlmRequestError, LlmTruncatedResponseError


logger = logging.getLogger(__name__)


class OllamaProvider:
    """Use the local chat API for text and structured application calls."""

    def __init__(
        self, *, model: str, base_url: str = "http://127.0.0.1:11434",
        num_ctx: int = 32_768, client: httpx.Client | None = None,
    ) -> None:
        if num_ctx < 4096:
            raise LlmConfigurationError("OLLAMA_NUM_CTX должен быть не меньше 4096")
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._num_ctx = num_ctx
        self._client = client

    def _bounded_messages(
        self, messages: Sequence[AgentMessage], max_tokens: int,
    ) -> list[AgentMessage]:
        # UTF-8 bytes provide a conservative token estimate for the selected
        # Qwen tokenizer. Reserve room for chat framing and the entire answer.
        budget = self._num_ctx - max_tokens - 512
        bounded = list(messages)
        def cost() -> int:
            return sum(len(message["content"].encode("utf-8")) + 32 for message in bounded)
        while cost() > budget:
            index = next((i for i, message in enumerate(bounded[:-1])
                          if message["role"] != "system"), None)
            if index is None:
                raise LlmRequestError(
                    "Контекст не помещается в локальную модель. Сократите запрос или память "
                    "либо увеличьте OLLAMA_NUM_CTX",
                )
            bounded.pop(index)
        return bounded

    def _request(
        self, messages: Sequence[AgentMessage], max_tokens: int, *, structured: bool = False,
        schema: dict[str, Any] | None = None,
    ) -> tuple[str, str | None]:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": self._bounded_messages(messages, max_tokens),
            "stream": False, "think": False, "keep_alive": "30m",
            "options": {"num_ctx": self._num_ctx, "num_predict": max_tokens},
        }
        if structured:
            payload["format"] = schema or "json"
            payload["options"]["temperature"] = 0
        try:
            if self._client is not None:
                response = self._client.post(f"{self._base_url}/api/chat", json=payload)
            else:
                with httpx.Client(timeout=httpx.Timeout(300, connect=5), trust_env=False) as client:
                    response = client.post(f"{self._base_url}/api/chat", json=payload)
            if response.status_code == 404:
                raise LlmRequestError(
                    f"Локальная модель {self._model} не найдена. Загрузите её через ollama pull",
                )
            response.raise_for_status()
            result = response.json()
            content = result["message"]["content"]
            if not isinstance(content, str) or not content.strip() or result.get("done") is not True:
                raise LlmRequestError("Ollama не вернула завершённый непустой ответ")
            return content, result.get("done_reason")
        except LlmRequestError:
            raise
        except httpx.TimeoutException as error:
            raise LlmRequestError("Ollama не успела ответить. Повторите запрос") from error
        except httpx.ConnectError as error:
            raise LlmRequestError("Ollama недоступна. Запустите локальный сервер ollama serve") from error
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as error:
            logger.warning("Ollama request failed: type=%s", type(error).__name__)
            raise LlmRequestError("Запрос к локальной модели завершился ошибкой") from error

    def generate(self, *, messages: Sequence[AgentMessage], max_tokens: int = 2000) -> str:
        conversation = list(messages)
        parts: list[str] = []
        for _ in range(MAX_RESPONSE_SEGMENTS):
            content, reason = self._request(conversation, max_tokens)
            parts.append(content)
            if reason != "length":
                return "".join(parts).strip()
            conversation.extend([
                {"role": "assistant", "content": content},
                {"role": "user", "content": CONTINUATION_PROMPT},
            ])
        raise LlmRequestError("Ответ локальной модели не завершился после нескольких продолжений")

    def generate_json(self, *, messages: Sequence[AgentMessage], max_tokens: int = 2000,
                      schema: dict[str, Any] | None = None) -> str:
        content, reason = self._request(messages, max_tokens, structured=True, schema=schema)
        if reason == "length":
            raise LlmTruncatedResponseError("Ollama")
        return content.strip()

    def complete(
        self, *, system_prompt: str, user_prompt: str,
        response_format: dict[str, Any] | None = None, max_tokens: int = 2000,
    ) -> str:
        messages: list[AgentMessage] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if response_format:
            schema = response_format.get("json_schema", {}).get("schema")
            return self.generate_json(messages=messages, max_tokens=max_tokens, schema=schema)
        return self.generate(messages=messages, max_tokens=max_tokens)
