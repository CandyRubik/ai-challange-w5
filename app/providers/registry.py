from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Literal

import httpx

from .deepseek import DeepSeekProvider
from .errors import LlmConfigurationError
from .ollama import OllamaProvider


ProviderId = Literal["ollama", "deepseek"]


@dataclass(frozen=True, slots=True)
class ModelSelection:
    provider: ProviderId
    model: str


class ModelRegistry:
    """Server-owned model configuration; every operation resolves its own runtime."""

    def __init__(self) -> None:
        self._models = {
            "ollama": os.getenv("OLLAMA_MODEL", "qwen3.5:9b-q4_K_M"),
            "deepseek": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        }
        self._ollama_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
        try:
            self._num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "32768"))
        except ValueError as error:
            raise LlmConfigurationError("OLLAMA_NUM_CTX должен быть целым числом") from error
        self.default_provider = os.getenv("LLM_DEFAULT_PROVIDER", "ollama")
        self.resolve(self.default_provider)

    def resolve(self, provider: str, model: str | None = None) -> ModelSelection:
        if provider not in self._models:
            raise LlmConfigurationError("Неизвестный провайдер модели")
        return ModelSelection(provider, model or self._models[provider])

    def build(self, selection: ModelSelection, *, thinking_enabled: bool = True):
        if selection.provider == "ollama":
            return OllamaProvider(model=selection.model, base_url=self._ollama_url, num_ctx=self._num_ctx)
        return DeepSeekProvider(model=selection.model, thinking_enabled=thinking_enabled)

    def catalog(self) -> dict:
        local_available = False
        local_detail = "Ollama недоступна — запустите ollama serve"
        try:
            with httpx.Client(timeout=2, trust_env=False) as client:
                response = client.get(f"{self._ollama_url}/api/tags")
                response.raise_for_status()
                local_available = any(isinstance(model, dict) and model.get("name") == self._models["ollama"]
                                      for model in response.json()["models"])
                local_detail = "Локальная модель готова" if local_available else "Модель не загружена — выполните ollama pull"
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            pass
        cloud_configured = bool(os.getenv("DEEPSEEK_API_KEY"))
        return {
            "default_provider": self.default_provider,
            "providers": [
                {"id": "ollama", "model": self._models["ollama"], "label": "Локальная",
                 "available": local_available, "detail": local_detail},
                {"id": "deepseek", "model": self._models["deepseek"], "label": "DeepSeek",
                 "available": cloud_configured, "detail": "Ключ настроен" if cloud_configured else "DEEPSEEK_API_KEY не задан"},
            ],
        }
