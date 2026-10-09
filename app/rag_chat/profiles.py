"""Measured generation profiles, scoped to local document answers."""

from dataclasses import dataclass
import os

from ..providers.errors import LlmConfigurationError


@dataclass(frozen=True)
class RagGenerationProfile:
    name: str
    num_ctx: int
    max_tokens: int
    temperature: float
    prompt_version: str
    max_num_ctx: int = 32768


PROFILES = {
    "baseline": RagGenerationProfile("baseline", 32768, 3000, 0, "baseline"),
    "compact": RagGenerationProfile("compact", 8192, 1000, 0, "baseline"),
    "optimized": RagGenerationProfile("optimized", 8192, 1000, 0, "complete-v3"),
}

CONFIGURATION_LABELS = {
    "baseline": "Исходный",
    "compact": "Только параметры",
    "optimized": "Оптимизированный",
    "q8": "Оптимизированный · Q8",
}


def configuration_profile(name: str) -> RagGenerationProfile:
    if name not in CONFIGURATION_LABELS:
        raise ValueError("Неизвестная конфигурация локальной LLM")
    return PROFILES["optimized" if name == "q8" else name]


def configuration_model(registry, name: str):
    configuration_profile(name)
    model = os.getenv("OLLAMA_Q8_MODEL", "qwen3.5:9b-q8_0") if name == "q8" else None
    return registry.resolve("ollama", model)


def generation_profile(name: str | None = None) -> RagGenerationProfile:
    name = name or os.getenv("RAG_LLM_PROFILE", "optimized")
    try:
        return PROFILES[name]
    except KeyError as error:
        raise LlmConfigurationError("RAG_LLM_PROFILE: выберите baseline, compact или optimized") from error
