"""Measured generation profiles, scoped to local document answers."""

from dataclasses import dataclass, replace
import os

from ..providers.errors import LlmConfigurationError
from ..private_service import ServiceLimits


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
    return _bounded_profile(PROFILES["optimized" if name == "q8" else name])


def _bounded_profile(profile):
    limits = ServiceLimits.from_env()
    if not limits.enabled:
        return profile
    return replace(profile, num_ctx=min(profile.num_ctx, limits.max_context),
                   max_num_ctx=min(profile.max_num_ctx, limits.max_context),
                   max_tokens=min(profile.max_tokens, limits.max_output))


def configuration_model(registry, name: str):
    configuration_profile(name)
    model = os.getenv("OLLAMA_Q8_MODEL", "qwen3.5:9b-q8_0") if name == "q8" else None
    return registry.resolve("ollama", model)


def generation_profile(name: str | None = None) -> RagGenerationProfile:
    name = name or os.getenv("RAG_LLM_PROFILE", "optimized")
    try:
        return _bounded_profile(PROFILES[name])
    except KeyError as error:
        raise LlmConfigurationError("RAG_LLM_PROFILE: выберите baseline, compact или optimized") from error
