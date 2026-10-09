"""Generate live A/B answers on one durable evidence and memory snapshot."""

from dataclasses import asdict
import hashlib
import json
from time import monotonic

from ..agents.agent import Agent
from ..indexing.grounding import GroundedAnswer
from ..indexing.store import SearchHit
from ..memory.context import MemoryContext
from .generation import GROUNDING_SYSTEM_PROMPT, generate_grounded_answer
from .profiles import RagGenerationProfile


def freeze_context(question, search_question, hits, memory):
    context = {
        "question": question, "search_question": search_question,
        "hits": [{"metadata": hit.metadata, "score": hit.score} for hit in hits],
        "memory": asdict(memory or MemoryContext()),
    }
    context["sha256"] = hashlib.sha256(
        json.dumps(context, ensure_ascii=False, sort_keys=True).encode(),
    ).hexdigest()
    return context


def generation_statistics(requests, registry=None, model=None):
    seconds = sum(r.get("eval_duration", 0) for r in requests) / 1e9
    tokens = sum(r.get("eval_count", 0) for r in requests)
    allocation = registry.loaded_local_model(model) if registry and requests else None
    options = requests[-1].get("options") if requests else None
    if allocation and options and allocation.get("context_length") != options.get("num_ctx"):
        allocation = None  # Another chat may have reconfigured this shared Ollama model.
    return {
        "output_tokens": tokens if requests else None,
        "tokens_per_second": tokens / seconds if seconds else None,
        "load_seconds": sum(r.get("load_duration", 0) for r in requests) / 1e9 if requests else None,
        "actual_options": options,
        # A point-in-time model allocation, not peak RSS or total machine memory.
        "model_allocation": allocation,
    }


def run_comparison(registry, request, context, results, progress):
    hits = [SearchHit(row["metadata"], row["score"]) for row in context["hits"]]
    memory = MemoryContext(**{key: tuple(value) for key, value in context["memory"].items()})
    for leg in ("reference", "candidate"):
        if leg in results:
            continue
        config = request[leg]
        profile = RagGenerationProfile(**config["profile"])
        model = registry.build(
            registry.resolve("ollama", config["model"]), thinking_enabled=False,
            num_ctx=profile.num_ctx, max_num_ctx=profile.max_num_ctx, temperature=profile.temperature,
        )
        metrics = {"generation_attempts": 0}
        started = monotonic()
        answer = generate_grounded_answer(
            Agent(model, system_prompt=GROUNDING_SYSTEM_PROMPT, max_tokens=profile.max_tokens),
            context["question"], context["search_question"], hits, memory=memory,
            prompt_version=profile.prompt_version, metrics=metrics,
        )
        elapsed = monotonic() - started
        requests = list(getattr(model, "request_metrics", []))
        results[leg] = {
            **config, "answer": asdict(answer), "elapsed_seconds": elapsed,
            "metrics": {**metrics, "ollama_requests": requests,
                        **generation_statistics(requests, registry, config["model"])},
            "context_sha256": context["sha256"],
        }
        progress()
    return GroundedAnswer(**results["candidate"]["answer"])
