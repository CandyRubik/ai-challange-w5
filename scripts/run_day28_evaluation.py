"""Compare real local/cloud generation on identical locally retrieved contexts."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import statistics
import sys
from tempfile import TemporaryDirectory
from time import monotonic

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.agents.agent import Agent
from app.indexing.rag import DocumentRag, RetrievalSettings
from app.indexing.rerank import LocalCrossEncoderReranker
from app.indexing.store import DocumentIndex
from app.providers.registry import ModelRegistry
from app.rag_chat.generation import GROUNDING_SYSTEM_PROMPT, generate_grounded_answer
from app.rag_chat.citation_options import prepare_citations
from app.rag_chat.models import RagTaskState
from app.rag_chat.service import RagChatService
from app.rag_chat.state import TurnInterpreter
from app.rag_chat.store import SQLiteRagChatRepository


def summarize(results: list[dict], providers: list[str]) -> dict:
    summary = {}
    for provider in providers:
        rows = [row for case in results for row in case["runs"] if row["provider"] == provider]
        timed = [row["elapsed_seconds"] for row in rows if row["metrics"].get("generation_attempts", 0)]
        statuses = [[row["status"] for row in case["runs"] if row["provider"] == provider] for case in results]
        summary[provider] = {
            "runs": len(rows), "provider_errors": sum(row["status"] == "provider_error" for row in rows),
            "grounding_failures": sum(row["status"] == "grounding_failed" for row in rows),
            "automatic_passes": sum(row["automatic_pass"] for row in rows),
            "generation_calls": len(timed),
            "generation_skipped": sum(not row["metrics"].get("generation_attempts", 0) for row in rows),
            "first_attempt_valid": sum(row["metrics"].get("generation_attempts") == 1 and row["status"] != "provider_error" for row in rows),
            "median_generation_seconds": round(statistics.median(timed), 3) if timed else None,
            "p95_generation_seconds": round(sorted(timed)[max(0, int(len(timed) * .95 + .999) - 1)], 3) if timed else None,
            "stable_status_questions": sum(len(set(items)) == 1 for items in statuses if items),
            "manual_quality_review": "pending",
        }
    return summary


def run(output: Path, repeats: int, providers: list[str], questions_path: Path) -> dict:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    registry = ModelRegistry()
    if "deepseek" in providers and not os.getenv("DEEPSEEK_API_KEY"):
        raise ValueError("Для сравнения с DeepSeek нужен ключ; используйте --providers ollama для локального прогона")
    cases = json.loads(questions_path.read_text(encoding="utf-8"))
    index = DocumentIndex()
    manifest = index.ensure_built()
    local = registry.build(registry.resolve("ollama"), thinking_enabled=False)
    interpreter = TurnInterpreter(local)
    rag = DocumentRag(index, local, reranker=LocalCrossEncoderReranker(), settings=RetrievalSettings(rewrite=False))
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(), "repeats": repeats,
        "models": {p: registry.resolve(p).model for p in providers},
        "platform": platform.platform(), "architecture": platform.machine(),
        "index": manifest, "pdf_sha256": hashlib.sha256(index.pdf_path.read_bytes()).hexdigest(),
        "method": "Local interpretation and retrieval once per question; identical document context and user question for each provider and repeat; no answer history or task memory. Ollama receives a native JSON Schema; DeepSeek receives the equivalent schema in its prompt with JSON mode.",
        "dependencies": {name: version(name) for name in (
            "sentence-transformers", "transformers", "torch", "faiss-cpu", "numpy", "httpx", "openai",
        )},
        "limitations": ["Exact quote validation does not establish semantic support; manual review is required.",
                        "No-hit refusals skip generation and measure application behavior, not LLM quality.",
                        "p95 is descriptive for this small fixed sample; model responses need not be text-identical."],
        "results": [],
    }
    assets = ROOT / "data/local-rag-assets.json"
    if assets.is_file():
        report["assets"] = json.loads(assets.read_text(encoding="utf-8"))
    output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        report["summary"] = summarize(report["results"], providers)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    started = monotonic()
    # Report process startup cost separately; do not count it as steady-state retrieval.
    warm_hits = index.search("Executor", k=3)
    rag.reranker.score("Executor", warm_hits)
    report["retrieval_cold_start_seconds"] = monotonic() - started
    with TemporaryDirectory(prefix="day28-evaluation-") as temporary:
        chat = RagChatService(SQLiteRagChatRepository(Path(temporary) / "rag.sqlite3"),
                              interpreter, rag, Agent(local))
        for case in cases:
            prepared = monotonic()
            decision = interpreter.interpret(case["question"], RagTaskState(), [])
            interpretation_seconds = monotonic() - prepared
            searched = monotonic()
            hits = chat.retrieve_context(decision)
            retrieval_seconds = monotonic() - searched
            prompt, _, _ = prepare_citations(hits)
            result = {**case, "search_question": decision.search_question,
                      "interpretation_seconds": interpretation_seconds, "retrieval_seconds": retrieval_seconds,
                      "context_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                      "selected": [{"score": hit.score, **hit.metadata} for hit in hits], "runs": []}
            report["results"].append(result)
            for repeat in range(1, repeats + 1):
                order = providers if (case["id"] + repeat) % 2 == 0 else list(reversed(providers))
                for provider in order:
                    selection = registry.resolve(provider)
                    model = registry.build(selection, thinking_enabled=False)
                    metrics = {"generation_attempts": 0}
                    started = monotonic()
                    try:
                        answer = generate_grounded_answer(
                            Agent(model, system_prompt=GROUNDING_SYSTEM_PROMPT, max_tokens=3_000),
                            case["question"], decision.search_question, hits, metrics=metrics,
                        )
                        data = asdict(answer)
                    except Exception as error:
                        data = {"content": "", "sources": [], "citations": [], "status": "provider_error",
                                "error": type(error).__name__}
                    elapsed = monotonic() - started
                    positive = bool(case["sections"])
                    automatic_pass = (
                        data["status"] == "answered" and any(
                            source["section"] in case["sections"] for source in data["sources"]
                        ) if positive else data["status"] == "insufficient_context" and not data["sources"]
                    )
                    result["runs"].append({
                        "provider": provider, "model": selection.model, "repeat": repeat,
                        "context_sha256": result["context_sha256"], "elapsed_seconds": elapsed,
                        "metrics": {**metrics, "ollama_requests": getattr(model, "request_metrics", [])},
                        "automatic_pass": automatic_pass, **data,
                    })
                    save()
                    print(f"Q{case['id']} {provider} #{repeat}: {data['status']} {elapsed:.2f}s auto={automatic_pass}", flush=True)
    report["complete"] = True
    save()
    return report


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/day28-artifacts/evaluation.json")
    parser.add_argument("--questions", type=Path, default=ROOT / "evaluation/day28-questions.json")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--providers", nargs="+", choices=["ollama", "deepseek"], default=["ollama", "deepseek"])
    args = parser.parse_args()
    run(args.output, args.repeats, list(dict.fromkeys(args.providers)), args.questions)


if __name__ == "__main__":
    main()
