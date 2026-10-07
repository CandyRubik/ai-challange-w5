"""Measure one real RAG generation after unloading local weights, then warm."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
from time import monotonic
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.agents.agent import Agent
from app.indexing.store import SearchHit
from app.providers.registry import ModelRegistry
from app.rag_chat.generation import GROUNDING_SYSTEM_PROMPT, generate_grounded_answer


def main():
    load_dotenv(ROOT / ".env", override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, default=ROOT / "docs/day28-artifacts/evaluation.json")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/day28-artifacts/cold-start.json")
    args = parser.parse_args()
    registry = ModelRegistry()
    selection = registry.resolve("ollama")
    base = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    if urlparse(base).hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("Cold-start measurement requires local Ollama")
    case = json.loads(args.evaluation.read_text())["results"][1]
    hits = [SearchHit({key: value for key, value in hit.items() if key != "score"}, hit["score"])
            for hit in case["selected"]]
    with httpx.Client(timeout=30, trust_env=False) as client:
        unloaded = client.post(base + "/api/chat", json={"model": selection.model, "messages": [], "keep_alive": 0})
        unloaded.raise_for_status()
    report = {"model": selection.model, "question": case["question"], "context_sha256": case["context_sha256"],
              "method": "Unload only the selected local model via keep_alive=0; generate twice on identical saved context.",
              "limitations": "One cold/warm observation; excludes interpretation and retrieval.", "runs": []}
    for label in ["cold", "warm"]:
        model = registry.build(selection)
        metrics = {}
        started = monotonic()
        answer = generate_grounded_answer(
            Agent(model, system_prompt=GROUNDING_SYSTEM_PROMPT, max_tokens=3000),
            case["question"], case["search_question"], hits, metrics=metrics,
        )
        report["runs"].append({"label": label, "elapsed_seconds": monotonic() - started,
                               "metrics": {**metrics, "ollama_requests": model.request_metrics}, **asdict(answer)})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        assert answer.status == "answered"
        print(label, round(report["runs"][-1]["elapsed_seconds"], 3), flush=True)


if __name__ == "__main__":
    main()
