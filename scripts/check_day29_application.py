"""Check real RAG HTTP flow and cold/warm generation after the isolated benchmark."""

from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
from time import monotonic

import httpx
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import main
from app.agents.agent import Agent
from app.indexing.store import SearchHit
from app.providers.ollama import OllamaProvider
from app.rag_chat.generation import GROUNDING_SYSTEM_PROMPT, generate_grounded_answer
from app.rag_chat.profiles import PROFILES
from scripts.run_day29_evaluation import power_state


def main_check():
    os.environ.update(DEEPSEEK_API_KEY="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1", LLM_DEFAULT_PROVIDER="ollama",
                      OLLAMA_MODEL="qwen3.5:9b-q4_K_M", OLLAMA_NUM_CTX="32768")
    base = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    cases = json.loads((ROOT / "evaluation/day29-cases.json").read_text())["cases"]
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "cold_warm": [], "http_runs": [],
              "power_state_start": power_state(),
              "method": "Real FastAPI routes with Qwen, local E5/FAISS/reranker and SQLite; cloud key absent. Model/context warmup excluded from HTTP timing; profile order alternates across three repeats.",
              "limitations": "Small descriptive timing sample; unloading weights leaves operating-system disk cache intact. External sockets are not blocked in this check."}
    target = ROOT / "docs/day29-artifacts/application-check.json"
    def save():
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    with httpx.Client(timeout=30, trust_env=False) as control:
        for name in ("baseline", "optimized"):
            profile = PROFILES[name]
            for loaded in control.get(base + "/api/ps").json()["models"]:
                control.post(base + "/api/generate", json={"model": loaded["name"], "keep_alive": 0}).raise_for_status()
            model = OllamaProvider(model="qwen3.5:9b-q4_K_M", base_url=base, num_ctx=profile.num_ctx,
                                   max_num_ctx=profile.max_num_ctx, temperature=profile.temperature)
            case = cases[2]
            for label in ("cold", "warm"):
                model.request_metrics.clear()
                started = monotonic()
                result = generate_grounded_answer(Agent(model, system_prompt=GROUNDING_SYSTEM_PROMPT, max_tokens=profile.max_tokens),
                                                  case["question"], case["search_question"],
                                                  [SearchHit(h, h.get("score", 1)) for h in case["hits"]],
                                                  prompt_version=profile.prompt_version)
                report["cold_warm"].append({"profile": name, "label": label, "question_id": case["id"],
                                             "question": case["question"], "elapsed_seconds": monotonic() - started,
                                             "requests": model.request_metrics[:], **asdict(result)})
                assert result.status == "answered"
                save()
        started = monotonic()
        main.get_document_index().search("Executor", k=3)
        hits = main.get_document_index().search("Executor", k=3)
        main.get_document_reranker().score("Executor", hits)
        report["retrieval_startup_seconds"] = monotonic() - started
        with TemporaryDirectory(prefix="day29-http-") as temporary:
            for repeat in range(1, 4):
                for name in (("baseline", "optimized") if repeat % 2 else ("optimized", "baseline")):
                    os.environ["RAG_LLM_PROFILE"] = name
                    os.environ["CHAT_DB_PATH"] = str(Path(temporary) / f"{name}.sqlite3")
                    main.get_rag_chat_service.cache_clear()
                    profile = PROFILES[name]
                    warm = OllamaProvider(model="qwen3.5:9b-q4_K_M", base_url=base, num_ctx=profile.num_ctx)
                    warm.generate(messages=[{"role": "user", "content": "Reply with the word ready."}], max_tokens=8)
                    with TestClient(main.app) as client:
                        session = client.post("/api/rag-chat/sessions", json={"provider": "ollama"}).json()
                        for case in (cases[2], cases[12], cases[8]):
                            question = case["question"]
                            started = monotonic()
                            response = client.post(f"/api/rag-chat/sessions/{session['id']}/turns", json={"content": question})
                            response.raise_for_status()
                            turn = response.json()
                            report["http_runs"].append({"profile": name, "repeat": repeat, "elapsed_seconds": monotonic() - started, "turn": turn})
                            assert turn["status"] == "done", turn.get("error")
                            assert turn["provider"] == "ollama"
                            assert turn["model"] == "qwen3.5:9b-q4_K_M"
                            expected = "insufficient_context" if case["expected_status"] == "unknown" else "answered"
                            assert turn["grounding_status"] == expected
                            assert turn["metrics"]["generation_profile"]["name"] == name
                            save()
                            print(f"HTTP {name} #{repeat}: {turn['grounding_status']} {turn['metrics']['total_seconds']:.2f}s", flush=True)
            # Preserve demonstration databases separate from the user's normal chat database.
            for name in ("baseline", "optimized"):
                shutil.copyfile(Path(temporary) / f"{name}.sqlite3", ROOT / f"data/day29-{name}.sqlite3")
    report["complete"] = True
    report["power_state_end"] = power_state()
    save()
    print(target)


if __name__ == "__main__":
    main_check()
