"""Exercise real local RAG against the isolated run_rag_offline.py server."""

import argparse
import hashlib
import json
from pathlib import Path
from time import monotonic

import httpx


ROOT = Path(__file__).resolve().parents[1]


def run(base_url: str, output: Path, resume_session: str | None = None) -> None:
    report = {"base_url": base_url, "real_llm": True, "records": []}
    output.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(base_url=base_url, timeout=300, trust_env=False) as client:
        def call(method, path, body=None):
            started = monotonic()
            response = client.request(method, path, json=body)
            response.raise_for_status()
            if response.headers.get("content-type", "").startswith("application/pdf"):
                data = {"bytes": len(response.content), "sha256": hashlib.sha256(response.content).hexdigest()}
                assert response.content.startswith(b"%PDF")
            else:
                data = response.json()
            report["records"].append({"method": method, "path": path,
                                      "elapsed_seconds": monotonic() - started, "response": data})
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(method, path, round(report["records"][-1]["elapsed_seconds"], 2), flush=True)
            return data
        assert not call("GET", "/api/health")["deepseek_configured"]
        models = call("GET", "/api/models")["providers"]
        assert next(m for m in models if m["id"] == "ollama")["available"]
        assert not next(m for m in models if m["id"] == "deepseek")["available"]
        if resume_session:
            session = call("GET", f"/api/rag-chat/sessions/{resume_session}")
            assert session["state"]["goal"] and session["turns"]
        else:
            session = call("POST", "/api/rag-chat/sessions", {"provider": "ollama"})
        report["session_id"] = session["id"]
        path = f"/api/rag-chat/sessions/{session['id']}"
        def send(content):
            turn = call("POST", path + "/turns", {"content": content, "provider": "ollama"})
            assert turn["status"] == "done" and turn["provider"] == "ollama", turn.get("error")
            return turn
        if not resume_session:
            send("Моя цель — объяснить Executor новичку. Отвечай кратко.")
            snapshot = call("GET", path)
            assert snapshot["state"]["goal"] and snapshot["state"]["constraints"]
            document = send("Почему обработчик ThreadPerTaskWebServer должен быть потокобезопасным?")
            assert document["grounding_status"] == "answered" and document["citations"]
            assert all(source["kind"] == "document" for source in document["sources"])
            send("Нет, отвечай подробно.")
            corrected = call("GET", path)
            assert corrected["state"]["revision"] > snapshot["state"]["revision"]
            assert any("подроб" in fact["value"].lower() for fact in corrected["state"]["constraints"])
        reminder = send("Какую цель мы зафиксировали?")
        assert reminder["grounding_status"] == "state" and reminder["sources"]
        assert all(source["kind"] == "message" for source in reminder["sources"])
        negative = send("Как в этой главе работает Java StructuredTaskScope?")
        assert negative["grounding_status"] == "insufficient_context"
        assert not negative["citations"] and not negative["sources"]
        assert negative["metrics"]["generation_attempts"] == 0
        call("GET", "/api/document-index/source")
        report["final_session"] = call("GET", path)
        report["complete"] = True
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("LOCAL_RAG_SMOKE_COMPLETE", report["session_id"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8769")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/day28-artifacts/offline-smoke.json")
    parser.add_argument("--resume-session")
    args = parser.parse_args()
    run(args.base_url, args.output, args.resume_session)
