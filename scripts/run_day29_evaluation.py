"""Paired, local-only generation benchmark on frozen calibration/validation evidence."""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
from threading import Event, Thread
from time import monotonic
from urllib.parse import urlparse

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.agents.agent import Agent
from app.indexing.store import SearchHit
from app.providers.ollama import OllamaProvider
from app.rag_chat.generation import GROUNDING_SYSTEM_PROMPT, generate_grounded_answer
from app.rag_chat.profiles import PROFILES


CANDIDATES = {
    **PROFILES,
    "ctx16k": replace(PROFILES["compact"], name="ctx16k", num_ctx=16384),
    "cap1500": replace(PROFILES["compact"], name="cap1500", max_tokens=1500),
    "temp01": replace(PROFILES["compact"], name="temp01", temperature=.1),
    "temp02": replace(PROFILES["compact"], name="temp02", temperature=.2),
    "promptv1": replace(PROFILES["optimized"], name="promptv1", prompt_version="complete-v1"),
    "promptv2": replace(PROFILES["optimized"], name="promptv2", prompt_version="complete-v2"),
    "q8": replace(PROFILES["optimized"], name="q8"),
}


def cpu_seconds(value: str) -> float:
    days, _, clock = value.partition("-")
    if not clock:
        clock, days = days, "0"
    result = 0.0
    for part in clock.split(":"):
        result = result * 60 + float(part)
    return int(days) * 86400 + result


def power_state() -> dict:
    if platform.system() != "Darwin":
        return {"source": "unavailable", "power_mode": None}
    battery = subprocess.check_output(["pmset", "-g", "batt"], text=True)
    source = re.search(r"Now drawing from '([^']+)'", battery).group(1)
    settings = subprocess.check_output(["pmset", "-g", "custom"], text=True)
    section = re.search(re.escape(source) + r":\n(.*?)(?=^[^\s].*?:|\Z)", settings, re.M | re.S)
    mode = re.search(r"powermode\s+(\d+)", section.group(1)) if section else None
    return {"source": source, "power_mode": int(mode.group(1)) if mode else None}


class ResourceSampler:
    """Sample Ollama allocation and server/runner RSS; never add unified memory twice."""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.stop = Event()
        self.samples = []
        self.error_types = set()
        self.thread = Thread(target=self._run, daemon=True)

    def _sample(self, client):
        processes = {}
        raw = subprocess.check_output(["ps", "-axo", "pid,ppid,rss,time,command"], text=True)
        inventory = {}
        selected = set()
        for line in raw.splitlines()[1:]:
            fields = line.strip().split(None, 4)
            if len(fields) == 5:
                inventory[fields[0]] = fields
                if "ollama serve" in fields[4]:
                    selected.add(fields[0])
        # GGUF models use a llama-server child, not necessarily an ollama runner.
        while True:
            descendants = {pid for pid, fields in inventory.items() if fields[1] in selected}
            if descendants <= selected:
                break
            selected |= descendants
        for pid in selected:
            fields = inventory[pid]
            processes[pid] = {"rss_bytes": int(fields[2]) * 1024,
                              "cpu_seconds": cpu_seconds(fields[3])}
        response = client.get(self.base_url + "/api/ps")
        response.raise_for_status()
        self.samples.append({"processes": processes, "models": response.json()["models"]})

    def _run(self):
        with httpx.Client(timeout=2, trust_env=False) as client:
            while not self.stop.is_set():
                try:
                    self._sample(client)
                except Exception as error:
                    self.error_types.add(type(error).__name__)
                self.stop.wait(.25)

    def finish(self) -> dict:
        self.stop.set()
        self.thread.join(timeout=3)
        initial = self.samples[0]["processes"] if self.samples else {}
        cpu = {}
        for sample in self.samples:
            for pid, data in sample["processes"].items():
                cpu[pid] = max(cpu.get(pid, 0), data["cpu_seconds"] - initial.get(pid, {}).get("cpu_seconds", 0))
        return {
            "process_scope": "ollama serve and all descendants (including llama-server)",
            "sample_interval_seconds": .25, "samples": len(self.samples),
            "peak_ollama_rss_bytes": max((sum(p["rss_bytes"] for p in s["processes"].values()) for s in self.samples), default=None),
            "peak_model_allocation_bytes": max((sum(m["size"] for m in s["models"]) for s in self.samples), default=None),
            "peak_model_vram_bytes": max((sum(m["size_vram"] for m in s["models"]) for s in self.samples), default=None),
            "sampled_context_lengths": sorted({m["context_length"] for s in self.samples for m in s["models"]}),
            "ollama_cpu_seconds": round(sum(cpu.values()), 3),
            "sampling_errors": sorted(self.error_types),
        }


def percentile(values, fraction):
    return sorted(values)[max(0, int(len(values) * fraction + .999) - 1)] if values else None


def summarize(rows):
    summary = {}
    for name in sorted({r["profile"] for r in rows}):
        group = [r for r in rows if r["profile"] == name]
        times = [r["elapsed_seconds"] for r in group]
        rates = [r["tokens_per_second"] for r in group if r["tokens_per_second"]]
        answer_times = [r["elapsed_seconds"] for r in group if r["expected_status"] == "answered"]
        refusal_times = [r["elapsed_seconds"] for r in group if r["expected_status"] == "unknown"]
        summary[name] = {
            "runs": len(group), "status_passes": sum(r["status_pass"] for r in group),
            "first_attempt_valid": sum(r["metrics"].get("generation_attempts") == 1 and r["status"] in ("answered", "insufficient_context") for r in group),
            "provider_errors": sum(r["status"] == "provider_error" for r in group),
            "median_seconds": round(statistics.median(times), 3), "p95_seconds": round(percentile(times, .95), 3),
            "median_tokens_per_second": round(statistics.median(rates), 2) if rates else None,
            "median_answer_seconds": round(statistics.median(answer_times), 3) if answer_times else None,
            "median_refusal_seconds": round(statistics.median(refusal_times), 3) if refusal_times else None,
            "peak_model_vram_bytes": max((r["resources"]["peak_model_vram_bytes"] or 0 for r in group)),
            "peak_ollama_rss_bytes": max((r["resources"]["peak_ollama_rss_bytes"] or 0 for r in group)),
            "quality_review": "pending; status/format validation is not semantic grading",
        }
    return summary


def run(args):
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    if urlparse(args.base_url).hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("Resource benchmark requires a loopback Ollama endpoint")
    case_bytes = args.cases.read_bytes()
    fixture = json.loads(case_bytes)
    cases = [c for c in fixture["cases"] if args.split == "all" or c["split"] == args.split]
    if args.question_ids:
        cases = [c for c in cases if c["id"] in args.question_ids]
        if len(cases) != len(set(args.question_ids)):
            raise ValueError("Unknown question IDs or IDs outside the selected split")
    definitions = {name: {**asdict(CANDIDATES[name]), "model": args.q8_model if name == "q8" else args.model} for name in args.profiles}
    config = {"cases_sha256": hashlib.sha256(case_bytes).hexdigest(), "split": args.split,
              "repeats": args.repeats, "profiles": definitions, "seed_policy": "1000 + repeat; same seed for paired profiles"}
    config["generation_code_sha256"] = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in (
        "app/providers/ollama.py", "app/rag_chat/citation_options.py", "app/rag_chat/generation.py",
        "app/agents/agent.py", "app/orchestration/prompt.py",
    )}
    if args.question_ids:
        config["question_ids"] = sorted(args.question_ids)
    initial_power = power_state() if args.power_guard else None
    if args.power_guard:
        if initial_power["source"] == "unavailable":
            raise ValueError("Power guard is currently supported on macOS only")
        config["power_state"] = initial_power
    with httpx.Client(timeout=30, trust_env=False) as client:
        tags = client.get(args.base_url + "/api/tags")
        tags.raise_for_status()
        installed = {m["name"]: m for m in tags.json()["models"]}
        for definition in definitions.values():
            if definition["model"] not in installed:
                raise ValueError("Model not installed: " + definition["model"])
        if args.output.exists():
            if not args.resume:
                raise ValueError("Output exists; choose another --output or use --resume")
            report = json.loads(args.output.read_text())
            if report["config"] != config:
                raise ValueError("Resume configuration differs from saved report")
            for model in report["models"]:
                if report["models"][model]["digest"] != installed[model]["digest"]:
                    raise ValueError("Model digest changed")
        else:
            report = {"created_at": datetime.now(timezone.utc).isoformat(), "config": config,
                      "platform": platform.platform(), "models": {m: installed[m] for m in {d["model"] for d in definitions.values()}},
                      "ollama_version": client.get(args.base_url + "/api/version").json(),
                      "method": fixture["method"], "cases": cases, "runs": [], "warmups": [],
                      "limitations": ["No retrieval or interpretation is included in generator timing.",
                                      "0.25s sampling can miss transient peaks. RSS and Metal unified-memory allocation overlap and must not be added.",
                                      "Warm blocks include prompt-prefix caching; unrelated warmup is excluded from timing. Profile block order rotates across repeats.",
                                      "Status correctness and exact citations do not establish semantic completeness; review stored answers against expected facts."]}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        def save():
            report["summary"] = summarize(report["runs"])
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        done = {(r["profile"], r["question_id"], r["repeat"]) for r in report["runs"]}
        for repeat in range(1, args.repeats + 1):
            offset = (repeat - 1) % len(args.profiles)
            order = args.profiles[offset:] + args.profiles[:offset]
            for name in order:
                if all((name, c["id"], repeat) in done for c in cases):
                    continue
                profile = CANDIDATES[name]
                model_name = definitions[name]["model"]
                # One resident model/context at a time; unloading also removes prefix cache.
                for loaded in client.get(args.base_url + "/api/ps").json()["models"]:
                    unloaded = client.post(args.base_url + "/api/generate", json={"model": loaded["name"], "keep_alive": 0})
                    unloaded.raise_for_status()
                model = OllamaProvider(model=model_name, base_url=args.base_url, num_ctx=profile.num_ctx,
                                       max_num_ctx=profile.max_num_ctx, temperature=profile.temperature, seed=1000 + repeat)
                warm_start = monotonic()
                model.generate(messages=[{"role": "user", "content": "Reply with the word ready."}], max_tokens=8)
                report["warmups"].append({"profile": name, "repeat": repeat, "elapsed_seconds": monotonic() - warm_start,
                                           "requests": model.request_metrics[:]})
                rotated = cases[(repeat - 1) % len(cases):] + cases[:(repeat - 1) % len(cases)]
                for case in rotated:
                    if (name, case["id"], repeat) in done:
                        continue
                    if args.power_guard and power_state() != initial_power:
                        raise ValueError("Power source/mode changed; choose a new output and rerun under stable power")
                    model.request_metrics.clear()
                    sampler = ResourceSampler(args.base_url)
                    sampler.thread.start()
                    metrics = {"generation_attempts": 0}
                    started = monotonic()
                    try:
                        answer = generate_grounded_answer(
                            Agent(model, system_prompt=GROUNDING_SYSTEM_PROMPT, max_tokens=profile.max_tokens),
                            case["question"], case["search_question"],
                            [SearchHit(h, h.get("score", 1)) for h in case["hits"]],
                            prompt_version=profile.prompt_version, metrics=metrics,
                        )
                        data = asdict(answer)
                    except Exception as error:
                        data = {"content": "", "sources": [], "citations": [], "status": "provider_error", "error": str(error), "error_type": type(error).__name__}
                    elapsed = monotonic() - started
                    resources = sampler.finish()
                    if args.power_guard and power_state() != initial_power:
                        raise ValueError("Power source/mode changed during generation; rerun under stable power")
                    metrics["ollama_requests"] = model.request_metrics[:]
                    count = sum(r.get("eval_count", 0) for r in model.request_metrics)
                    duration = sum(r.get("eval_duration", 0) for r in model.request_metrics) / 1e9
                    expected = "insufficient_context" if case["expected_status"] == "unknown" else "answered"
                    row = {"profile": name, "question_id": case["id"], "split": case["split"], "repeat": repeat,
                           "expected_status": case["expected_status"],
                           "model": model_name, "elapsed_seconds": elapsed, "status_pass": data["status"] == expected,
                           "tokens_per_second": count / duration if duration else None, "resources": resources, "metrics": metrics, **data}
                    if args.power_guard:
                        row["power_state"] = initial_power
                    report["runs"].append(row)
                    done.add((name, case["id"], repeat))
                    save()
                    print(f"{name} Q{case['id']} #{repeat}: {data['status']} {elapsed:.2f}s status={row['status_pass']}", flush=True)
        report["complete"] = True
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=ROOT / "evaluation/day29-cases.json")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/day29-artifacts/evaluation.json")
    parser.add_argument("--split", choices=["calibration", "validation", "all"], default="all")
    parser.add_argument("--profiles", nargs="+", choices=list(CANDIDATES), default=["baseline", "compact", "optimized", "q8"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--model", default="qwen3.5:9b-q4_K_M")
    parser.add_argument("--q8-model", default="qwen3.5:9b-q8_0")
    parser.add_argument("--base-url", default=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--question-ids", nargs="+", type=int)
    parser.add_argument("--power-guard", action="store_true", help="Require unchanged power source/mode before and after every call")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
