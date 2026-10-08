"""Apply explicit human/agent review annotations to every saved benchmark response."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "docs/day29-artifacts"


def response_key(row):
    value = {key: row[key] for key in ("status", "content", "citations")}
    digest = hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return str(row["question_id"]) + ":" + digest


def main():
    raw = (ARTIFACTS / "evaluation.json").read_bytes()
    report = json.loads(raw)
    if not report.get("complete"):
        raise ValueError("Benchmark is not complete")
    rubric_raw = (ROOT / "evaluation/day29-rubric.json").read_bytes()
    rubric = {c["id"]: c for c in json.loads(rubric_raw)["cases"]}
    annotations = json.loads((ARTIFACTS / "quality-notes.json").read_text())
    runs = []
    for row in report["runs"]:
        expected = rubric[row["question_id"]]
        key = response_key(row)
        if key not in annotations["responses"]:
            raise ValueError("Unreviewed response: " + key)
        note = annotations["responses"][key]
        facts = note["facts_present"]
        required = len(expected["required_facts"])
        if len(set(facts)) != len(facts) or any(f < 1 or f > required for f in facts):
            raise ValueError("Invalid fact index in " + key)
        refusal = expected["expected_status"] == "unknown"
        runs.append({"profile": row["profile"], "question_id": row["question_id"], "repeat": row["repeat"],
                     "split": row["split"], "response_key": key, "facts_present": facts,
                     "complete": not refusal and row["status"] == "answered" and len(facts) == required,
                     "answered": row["status"] == "answered", "expected_refusal": refusal,
                     "correct_refusal": refusal and row["status"] == "insufficient_context" and not row["citations"],
                     "factually_correct": note["factually_correct"], "citation_support": note["citation_support"],
                     "note": note["note"]})
    def summarize(rows):
        result = {}
        for name in report["config"]["profiles"]:
            group = [r for r in rows if r["profile"] == name]
            answers = [r for r in group if not r["expected_refusal"]]
            refusals = [r for r in group if r["expected_refusal"]]
            result[name] = {"answer_runs": len(answers), "refusal_runs": len(refusals),
                            "complete_runs": sum(r["complete"] for r in answers),
                            "correct_answers": sum(r["answered"] and r["factually_correct"] for r in answers),
                            "supported_answers": sum(r["answered"] and r["citation_support"] for r in answers),
                            "acceptable_answers": sum(r["complete"] and r["factually_correct"] and r["citation_support"] for r in answers),
                            "correct_refusals": sum(r["correct_refusal"] for r in refusals)}
        return result
    review = {"evaluation_sha256": hashlib.sha256(raw).hexdigest(),
              "rubric_sha256": hashlib.sha256(rubric_raw).hexdigest(), "reviewer": annotations["reviewer"],
              "method": "Each distinct status/content/citation response was reviewed against the frozen required facts and its actual cited source excerpts. Identical responses for the same question reuse the same annotation across profiles/repeats.",
              "limitations": "Codex semantic review, no independent human review. Small fixed sample from one chapter. Completeness, factual correctness and citation support are recorded separately.",
              "summary": summarize(runs), "by_split": {split: summarize([r for r in runs if r["split"] == split]) for split in ("calibration", "validation")},
              "runs": runs}
    target = ARTIFACTS / "quality-review.json"
    target.write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(review["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
