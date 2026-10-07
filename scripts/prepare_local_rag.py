"""Copy the W4 index, PDF, and cached model snapshots without downloading anything."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
EMBEDDER_ID = "intfloat/multilingual-e5-small"
RERANKER_ID = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def snapshot(cache: Path, model: str) -> Path:
    model_cache = cache / ("models--" + model.replace("/", "--"))
    ref = model_cache / "refs/main"
    if ref.is_file():
        path = model_cache / "snapshots" / ref.read_text().strip()
        if path.is_dir():
            return path
    raise FileNotFoundError(f"Нет локального snapshot {model} в {cache}; укажите его явно")


def copy_verified(source: Path, target: Path) -> None:
    if target.exists():
        if digest(source) != digest(target):
            raise ValueError(f"Существующий файл отличается от источника: {target}")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def copy_model(source: Path, target: Path) -> dict:
    if not (source / "config.json").is_file() or not (source / "tokenizer.json").is_file():
        raise ValueError(f"Неполный локальный snapshot: {source}")
    files = [p for p in source.rglob("*") if p.is_file()
             and not any(part in {"onnx", "openvino"} for part in p.relative_to(source).parts)]
    if not any(p.suffix in {".bin", ".safetensors"} for p in files):
        raise ValueError(f"Нет весов модели: {source}")
    hashes = {}
    for path in files:
        relative = path.relative_to(source)
        copy_verified(path, target / relative)
        hashes[str(relative)] = digest(target / relative)
    return {"revision": source.name, "files_sha256": hashes}


def prepare(source: Path, data: Path, embedding_snapshot: Path, reranker_snapshot: Path) -> dict:
    index_root = source / "data/document_index"
    manifest = json.loads((index_root / "active.json").read_text())
    if manifest["model"] != EMBEDDER_ID:
        raise ValueError("Ожидается индекс multilingual-e5-small из W4")
    version = manifest["version"]
    directory = (index_root / version).resolve()
    if not directory.is_relative_to(index_root.resolve()):
        raise ValueError("Некорректный путь версии индекса")
    # Validate all inputs before starting the copy.
    inputs = [(source / "data/documents/jcip-sample.pdf", data / "documents/jcip-sample.pdf")]
    inputs += [(directory / f"{strategy}.{ext}", data / "document_index" / version / f"{strategy}.{ext}")
               for strategy in ("fixed", "structure") for ext in ("faiss", "json")]
    for path, _ in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    models = {
        EMBEDDER_ID: copy_model(embedding_snapshot, data / "models/e5"),
        RERANKER_ID: copy_model(reranker_snapshot, data / "models/reranker"),
    }
    for origin, destination in inputs:
        copy_verified(origin, destination)
    copy_verified(index_root / "active.json", data / "document_index/active.json")
    report = {"index_version": version, "embedding_model": EMBEDDER_ID,
              "models": models, "files_sha256": {str(target.relative_to(data)): digest(target) for _, target in inputs}}
    (data / "local-rag-assets.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT.parent / "ai-challange-w4")
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--embedding-snapshot", type=Path)
    parser.add_argument("--reranker-snapshot", type=Path)
    args = parser.parse_args()
    embedding = args.embedding_snapshot or snapshot(Path.home() / ".cache/huggingface/hub", EMBEDDER_ID)
    reranker = args.reranker_snapshot or snapshot(args.source / "data/model_cache", RERANKER_ID)
    report = prepare(args.source, args.data, embedding, reranker)
    print(f"Готово: {args.data}; версия индекса {report['index_version']}; все веса локальные")


if __name__ == "__main__":
    main()
