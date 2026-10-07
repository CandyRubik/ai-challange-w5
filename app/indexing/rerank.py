from __future__ import annotations

import os
from pathlib import Path
from threading import Lock
from typing import Protocol, Sequence

from .store import DocumentIndexError, SearchHit


DEFAULT_RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
DEFAULT_MODEL_CACHE = Path(__file__).resolve().parents[2] / "data/model_cache"


class RelevanceScorer(Protocol):
    def score(self, question: str, hits: Sequence[SearchHit]) -> list[float]: ...


class LocalCrossEncoderReranker:
    """Score candidate passages against the user's original question."""

    def __init__(self, model_name: str | None = None) -> None:
        local_model = Path(__file__).resolve().parents[2] / "data/models/reranker"
        self.model_name = model_name or os.getenv(
            "RAG_RERANKER_MODEL", str(local_model) if local_model.is_dir() else DEFAULT_RERANKER_MODEL,
        )
        self.cache_dir = Path(os.getenv("RAG_MODEL_CACHE_DIR", str(DEFAULT_MODEL_CACHE)))
        self._model = None
        self._lock = Lock()

    def score(self, question: str, hits: Sequence[SearchHit]) -> list[float]:
        if not hits:
            return []
        with self._lock:
            if self._model is None:
                try:
                    import torch
                    from sentence_transformers import CrossEncoder
                except ImportError as error:
                    raise DocumentIndexError("Установите зависимости: pip install -r requirements-indexing.txt") from error

                torch.set_num_threads(1)
                options = {
                    "device": "cpu",
                    "cache_folder": str(self.cache_dir),
                    "activation_fn": torch.nn.Sigmoid(),
                }
                try:
                    self._model = CrossEncoder(
                        self.model_name, local_files_only=True, **options,
                    )
                except (OSError, ValueError) as error:
                    raise DocumentIndexError(
                        "Локальные веса reranker не найдены. Выполните scripts/prepare_local_rag.py"
                    ) from error
            pairs = [
                (question, f"{hit.metadata['section']}\n{hit.metadata['text']}")
                for hit in hits
            ]
            values = self._model.predict(pairs, batch_size=8, show_progress_bar=False)
        return [float(value) for value in values]
