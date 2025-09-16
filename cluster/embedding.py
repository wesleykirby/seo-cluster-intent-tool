"""Utility helpers for working with the sentence-transformer encoder.

The project relies on a sentence-transformer model that has been fine-tuned
with a triplet loss objective on curated (anchor, positive, negative) keyword
examples.  The fine-tuned weights are expected to live in
``cluster/models/intent-encoder`` or a path supplied through the
``INTENT_ENCODER_PATH`` environment variable.  When those weights are not
available the helpers gracefully fall back to the base model so the rest of the
pipeline can still operate, albeit with lower quality clusters.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_FINETUNED_DIR = Path(__file__).resolve().parent / "models" / "intent-encoder"
ENV_MODEL_PATH = "INTENT_ENCODER_PATH"


def _resolve_model_path(model_path: str | os.PathLike[str] | None = None) -> Path | None:
    """Return the most appropriate location of the fine-tuned model weights."""

    if model_path is not None:
        path = Path(model_path)
        return path if path.exists() else None

    env_path = os.environ.get(ENV_MODEL_PATH)
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path

    if DEFAULT_FINETUNED_DIR.exists():
        return DEFAULT_FINETUNED_DIR

    return None


@lru_cache(maxsize=2)
def load_encoder(model_path: str | os.PathLike[str] | None = None) -> SentenceTransformer:
    """Load the sentence-transformer encoder used throughout the pipeline.

    Parameters
    ----------
    model_path:
        Optional path pointing to a directory containing the fine-tuned
        sentence-transformer weights.  When omitted the helper looks for the
        weights inside ``cluster/models/intent-encoder`` and finally falls back
        to the base model shipped with SentenceTransformers.
    """

    resolved = _resolve_model_path(model_path)
    if resolved is not None:
        return SentenceTransformer(str(resolved))
    return SentenceTransformer(DEFAULT_BASE_MODEL)


def encode_texts(
    texts: Sequence[str] | Iterable[str],
    *,
    normalize: bool = True,
    batch_size: int = 32,
    show_progress_bar: bool = False,
    model_path: str | os.PathLike[str] | None = None,
) -> np.ndarray:
    """Encode a collection of texts using the fine-tuned encoder.

    Parameters
    ----------
    texts:
        Iterable containing the strings that need to be embedded.
    normalize:
        Whether to L2-normalize the resulting embeddings.  Cosine similarity and
        HDBSCAN clustering perform better with normalized vectors, so the
        default is ``True``.
    batch_size:
        Batch size forwarded to :meth:`SentenceTransformer.encode`.
    show_progress_bar:
        When ``True`` the underlying model will show a progress bar while
        encoding; disabled by default to keep the command-line interface quiet.
    model_path:
        Optional override pointing to the model that should be used.

    Returns
    -------
    numpy.ndarray
        Two dimensional array where each row corresponds to the embedding of the
        matching text input.
    """

    model = load_encoder(model_path=model_path)
    vectors = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
    )
    if normalize:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero for empty or pathological vectors.
        norms[norms == 0.0] = 1.0
        vectors = vectors / norms
    return vectors


__all__ = ["load_encoder", "encode_texts", "DEFAULT_BASE_MODEL", "DEFAULT_FINETUNED_DIR"]
