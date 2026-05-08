"""Phase 2: text embeddings for semantic bio similarity.

Design principles
-----------------
This module is intentionally lightweight in its core shape so that:

- Unit tests run without `sentence-transformers` or `numpy` installed.
- CI is fast and never downloads model weights.
- Production code can swap in any embedder (sentence-transformers,
  OpenAI, Cohere, a local model, etc.) by implementing the `Embedder`
  Protocol below.

The matching engine does not load any model itself. Instead it accepts a
precomputed `{user_id: vector}` dict via
`compute_match_score(..., bio_embeddings=...)`. That separation keeps the
ranker pure (no I/O, no model state) and makes batch precomputation easy.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Protocol

from .models import UserProfile

# A vector is any sequence of floats. Using `Sequence[float]` lets callers
# pass plain Python lists (in tests) or `numpy.ndarray` (in production)
# without the core needing to import numpy.
Vector = Sequence[float]


def cosine_similarity(a: Vector, b: Vector) -> float:
    """Return cosine similarity in [-1.0, 1.0].

    Returns 0.0 for empty vectors, mismatched lengths, or zero-magnitude
    inputs. Pure-Python so this works with or without numpy installed.

    Cosine similarity formula:
        sim(a, b) = (a . b) / (||a|| * ||b||)
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class Embedder(Protocol):
    """Anything with an `embed(text) -> Vector` method.

    Keep this Protocol tiny so swapping providers is trivial.
    """

    def embed(self, text: str) -> Vector: ...


class SentenceTransformerEmbedder:
    """Concrete embedder backed by a Hugging Face sentence-transformer model.

    `sentence-transformers` is imported lazily inside `__init__` so that
    importing this module never pulls in the heavy ML stack. Install with:

        pip install -e ".[ml]"

    The default model (`all-MiniLM-L6-v2`) is a strong, ~90MB baseline:
    fast, multilingual-friendly, and standard in semantic-search tutorials.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerEmbedder. "
                'Install with: pip install -e ".[ml]"'
            ) from exc
        self._model = SentenceTransformer(model_name)

    def embed(self, text: str) -> Vector:
        """Encode `text` into a list of floats.

        Returns a `list` rather than `numpy.ndarray` so downstream
        consumers (and tests) don't need numpy to handle the result.
        """
        vector = self._model.encode(text)
        return [float(x) for x in vector]


def embed_profiles(
    profiles: Iterable[UserProfile],
    embedder: Embedder,
) -> dict[str, Vector]:
    """Precompute `{user_id: vector}` for every profile that has a bio.

    Profiles without a bio are skipped — the ranker treats missing
    embeddings as a 0.0 bio_similarity contribution, never an error.
    """
    return {profile.user_id: embedder.embed(profile.bio) for profile in profiles if profile.bio}
