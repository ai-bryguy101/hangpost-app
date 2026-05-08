"""Phase 2: text embeddings for semantic profile similarity.

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
`compute_match_score(..., profile_embeddings=...)`. That separation keeps
the ranker pure (no I/O, no model state) and makes batch precomputation
easy.

Hangpost users do NOT write free-text bios. The text that gets embedded
is auto-synthesized from the structured fields each user has already
provided (interests, liked topics, hometown, age). See `profile_to_text`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
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


def profile_to_text(profile: UserProfile) -> str:
    """Compose a deterministic natural-language summary of a profile.

    This intentionally NEVER consumes a hand-written bio — Hangpost users
    don't write bios. The semantic representation is built from the same
    structured fields the rest of the engine already uses, so every user
    automatically gets a useful embedding without extra typing.

    The fields are joined in a fixed order with sorted set contents so the
    output is reproducible (re-embedding the same profile twice yields the
    same vector, which matters for caching and evaluation).

    Returns an empty string if the profile has no usable fields, in which
    case `embed_profiles` will skip it.
    """
    parts: list[str] = []
    if profile.age is not None:
        parts.append(f"{profile.age} years old")
    if profile.location:
        parts.append(f"from {profile.location}")
    if profile.interests:
        parts.append(f"enjoys {', '.join(sorted(profile.interests))}")
    if profile.liked_topics:
        parts.append(f"likes {', '.join(sorted(profile.liked_topics))}")
    if not parts:
        return ""
    return ". ".join(parts) + "."


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
            from sentence_transformers import SentenceTransformer
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
    text_fn: Callable[[UserProfile], str] = profile_to_text,
) -> dict[str, Vector]:
    """Precompute `{user_id: vector}` for every profile.

    Each profile is run through `text_fn` (default: `profile_to_text`) to
    produce the string that gets embedded. Profiles whose `text_fn` output
    is empty are skipped — the ranker treats missing embeddings as a 0.0
    contribution, never an error.

    Pass a custom `text_fn` to experiment with different ways of describing
    a profile (e.g., for ablation studies in evaluation).
    """
    result: dict[str, Vector] = {}
    for profile in profiles:
        text = text_fn(profile)
        if not text:
            continue
        result[profile.user_id] = embedder.embed(text)
    return result
