"""Tests for the Phase 2 embedding utilities and semantic_similarity scoring.

These tests deliberately avoid loading any real model — they pass plain
Python lists into `cosine_similarity` and `compute_match_score`, so the
suite runs fast and works without `sentence-transformers` or `numpy`.
"""

import math

from hangpost_matching import (
    UserProfile,
    compute_match_score,
    cosine_similarity,
    embed_profiles,
    profile_to_text,
    rank_candidates,
)
from hangpost_matching.embeddings import Vector


class _StubEmbedder:
    """Records every text it sees so tests can verify the synthesized input."""

    def __init__(self) -> None:
        self.texts_seen: list[str] = []

    def embed(self, text: str) -> Vector:
        self.texts_seen.append(text)
        # Return a vector whose first dim encodes string length so different
        # texts produce different vectors. Good enough for tests.
        return [float(len(text)), 0.0, 0.0]


def test_cosine_similarity_identical_vectors_is_one() -> None:
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0


def test_cosine_similarity_orthogonal_vectors_is_zero() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_similarity_opposite_vectors_is_negative_one() -> None:
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == -1.0


def test_cosine_similarity_handles_empty_and_mismatched_lengths() -> None:
    assert cosine_similarity([], [1.0]) == 0.0
    assert cosine_similarity([1.0, 2.0], [1.0]) == 0.0


def test_cosine_similarity_zero_vector_returns_zero() -> None:
    assert cosine_similarity([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]) == 0.0


def test_cosine_similarity_known_value() -> None:
    # angle of 60 degrees -> cosine = 0.5
    sim = cosine_similarity([1.0, 0.0], [0.5, math.sqrt(3) / 2])
    assert math.isclose(sim, 0.5, rel_tol=1e-9)


def test_profile_to_text_uses_structured_fields() -> None:
    profile = UserProfile(
        user_id="u",
        interests={"hiking", "coding"},
        liked_topics={"tech", "travel"},
        location="austin",
        age=30,
    )

    text = profile_to_text(profile)

    assert "30 years old" in text
    assert "from austin" in text
    assert "coding" in text and "hiking" in text
    assert "tech" in text and "travel" in text


def test_profile_to_text_is_deterministic() -> None:
    """Same inputs must produce the same string so embeddings are reproducible."""
    profile_a = UserProfile(
        user_id="a",
        interests={"chess", "yoga", "coding"},
        liked_topics={"music", "art"},
        location="boston",
        age=25,
    )
    profile_b = UserProfile(
        user_id="b",
        interests={"yoga", "coding", "chess"},  # different insertion order
        liked_topics={"art", "music"},
        location="boston",
        age=25,
    )

    assert profile_to_text(profile_a) == profile_to_text(profile_b)


def test_profile_to_text_returns_empty_for_blank_profile() -> None:
    assert profile_to_text(UserProfile(user_id="empty")) == ""


def test_embed_profiles_synthesizes_text_from_structured_fields() -> None:
    """`embed_profiles` should not require a bio — it builds text itself."""
    profiles = [
        UserProfile(user_id="a", interests={"hiking"}, age=28),
        UserProfile(user_id="b", interests={"chess"}, age=40),
        UserProfile(user_id="empty"),  # no fields -> skipped
    ]
    embedder = _StubEmbedder()

    embeddings = embed_profiles(profiles, embedder)

    assert set(embeddings.keys()) == {"a", "b"}  # empty profile skipped
    assert any("hiking" in t for t in embedder.texts_seen)
    assert any("chess" in t for t in embedder.texts_seen)


def test_semantic_similarity_zero_when_no_embeddings_provided() -> None:
    source = UserProfile(user_id="a", interests={"hiking"})
    candidate = UserProfile(user_id="b", interests={"hiking"})

    breakdown = compute_match_score(source, candidate)

    assert breakdown.semantic_similarity == 0.0


def test_semantic_similarity_uses_provided_embeddings() -> None:
    source = UserProfile(user_id="a", interests={"hiking"})
    candidate = UserProfile(user_id="b", interests={"hiking"})
    embeddings = {"a": [1.0, 0.0, 0.0], "b": [1.0, 0.0, 0.0]}

    breakdown = compute_match_score(source, candidate, profile_embeddings=embeddings)

    assert breakdown.semantic_similarity == 1.0


def test_semantic_similarity_ignores_negative_cosine() -> None:
    source = UserProfile(user_id="a")
    candidate = UserProfile(user_id="b")
    embeddings = {"a": [1.0, 0.0], "b": [-1.0, 0.0]}

    breakdown = compute_match_score(source, candidate, profile_embeddings=embeddings)

    assert breakdown.semantic_similarity == 0.0


def test_semantic_similarity_zero_when_one_user_missing_embedding() -> None:
    source = UserProfile(user_id="a")
    candidate = UserProfile(user_id="b")
    embeddings = {"a": [1.0, 0.0]}  # no entry for "b"

    breakdown = compute_match_score(source, candidate, profile_embeddings=embeddings)

    assert breakdown.semantic_similarity == 0.0


def test_semantic_similarity_breaks_ties_in_ranking() -> None:
    """Two candidates equal on every other signal — embedding decides."""
    source = UserProfile(
        user_id="source",
        interests={"hiking"},
        liked_topics={"music"},
        location="austin",
        age=30,
    )
    similar_profile = UserProfile(
        user_id="similar",
        interests={"hiking"},
        liked_topics={"music"},
        location="austin",
        age=30,
    )
    different_profile = UserProfile(
        user_id="different",
        interests={"hiking"},
        liked_topics={"music"},
        location="austin",
        age=30,
    )
    embeddings = {
        "source": [1.0, 0.0, 0.0],
        "similar": [0.9, 0.1, 0.0],
        "different": [0.0, 1.0, 0.0],
    }

    ranked = rank_candidates(
        source,
        [different_profile, similar_profile],
        profile_embeddings=embeddings,
    )

    assert ranked[0][0].user_id == "similar"
    assert ranked[1][0].user_id == "different"
    assert ranked[0][1].semantic_similarity > ranked[1][1].semantic_similarity
