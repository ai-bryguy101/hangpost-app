"""Tests for the Phase 2 embedding utilities and bio_similarity scoring.

These tests deliberately avoid loading any real model — they pass plain
Python lists into `cosine_similarity` and `compute_match_score`, so the
suite runs fast and works without `sentence-transformers` or `numpy`.
"""

import math

from hangpost_matching import (
    UserProfile,
    compute_match_score,
    cosine_similarity,
    rank_candidates,
)


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


def test_bio_similarity_zero_when_no_embeddings_provided() -> None:
    source = UserProfile(user_id="a", bio="loves hiking and coffee")
    candidate = UserProfile(user_id="b", bio="also loves hiking and coffee")

    breakdown = compute_match_score(source, candidate)

    assert breakdown.bio_similarity == 0.0


def test_bio_similarity_uses_provided_embeddings() -> None:
    source = UserProfile(user_id="a", bio="hiking")
    candidate = UserProfile(user_id="b", bio="hiking")
    embeddings = {"a": [1.0, 0.0, 0.0], "b": [1.0, 0.0, 0.0]}

    breakdown = compute_match_score(source, candidate, bio_embeddings=embeddings)

    assert breakdown.bio_similarity == 1.0


def test_bio_similarity_ignores_negative_cosine() -> None:
    source = UserProfile(user_id="a")
    candidate = UserProfile(user_id="b")
    embeddings = {"a": [1.0, 0.0], "b": [-1.0, 0.0]}

    breakdown = compute_match_score(source, candidate, bio_embeddings=embeddings)

    assert breakdown.bio_similarity == 0.0


def test_bio_similarity_zero_when_one_user_missing_embedding() -> None:
    source = UserProfile(user_id="a")
    candidate = UserProfile(user_id="b")
    embeddings = {"a": [1.0, 0.0]}  # no entry for "b"

    breakdown = compute_match_score(source, candidate, bio_embeddings=embeddings)

    assert breakdown.bio_similarity == 0.0


def test_bio_similarity_breaks_ties_in_ranking() -> None:
    """Two candidates equal on every other signal — bio embedding decides."""
    source = UserProfile(
        user_id="source",
        interests={"hiking"},
        liked_topics={"music"},
        location="austin",
        age=30,
    )
    similar_bio = UserProfile(
        user_id="similar",
        interests={"hiking"},
        liked_topics={"music"},
        location="austin",
        age=30,
    )
    different_bio = UserProfile(
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

    ranked = rank_candidates(source, [different_bio, similar_bio], bio_embeddings=embeddings)

    assert ranked[0][0].user_id == "similar"
    assert ranked[1][0].user_id == "different"
    assert ranked[0][1].bio_similarity > ranked[1][1].bio_similarity
