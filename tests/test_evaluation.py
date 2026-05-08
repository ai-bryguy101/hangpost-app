"""Tests for the offline evaluation harness.

Each metric is exercised against small hand-computed cases so the
expected values are obvious from the test names alone.
"""

import math

from hangpost_matching import (
    UserProfile,
    average_precision_at_k,
    evaluate_ranker,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    synthesize_relevance,
)

# ---------- precision@k ----------


def test_precision_at_k_perfect_ranking() -> None:
    assert precision_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3) == 1.0


def test_precision_at_k_partial() -> None:
    # 2 of top 4 are relevant -> 0.5
    assert precision_at_k(["a", "x", "b", "y"], {"a", "b"}, k=4) == 0.5


def test_precision_at_k_zero_when_no_hits() -> None:
    assert precision_at_k(["x", "y", "z"], {"a", "b"}, k=3) == 0.0


def test_precision_at_k_handles_short_retrieval() -> None:
    # only 2 items retrieved but k=10 — denominator is 2, not 10
    assert precision_at_k(["a", "b"], {"a"}, k=10) == 0.5


def test_precision_at_k_zero_for_non_positive_k() -> None:
    assert precision_at_k(["a", "b"], {"a"}, k=0) == 0.0


# ---------- recall@k ----------


def test_recall_at_k_perfect() -> None:
    assert recall_at_k(["a", "b"], {"a", "b"}, k=2) == 1.0


def test_recall_at_k_partial() -> None:
    # 1 of 2 relevant items in top 3 -> 0.5
    assert recall_at_k(["a", "x", "y"], {"a", "b"}, k=3) == 0.5


def test_recall_at_k_zero_when_no_relevant_set() -> None:
    assert recall_at_k(["a", "b"], set(), k=2) == 0.0


# ---------- average precision@k ----------


def test_average_precision_at_k_perfect_top() -> None:
    # all relevant items at the very top
    # P@1=1, P@2=1 -> AP = (1+1)/2 = 1.0
    assert average_precision_at_k(["a", "b", "x"], {"a", "b"}, k=3) == 1.0


def test_average_precision_at_k_known_value() -> None:
    # ranks 1 and 3 are relevant out of 2 total relevant
    # P@1 = 1/1 = 1.0
    # P@3 = 2/3
    # AP = (1.0 + 2/3) / 2 = 0.8333...
    result = average_precision_at_k(["a", "x", "b"], {"a", "b"}, k=3)
    assert math.isclose(result, (1.0 + 2 / 3) / 2, rel_tol=1e-9)


def test_average_precision_at_k_zero_when_no_hits() -> None:
    assert average_precision_at_k(["x", "y"], {"a"}, k=2) == 0.0


# ---------- NDCG@k ----------


def test_ndcg_at_k_perfect_ranking_is_one() -> None:
    assert ndcg_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3) == 1.0


def test_ndcg_at_k_known_value() -> None:
    # only one relevant item at rank 2 -> DCG = 1/log2(3)
    # IDCG (1 relevant, ideal at rank 1) = 1/log2(2) = 1
    # NDCG = (1/log2(3)) / 1 = 0.6309...
    result = ndcg_at_k(["x", "a", "y"], {"a"}, k=3)
    assert math.isclose(result, 1.0 / math.log2(3), rel_tol=1e-9)


def test_ndcg_at_k_zero_with_no_hits() -> None:
    assert ndcg_at_k(["x", "y", "z"], {"a"}, k=3) == 0.0


# ---------- evaluate_ranker ----------


def test_evaluate_ranker_aggregates_across_queries() -> None:
    profile_a = UserProfile(user_id="a")
    profile_b = UserProfile(user_id="b")
    profile_c = UserProfile(user_id="c")

    def perfect_ranker(_source: UserProfile, candidates: list[UserProfile]) -> list[str]:
        # Always returns relevant items first.
        return [p.user_id for p in candidates]

    queries = [
        (profile_a, [profile_b, profile_c], {"b", "c"}),
        (profile_a, [profile_b], {"b"}),
    ]

    result = evaluate_ranker(perfect_ranker, queries, k=2)

    assert result.n_queries == 2
    assert result.k == 2
    assert result.precision == 1.0
    assert result.ndcg == 1.0
    assert result.map == 1.0


def test_evaluate_ranker_handles_empty_query_set() -> None:
    def ranker(_source: UserProfile, _candidates: list[UserProfile]) -> list[str]:
        return []

    result = evaluate_ranker(ranker, [], k=10)

    assert result.n_queries == 0
    assert result.precision == 0.0


# ---------- synthesize_relevance ----------


def test_synthesize_relevance_true_when_three_signals_align() -> None:
    source = UserProfile(
        user_id="src",
        interests={"a", "b", "c"},
        liked_topics={"x", "y"},
        location="austin",
        age=30,
    )
    matched = UserProfile(
        user_id="match",
        interests={"a", "b"},  # ≥2 shared
        liked_topics={"x", "y"},  # ≥2 shared
        location="austin",  # same hometown -> 3 signals
        age=50,
    )

    assert synthesize_relevance(source, matched) is True


def test_synthesize_relevance_false_when_only_two_signals() -> None:
    source = UserProfile(
        user_id="src",
        interests={"a", "b"},
        liked_topics={"x", "y"},
        location="austin",
        age=30,
    )
    weak = UserProfile(
        user_id="weak",
        interests={"a", "b"},  # ≥2 shared (signal 1)
        liked_topics={"q", "r"},
        location="seattle",
        age=29,  # age close (signal 2)
    )

    assert synthesize_relevance(source, weak) is False
