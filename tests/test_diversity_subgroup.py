"""Tests for the diversity, subgroup, and graded-relevance additions.

These cover the resume-signal features added on top of the original
evaluation harness: per-subgroup metrics, intra-list diversity, MMR
re-ranking, graded NDCG, and the cold-start fallback path.
"""

from __future__ import annotations

import math

import pytest

from hangpost_matching import (
    UserProfile,
    age_band,
    evaluate_ranker_by_subgroup,
    intra_list_diversity,
    is_cold_start,
    make_mmr_reranker,
    mutual_friend_density_band,
    ndcg_at_k_graded,
    rank_candidates_with_cold_start,
)

# ---------- graded NDCG ----------


def test_ndcg_at_k_graded_perfect_ranking_is_one() -> None:
    gains = {"a": 15.0, "b": 7.0, "c": 1.0}  # 2**4-1, 2**3-1, 2**1-1
    assert ndcg_at_k_graded(["a", "b", "c"], gains, k=3) == 1.0


def test_ndcg_at_k_graded_penalises_swapped_high_value() -> None:
    """Swapping the top-rated item below a lower-rated one lowers NDCG."""
    gains = {"a": 15.0, "b": 7.0, "c": 1.0}
    perfect = ndcg_at_k_graded(["a", "b", "c"], gains, k=3)
    swapped = ndcg_at_k_graded(["b", "a", "c"], gains, k=3)
    assert swapped < perfect
    # Sanity: still positive, and the absolute value is computable.
    assert swapped > 0.0


def test_ndcg_at_k_graded_zero_when_no_gains() -> None:
    assert ndcg_at_k_graded(["x", "y"], {}, k=2) == 0.0


def test_ndcg_at_k_graded_known_value() -> None:
    # One rated item, gain 3, sitting at rank 2 → DCG = 3/log2(3),
    # IDCG = 3/log2(2) = 3, NDCG = 1/log2(3).
    gains = {"a": 3.0}
    result = ndcg_at_k_graded(["x", "a", "y"], gains, k=3)
    assert math.isclose(result, 1.0 / math.log2(3), rel_tol=1e-9)


# ---------- subgroup evaluation ----------


def test_evaluate_ranker_by_subgroup_partitions_queries() -> None:
    young = UserProfile(user_id="young", age=22)
    older = UserProfile(user_id="older", age=55)
    cand_a = UserProfile(user_id="a")
    cand_b = UserProfile(user_id="b")

    def perfect_ranker(_source: UserProfile, candidates: list[UserProfile]) -> list[str]:
        return [c.user_id for c in candidates]

    queries = [
        (young, [cand_a, cand_b], {"a"}),
        (older, [cand_a, cand_b], {"b"}),
    ]
    by_group = evaluate_ranker_by_subgroup(perfect_ranker, queries, age_band, k=2)

    assert set(by_group) == {"<25", "50+"}
    assert by_group["<25"].n_queries == 1
    assert by_group["50+"].n_queries == 1


def test_age_band_buckets() -> None:
    assert age_band(UserProfile(user_id="u", age=22)) == "<25"
    assert age_band(UserProfile(user_id="u", age=30)) == "25-34"
    assert age_band(UserProfile(user_id="u", age=40)) == "35-49"
    assert age_band(UserProfile(user_id="u", age=70)) == "50+"
    assert age_band(UserProfile(user_id="u")) == "unknown"


def test_mutual_friend_density_band_buckets() -> None:
    assert mutual_friend_density_band(UserProfile(user_id="u")) == "cold (0)"
    assert (
        mutual_friend_density_band(UserProfile(user_id="u", mutual_friend_ids={"a", "b"}))
        == "sparse (1-3)"
    )
    assert (
        mutual_friend_density_band(
            UserProfile(user_id="u", mutual_friend_ids={f"f{i}" for i in range(7)})
        )
        == "warm (4-10)"
    )
    assert (
        mutual_friend_density_band(
            UserProfile(user_id="u", mutual_friend_ids={f"f{i}" for i in range(15)})
        )
        == "dense (11+)"
    )


# ---------- diversity ----------


def test_intra_list_diversity_zero_when_all_identical() -> None:
    embeddings = {"a": [1.0, 0.0], "b": [1.0, 0.0], "c": [1.0, 0.0]}
    assert intra_list_diversity(["a", "b", "c"], embeddings, k=3) == 0.0


def test_intra_list_diversity_positive_when_orthogonal() -> None:
    embeddings = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    # cosine = 0, dissimilarity = 1.0
    assert math.isclose(intra_list_diversity(["a", "b"], embeddings, k=2), 1.0, rel_tol=1e-9)


def test_intra_list_diversity_zero_when_fewer_than_two_vectors() -> None:
    embeddings = {"a": [1.0, 0.0]}
    assert intra_list_diversity(["a", "b"], embeddings, k=2) == 0.0


# ---------- MMR re-rank ----------


def test_mmr_reranker_pure_relevance_matches_base() -> None:
    """lambda=1.0 should preserve the base ordering exactly."""

    def base(_source: UserProfile, candidates: list[UserProfile]) -> list[str]:
        return [c.user_id for c in candidates]

    embeddings = {"a": [1.0, 0.0], "b": [1.0, 0.0], "c": [0.0, 1.0]}
    reranker = make_mmr_reranker(base, embeddings, lambda_relevance=1.0)
    source = UserProfile(user_id="src")
    candidates = [
        UserProfile(user_id="a"),
        UserProfile(user_id="b"),
        UserProfile(user_id="c"),
    ]
    assert reranker(source, candidates) == ["a", "b", "c"]


def test_mmr_reranker_diversity_promotes_orthogonal_item() -> None:
    """A pure-diversity MMR should pull the orthogonal item up."""

    def base(_source: UserProfile, candidates: list[UserProfile]) -> list[str]:
        return [c.user_id for c in candidates]

    embeddings = {"a": [1.0, 0.0], "b": [1.0, 0.0], "c": [0.0, 1.0]}
    reranker = make_mmr_reranker(base, embeddings, lambda_relevance=0.0)
    source = UserProfile(user_id="src")
    candidates = [
        UserProfile(user_id="a"),
        UserProfile(user_id="b"),
        UserProfile(user_id="c"),
    ]
    # First pick: "a" (top of base order). Then MMR should prefer "c"
    # (orthogonal to "a") over "b" (identical to "a").
    result = reranker(source, candidates)
    assert result.index("c") < result.index("b")


def test_mmr_reranker_rejects_invalid_lambda() -> None:
    def base(_source: UserProfile, candidates: list[UserProfile]) -> list[str]:
        return [c.user_id for c in candidates]

    with pytest.raises(ValueError):
        make_mmr_reranker(base, {}, lambda_relevance=1.5)


# ---------- cold-start ----------


def test_is_cold_start_true_for_sparse_profile() -> None:
    sparse = UserProfile(user_id="new", interests={"music"})
    assert is_cold_start(sparse) is True


def test_is_cold_start_false_for_populated_profile() -> None:
    full = UserProfile(
        user_id="full",
        age=30,
        hometown="austin",
        college="ut",
        interests={"hiking", "music"},
        liked_topics={"coffee", "indie"},
        mutual_friend_ids={"f1"},
    )
    assert is_cold_start(full) is False


def test_rank_candidates_with_cold_start_falls_back_to_popularity() -> None:
    """A cold-start source should see well-connected candidates first."""
    source = UserProfile(user_id="cold")  # nothing populated
    popular = UserProfile(
        user_id="popular",
        mutual_friend_ids={f"f{i}" for i in range(10)},
        interests={"a", "b", "c"},
    )
    quiet = UserProfile(user_id="quiet")
    ranked = rank_candidates_with_cold_start(source, [quiet, popular])
    assert [c.user_id for c, _ in ranked] == ["popular", "quiet"]


def test_rank_candidates_with_cold_start_uses_main_ranker_for_warm_source() -> None:
    """A populated source must get the normal three-lane sort, not popularity."""
    warm = UserProfile(
        user_id="warm",
        age=30,
        hometown="austin",
        college="ut",
        interests={"hiking", "music"},
        mutual_friend_ids={"f1"},
    )
    friend_of_friend = UserProfile(
        user_id="fof",
        mutual_friend_ids={"f1"},  # shared friend → Lane A
    )
    popular_stranger = UserProfile(
        user_id="popular_stranger",
        mutual_friend_ids={f"x{i}" for i in range(50)},
        interests={"hiking"},
    )
    ranked = rank_candidates_with_cold_start(warm, [popular_stranger, friend_of_friend])
    # Lane A must come first regardless of raw friend-list size.
    assert ranked[0][0].user_id == "fof"
