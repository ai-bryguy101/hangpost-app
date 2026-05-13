"""Tests for the offline evaluation harness.

Each metric is exercised against small hand-computed cases so the
expected values are obvious from the test names alone.
"""

import math

from hangpost_matching import (
    ABLATABLE_WEIGHT_FIELDS,
    UserProfile,
    ablate_weights,
    average_precision_at_k,
    evaluate_ranker,
    get_relevance_fn,
    make_noisy_relevance_fn,
    make_simulated_outcome_fn,
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
        hometown="austin",
        age=30,
    )
    matched = UserProfile(
        user_id="match",
        interests={"a", "b"},  # ≥2 shared
        liked_topics={"x", "y"},  # ≥2 shared
        hometown="austin",  # same hometown -> 3 signals
        age=50,
    )

    assert synthesize_relevance(source, matched) is True


def test_synthesize_relevance_false_when_only_two_signals() -> None:
    source = UserProfile(
        user_id="src",
        interests={"a", "b"},
        liked_topics={"x", "y"},
        hometown="austin",
        age=30,
    )
    weak = UserProfile(
        user_id="weak",
        interests={"a", "b"},  # ≥2 shared (signal 1)
        liked_topics={"q", "r"},
        hometown="seattle",
        age=29,  # age close (signal 2)
    )

    assert synthesize_relevance(source, weak) is False


# ---------- noisy relevance generator ----------


def test_noisy_relevance_is_deterministic_per_pair() -> None:
    """Calling the same labeller twice on the same pair gives the same answer."""
    fn = make_noisy_relevance_fn(noise_level=0.5, seed=7)
    source = UserProfile(user_id="src", interests={"a"}, age=30)
    candidate = UserProfile(user_id="cand", interests={"a"}, age=30)

    assert fn(source, candidate) == fn(source, candidate)


def test_noisy_relevance_with_zero_noise_matches_base() -> None:
    """noise_level=0 must return the underlying label every time."""
    fn = make_noisy_relevance_fn(noise_level=0.0, seed=0)
    source = UserProfile(
        user_id="src",
        interests={"a", "b", "c"},
        liked_topics={"x", "y"},
        hometown="austin",
        age=30,
    )
    cand_match = UserProfile(
        user_id="match",
        interests={"a", "b"},
        liked_topics={"x", "y"},
        hometown="austin",
        age=50,
    )
    cand_no_match = UserProfile(user_id="other", interests=set(), age=99)

    assert fn(source, cand_match) is True
    assert fn(source, cand_no_match) is False


def test_noisy_relevance_flips_at_least_some_labels() -> None:
    """Across many pairs, high noise must actually flip labels."""
    base = make_noisy_relevance_fn(noise_level=0.0, seed=0)
    noisy = make_noisy_relevance_fn(noise_level=0.5, seed=0)
    source = UserProfile(user_id="src", interests={"a", "b", "c"}, age=30)
    flips = 0
    for i in range(200):
        cand = UserProfile(user_id=f"c_{i}", interests={"a"}, age=30)
        if base(source, cand) != noisy(source, cand):
            flips += 1
    assert flips > 0


def test_noisy_relevance_rejects_invalid_noise_level() -> None:
    import pytest

    with pytest.raises(ValueError):
        make_noisy_relevance_fn(noise_level=1.5)


# ---------- simulated outcome generator ----------


def test_simulated_outcome_is_deterministic_per_pair() -> None:
    fn = make_simulated_outcome_fn(seed=42)
    source = UserProfile(user_id="src", interests={"a"}, age=30)
    candidate = UserProfile(user_id="cand", interests={"a"}, age=30)

    assert fn(source, candidate) == fn(source, candidate)


def test_simulated_outcome_depends_on_user_id_not_only_features() -> None:
    """Two candidates with IDENTICAL observable features can get different labels.

    This is the core property: the simulator's hidden personality vector
    depends on user_id, so the ranker cannot perfectly recover labels
    from features alone. Without this, the learned ranker has no
    realistic ceiling above the rules baseline.
    """
    fn = make_simulated_outcome_fn(seed=1, noise_level=0.0)
    source = UserProfile(user_id="src", interests={"a", "b"}, age=30, hometown="austin")
    labels = {
        fn(
            source,
            UserProfile(
                user_id=f"cand_{i}",
                interests={"a", "b"},
                age=30,
                hometown="austin",
            ),
        )
        for i in range(50)
    }
    # We expect both True and False to appear, because hidden personality
    # vectors differ across the synthetic user_ids.
    assert labels == {True, False}


def test_simulated_outcome_produces_mix_of_labels() -> None:
    fn = make_simulated_outcome_fn(seed=3)
    source = UserProfile(user_id="src", interests={"a", "b"}, age=30, hometown="austin")
    labels = [
        fn(
            source,
            UserProfile(
                user_id=f"c_{i}",
                interests={"a"} if i % 2 == 0 else {"x"},
                age=30 + (i % 10),
                hometown="austin" if i % 3 == 0 else "seattle",
            ),
        )
        for i in range(100)
    ]
    n_true = sum(labels)
    assert 0 < n_true < len(labels), "simulator should produce a mix of labels"


def test_simulated_outcome_rejects_invalid_noise_level() -> None:
    import pytest

    with pytest.raises(ValueError):
        make_simulated_outcome_fn(noise_level=-0.1)


# ---------- get_relevance_fn registry ----------


def test_get_relevance_fn_known_names() -> None:
    for name in ("rule_based", "noisy", "simulated"):
        fn = get_relevance_fn(name, seed=0)
        assert callable(fn)


def test_get_relevance_fn_unknown_name_raises() -> None:
    import pytest

    with pytest.raises(ValueError, match="Unknown relevance fn"):
        get_relevance_fn("not_a_real_one")


# ---------- ablate_weights ----------


def test_ablate_weights_returns_baseline_first_then_one_row_per_feature() -> None:
    source = UserProfile(
        user_id="src",
        interests={"a", "b", "c"},
        liked_topics={"x", "y"},
        hometown="austin",
        college="ut",
        age=30,
        mutual_friend_ids={"f1", "f2"},
    )
    candidates = [
        UserProfile(
            user_id=f"c{i}",
            interests={"a"} if i % 2 == 0 else {"z"},
            liked_topics={"x"} if i % 3 == 0 else {"q"},
            hometown="austin" if i % 4 == 0 else "seattle",
            college="ut" if i % 5 == 0 else "harvard",
            age=30 + (i - 5),
            mutual_friend_ids={"f1"} if i % 2 == 0 else set(),
        )
        for i in range(10)
    ]
    relevant = {"c0", "c2", "c4"}
    queries = [(source, candidates, relevant)]

    rows = ablate_weights(queries, k=5)

    assert rows[0].feature == "<full>"
    assert [r.feature for r in rows[1:]] == list(ABLATABLE_WEIGHT_FIELDS)
    # Baseline deltas are zero by construction.
    assert rows[0].delta_precision == 0.0
    assert rows[0].delta_ndcg == 0.0


def test_ablate_weights_rejects_unknown_field() -> None:
    import pytest

    with pytest.raises(ValueError, match="Unknown ScoringWeights field"):
        ablate_weights([], features=("not_a_real_weight",))
