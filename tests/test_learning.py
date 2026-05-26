"""Tests for Phase 3 learning-to-rank.

These tests run without `lightgbm` installed by injecting a stub
predictor that satisfies the `Predictor` Protocol. The real LightGBM
training path is exercised by `scripts/train.py` against the [ml] extra.
"""

from collections.abc import Sequence

import pytest

from hangpost_matching import (
    FEATURE_NAMES,
    LearnedRanker,
    UserProfile,
    extract_features,
)
from hangpost_matching.learning import Predictor


class _ScoreSumPredictor:
    """Stand-in model that ranks by the sum of all features.

    Implements the `Predictor` Protocol so `LearnedRanker.rank` works
    without lightgbm. The exact scoring rule is irrelevant — what
    matters is that the wrapper feeds features in correctly and orders
    by descending score.
    """

    def predict(self, x: Sequence[Sequence[float]]) -> Sequence[float]:
        return [sum(row) for row in x]


def _toy_profile(user_id: str, age: int, interests: set[str]) -> UserProfile:
    return UserProfile(user_id=user_id, age=age, interests=interests)


# ---------- extract_features ----------


def test_extract_features_returns_one_value_per_feature_name() -> None:
    source = _toy_profile("a", 30, {"hiking"})
    candidate = _toy_profile("b", 30, {"hiking"})

    features = extract_features(source, candidate)

    assert len(features) == len(FEATURE_NAMES)


def test_extract_features_uses_embeddings_when_provided() -> None:
    source = _toy_profile("a", 30, {"hiking"})
    candidate = _toy_profile("b", 30, {"hiking"})
    embeddings = {"a": [1.0, 0.0, 0.0], "b": [1.0, 0.0, 0.0]}

    without = extract_features(source, candidate)
    with_embeds = extract_features(source, candidate, profile_embeddings=embeddings)

    semantic_idx = FEATURE_NAMES.index("semantic_similarity")
    assert without[semantic_idx] == 0.0
    assert with_embeds[semantic_idx] == 1.0


def test_extract_features_has_mutual_friends_is_one_or_zero() -> None:
    source = UserProfile(user_id="a", mutual_friend_ids={"f1"})
    shared = UserProfile(user_id="b", mutual_friend_ids={"f1"})
    none = UserProfile(user_id="c", mutual_friend_ids={"other"})

    flag_idx = FEATURE_NAMES.index("has_mutual_friends")
    assert extract_features(source, shared)[flag_idx] == 1.0
    assert extract_features(source, none)[flag_idx] == 0.0


# ---------- LearnedRanker (stub model) ----------


def test_learned_ranker_orders_by_predicted_score() -> None:
    source = _toy_profile("source", 30, {"hiking", "coding"})
    strong = _toy_profile("strong", 30, {"hiking", "coding"})
    weak = _toy_profile("weak", 50, {"chess"})

    ranker = LearnedRanker(model=_ScoreSumPredictor())

    order = ranker.rank(source, [weak, strong])

    assert order == ["strong", "weak"]


def test_learned_ranker_handles_empty_candidate_list() -> None:
    source = _toy_profile("source", 30, set())
    ranker = LearnedRanker(model=_ScoreSumPredictor())

    assert ranker.rank(source, []) == []


def test_learned_ranker_raises_when_used_before_fit() -> None:
    ranker = LearnedRanker()

    with pytest.raises(RuntimeError, match="has not been fit"):
        ranker.rank(_toy_profile("a", 30, set()), [_toy_profile("b", 30, set())])


def test_learned_ranker_as_ranker_satisfies_ranker_protocol() -> None:
    """as_ranker() returns something the evaluation harness can call."""
    source = _toy_profile("source", 30, {"hiking"})
    candidate = _toy_profile("b", 30, {"hiking"})

    ranker_fn = LearnedRanker(model=_ScoreSumPredictor()).as_ranker()
    result = ranker_fn(source, [candidate])

    assert result == ["b"]


def test_learned_ranker_passes_embedding_features_to_model() -> None:
    """If embeddings are provided, the semantic_similarity feature reaches the model."""
    source = _toy_profile("a", 30, set())
    candidate = _toy_profile("b", 30, set())
    embeddings = {"a": [1.0, 0.0], "b": [1.0, 0.0]}

    captured: list[list[list[float]]] = []

    class _CapturingPredictor:
        def predict(self, x: Sequence[Sequence[float]]) -> Sequence[float]:
            captured.append([list(row) for row in x])
            return [0.0 for _ in x]

    ranker = LearnedRanker(profile_embeddings=embeddings, model=_CapturingPredictor())
    ranker.rank(source, [candidate])

    semantic_idx = FEATURE_NAMES.index("semantic_similarity")
    assert captured[0][0][semantic_idx] == 1.0


def test_predictor_protocol_is_satisfied_by_stub() -> None:
    """Sanity check that the stub matches the Protocol shape."""

    def _accepts_predictor(_: Predictor) -> None:
        pass

    _accepts_predictor(_ScoreSumPredictor())


def test_learned_ranker_raises_on_score_count_mismatch() -> None:
    source = _toy_profile("source", 30, {"hiking"})
    candidate = _toy_profile("c1", 31, {"hiking"})

    class _BadPredictor:
        def predict(self, x: Sequence[Sequence[float]]) -> Sequence[float]:
            return [0.1, 0.2]

    ranker = LearnedRanker(model=_BadPredictor())
    with pytest.raises(ValueError, match="score count that does not match candidates"):
        ranker.rank(source, [candidate])
