"""Tests for the hangpost matching engine.

Covers:
- Happy path scoring and ranking
- Edge cases (None values, empty sets, boundary conditions)
- Input validation
- Loader utilities
- Tie-breaking and ranking stability
- Case-insensitive location matching
"""

import pytest

from hangpost_matching import (
    MAX_MUTUAL_FRIENDS,
    ScoringWeights,
    UserProfile,
    compute_match_score,
    rank_candidates,
)
from hangpost_matching.loader import tokenize


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def source():
    return UserProfile(
        user_id="source",
        interests={"hiking", "coding"},
        liked_topics={"tech", "travel"},
        location="austin",
        age=30,
        mutual_friend_ids={"f1", "f2", "f3"},
    )


@pytest.fixture
def strong_candidate():
    return UserProfile(
        user_id="strong",
        interests={"hiking", "coding", "reading"},
        liked_topics={"tech", "travel"},
        location="austin",
        age=29,
        mutual_friend_ids={"f2", "f3"},
    )


@pytest.fixture
def weak_candidate():
    return UserProfile(
        user_id="weak",
        interests={"chess"},
        liked_topics={"finance"},
        location="seattle",
        age=50,
        mutual_friend_ids=set(),
    )


# ---------------------------------------------------------------------------
# Original tests (preserved)
# ---------------------------------------------------------------------------

class TestComputeMatchScore:
    def test_prefers_overlap(self, source, strong_candidate, weak_candidate):
        strong_score = compute_match_score(source, strong_candidate).total_score
        weak_score = compute_match_score(source, weak_candidate).total_score
        assert strong_score > weak_score

    def test_default_weights_prioritize_mutual_friends_then_age(self):
        weights = ScoringWeights()
        assert weights.mutual_friends > weights.age_compatibility
        assert weights.age_compatibility > weights.interest_overlap

    def test_mutual_friends_get_social_boost_and_priority(self, source):
        no_mutual = UserProfile(
            user_id="no_mutual",
            interests={"hiking", "coding"},
            liked_topics={"tech", "travel"},
            location="austin",
            age=30,
            mutual_friend_ids=set(),
        )
        has_mutual = UserProfile(
            user_id="with_mutual",
            interests={"gaming"},
            liked_topics={"esports"},
            location="seattle",
            age=39,
            mutual_friend_ids={"f2"},
        )

        scored_no = compute_match_score(source, no_mutual)
        scored_yes = compute_match_score(source, has_mutual)

        assert scored_no.has_mutual_friends is False
        assert scored_yes.has_mutual_friends is True
        assert scored_yes.social_boost > 0.0

        ranked = rank_candidates(source, [no_mutual, has_mutual])
        assert ranked[0][0].user_id == "with_mutual"

    def test_age_compatibility_step_down(self):
        source = UserProfile(user_id="source", age=30)

        assert compute_match_score(source, UserProfile(user_id="s", age=30)).age_compatibility == 1.0
        assert compute_match_score(source, UserProfile(user_id="s1", age=29)).age_compatibility == 0.9
        assert compute_match_score(source, UserProfile(user_id="s2", age=28)).age_compatibility == 0.8
        assert compute_match_score(source, UserProfile(user_id="s10", age=20)).age_compatibility == 0.0


class TestRankCandidates:
    def test_descending_order(self):
        source = UserProfile(user_id="source", interests={"a"}, liked_topics={"x"}, location="nyc", age=24)
        c1 = UserProfile(user_id="c1", interests={"a"}, liked_topics={"x"}, location="nyc", age=24)
        c2 = UserProfile(user_id="c2", interests={"b"}, liked_topics={"y"}, location="la", age=40)

        ranked = rank_candidates(source, [c2, c1])
        assert ranked[0][0].user_id == "c1"
        assert ranked[1][0].user_id == "c2"


# ---------------------------------------------------------------------------
# Edge case tests: None / missing values
# ---------------------------------------------------------------------------

class TestNoneAndEmptyEdgeCases:
    def test_both_ages_none(self):
        a = UserProfile(user_id="a", age=None)
        b = UserProfile(user_id="b", age=None)
        result = compute_match_score(a, b)
        assert result.age_compatibility == 0.0

    def test_one_age_none(self):
        a = UserProfile(user_id="a", age=25)
        b = UserProfile(user_id="b", age=None)
        result = compute_match_score(a, b)
        assert result.age_compatibility == 0.0

    def test_both_locations_none(self):
        a = UserProfile(user_id="a", location=None)
        b = UserProfile(user_id="b", location=None)
        result = compute_match_score(a, b)
        assert result.location_match == 0.0

    def test_one_location_none(self):
        a = UserProfile(user_id="a", location="austin")
        b = UserProfile(user_id="b", location=None)
        result = compute_match_score(a, b)
        assert result.location_match == 0.0

    def test_empty_interests_both(self):
        a = UserProfile(user_id="a")
        b = UserProfile(user_id="b")
        result = compute_match_score(a, b)
        assert result.interest_overlap == 0.0
        assert result.liked_topic_overlap == 0.0

    def test_empty_interests_one_side(self):
        a = UserProfile(user_id="a", interests={"hiking"})
        b = UserProfile(user_id="b", interests=set())
        result = compute_match_score(a, b)
        assert result.interest_overlap == 0.0

    def test_minimal_profiles_no_crash(self):
        """Two profiles with only user_id should score without error."""
        a = UserProfile(user_id="a")
        b = UserProfile(user_id="b")
        result = compute_match_score(a, b)
        assert result.total_score == 0.0
        assert result.has_mutual_friends is False

    def test_empty_candidate_list(self):
        source = UserProfile(user_id="source")
        ranked = rank_candidates(source, [])
        assert ranked == []


# ---------------------------------------------------------------------------
# Boundary / numeric edge cases
# ---------------------------------------------------------------------------

class TestBoundaryConditions:
    def test_score_capped_at_1(self):
        """Even with perfect overlap + social boost, score should not exceed 1.0."""
        friends = {f"f{i}" for i in range(25)}
        a = UserProfile(
            user_id="a",
            interests={"x", "y", "z"},
            liked_topics={"p", "q"},
            location="austin",
            age=25,
            mutual_friend_ids=friends,
        )
        b = UserProfile(
            user_id="b",
            interests={"x", "y", "z"},
            liked_topics={"p", "q"},
            location="austin",
            age=25,
            mutual_friend_ids=friends,
        )
        result = compute_match_score(a, b)
        assert result.total_score <= 1.0

    def test_perfect_interest_overlap(self):
        interests = {"hiking", "coding", "music"}
        a = UserProfile(user_id="a", interests=interests)
        b = UserProfile(user_id="b", interests=interests)
        result = compute_match_score(a, b)
        assert result.interest_overlap == 1.0

    def test_no_interest_overlap(self):
        a = UserProfile(user_id="a", interests={"hiking"})
        b = UserProfile(user_id="b", interests={"chess"})
        result = compute_match_score(a, b)
        assert result.interest_overlap == 0.0

    def test_age_gap_exactly_10(self):
        a = UserProfile(user_id="a", age=30)
        b = UserProfile(user_id="b", age=20)
        result = compute_match_score(a, b)
        assert result.age_compatibility == 0.0

    def test_age_gap_exactly_9(self):
        a = UserProfile(user_id="a", age=30)
        b = UserProfile(user_id="b", age=21)
        result = compute_match_score(a, b)
        assert result.age_compatibility == pytest.approx(0.1)

    def test_age_gap_over_10(self):
        a = UserProfile(user_id="a", age=30)
        b = UserProfile(user_id="b", age=10)
        result = compute_match_score(a, b)
        assert result.age_compatibility == 0.0

    def test_mutual_friends_at_max_cap(self):
        """When mutual friends = MAX_MUTUAL_FRIENDS, ratio should be 1.0."""
        friends = {f"f{i}" for i in range(MAX_MUTUAL_FRIENDS)}
        a = UserProfile(user_id="a", mutual_friend_ids=friends)
        b = UserProfile(user_id="b", mutual_friend_ids=friends)
        result = compute_match_score(a, b)
        assert result.mutual_friends == 1.0

    def test_mutual_friends_above_cap(self):
        """More than MAX_MUTUAL_FRIENDS should still cap at 1.0."""
        friends = {f"f{i}" for i in range(MAX_MUTUAL_FRIENDS + 10)}
        a = UserProfile(user_id="a", mutual_friend_ids=friends)
        b = UserProfile(user_id="b", mutual_friend_ids=friends)
        result = compute_match_score(a, b)
        assert result.mutual_friends == 1.0

    def test_custom_weights(self):
        """Scoring should respect custom weights."""
        a = UserProfile(user_id="a", interests={"x"}, age=25)
        b = UserProfile(user_id="b", interests={"x"}, age=25)

        # All weight on interests
        w = ScoringWeights(
            interest_overlap=1.0,
            liked_topic_overlap=0.0,
            mutual_friends=0.0,
            location_match=0.0,
            age_compatibility=0.0,
            friend_common_boost=0.0,
        )
        result = compute_match_score(a, b, w)
        assert result.total_score == pytest.approx(1.0)

    def test_score_components_are_rounded(self):
        """All component scores should be rounded to 6 decimal places."""
        a = UserProfile(user_id="a", interests={"x", "y", "z"})
        b = UserProfile(user_id="b", interests={"x"})
        result = compute_match_score(a, b)
        # Jaccard of {x,y,z} & {x} = 1/3 ≈ 0.333333...
        assert result.interest_overlap == round(1 / 3, 6)


# ---------------------------------------------------------------------------
# Location case-sensitivity
# ---------------------------------------------------------------------------

class TestLocationCaseInsensitive:
    def test_same_location_different_case(self):
        a = UserProfile(user_id="a", location="Austin")
        b = UserProfile(user_id="b", location="austin")
        result = compute_match_score(a, b)
        assert result.location_match == 1.0

    def test_location_with_whitespace(self):
        a = UserProfile(user_id="a", location="  Austin  ")
        b = UserProfile(user_id="b", location="austin")
        result = compute_match_score(a, b)
        assert result.location_match == 1.0


# ---------------------------------------------------------------------------
# Ranking stability / tie-breaking
# ---------------------------------------------------------------------------

class TestRankingStability:
    def test_identical_candidates_all_returned(self):
        """Identical candidates should all appear in results."""
        source = UserProfile(user_id="source", interests={"a"}, age=25)
        candidates = [
            UserProfile(user_id=f"c{i}", interests={"a"}, age=25)
            for i in range(5)
        ]
        ranked = rank_candidates(source, candidates)
        assert len(ranked) == 5

    def test_mutual_friends_lane_always_first(self):
        """Candidates with mutual friends should always rank above those without,
        even if the non-mutual candidate has higher base compatibility."""
        source = UserProfile(
            user_id="source",
            interests={"hiking", "coding", "reading"},
            liked_topics={"tech", "travel", "food"},
            location="austin",
            age=25,
            mutual_friend_ids={"f1"},
        )
        # Perfect match but no mutual friends
        perfect_no_mutual = UserProfile(
            user_id="perfect",
            interests={"hiking", "coding", "reading"},
            liked_topics={"tech", "travel", "food"},
            location="austin",
            age=25,
            mutual_friend_ids=set(),
        )
        # Poor match but has mutual friends
        poor_with_mutual = UserProfile(
            user_id="poor",
            interests={"chess"},
            liked_topics={"finance"},
            location="seattle",
            age=50,
            mutual_friend_ids={"f1"},
        )
        ranked = rank_candidates(source, [perfect_no_mutual, poor_with_mutual])
        assert ranked[0][0].user_id == "poor"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_empty_user_id_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            UserProfile(user_id="")

    def test_whitespace_user_id_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            UserProfile(user_id="   ")

    def test_non_string_user_id_raises(self):
        with pytest.raises(TypeError, match="must be a str"):
            UserProfile(user_id=123)  # type: ignore[arg-type]

    def test_negative_age_raises(self):
        with pytest.raises(ValueError, match="between 0 and 150"):
            UserProfile(user_id="a", age=-1)

    def test_age_over_150_raises(self):
        with pytest.raises(ValueError, match="between 0 and 150"):
            UserProfile(user_id="a", age=200)

    def test_age_zero_valid(self):
        p = UserProfile(user_id="a", age=0)
        assert p.age == 0

    def test_age_150_valid(self):
        p = UserProfile(user_id="a", age=150)
        assert p.age == 150

    def test_none_age_valid(self):
        p = UserProfile(user_id="a", age=None)
        assert p.age is None


# ---------------------------------------------------------------------------
# Loader / tokenizer tests
# ---------------------------------------------------------------------------

class TestTokenize:
    def test_basic_tokenization(self):
        assert tokenize("Hiking; Cooking; Yoga") == {"hiking", "cooking", "yoga"}

    def test_empty_string(self):
        assert tokenize("") == set()

    def test_whitespace_only(self):
        assert tokenize("  ;  ;  ") == set()

    def test_single_item(self):
        assert tokenize("Hiking") == {"hiking"}

    def test_duplicate_items(self):
        assert tokenize("Hiking; hiking; HIKING") == {"hiking"}
