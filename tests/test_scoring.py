"""Tests for the hangpost matching engine.

WHAT THIS COVERS:
- All 9 scoring signals (hobby, interest, fan_of, mutual_friends, location,
  age, college, faith, travel) — both happy path and edge cases
- Tiered location scoring (same city+state, same state only, different state)
- The Location dataclass (city+state pair)
- Social boost and two-lane ranking strategy
- Input validation (bad user_id, negative age, etc.)
- Edge cases (None values, empty sets, boundary conditions)
- Loader/tokenizer utilities

WHAT CHANGED AND WHY (v0.2.0):
- Tests now use the new UserProfile fields: hobbies, interests, fan_of,
  Location, college, faith, travel_wishlist. The old `liked_topics` and
  bare-string `location` are gone.
- New test classes for tiered location scoring, college/faith exact matching,
  and travel overlap — these signals didn't exist before.
- ScoringWeights tests updated for the 9+1 weight structure (9 base weights
  + friend_common_boost).

WHY SO MANY TESTS:
The matching engine is the core of Hangpost. If it silently breaks, users get
bad recommendations and we'd never know. Every signal, edge case, and boundary
condition has a test so we catch regressions immediately. Each test is named
to describe WHAT it verifies and WHY that matters.
"""

import pytest

from hangpost_matching import (
    Location,
    MAX_MUTUAL_FRIENDS,
    SAME_STATE_SCORE,
    ScoringWeights,
    UserProfile,
    compute_match_score,
    rank_candidates,
)
from hangpost_matching.loader import tokenize


# ---------------------------------------------------------------------------
# Fixtures: reusable test profiles
# ---------------------------------------------------------------------------
# WHY fixtures: Every test that needs a "source" and "candidate" would
# otherwise have to construct them from scratch. Fixtures keep tests clean
# and consistent — if we change the data model, we only update the fixtures.


@pytest.fixture
def source():
    """A well-rounded source profile with values in all fields.

    This profile has data in every field so it can exercise every scoring
    signal. Tests that need specific missing/empty fields create their own
    profiles instead.
    """
    return UserProfile(
        user_id="source",
        hobbies={"hiking", "coding"},
        interests={"tech", "outdoor adventure"},
        fan_of={"kendrick lamar", "the bear"},
        location=Location(city="Austin", state="Texas"),
        age=30,
        mutual_friend_ids={"f1", "f2", "f3"},
        college="University of Texas at Austin",
        faith="Agnostic",
        travel_wishlist={"japan", "iceland"},
    )


@pytest.fixture
def strong_candidate():
    """A candidate who shares many attributes with the source.

    Strong match on: hobbies (2/3 overlap), interests (2/2), fan_of (2/3),
    same city+state, similar age, 2 mutual friends, same college, same faith,
    overlapping travel. Should always outscore the weak_candidate.
    """
    return UserProfile(
        user_id="strong",
        hobbies={"hiking", "coding", "reading"},
        interests={"tech", "outdoor adventure"},
        fan_of={"kendrick lamar", "the bear", "marvel"},
        location=Location(city="Austin", state="Texas"),
        age=29,
        mutual_friend_ids={"f2", "f3"},
        college="University of Texas at Austin",
        faith="Agnostic",
        travel_wishlist={"japan", "spain"},
    )


@pytest.fixture
def weak_candidate():
    """A candidate with almost nothing in common with the source.

    No hobby overlap, no interest overlap, different location, big age gap,
    no mutual friends, different college, different faith, no travel overlap.
    Should always score near 0.
    """
    return UserProfile(
        user_id="weak",
        hobbies={"basketball"},
        interests={"hip hop"},
        fan_of={"nba"},
        location=Location(city="Miami", state="Florida"),
        age=50,
        mutual_friend_ids=set(),
        college="University of Miami",
        faith="Christian",
        travel_wishlist={"brazil"},
    )


# ---------------------------------------------------------------------------
# Core scoring tests
# ---------------------------------------------------------------------------

class TestComputeMatchScore:
    """Tests for the main compute_match_score function.

    These verify that the overall scoring pipeline works correctly:
    signals are computed, weighted, and combined properly.
    """

    def test_strong_beats_weak(self, source, strong_candidate, weak_candidate):
        """A profile with high overlap should always outscore one with low overlap.

        WHY this matters: This is the most basic sanity check. If the algorithm
        can't distinguish a good match from a bad one, nothing else matters.
        """
        strong_score = compute_match_score(source, strong_candidate).total_score
        weak_score = compute_match_score(source, weak_candidate).total_score
        assert strong_score > weak_score

    def test_default_weights_priority_order(self):
        """Default weights reflect the friendship tier hierarchy:
        location > college > mutual_friends > hobbies > interests.

        WHY this order: The tier list is:
          Tier 1 (instant friends): mutual friends — handled by base weight
            PLUS the separate friend_common_boost, which is the real driver.
          Tier 2 (probable friends): hometown — location_match is the
            single largest base weight so it dominates when no mutual friends.
          Tier 3 (probable friends): college — second-largest base weight.
          Tier 4 (sometimes friends): hobbies.
          Tier 5 (sometimes friends): interests.
        """
        w = ScoringWeights()
        # Tier 2: location is the biggest base weight (no boost of its own).
        assert w.location_match > w.mutual_friends
        # Tier 1: mutual_friends base weight sits above college. It also has
        # the separate friend_common_boost, making it truly dominant overall.
        assert w.mutual_friends > w.college_match
        # Tier 3 → 4 → 5:
        assert w.college_match > w.hobby_overlap
        assert w.hobby_overlap > w.interest_overlap
        assert w.interest_overlap > w.fan_of_overlap

    def test_all_nine_weights_sum_near_one(self):
        """The 9 base weights should sum to approximately 1.0.

        WHY: If they sum to 1.0, the base score (before social boost) stays
        in the 0.0-1.0 range, which makes it interpretable and consistent.
        The friend_common_boost is intentionally excluded from this sum because
        it's a separate additive boost.

        Current layout: location(0.22) + college(0.18) + mutual_friends(0.20)
        + hobby(0.15) + interest(0.08) + fan_of(0.05) + age(0.07)
        + faith(0.03) + travel(0.02) = 1.00
        """
        w = ScoringWeights()
        total = (
            w.hobby_overlap + w.interest_overlap + w.fan_of_overlap
            + w.mutual_friends + w.location_match + w.age_compatibility
            + w.college_match + w.faith_match + w.travel_overlap
        )
        assert total == pytest.approx(1.0, abs=0.01)

    def test_score_components_are_rounded(self):
        """All component scores should be rounded to 6 decimal places.

        WHY: Floating-point arithmetic produces noise (e.g., 0.33333333333337).
        Rounding to 6 places keeps scores clean and test assertions reliable.
        """
        a = UserProfile(user_id="a", hobbies={"x", "y", "z"})
        b = UserProfile(user_id="b", hobbies={"x"})
        result = compute_match_score(a, b)
        # Jaccard of {x,y,z} & {x} = 1/3 ≈ 0.333333
        assert result.hobby_overlap == round(1 / 3, 6)


# ---------------------------------------------------------------------------
# Hobby, Interest, and Fan-Of overlap tests (Jaccard similarity)
# ---------------------------------------------------------------------------

class TestJaccardOverlapScoring:
    """Tests for the three Jaccard-based signals: hobbies, interests, fan_of.

    All three use the same Jaccard formula: |intersection| / |union|.
    We test each independently to verify they're wired correctly.
    """

    def test_perfect_hobby_overlap(self):
        """Identical hobby sets → 1.0 overlap."""
        hobbies = {"hiking", "coding", "cooking"}
        a = UserProfile(user_id="a", hobbies=hobbies)
        b = UserProfile(user_id="b", hobbies=hobbies)
        assert compute_match_score(a, b).hobby_overlap == 1.0

    def test_no_hobby_overlap(self):
        """Completely disjoint hobby sets → 0.0 overlap."""
        a = UserProfile(user_id="a", hobbies={"hiking"})
        b = UserProfile(user_id="b", hobbies={"chess"})
        assert compute_match_score(a, b).hobby_overlap == 0.0

    def test_partial_hobby_overlap(self):
        """Partial overlap should produce a fraction between 0 and 1.

        {hiking, coding} & {hiking, chess} = {hiking} / {hiking, coding, chess}
        = 1/3 ≈ 0.333333
        """
        a = UserProfile(user_id="a", hobbies={"hiking", "coding"})
        b = UserProfile(user_id="b", hobbies={"hiking", "chess"})
        result = compute_match_score(a, b)
        assert result.hobby_overlap == pytest.approx(1 / 3, abs=0.001)

    def test_empty_hobbies_both(self):
        """Both have empty hobbies → 0.0 (can't compute overlap on nothing)."""
        a = UserProfile(user_id="a", hobbies=set())
        b = UserProfile(user_id="b", hobbies=set())
        assert compute_match_score(a, b).hobby_overlap == 0.0

    def test_empty_hobbies_one_side(self):
        """One side has hobbies, other doesn't → 0.0 overlap."""
        a = UserProfile(user_id="a", hobbies={"hiking"})
        b = UserProfile(user_id="b", hobbies=set())
        assert compute_match_score(a, b).hobby_overlap == 0.0

    def test_perfect_interest_overlap(self):
        """Identical interest sets → 1.0 overlap."""
        interests = {"tech", "outdoor adventure"}
        a = UserProfile(user_id="a", interests=interests)
        b = UserProfile(user_id="b", interests=interests)
        assert compute_match_score(a, b).interest_overlap == 1.0

    def test_no_interest_overlap(self):
        """Disjoint interest sets → 0.0."""
        a = UserProfile(user_id="a", interests={"tech"})
        b = UserProfile(user_id="b", interests={"hip hop"})
        assert compute_match_score(a, b).interest_overlap == 0.0

    def test_perfect_fan_of_overlap(self):
        """Identical fan_of sets → 1.0 overlap."""
        fan_of = {"kendrick lamar", "the bear"}
        a = UserProfile(user_id="a", fan_of=fan_of)
        b = UserProfile(user_id="b", fan_of=fan_of)
        assert compute_match_score(a, b).fan_of_overlap == 1.0

    def test_no_fan_of_overlap(self):
        """Completely different fandoms → 0.0."""
        a = UserProfile(user_id="a", fan_of={"kendrick lamar"})
        b = UserProfile(user_id="b", fan_of={"taylor swift"})
        assert compute_match_score(a, b).fan_of_overlap == 0.0


# ---------------------------------------------------------------------------
# Tiered location scoring tests
# ---------------------------------------------------------------------------

class TestTieredLocationScoring:
    """Tests for the city+state tiered scoring system.

    The tiers:
    - Same city + same state → 1.0
    - Different city, same state → SAME_STATE_SCORE (0.4)
    - Different state → 0.0
    - Missing location → 0.0

    WHY tiered: Binary matching (old system) gave 0.0 to someone in Houston
    when you're in Dallas. But both are in Texas — that's worth something.
    """

    def test_same_city_same_state(self):
        """Same city + same state → full match (1.0)."""
        a = UserProfile(user_id="a", location=Location(city="Austin", state="Texas"))
        b = UserProfile(user_id="b", location=Location(city="Austin", state="Texas"))
        assert compute_match_score(a, b).location_match == 1.0

    def test_different_city_same_state(self):
        """Different city, same state → partial match (SAME_STATE_SCORE).

        Austin, TX vs Dallas, TX: they share state culture but can't easily
        grab coffee together.
        """
        a = UserProfile(user_id="a", location=Location(city="Austin", state="Texas"))
        b = UserProfile(user_id="b", location=Location(city="Dallas", state="Texas"))
        assert compute_match_score(a, b).location_match == SAME_STATE_SCORE

    def test_different_state(self):
        """Different states → no location signal (0.0)."""
        a = UserProfile(user_id="a", location=Location(city="Austin", state="Texas"))
        b = UserProfile(user_id="b", location=Location(city="Miami", state="Florida"))
        assert compute_match_score(a, b).location_match == 0.0

    def test_both_locations_none(self):
        """Both have no location → 0.0 (can't score what we don't know)."""
        a = UserProfile(user_id="a", location=None)
        b = UserProfile(user_id="b", location=None)
        assert compute_match_score(a, b).location_match == 0.0

    def test_one_location_none(self):
        """One has location, other doesn't → 0.0."""
        a = UserProfile(user_id="a", location=Location(city="Austin", state="Texas"))
        b = UserProfile(user_id="b", location=None)
        assert compute_match_score(a, b).location_match == 0.0

    def test_location_case_insensitive(self):
        """Location matching should be case-insensitive.

        WHY: Users might type "austin" or "Austin" — both mean the same city.
        The scoring function normalizes to lowercase before comparing.
        """
        a = UserProfile(user_id="a", location=Location(city="Austin", state="Texas"))
        b = UserProfile(user_id="b", location=Location(city="austin", state="texas"))
        assert compute_match_score(a, b).location_match == 1.0

    def test_location_whitespace_trimmed(self):
        """Extra whitespace should be ignored in location comparison."""
        a = UserProfile(user_id="a", location=Location(city="  Austin  ", state="  Texas  "))
        b = UserProfile(user_id="b", location=Location(city="Austin", state="Texas"))
        assert compute_match_score(a, b).location_match == 1.0

    def test_same_city_name_different_state(self):
        """Cities with the same name in different states should NOT match.

        WHY: Portland, Oregon and Portland, Maine are completely different
        places. The old single-string location would have matched them.
        """
        a = UserProfile(user_id="a", location=Location(city="Portland", state="Oregon"))
        b = UserProfile(user_id="b", location=Location(city="Portland", state="Maine"))
        # Different states → 0.0, NOT 1.0 even though city names match.
        assert compute_match_score(a, b).location_match == 0.0


# ---------------------------------------------------------------------------
# Age compatibility tests
# ---------------------------------------------------------------------------

class TestAgeCompatibility:
    """Tests for the step-down age compatibility ladder.

    The formula: 1.0 - (0.1 × gap), floored at 0.0.
    This means 10% penalty per year of age difference, zero at 10+ years.
    """

    def test_same_age(self):
        """Same age → maximum compatibility (1.0)."""
        a = UserProfile(user_id="a", age=30)
        b = UserProfile(user_id="b", age=30)
        assert compute_match_score(a, b).age_compatibility == 1.0

    def test_one_year_gap(self):
        """1-year gap → 0.9 (10% penalty)."""
        a = UserProfile(user_id="a", age=30)
        b = UserProfile(user_id="b", age=29)
        assert compute_match_score(a, b).age_compatibility == 0.9

    def test_five_year_gap(self):
        """5-year gap → 0.5 (50% penalty)."""
        a = UserProfile(user_id="a", age=30)
        b = UserProfile(user_id="b", age=25)
        assert compute_match_score(a, b).age_compatibility == 0.5

    def test_nine_year_gap(self):
        """9-year gap → 0.1 (90% penalty, but not zero)."""
        a = UserProfile(user_id="a", age=30)
        b = UserProfile(user_id="b", age=21)
        assert compute_match_score(a, b).age_compatibility == pytest.approx(0.1)

    def test_ten_year_gap(self):
        """10-year gap → 0.0 (cutoff reached)."""
        a = UserProfile(user_id="a", age=30)
        b = UserProfile(user_id="b", age=20)
        assert compute_match_score(a, b).age_compatibility == 0.0

    def test_over_ten_year_gap(self):
        """More than 10 years → still 0.0 (can't go negative)."""
        a = UserProfile(user_id="a", age=30)
        b = UserProfile(user_id="b", age=10)
        assert compute_match_score(a, b).age_compatibility == 0.0

    def test_both_ages_none(self):
        """Both ages unknown → 0.0 (can't compute gap)."""
        a = UserProfile(user_id="a", age=None)
        b = UserProfile(user_id="b", age=None)
        assert compute_match_score(a, b).age_compatibility == 0.0

    def test_one_age_none(self):
        """One age unknown → 0.0."""
        a = UserProfile(user_id="a", age=25)
        b = UserProfile(user_id="b", age=None)
        assert compute_match_score(a, b).age_compatibility == 0.0


# ---------------------------------------------------------------------------
# Mutual friends and social boost tests
# ---------------------------------------------------------------------------

class TestMutualFriendsAndSocialBoost:
    """Tests for mutual friend scoring and the two-lane social boost.

    The mutual friends signal has two parts:
    1. A bounded ratio score: min(shared_count / MAX_MUTUAL_FRIENDS, 1.0)
    2. A social boost: a flat additive bonus when ANY mutual friends exist

    WHY two parts: The ratio rewards having MORE mutual friends (deeper social
    embedding). The boost ensures that even 1 mutual friend significantly
    raises the profile above strangers.
    """

    def test_no_mutual_friends(self, source):
        """No shared friend IDs → 0.0 score, no boost, has_mutual_friends=False."""
        no_friends = UserProfile(user_id="no_friends", mutual_friend_ids=set())
        result = compute_match_score(source, no_friends)
        assert result.has_mutual_friends is False
        assert result.mutual_friends == 0.0
        assert result.social_boost == 0.0

    def test_has_mutual_friends(self, source):
        """Shared friend IDs → positive score, has_mutual_friends=True, boost applied."""
        with_friends = UserProfile(user_id="with_friends", mutual_friend_ids={"f2"})
        result = compute_match_score(source, with_friends)
        assert result.has_mutual_friends is True
        assert result.mutual_friends > 0.0
        assert result.social_boost > 0.0

    def test_mutual_friends_at_cap(self):
        """When mutual friends = MAX_MUTUAL_FRIENDS, ratio should be 1.0."""
        friends = {f"f{i}" for i in range(MAX_MUTUAL_FRIENDS)}
        a = UserProfile(user_id="a", mutual_friend_ids=friends)
        b = UserProfile(user_id="b", mutual_friend_ids=friends)
        assert compute_match_score(a, b).mutual_friends == 1.0

    def test_mutual_friends_above_cap(self):
        """More than MAX_MUTUAL_FRIENDS should still cap at 1.0."""
        friends = {f"f{i}" for i in range(MAX_MUTUAL_FRIENDS + 10)}
        a = UserProfile(user_id="a", mutual_friend_ids=friends)
        b = UserProfile(user_id="b", mutual_friend_ids=friends)
        assert compute_match_score(a, b).mutual_friends == 1.0

    def test_mutual_friend_lane_priority(self, source):
        """Candidates with mutual friends should rank above those without,
        even if the no-mutual candidate has better compatibility.

        WHY: The two-lane sort ensures socially-connected people always
        appear first. In real life, a friend-of-a-friend is almost always
        a safer recommendation than a total stranger.
        """
        # Perfect compatibility but zero mutual friends.
        perfect_stranger = UserProfile(
            user_id="perfect_stranger",
            hobbies={"hiking", "coding"},
            interests={"tech", "outdoor adventure"},
            fan_of={"kendrick lamar", "the bear"},
            location=Location(city="Austin", state="Texas"),
            age=30,
            mutual_friend_ids=set(),
            college="University of Texas at Austin",
            faith="Agnostic",
            travel_wishlist={"japan", "iceland"},
        )
        # Terrible compatibility but has mutual friends.
        poor_with_friends = UserProfile(
            user_id="poor_with_friends",
            hobbies={"basketball"},
            interests={"hip hop"},
            fan_of={"nba"},
            location=Location(city="Miami", state="Florida"),
            age=50,
            mutual_friend_ids={"f1"},
        )

        ranked = rank_candidates(source, [perfect_stranger, poor_with_friends])
        # The poor match WITH mutual friends should rank first.
        assert ranked[0][0].user_id == "poor_with_friends"


# ---------------------------------------------------------------------------
# College match tests (exact match)
# ---------------------------------------------------------------------------

class TestCollegeMatch:
    """Tests for college exact-match scoring.

    WHY exact match: You either went to the same school or you didn't.
    There's no "partial" college match (unlike hobbies where partial overlap
    makes sense). Case-insensitive because "UCLA" and "ucla" are the same.
    """

    def test_same_college(self):
        """Same college → 1.0."""
        a = UserProfile(user_id="a", college="UCLA")
        b = UserProfile(user_id="b", college="UCLA")
        assert compute_match_score(a, b).college_match == 1.0

    def test_same_college_case_insensitive(self):
        """College matching should be case-insensitive."""
        a = UserProfile(user_id="a", college="UCLA")
        b = UserProfile(user_id="b", college="ucla")
        assert compute_match_score(a, b).college_match == 1.0

    def test_different_college(self):
        """Different colleges → 0.0."""
        a = UserProfile(user_id="a", college="UCLA")
        b = UserProfile(user_id="b", college="NYU")
        assert compute_match_score(a, b).college_match == 0.0

    def test_both_college_none(self):
        """Both have no college → 0.0 (no signal, no credit)."""
        a = UserProfile(user_id="a", college=None)
        b = UserProfile(user_id="b", college=None)
        assert compute_match_score(a, b).college_match == 0.0

    def test_one_college_none(self):
        """One has college, other doesn't → 0.0."""
        a = UserProfile(user_id="a", college="UCLA")
        b = UserProfile(user_id="b", college=None)
        assert compute_match_score(a, b).college_match == 0.0


# ---------------------------------------------------------------------------
# Faith match tests (exact match)
# ---------------------------------------------------------------------------

class TestFaithMatch:
    """Tests for faith exact-match scoring.

    Same logic as college: exact match → 1.0, anything else → 0.0.
    """

    def test_same_faith(self):
        """Same faith → 1.0."""
        a = UserProfile(user_id="a", faith="Christian")
        b = UserProfile(user_id="b", faith="Christian")
        assert compute_match_score(a, b).faith_match == 1.0

    def test_same_faith_case_insensitive(self):
        """Faith matching should be case-insensitive."""
        a = UserProfile(user_id="a", faith="Christian")
        b = UserProfile(user_id="b", faith="christian")
        assert compute_match_score(a, b).faith_match == 1.0

    def test_different_faith(self):
        """Different faiths → 0.0."""
        a = UserProfile(user_id="a", faith="Christian")
        b = UserProfile(user_id="b", faith="Buddhist")
        assert compute_match_score(a, b).faith_match == 0.0

    def test_both_faith_none(self):
        """Both have no faith → 0.0."""
        a = UserProfile(user_id="a", faith=None)
        b = UserProfile(user_id="b", faith=None)
        assert compute_match_score(a, b).faith_match == 0.0

    def test_one_faith_none(self):
        """One has faith, other doesn't → 0.0."""
        a = UserProfile(user_id="a", faith="Jewish")
        b = UserProfile(user_id="b", faith=None)
        assert compute_match_score(a, b).faith_match == 0.0


# ---------------------------------------------------------------------------
# Travel overlap tests (Jaccard similarity)
# ---------------------------------------------------------------------------

class TestTravelOverlap:
    """Tests for travel wishlist Jaccard scoring.

    Works the same as hobbies/interests/fan_of — Jaccard similarity on
    two sets of destination strings.
    """

    def test_perfect_travel_overlap(self):
        """Identical travel lists → 1.0."""
        travel = {"japan", "iceland", "portugal"}
        a = UserProfile(user_id="a", travel_wishlist=travel)
        b = UserProfile(user_id="b", travel_wishlist=travel)
        assert compute_match_score(a, b).travel_overlap == 1.0

    def test_no_travel_overlap(self):
        """Completely different travel lists → 0.0."""
        a = UserProfile(user_id="a", travel_wishlist={"japan"})
        b = UserProfile(user_id="b", travel_wishlist={"brazil"})
        assert compute_match_score(a, b).travel_overlap == 0.0

    def test_partial_travel_overlap(self):
        """Partial overlap on travel wishlists."""
        a = UserProfile(user_id="a", travel_wishlist={"japan", "iceland"})
        b = UserProfile(user_id="b", travel_wishlist={"japan", "spain"})
        # Jaccard: {japan} / {japan, iceland, spain} = 1/3
        result = compute_match_score(a, b)
        assert result.travel_overlap == pytest.approx(1 / 3, abs=0.001)

    def test_both_travel_empty(self):
        """Both have empty travel lists → 0.0."""
        a = UserProfile(user_id="a", travel_wishlist=set())
        b = UserProfile(user_id="b", travel_wishlist=set())
        assert compute_match_score(a, b).travel_overlap == 0.0


# ---------------------------------------------------------------------------
# Ranking tests
# ---------------------------------------------------------------------------

class TestRankCandidates:
    """Tests for the rank_candidates function (two-lane sort)."""

    def test_descending_order(self):
        """Higher-scoring candidates should appear first."""
        source = UserProfile(
            user_id="source",
            hobbies={"hiking"},
            interests={"tech"},
            location=Location(city="Austin", state="Texas"),
            age=24,
        )
        c1 = UserProfile(
            user_id="c1",
            hobbies={"hiking"},
            interests={"tech"},
            location=Location(city="Austin", state="Texas"),
            age=24,
        )
        c2 = UserProfile(
            user_id="c2",
            hobbies={"basketball"},
            interests={"hip hop"},
            location=Location(city="Miami", state="Florida"),
            age=40,
        )

        ranked = rank_candidates(source, [c2, c1])
        # c1 has more overlap → should rank first.
        assert ranked[0][0].user_id == "c1"
        assert ranked[1][0].user_id == "c2"

    def test_empty_candidate_list(self):
        """Empty candidate list → empty results (no crash)."""
        source = UserProfile(user_id="source")
        assert rank_candidates(source, []) == []

    def test_identical_candidates_all_returned(self):
        """Identical candidates should all appear in results (no deduplication)."""
        source = UserProfile(user_id="source", hobbies={"hiking"}, age=25)
        candidates = [
            UserProfile(user_id=f"c{i}", hobbies={"hiking"}, age=25)
            for i in range(5)
        ]
        ranked = rank_candidates(source, candidates)
        assert len(ranked) == 5


# ---------------------------------------------------------------------------
# Boundary / score cap tests
# ---------------------------------------------------------------------------

class TestBoundaryConditions:
    """Tests for edge cases and numerical boundaries."""

    def test_score_capped_at_1(self):
        """Even with perfect overlap + social boost, score should not exceed 1.0.

        WHY: Scores above 1.0 would break the interpretability of the system.
        The cap ensures 1.0 = best possible match, always.
        """
        friends = {f"f{i}" for i in range(25)}
        a = UserProfile(
            user_id="a",
            hobbies={"x", "y", "z"},
            interests={"p", "q"},
            fan_of={"r", "s"},
            location=Location(city="Austin", state="Texas"),
            age=25,
            mutual_friend_ids=friends,
            college="UCLA",
            faith="Agnostic",
            travel_wishlist={"japan", "iceland"},
        )
        b = UserProfile(
            user_id="b",
            hobbies={"x", "y", "z"},
            interests={"p", "q"},
            fan_of={"r", "s"},
            location=Location(city="Austin", state="Texas"),
            age=25,
            mutual_friend_ids=friends,
            college="UCLA",
            faith="Agnostic",
            travel_wishlist={"japan", "iceland"},
        )
        result = compute_match_score(a, b)
        assert result.total_score <= 1.0

    def test_minimal_profiles_no_crash(self):
        """Two profiles with only user_id should score without error.

        WHY: In production, some profiles may have missing data. The algorithm
        should gracefully return 0.0 instead of crashing.
        """
        a = UserProfile(user_id="a")
        b = UserProfile(user_id="b")
        result = compute_match_score(a, b)
        assert result.total_score == 0.0
        assert result.has_mutual_friends is False

    def test_custom_weights(self):
        """Scoring should respect custom weights.

        WHY: Different applications might want to weight signals differently.
        A college alumni app might heavily weight college_match. This test
        verifies that custom weights are actually used, not ignored.
        """
        a = UserProfile(user_id="a", hobbies={"x"}, age=25)
        b = UserProfile(user_id="b", hobbies={"x"}, age=25)

        # Put ALL weight on hobby_overlap, zero everything else.
        w = ScoringWeights(
            hobby_overlap=1.0,
            interest_overlap=0.0,
            fan_of_overlap=0.0,
            mutual_friends=0.0,
            location_match=0.0,
            age_compatibility=0.0,
            college_match=0.0,
            faith_match=0.0,
            travel_overlap=0.0,
            friend_common_boost=0.0,
        )
        result = compute_match_score(a, b, w)
        # Perfect hobby overlap with weight 1.0 → total_score should be 1.0.
        assert result.total_score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------

class TestInputValidation:
    """Tests for UserProfile validation.

    WHY validate: Bad data should fail loudly at creation time, not silently
    produce wrong scores downstream. These tests ensure the guardrails work.
    """

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
        """Age 0 is technically valid (for data integrity, not for matching)."""
        p = UserProfile(user_id="a", age=0)
        assert p.age == 0

    def test_age_150_valid(self):
        """Age 150 is the upper boundary — should be accepted."""
        p = UserProfile(user_id="a", age=150)
        assert p.age == 150

    def test_none_age_valid(self):
        """None age means unknown — should be accepted."""
        p = UserProfile(user_id="a", age=None)
        assert p.age is None


# ---------------------------------------------------------------------------
# Loader / tokenizer tests
# ---------------------------------------------------------------------------

class TestTokenize:
    """Tests for the semicolon tokenizer used in CSV loading.

    The tokenizer splits "Hiking; Cooking; Yoga" into {"hiking", "cooking", "yoga"}.
    It's the bridge between human-readable CSV data and the set fields on
    UserProfile.
    """

    def test_basic_tokenization(self):
        assert tokenize("Hiking; Cooking; Yoga") == {"hiking", "cooking", "yoga"}

    def test_empty_string(self):
        assert tokenize("") == set()

    def test_whitespace_only(self):
        assert tokenize("  ;  ;  ") == set()

    def test_single_item(self):
        assert tokenize("Hiking") == {"hiking"}

    def test_duplicate_items(self):
        """Duplicates should be collapsed (sets have no duplicates)."""
        assert tokenize("Hiking; hiking; HIKING") == {"hiking"}
