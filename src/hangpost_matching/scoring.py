"""Core scoring and ranking logic.

High-level flow:
1) Compute component scores (overlap, age closeness, etc.)
2) Build a weighted base compatibility score
3) Apply separate social boost when mutual friends exist
4) Sort candidates with a two-lane strategy:
   - lane A: profiles with mutual friends
   - lane B: profiles without mutual friends
"""

from .models import MatchBreakdown, ScoringWeights, UserProfile

# Maximum mutual-friend count used for normalization. Profiles with this many
# (or more) mutual friends receive the maximum mutual-friends component score
# of 1.0. Adjust this if your social graph typically has denser connections.
MAX_MUTUAL_FRIENDS = 20


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    """Return overlap ratio between two sets using Jaccard similarity.

    Formula:
        |intersection(left, right)| / |union(left, right)|

    Range: 0.0 to 1.0
    """
    if not left and not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _bounded_ratio(value: float, max_value: float) -> float:
    """Normalize value/max_value into [0.0, 1.0]."""
    if max_value <= 0:
        return 0.0
    return min(max(value / max_value, 0.0), 1.0)


def _location_score(source: UserProfile, candidate: UserProfile) -> float:
    """Case-insensitive exact-match location score."""
    if (
        source.location
        and candidate.location
        and source.location.strip().lower() == candidate.location.strip().lower()
    ):
        return 1.0
    return 0.0


def _age_compatibility_score(source: UserProfile, candidate: UserProfile) -> float:
    """Return step-down age compatibility with a 10% drop per year.

    Requested behavior:
    - same age (gap 0) -> 1.0 (100% of age weight)
    - 1 year apart -> 0.9
    - 2 years apart -> 0.8
    - ...
    - 10+ years apart -> 0.0

    This is a sequential/ladder score rather than a continuous normalization.
    """
    if source.age is None or candidate.age is None:
        return 0.0

    gap = abs(source.age - candidate.age)
    compatibility = 1.0 - (0.1 * gap)
    return max(compatibility, 0.0)


def compute_match_score(
    source: UserProfile,
    candidate: UserProfile,
    weights: ScoringWeights | None = None,
) -> MatchBreakdown:
    """Compute full explainable score breakdown for a source/candidate pair.

    This function is intentionally explicit and verbose for readability.
    """
    active_weights = weights or ScoringWeights()

    # 1) Component-level similarities/signals.
    interest_overlap = _jaccard_similarity(source.interests, candidate.interests)
    liked_topic_overlap = _jaccard_similarity(source.liked_topics, candidate.liked_topics)

    # Mutual-friend details are split into:
    # - count (for explanation)
    # - normalized ratio (for weighted base score)
    # - boolean trigger (for separate social-boost lane)
    mutual_friend_count = len(source.mutual_friend_ids & candidate.mutual_friend_ids)
    mutual_friends = _bounded_ratio(mutual_friend_count, MAX_MUTUAL_FRIENDS)
    has_mutual_friends = mutual_friend_count > 0

    location_match = _location_score(source, candidate)
    age_compatibility = _age_compatibility_score(source, candidate)

    # 2) Weighted base score (normal compatibility lane).
    base_score = (
        active_weights.interest_overlap * interest_overlap
        + active_weights.liked_topic_overlap * liked_topic_overlap
        + active_weights.mutual_friends * mutual_friends
        + active_weights.location_match * location_match
        + active_weights.age_compatibility * age_compatibility
    )

    # 3) Separate social boost lane.
    # If user has mutual friends with candidate, push them upward significantly.
    social_boost = active_weights.friend_common_boost if has_mutual_friends else 0.0

    # Cap final score at 1.0 for easier interpretation.
    total_score = min(base_score + social_boost, 1.0)

    # 4) Return rounded, explainable values.
    return MatchBreakdown(
        total_score=round(total_score, 6),
        has_mutual_friends=has_mutual_friends,
        social_boost=round(social_boost, 6),
        interest_overlap=round(interest_overlap, 6),
        liked_topic_overlap=round(liked_topic_overlap, 6),
        mutual_friends=round(mutual_friends, 6),
        location_match=round(location_match, 6),
        age_compatibility=round(age_compatibility, 6),
    )


def rank_candidates(
    source: UserProfile,
    candidates: list[UserProfile],
    weights: ScoringWeights | None = None,
) -> list[tuple[UserProfile, MatchBreakdown]]:
    """Rank candidates for one source profile.

    Sort strategy (descending):
    1) `has_mutual_friends` (True before False)
    2) `total_score`

    This creates a clear two-lane policy:
    - socially-connected candidates first
    - then non-connected candidates by compatibility
    """
    scored = [(candidate, compute_match_score(source, candidate, weights)) for candidate in candidates]
    return sorted(
        scored,
        key=lambda item: (item[1].has_mutual_friends, item[1].total_score),
        reverse=True,
    )
