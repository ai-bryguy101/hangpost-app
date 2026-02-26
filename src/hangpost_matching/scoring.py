from .models import MatchBreakdown, ScoringWeights, UserProfile


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _bounded_ratio(value: float, max_value: float) -> float:
    if max_value <= 0:
        return 0.0
    return min(max(value / max_value, 0.0), 1.0)


def _location_score(source: UserProfile, candidate: UserProfile) -> float:
    if source.location and candidate.location and source.location == candidate.location:
        return 1.0
    return 0.0


def _age_compatibility_score(source: UserProfile, candidate: UserProfile, max_gap: int = 15) -> float:
    if source.age is None or candidate.age is None:
        return 0.0
    gap = abs(source.age - candidate.age)
    return 1.0 - _bounded_ratio(gap, max_gap)


def compute_match_score(
    source: UserProfile,
    candidate: UserProfile,
    weights: ScoringWeights | None = None,
) -> MatchBreakdown:
    active_weights = weights or ScoringWeights()

    interest_overlap = _jaccard_similarity(source.interests, candidate.interests)
    liked_topic_overlap = _jaccard_similarity(source.liked_topics, candidate.liked_topics)
    mutual_friend_count = len(source.mutual_friend_ids & candidate.mutual_friend_ids)
    mutual_friends = _bounded_ratio(mutual_friend_count, 20)
    has_mutual_friends = mutual_friend_count > 0

    location_match = _location_score(source, candidate)
    age_compatibility = _age_compatibility_score(source, candidate)

    base_score = (
        active_weights.interest_overlap * interest_overlap
        + active_weights.liked_topic_overlap * liked_topic_overlap
        + active_weights.mutual_friends * mutual_friends
        + active_weights.location_match * location_match
        + active_weights.age_compatibility * age_compatibility
    )

    social_boost = active_weights.friend_common_boost if has_mutual_friends else 0.0
    total_score = min(base_score + social_boost, 1.0)

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
    scored = [(candidate, compute_match_score(source, candidate, weights)) for candidate in candidates]
    return sorted(
        scored,
        key=lambda item: (item[1].has_mutual_friends, item[1].total_score),
        reverse=True,
    )
