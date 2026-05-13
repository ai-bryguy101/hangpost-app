"""Core scoring and ranking logic.

High-level flow:
1) Compute component scores (overlap, age closeness, semantic similarity, etc.)
2) Build a weighted base compatibility score
3) Apply separate social boost when mutual friends exist
4) Sort candidates with a two-lane strategy:
   - lane A: profiles with mutual friends
   - lane B: profiles without mutual friends
"""

from collections.abc import Mapping

from .embeddings import Vector, cosine_similarity
from .models import MatchBreakdown, ScoringWeights, UserProfile


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


def _hometown_score(source: UserProfile, candidate: UserProfile) -> float:
    """Simple exact-match hometown score.

    Hometown is the place a user grew up — a soft matching signal. This is
    NOT the radius pre-filter (the radius is applied upstream of the
    ranker; see PRODUCT_VISION.md).
    """
    if source.hometown and candidate.hometown and source.hometown == candidate.hometown:
        return 1.0
    return 0.0


def _college_score(source: UserProfile, candidate: UserProfile) -> float:
    """Simple exact-match college score.

    Same-college is a peer-strength friendship cue to same-hometown — two
    users who went to the same university have a strong conversation
    starter even when they grew up in different cities. Hometown and
    college contribute independently, so a user can match on one, the
    other, or both.
    """
    if source.college and candidate.college and source.college == candidate.college:
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


def _semantic_similarity_score(
    source: UserProfile,
    candidate: UserProfile,
    profile_embeddings: Mapping[str, Vector] | None,
) -> float:
    """Cosine similarity between precomputed profile embeddings, clamped to [0, 1].

    Returns 0.0 when no embedding map is supplied or either user is absent.
    Negative cosine values are zeroed because they represent semantic
    *opposition*, which we don't want to reward or penalize at this stage.

    The text that produced these embeddings is auto-synthesized from each
    user's structured fields — see `hangpost_matching.embeddings.profile_to_text`.
    """
    if profile_embeddings is None:
        return 0.0
    source_vec = profile_embeddings.get(source.user_id)
    candidate_vec = profile_embeddings.get(candidate.user_id)
    if source_vec is None or candidate_vec is None:
        return 0.0
    return max(cosine_similarity(source_vec, candidate_vec), 0.0)


def compute_match_score(
    source: UserProfile,
    candidate: UserProfile,
    weights: ScoringWeights | None = None,
    profile_embeddings: Mapping[str, Vector] | None = None,
) -> MatchBreakdown:
    """Compute full explainable score breakdown for a source/candidate pair.

    Pass `profile_embeddings={user_id: vector, ...}` to enable Phase 2
    semantic similarity. The ranker itself never loads a model — see
    `hangpost_matching.embeddings.embed_profiles` for how to precompute the
    map once for a batch of candidates.

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
    mutual_friends = _bounded_ratio(mutual_friend_count, 20)
    has_mutual_friends = mutual_friend_count > 0

    hometown_match = _hometown_score(source, candidate)
    college_match = _college_score(source, candidate)
    # "Shared background" trigger for the second sort lane: same hometown
    # OR same college is enough. The two cues are independent — see
    # PRODUCT_VISION.md — so a candidate that matches on either qualifies.
    has_shared_background = hometown_match > 0.0 or college_match > 0.0
    age_compatibility = _age_compatibility_score(source, candidate)
    semantic_similarity = _semantic_similarity_score(source, candidate, profile_embeddings)

    # 2) Weighted base score (normal compatibility lane).
    # The cap on `total_score` was removed when `semantic_similarity` was
    # introduced. With seven weighted components plus the social boost, capping
    # at 1.0 silently compressed the top of the distribution and hid signal.
    # The ranker only cares about ordering, so absolute magnitude is fine.
    base_score = (
        active_weights.interest_overlap * interest_overlap
        + active_weights.liked_topic_overlap * liked_topic_overlap
        + active_weights.mutual_friends * mutual_friends
        + active_weights.hometown_match * hometown_match
        + active_weights.college_match * college_match
        + active_weights.age_compatibility * age_compatibility
        + active_weights.semantic_similarity * semantic_similarity
    )

    # 3) Separate social boost lane.
    # If user has mutual friends with candidate, push them upward significantly.
    social_boost = active_weights.friend_common_boost if has_mutual_friends else 0.0

    total_score = base_score + social_boost

    # 4) Return rounded, explainable values.
    return MatchBreakdown(
        total_score=round(total_score, 6),
        has_mutual_friends=has_mutual_friends,
        has_shared_background=has_shared_background,
        social_boost=round(social_boost, 6),
        interest_overlap=round(interest_overlap, 6),
        liked_topic_overlap=round(liked_topic_overlap, 6),
        mutual_friends=round(mutual_friends, 6),
        hometown_match=round(hometown_match, 6),
        college_match=round(college_match, 6),
        age_compatibility=round(age_compatibility, 6),
        semantic_similarity=round(semantic_similarity, 6),
    )


def rank_candidates(
    source: UserProfile,
    candidates: list[UserProfile],
    weights: ScoringWeights | None = None,
    profile_embeddings: Mapping[str, Vector] | None = None,
) -> list[tuple[UserProfile, MatchBreakdown]]:
    """Rank candidates for one source profile.

    Sort strategy (descending):
    1) `has_mutual_friends`         (True before False) — tier 1
    2) `has_shared_background`      (True before False) — tier 2
    3) `total_score`                (weighted compatibility)

    This creates a clear three-lane policy that matches the product
    tiering described in PRODUCT_VISION.md:

    - Lane A — socially-connected candidates (≥1 mutual friend) always
      rank first, regardless of any other signal.
    - Lane B — no mutual friends, but same hometown OR same college.
      Always ranks above Lane C even when Lane C has higher hobby /
      age / semantic compatibility.
    - Lane C — neither tier 1 nor tier 2. Ordered by the weighted
      `total_score` so age, hobbies, and semantic similarity still
      decide who's surfaced first.
    """
    scored = [
        (candidate, compute_match_score(source, candidate, weights, profile_embeddings))
        for candidate in candidates
    ]
    return sorted(
        scored,
        key=lambda item: (
            item[1].has_mutual_friends,
            item[1].has_shared_background,
            item[1].total_score,
        ),
        reverse=True,
    )
