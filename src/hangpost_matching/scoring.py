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

# A source is "cold-start" when it has fewer than this many populated
# signal fields. The threshold is deliberately permissive: a user with
# only an age and an interest list still falls below it, which is the
# population a cold-start fallback is meant to help.
COLD_START_FIELD_THRESHOLD = 3


def is_cold_start(profile: UserProfile) -> bool:
    """Return True if `profile` has too few signals for the main ranker.

    "Cold start" here means a brand-new user the engine knows almost
    nothing about — no mutual friends, no hometown/college, and at most
    a couple of interests. The full ranker will dump them straight to
    Lane C with a near-zero score; the cold-start fallback exists to
    give them a useful list anyway.
    """
    signals = sum(
        [
            bool(profile.mutual_friend_ids),
            bool(profile.hometown),
            bool(profile.college),
            len(profile.interests) >= 2,
            len(profile.liked_topics) >= 2,
            profile.age is not None,
        ]
    )
    return signals < COLD_START_FIELD_THRESHOLD


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
    # "Shared background" trigger: same hometown OR same college is enough
    # to qualify. `has_both_shared_background` is the stricter sub-tier —
    # candidates that hit BOTH outrank candidates that hit only one,
    # regardless of how strong their hobby / age signals are. See
    # PRODUCT_VISION.md.
    has_shared_background = hometown_match > 0.0 or college_match > 0.0
    has_both_shared_background = hometown_match > 0.0 and college_match > 0.0
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
        has_both_shared_background=has_both_shared_background,
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

    Sort strategy (descending), with four lexicographic tiers:
    1) `has_mutual_friends`             (True before False)
    2) `has_both_shared_background`     (True before False)
    3) `has_shared_background`          (True before False)
    4) `total_score`                    (weighted compatibility)

    Combined, that produces six effective tiers:

    - Tier 1: mutual friend + same hometown AND same college   (Lane A++)
    - Tier 2: mutual friend + same hometown OR same college    (Lane A+)
    - Tier 3: mutual friend, no background match               (Lane A)
    - Tier 4: no mutual + same hometown AND same college       (Lane B+)
    - Tier 5: no mutual + same hometown OR same college        (Lane B)
    - Tier 6: neither                                          (Lane C)

    Both-background is a HARD tier above either-background — a candidate
    who matches on both hometown and college will outrank a candidate
    who matches on just one, regardless of how much higher the latter
    scores on hobbies / age / semantic similarity. Within each tier,
    `total_score` still decides ordering, so the soft signals do
    meaningful work for the bulk of candidates.

    The combination of mutual friend + same hometown + same college
    (Tier 1) is, by construction, the highest possible position any
    candidate can occupy — and within Tier 1 the additive total_score
    promotes the same-age candidate to the very top.
    """
    scored = [
        (candidate, compute_match_score(source, candidate, weights, profile_embeddings))
        for candidate in candidates
    ]
    return sorted(
        scored,
        key=lambda item: (
            item[1].has_mutual_friends,
            item[1].has_both_shared_background,
            item[1].has_shared_background,
            item[1].total_score,
        ),
        reverse=True,
    )


def rank_candidates_with_cold_start(
    source: UserProfile,
    candidates: list[UserProfile],
    weights: ScoringWeights | None = None,
    profile_embeddings: Mapping[str, Vector] | None = None,
) -> list[tuple[UserProfile, MatchBreakdown]]:
    """Like `rank_candidates`, but for very sparse source profiles falls
    back to a candidate-popularity prior (most-connected first).

    Behaviour:
    - If the source profile is NOT cold-start: identical to
      `rank_candidates`.
    - If the source IS cold-start: ranks by candidate's own
      `mutual_friend_ids` size, then by interest-list size — both proxies
      for "this candidate is active and well-connected, surfacing them
      to a new user is a safer bet than surfacing a similarly-empty
      stranger." The full `MatchBreakdown` is still returned so the UI
      stays explainable.

    Why this matters: a brand-new user with no friends, no hometown, and
    one interest gets dumped into Lane C by the main ranker with a
    near-zero score, which means their feed is effectively random. The
    fallback gives them a populated, defensible list on day one.
    """
    if not is_cold_start(source):
        return rank_candidates(source, candidates, weights, profile_embeddings)

    scored = [
        (candidate, compute_match_score(source, candidate, weights, profile_embeddings))
        for candidate in candidates
    ]
    return sorted(
        scored,
        key=lambda item: (
            len(item[0].mutual_friend_ids),
            len(item[0].interests),
            item[1].total_score,
        ),
        reverse=True,
    )
