"""Data models for the matching engine.

This file only defines *data shapes* (no ranking logic):
- what a user profile looks like
- what weight settings look like
- what a scoring result looks like
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class UserProfile:
    """Normalized profile fields used by the ranking logic.

    Notes for beginners:
    - `frozen=True` means instances are immutable after creation.
      That helps avoid accidental changes during scoring.
    - `set[str]` fields are used for overlap comparisons (e.g., Jaccard similarity).
      Sets make intersections/unions easy and fast.
    """

    # Stable identifier for this profile in your system.
    user_id: str

    # Broad activities/skills bucket (from hobbies + activities + sports + games + certs).
    interests: set[str] = field(default_factory=set)

    # Likes/preferences bucket (food/music/philosophy/etc.).
    liked_topics: set[str] = field(default_factory=set)

    # Hometown — where the user grew up. Soft matching signal (NOT the
    # current-location radius pre-filter; that happens upstream).
    hometown: str | None = None

    # College / university the user attended. A peer-strength signal to
    # `hometown`: same-college is independently strong even when the
    # hometowns differ (e.g., two BU grads from different cities).
    college: str | None = None

    # Numeric age used for age-closeness scoring.
    age: int | None = None

    # IDs representing social graph overlap.
    # Any intersection with another profile indicates mutual friends.
    mutual_friend_ids: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class ScoringWeights:
    """Tunable knobs for how strongly each signal influences ranking.

    Current behavior:
    - `friend_common_boost` is a separate boost lane that activates when
      there is at least 1 mutual friend.
    - The other weights build the normal compatibility base score.

    Tip:
    Keep weights understandable and product-driven first.
    Later (with real outcome labels) you can learn these weights via ML.
    """

    # Overlap on hobbies/activities/skills bucket.
    interest_overlap: float = 0.20

    # Overlap on likes/preferences bucket.
    liked_topic_overlap: float = 0.15

    # Strength of mutual-friend count ratio inside base score.
    mutual_friends: float = 0.30

    # Exact same-hometown bonus inside base score.
    hometown_match: float = 0.10

    # Exact same-college bonus inside base score.
    # Same weight as `hometown_match` on purpose: shared origin and shared
    # alma mater are both top-tier friendship cues, and they're independent
    # (you can match on one without the other).
    college_match: float = 0.10

    # Age-closeness strength inside base score.
    age_compatibility: float = 0.25

    # Phase 2 signal: cosine similarity between auto-synthesized profile
    # embeddings. The "profile text" is built deterministically from the
    # structured fields above — users do NOT hand-write a bio.
    # See `hangpost_matching.embeddings.profile_to_text`.
    semantic_similarity: float = 0.20

    # Separate social boost if candidate shares any mutual friends.
    friend_common_boost: float = 0.35


@dataclass(frozen=True)
class MatchBreakdown:
    """Explainable output from `compute_match_score`.

    This object is intentionally verbose so you can:
    - debug why a profile ranked where it did
    - show transparent explanations in the product UI
    - inspect or log component scores for future ML training data
    """

    # Final score used for ordering (weighted base + social boost).
    total_score: float

    # Whether at least 1 mutual friend exists.
    has_mutual_friends: bool

    # Whether the candidate shares hometown OR college with the source.
    # Drives one tier in `rank_candidates`'s sort: after mutual-friend
    # candidates, candidates with shared background rank above candidates
    # with neither, regardless of how well the latter score on hobbies /
    # age / semantic similarity.
    has_shared_background: bool

    # Whether the candidate shares BOTH hometown AND college (strict
    # version of `has_shared_background`). Drives a stronger sub-tier:
    # within each mutual-friends bucket, candidates who hit both signals
    # outrank candidates who hit only one, regardless of hobbies. This
    # is a hard guarantee, not a score nudge — see `rank_candidates`.
    has_both_shared_background: bool

    # The extra social boost applied when `has_mutual_friends` is True.
    social_boost: float

    # Component scores used by the base model.
    interest_overlap: float
    liked_topic_overlap: float
    mutual_friends: float
    hometown_match: float
    college_match: float
    age_compatibility: float
    semantic_similarity: float
