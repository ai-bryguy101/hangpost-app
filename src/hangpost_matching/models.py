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

    # Location value used by simple exact-match scoring.
    location: str | None = None

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

    # Exact same-location bonus inside base score.
    location_match: float = 0.10

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

    # The extra social boost applied when `has_mutual_friends` is True.
    social_boost: float

    # Component scores used by the base model.
    interest_overlap: float
    liked_topic_overlap: float
    mutual_friends: float
    location_match: float
    age_compatibility: float
    semantic_similarity: float
