from dataclasses import dataclass, field


@dataclass(frozen=True)
class UserProfile:
    user_id: str
    interests: set[str] = field(default_factory=set)
    liked_topics: set[str] = field(default_factory=set)
    location: str | None = None
    age: int | None = None
    mutual_friend_ids: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class ScoringWeights:
    interest_overlap: float = 0.20
    liked_topic_overlap: float = 0.15
    mutual_friends: float = 0.30
    location_match: float = 0.10
    age_compatibility: float = 0.25
    friend_common_boost: float = 0.35


@dataclass(frozen=True)
class MatchBreakdown:
    total_score: float
    has_mutual_friends: bool
    social_boost: float
    interest_overlap: float
    liked_topic_overlap: float
    mutual_friends: float
    location_match: float
    age_compatibility: float
