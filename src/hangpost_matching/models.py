"""Data models for the matching engine.

This file only defines *data shapes* (no ranking logic):
- what a user profile looks like
- what weight settings look like
- what a scoring result looks like

WHAT CHANGED AND WHY (v0.2.0):
- Location is now a city+state pair instead of a single string.
  WHY: In the real world, two people in different Texas cities have more in
  common than two people in different states. A single string ("austin") gave
  us only binary matching (0 or 1). Now we can score city-match vs state-match
  at different strengths.

- The old `interests` and `liked_topics` fields are replaced by three fields:
  `hobbies`, `interests`, and `fan_of`.
  WHY: The old model mixed *things you do* (hiking), *categories you like*
  (hip hop), and *specific things you're a fan of* (Kendrick Lamar) into two
  overlapping buckets. That's like mixing nouns and verbs — the algorithm
  can't tell if two people share an activity vs a taste vs a specific fandom.
  Splitting into three gives much better signal:
    - hobbies  = activities you actively participate in  (Hiking, Chess, Guitar)
    - interests = broad categories/types you enjoy       (Hip Hop, Japanese Food, Film)
    - fan_of   = specific named things you love          (Kendrick Lamar, The Bear, NFL)

- New fields: `college`, `faith`, `travel_wishlist`.
  WHY: These were already in our CSV data but never loaded into UserProfile,
  which means the scoring engine completely ignored them. Alumni connections,
  shared faith, and travel overlap are all real friendship signals.
"""

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
# These run inside __post_init__ on every UserProfile creation. They exist
# to catch bad data early (e.g., negative ages, empty IDs) rather than
# letting it silently produce wrong scores downstream.

def _validate_age(age: int | None) -> None:
    """Raise ValueError if age is present but invalid."""
    if age is not None:
        if not isinstance(age, int):
            raise TypeError(f"age must be an int or None, got {type(age).__name__}")
        if age < 0 or age > 150:
            raise ValueError(f"age must be between 0 and 150, got {age}")


def _validate_user_id(user_id: str) -> None:
    """Raise ValueError if user_id is empty or not a string."""
    if not isinstance(user_id, str):
        raise TypeError(f"user_id must be a str, got {type(user_id).__name__}")
    if not user_id.strip():
        raise ValueError("user_id must not be empty or whitespace-only")


# ---------------------------------------------------------------------------
# Location model
# ---------------------------------------------------------------------------
# Previously, location was just a string like "austin". That only allowed
# exact-match scoring (1.0 or 0.0). By splitting into city + state, we can
# do tiered scoring: same city = full credit, same state = partial credit.

@dataclass(frozen=True)
class Location:
    """A city+state pair representing where someone lives.

    WHY a separate dataclass instead of two loose strings on UserProfile?
    Because city and state are inherently linked — "Austin" only means
    something when paired with "Texas" (there's also an Austin, Minnesota).
    Grouping them prevents bugs where city and state get out of sync.
    """
    city: str    # e.g. "Austin"
    state: str   # e.g. "Texas"


# ---------------------------------------------------------------------------
# User profile
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UserProfile:
    """Normalized profile fields used by the ranking logic.

    DESIGN PRINCIPLES:
    - `frozen=True` means instances are immutable after creation.
      This prevents accidental changes during scoring.
    - `set[str]` fields are used for overlap comparisons (Jaccard similarity).
      Sets make intersections and unions fast and easy.
    - Every field here either feeds directly into scoring or identifies the user.
      We don't store display-only fields here — those stay in the raw CSV row.

    FIELD MAPPING (what feeds into which scoring signal):
    - hobbies          → hobby_overlap      (Jaccard similarity)
    - interests        → interest_overlap   (Jaccard similarity)
    - fan_of           → fan_of_overlap     (Jaccard similarity)
    - location         → location_match     (tiered: city > state > none)
    - age              → age_compatibility  (step-down ladder)
    - mutual_friend_ids → mutual_friends    (bounded ratio + social boost)
    - college          → college_match      (exact match)
    - faith            → faith_match        (exact match)
    - travel_wishlist  → travel_overlap     (Jaccard similarity)
    """

    # ── Identity ──
    user_id: str

    # ── Activity signals (things you DO) ──
    # Examples: Hiking, Chess, Guitar, Photography
    # WHY separate from interests: doing an activity together is a stronger
    # friendship signal than both liking the same broad category.
    hobbies: set[str] = field(default_factory=set)

    # ── Taste signals (broad categories you enjoy) ──
    # Examples: Hip Hop, Japanese Food, Outdoor Sports, Film/TV
    # WHY separate from fan_of: these capture general taste alignment.
    # Two people who both like "Hip Hop" have compatible taste even if
    # they listen to different specific artists.
    interests: set[str] = field(default_factory=set)

    # ── Fandom signals (specific things you love) ──
    # Examples: Kendrick Lamar, The Bear, NFL, Zelda, Atomic Habits
    # WHY this matters: sharing a specific fandom is a strong conversation
    # starter. "Oh you watch The Bear too?" is instant connection.
    fan_of: set[str] = field(default_factory=set)

    # ── Location ──
    # Now a structured city+state pair for tiered scoring.
    # None means location is unknown/not provided.
    location: Location | None = None

    # ── Demographics ──
    age: int | None = None

    # ── Social graph ──
    # IDs of friends this user has. When two profiles share IDs in this set,
    # they have mutual friends — the strongest friendship predictor.
    mutual_friend_ids: set[str] = field(default_factory=set)

    # ── Education ──
    # College name for alumni matching. Exact-match only for now.
    # WHY: People who went to the same school share a built-in social context
    # (same campus, same traditions, overlapping friend networks).
    college: str | None = None

    # ── Values ──
    # Faith/religion for values-alignment matching. Exact-match only.
    # WHY: For many people, shared faith is a core part of their social life
    # (church groups, community events, shared holidays).
    faith: str | None = None

    # ── Travel ──
    # Places the user wants to visit or has visited. Jaccard similarity.
    # WHY: Shared travel interests signal compatible lifestyles and give
    # people something to bond over ("I've always wanted to go to Japan too!").
    travel_wishlist: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        """Validate fields after initialization.

        WHY validate here: frozen dataclasses can't be modified after creation,
        so if bad data gets in, it's stuck. Better to catch it immediately.
        """
        _validate_user_id(self.user_id)
        _validate_age(self.age)


# ---------------------------------------------------------------------------
# Scoring weights
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScoringWeights:
    """Tunable knobs for how strongly each signal influences ranking.

    HOW THE WEIGHTS WORK:
    Each weight controls what fraction of the base score comes from that signal.
    The base score weights should roughly sum to 1.0 so the base score stays
    in the 0.0-1.0 range before the social boost is applied.

    WHY THESE DEFAULTS:
    - mutual_friends (0.25): Strongest real-world friendship predictor. If you
      already share friends, you're very likely to get along.
    - age_compatibility (0.15): Age matters for shared life stage, but it's not
      everything — a 25-year-old and 30-year-old can absolutely be friends.
    - hobby_overlap (0.15): Shared activities = shared time together.
    - interest_overlap (0.12): Broad taste alignment matters for conversation.
    - fan_of_overlap (0.08): Specific shared fandoms are great ice-breakers.
    - location_match (0.08): Proximity matters for hanging out in person.
    - college_match (0.05): Alumni connection is a nice bonus, not a dealbreaker.
    - faith_match (0.05): Shared values matter for some, not all.
    - travel_overlap (0.07): Shared wanderlust signals lifestyle compatibility.
    - friend_common_boost (0.35): Separate lane — big boost when mutual friends exist.

    DESIGN NOTE:
    `friend_common_boost` is NOT part of the base score sum. It's a separate
    additive boost that fires only when mutual friends exist. This creates the
    "two-lane" ranking: people with mutual friends get pushed significantly
    higher. The total is capped at 1.0 so scores stay interpretable.
    """

    # ── Base score components (should roughly sum to 1.0) ──

    # Things you DO together — strong friendship signal.
    hobby_overlap: float = 0.15

    # Broad taste categories — "we're into the same kind of stuff."
    interest_overlap: float = 0.12

    # Specific fandoms — "oh you watch that too?!" moments.
    fan_of_overlap: float = 0.08

    # Mutual friend count, normalized. The more friends you share,
    # the more likely you'll get along.
    mutual_friends: float = 0.25

    # Tiered location: same city (full), same state (partial), else 0.
    location_match: float = 0.08

    # Age closeness — similar life stages.
    age_compatibility: float = 0.15

    # Same college — shared institutional context.
    college_match: float = 0.05

    # Same faith — shared values and community.
    faith_match: float = 0.05

    # Overlapping travel bucket lists.
    travel_overlap: float = 0.07

    # ── Separate social boost (not part of the base score sum) ──
    friend_common_boost: float = 0.35


# ---------------------------------------------------------------------------
# Match breakdown (explainable output)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MatchBreakdown:
    """Explainable output from `compute_match_score`.

    WHY this exists:
    Every matching system needs transparency. When you look at a match and
    think "why is this person ranked #3?", this breakdown answers that question.
    Each field shows exactly how much that signal contributed.

    This is also critical for future ML work — these component scores become
    training features when you start logging outcomes (clicked, friended, etc.).
    """

    # Final score used for ordering (after social boost, capped at 1.0).
    total_score: float

    # Whether at least 1 mutual friend exists (drives the two-lane sort).
    has_mutual_friends: bool

    # The extra social boost applied when `has_mutual_friends` is True.
    social_boost: float

    # ── Individual component scores (each 0.0 to 1.0) ──
    hobby_overlap: float
    interest_overlap: float
    fan_of_overlap: float
    mutual_friends: float
    location_match: float
    age_compatibility: float
    college_match: float
    faith_match: float
    travel_overlap: float
