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

    FRIENDSHIP TIER RATIONALE (why these weights):
    The weights encode a real-world friendship probability hierarchy:

    TIER 1 — "Instant friends" (friends in common)
    - mutual_friends (0.20) + friend_common_boost (0.35):
      The combo makes mutual-friend candidates strongly dominant. The base
      weight rewards *how many* shared friends you have; the boost fires even
      if you share just one. Together they can add up to 0.55 before other
      signals are counted.

    TIER 2 — "Probable friends" (hometown)
    - location_match (0.22): Now the single largest base weight.
      Being from the same city means you literally have the option to meet.
      Being in the same state is a meaningful partial signal (0.4 × 0.22 = 0.088).
      WHY so high: A friend in your city is infinitely more actionable than
      a perfect-taste-match living 2,000 miles away.

    TIER 3 — "Probable friends" (college)
    - college_match (0.18): Second-largest base weight.
      Alumni connections carry pre-built social context: shared campus,
      traditions, overlapping friend networks, and a built-in conversation
      opener ("Oh you went to UT too? Did you know...").
      WHY almost as high as location: same-school bonds are nearly as strong
      as same-city proximity — you already have a shared identity.

    TIER 4 — "Sometimes friends" (hobbies)
    - hobby_overlap (0.15): Shared activities = shared time together.
      Kept at the same weight — this is the right level for "sometimes friends."

    TIER 5 — "Sometimes friends" (interests & taste)
    - interest_overlap (0.08): Reduced. Broad taste alignment is nice but
      not as predictive as shared activities.
    - fan_of_overlap (0.05): Reduced. Good ice-breaker but shallow signal.

    SUPPORTING SIGNALS (not in user's tier list, still meaningful):
    - age_compatibility (0.07): Reduced significantly. Similar life stages
      matter, but not enough to override location or college.
    - faith_match (0.03): Small — relevant to those who care deeply.
    - travel_overlap (0.02): Lifestyle compatibility signal, but thin.

    BASE WEIGHT SUM: 0.20+0.22+0.18+0.15+0.08+0.05+0.07+0.03+0.02 = 1.00

    DESIGN NOTE:
    `friend_common_boost` is NOT part of the base score sum. It's a separate
    additive boost that fires only when mutual friends exist. This creates the
    "two-lane" ranking: people with mutual friends get pushed significantly
    higher. The total is capped at 1.0 so scores stay interpretable.
    """

    # ── Base score components (sum to 1.0) ──

    # TIER 2: Hometown — now the largest single base weight.
    # Same city = full credit. Same state = 40% credit (still meaningful).
    location_match: float = 0.22

    # TIER 3: College alumni match — nearly as strong as location.
    # Exact match only: you either went to the same school or you didn't.
    college_match: float = 0.18

    # TIER 1: Mutual friend count, normalized to MAX_MUTUAL_FRIENDS.
    # Combined with friend_common_boost below, mutual friends dominate ranking.
    mutual_friends: float = 0.20

    # TIER 4: Hobbies — shared activities you can actually do together.
    hobby_overlap: float = 0.15

    # TIER 5: Broad taste categories — "we're into the same kind of stuff."
    interest_overlap: float = 0.08

    # TIER 5: Specific fandoms — "oh you watch that too?!" moments.
    fan_of_overlap: float = 0.05

    # Supporting: age closeness. Important but not a tier in the user's model.
    age_compatibility: float = 0.07

    # Supporting: shared faith/values. Small but meaningful to those it applies.
    faith_match: float = 0.03

    # Supporting: travel wishlist overlap — thin lifestyle signal.
    travel_overlap: float = 0.02

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
