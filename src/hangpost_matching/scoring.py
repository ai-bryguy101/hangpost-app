"""Core scoring and ranking logic.

High-level flow:
1) Compute component scores for each signal (overlap, age closeness, etc.)
2) Build a weighted base compatibility score from all components
3) Apply a separate social boost when mutual friends exist
4) Sort candidates with a two-lane strategy:
   - Lane A: profiles WITH mutual friends (sorted by score)
   - Lane B: profiles WITHOUT mutual friends (sorted by score)

WHAT CHANGED AND WHY (v0.2.0):
- Location scoring is now TIERED instead of binary.
  OLD: "austin" == "austin" → 1.0, anything else → 0.0
  NEW: same city+state → 1.0, different city but same state → 0.4, else → 0.0
  WHY: Two people in Dallas and Houston (both Texas) have more in common than
  someone in Dallas and someone in Seattle. The 0.4 value for same-state is a
  product decision — it says "being in the same state is worth 40% of being in
  the same city." This can be tuned later with real outcome data.

- Three new Jaccard-scored fields: hobbies, interests, fan_of (replaces the
  old interests + liked_topics that mixed everything together).

- Three new exact-match signals: college, faith, travel (were in the data but
  completely ignored before — now they contribute to the score).
"""

from .models import Location, MatchBreakdown, ScoringWeights, UserProfile

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum mutual-friend count used for normalization. Profiles with this many
# (or more) mutual friends receive the maximum mutual-friends component score
# of 1.0. We cap at 20 because in most social networks, having 20+ mutual
# friends means you're deeply embedded in the same social circle.
MAX_MUTUAL_FRIENDS = 20

# How much credit to give for same-state-but-different-city.
# 0.4 means "40% as good as being in the exact same city."
# WHY 0.4? It's a balance: same state means similar culture, sports teams,
# maybe weekend trips — but you can't casually grab coffee like you can
# with someone in your own city. Tune this based on user feedback.
SAME_STATE_SCORE = 0.4


# ---------------------------------------------------------------------------
# Component scoring functions
# ---------------------------------------------------------------------------
# Each function below computes one signal on a 0.0-1.0 scale.
# They're intentionally simple and stateless — no side effects, no database
# calls, just math. This makes them easy to test and reason about.

def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    """Return overlap ratio between two sets using Jaccard similarity.

    Formula: |intersection(A, B)| / |union(A, B)|

    WHY Jaccard: It naturally handles sets of different sizes. If you have
    3 hobbies and I have 8, we're not penalized for the size difference —
    we're measured on what fraction of our combined hobbies we share.

    Range: 0.0 (nothing in common) to 1.0 (identical sets)
    Edge case: both empty → 0.0 (can't compute overlap on nothing)
    """
    if not left and not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _bounded_ratio(value: float, max_value: float) -> float:
    """Normalize value/max_value into [0.0, 1.0].

    Used for mutual friends: if you share 10 friends out of a max of 20,
    your score is 0.5. Anything above max_value caps at 1.0.
    """
    if max_value <= 0:
        return 0.0
    return min(max(value / max_value, 0.0), 1.0)


def _location_score(source: UserProfile, candidate: UserProfile) -> float:
    """Tiered location scoring: city match > state match > no match.

    Returns:
        1.0 — same city AND same state (best: you can hang out easily)
        0.4 — different city, same state (decent: road trip distance)
        0.0 — different state or missing location (no proximity signal)

    WHY tiered instead of binary:
    The old version gave 0.0 to someone in Houston when you're in Dallas.
    But those two people are both in Texas — they share culture, sports teams,
    maybe regional events. That's worth *something*, just not as much as
    being in the same city. The tier captures this nuance.

    Comparisons are case-insensitive and whitespace-trimmed so that
    "Austin" matches "austin" and " Austin " matches "Austin".
    """
    # If either profile has no location, we can't score this signal.
    if source.location is None or candidate.location is None:
        return 0.0

    source_city = source.location.city.strip().lower()
    source_state = source.location.state.strip().lower()
    cand_city = candidate.location.city.strip().lower()
    cand_state = candidate.location.state.strip().lower()

    # Same city AND same state = full match.
    # We check both because city names aren't unique across states
    # (e.g., "Portland, Oregon" vs "Portland, Maine").
    if source_city == cand_city and source_state == cand_state:
        return 1.0

    # Different city but same state = partial credit.
    if source_state == cand_state:
        return SAME_STATE_SCORE

    # Different state = no location signal.
    return 0.0


def _age_compatibility_score(source: UserProfile, candidate: UserProfile) -> float:
    """Return step-down age compatibility: 10% drop per year of age gap.

    Score ladder:
        0 years apart → 1.0  (same age, maximum compatibility)
        1 year apart  → 0.9
        2 years apart → 0.8
        ...
        9 years apart → 0.1
        10+ years     → 0.0  (too far apart for this signal to help)

    WHY a step-down ladder instead of a continuous curve:
    It's simple, transparent, and easy to explain to users. "You lost 10%
    because you're 1 year apart" is clearer than a Gaussian decay formula.

    WHY 10-year cutoff:
    A 25-year-old and a 35-year-old are in very different life stages.
    The algorithm doesn't *prevent* this match — it just means age
    contributes 0.0 to the score. Other signals (shared hobbies, mutual
    friends) can still make it a good match.

    Returns 0.0 if either age is unknown (we can't compute a gap).
    """
    if source.age is None or candidate.age is None:
        return 0.0

    gap = abs(source.age - candidate.age)
    compatibility = 1.0 - (0.1 * gap)
    return max(compatibility, 0.0)


def _exact_match_score(
    source_value: str | None,
    candidate_value: str | None,
) -> float:
    """Case-insensitive exact match: 1.0 if same, 0.0 otherwise.

    Used for college and faith where partial matching doesn't make sense —
    you either went to the same school or you didn't.
    """
    if source_value and candidate_value:
        if source_value.strip().lower() == candidate_value.strip().lower():
            return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def compute_match_score(
    source: UserProfile,
    candidate: UserProfile,
    weights: ScoringWeights | None = None,
) -> MatchBreakdown:
    """Compute a full, explainable score breakdown for a source/candidate pair.

    This is the heart of the matching engine. It:
    1. Computes each signal independently (so you can see what contributed)
    2. Multiplies each signal by its weight
    3. Sums them into a base score
    4. Adds a social boost if mutual friends exist
    5. Caps at 1.0 and returns everything in a MatchBreakdown

    WHY explicit and verbose:
    This function intentionally doesn't use loops or abstractions. Each signal
    is computed on its own named line. This makes it trivially easy to:
    - Debug ("why did this person score 0.6?")
    - Explain to users ("you matched because of shared hobbies and location")
    - Log for future ML training
    """
    active_weights = weights or ScoringWeights()

    # ── Step 1: Compute each component signal (all 0.0 to 1.0) ──

    # Jaccard overlap on hobbies (things you DO).
    hobby_overlap = _jaccard_similarity(source.hobbies, candidate.hobbies)

    # Jaccard overlap on interest categories (broad tastes).
    interest_overlap = _jaccard_similarity(source.interests, candidate.interests)

    # Jaccard overlap on specific fandoms (artists, shows, teams).
    fan_of_overlap = _jaccard_similarity(source.fan_of, candidate.fan_of)

    # Mutual friends: how many friend IDs appear in both profiles.
    # We split this into a count (for the boolean check) and a ratio (for scoring).
    mutual_friend_count = len(source.mutual_friend_ids & candidate.mutual_friend_ids)
    mutual_friends_score = _bounded_ratio(mutual_friend_count, MAX_MUTUAL_FRIENDS)
    has_mutual_friends = mutual_friend_count > 0

    # Tiered location: full credit for same city, partial for same state.
    location_match = _location_score(source, candidate)

    # Age ladder: 10% penalty per year of gap.
    age_compatibility = _age_compatibility_score(source, candidate)

    # Exact matches: same college? same faith?
    college_match = _exact_match_score(source.college, candidate.college)
    faith_match = _exact_match_score(source.faith, candidate.faith)

    # Jaccard overlap on travel wishlists.
    travel_overlap = _jaccard_similarity(source.travel_wishlist, candidate.travel_wishlist)

    # ── Step 2: Weighted base score ──
    # Each signal contributes its weight × its score. The weights are set
    # in ScoringWeights and should roughly sum to 1.0 so the base score
    # stays in a 0.0-1.0 range.
    base_score = (
        active_weights.hobby_overlap * hobby_overlap
        + active_weights.interest_overlap * interest_overlap
        + active_weights.fan_of_overlap * fan_of_overlap
        + active_weights.mutual_friends * mutual_friends_score
        + active_weights.location_match * location_match
        + active_weights.age_compatibility * age_compatibility
        + active_weights.college_match * college_match
        + active_weights.faith_match * faith_match
        + active_weights.travel_overlap * travel_overlap
    )

    # ── Step 3: Social boost ──
    # This is a separate "lane" that fires only when mutual friends exist.
    # WHY separate: Having mutual friends is such a strong real-world signal
    # that we want it to significantly boost the score beyond what the base
    # score alone would give. It's additive, not multiplicative, so even a
    # low-compatibility profile with mutual friends gets a meaningful push.
    social_boost = active_weights.friend_common_boost if has_mutual_friends else 0.0

    # Cap at 1.0 so scores stay easy to interpret (0 = no match, 1 = perfect).
    total_score = min(base_score + social_boost, 1.0)

    # ── Step 4: Return rounded, explainable breakdown ──
    # We round to 6 decimal places to avoid floating-point noise
    # (e.g., 0.33333333333333337 becomes 0.333333).
    return MatchBreakdown(
        total_score=round(total_score, 6),
        has_mutual_friends=has_mutual_friends,
        social_boost=round(social_boost, 6),
        hobby_overlap=round(hobby_overlap, 6),
        interest_overlap=round(interest_overlap, 6),
        fan_of_overlap=round(fan_of_overlap, 6),
        mutual_friends=round(mutual_friends_score, 6),
        location_match=round(location_match, 6),
        age_compatibility=round(age_compatibility, 6),
        college_match=round(college_match, 6),
        faith_match=round(faith_match, 6),
        travel_overlap=round(travel_overlap, 6),
    )


# ---------------------------------------------------------------------------
# Ranking function
# ---------------------------------------------------------------------------

def rank_candidates(
    source: UserProfile,
    candidates: list[UserProfile],
    weights: ScoringWeights | None = None,
) -> list[tuple[UserProfile, MatchBreakdown]]:
    """Rank candidates for one source profile using a two-lane strategy.

    Sort priority (descending):
    1) has_mutual_friends (True comes before False)
    2) total_score (higher is better)

    WHY two lanes:
    In real social networks, someone you share mutual friends with is almost
    always a better recommendation than a stranger — even if the stranger has
    slightly higher taste compatibility. This sort ensures socially-connected
    candidates always appear first, then non-connected candidates by score.
    """
    scored = [
        (candidate, compute_match_score(source, candidate, weights))
        for candidate in candidates
    ]
    return sorted(
        scored,
        key=lambda item: (item[1].has_mutual_friends, item[1].total_score),
        reverse=True,
    )
