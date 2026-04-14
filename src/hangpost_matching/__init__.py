"""Hangpost matching engine package.

Public API:

Data models:
- UserProfile, Location: profile data structures
- ScoringWeights: tunable scoring configuration
- MatchBreakdown: explainable scoring output

Scoring:
- compute_match_score: score a single source/candidate pair
- rank_candidates: rank a list of candidates for a source
- MAX_MUTUAL_FRIENDS, SAME_STATE_SCORE: scoring constants

CSV loading (original, still works for scripts and testing):
- load_profiles, profile_from_row, tokenize

Database layer (new — SQLite-backed storage with geo queries):
- get_connection, init_schema: DB setup
- insert_profile, insert_friendship: writing data
- find_nearby_profiles, get_friend_ids, row_to_profile: reading data
- find_and_rank_candidates: high-level "open app → see matches" flow
- haversine_miles, bounding_box: geolocation math
"""

from .db import (
    bounding_box,
    find_and_rank_candidates,
    find_nearby_profiles,
    get_connection,
    get_friend_ids,
    haversine_miles,
    init_schema,
    insert_friendship,
    insert_profile,
    row_to_profile,
)
from .loader import load_profiles, profile_from_row, tokenize
from .models import Location, MatchBreakdown, ScoringWeights, UserProfile
from .scoring import MAX_MUTUAL_FRIENDS, SAME_STATE_SCORE, compute_match_score, rank_candidates

__all__ = [
    # Models
    "UserProfile",
    "Location",
    "ScoringWeights",
    "MatchBreakdown",
    # Scoring
    "MAX_MUTUAL_FRIENDS",
    "SAME_STATE_SCORE",
    "compute_match_score",
    "rank_candidates",
    # CSV loading
    "load_profiles",
    "profile_from_row",
    "tokenize",
    # Database
    "get_connection",
    "init_schema",
    "insert_profile",
    "insert_friendship",
    "find_nearby_profiles",
    "get_friend_ids",
    "row_to_profile",
    "find_and_rank_candidates",
    "haversine_miles",
    "bounding_box",
]
