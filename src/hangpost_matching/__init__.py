"""Hangpost matching engine package.

Public API:
- UserProfile, Location: data models for profiles
- ScoringWeights: tunable scoring configuration
- MatchBreakdown: explainable scoring output
- compute_match_score: score a single source/candidate pair
- rank_candidates: rank a list of candidates for a source
- load_profiles, profile_from_row, tokenize: CSV loading utilities
- MAX_MUTUAL_FRIENDS, SAME_STATE_SCORE: scoring constants
"""

from .loader import load_profiles, profile_from_row, tokenize
from .models import Location, MatchBreakdown, ScoringWeights, UserProfile
from .scoring import MAX_MUTUAL_FRIENDS, SAME_STATE_SCORE, compute_match_score, rank_candidates

__all__ = [
    "UserProfile",
    "Location",
    "ScoringWeights",
    "MatchBreakdown",
    "MAX_MUTUAL_FRIENDS",
    "SAME_STATE_SCORE",
    "compute_match_score",
    "rank_candidates",
    "load_profiles",
    "profile_from_row",
    "tokenize",
]
