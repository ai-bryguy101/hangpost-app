"""Hangpost matching engine package."""

from .embeddings import (
    Embedder,
    SentenceTransformerEmbedder,
    Vector,
    cosine_similarity,
    embed_profiles,
    profile_to_text,
)
from .models import MatchBreakdown, ScoringWeights, UserProfile
from .scoring import compute_match_score, rank_candidates

__all__ = [
    "Embedder",
    "MatchBreakdown",
    "ScoringWeights",
    "SentenceTransformerEmbedder",
    "UserProfile",
    "Vector",
    "compute_match_score",
    "cosine_similarity",
    "embed_profiles",
    "profile_to_text",
    "rank_candidates",
]
