"""Hangpost matching engine package."""

from .data import load_profiles_from_csv
from .embeddings import (
    Embedder,
    SentenceTransformerEmbedder,
    Vector,
    cosine_similarity,
    embed_profiles,
    profile_to_text,
)
from .evaluation import (
    EvaluationResult,
    Query,
    Ranker,
    average_precision_at_k,
    evaluate_ranker,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    synthesize_relevance,
)
from .models import MatchBreakdown, ScoringWeights, UserProfile
from .scoring import compute_match_score, rank_candidates

__all__ = [
    "Embedder",
    "EvaluationResult",
    "MatchBreakdown",
    "Query",
    "Ranker",
    "ScoringWeights",
    "SentenceTransformerEmbedder",
    "UserProfile",
    "Vector",
    "average_precision_at_k",
    "compute_match_score",
    "cosine_similarity",
    "embed_profiles",
    "evaluate_ranker",
    "load_profiles_from_csv",
    "ndcg_at_k",
    "precision_at_k",
    "profile_to_text",
    "rank_candidates",
    "recall_at_k",
    "synthesize_relevance",
]
