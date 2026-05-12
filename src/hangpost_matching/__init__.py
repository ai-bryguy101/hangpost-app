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
    RELEVANCE_GENERATORS,
    EvaluationResult,
    Query,
    Ranker,
    average_precision_at_k,
    build_queries,
    evaluate_ranker,
    get_relevance_fn,
    make_noisy_relevance_fn,
    make_random_ranker,
    make_rules_ranker,
    make_simulated_outcome_fn,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    split_queries,
    synthesize_relevance,
)
from .learning import FEATURE_NAMES, LearnedRanker, Predictor, extract_features
from .models import MatchBreakdown, ScoringWeights, UserProfile
from .scoring import compute_match_score, rank_candidates

__all__ = [
    "FEATURE_NAMES",
    "RELEVANCE_GENERATORS",
    "Embedder",
    "EvaluationResult",
    "LearnedRanker",
    "MatchBreakdown",
    "Predictor",
    "Query",
    "Ranker",
    "ScoringWeights",
    "SentenceTransformerEmbedder",
    "UserProfile",
    "Vector",
    "average_precision_at_k",
    "build_queries",
    "compute_match_score",
    "cosine_similarity",
    "embed_profiles",
    "evaluate_ranker",
    "extract_features",
    "get_relevance_fn",
    "load_profiles_from_csv",
    "make_noisy_relevance_fn",
    "make_random_ranker",
    "make_rules_ranker",
    "make_simulated_outcome_fn",
    "ndcg_at_k",
    "precision_at_k",
    "profile_to_text",
    "rank_candidates",
    "recall_at_k",
    "split_queries",
    "synthesize_relevance",
]
