"""Offline evaluation harness for the matching engine.

Why this module exists
----------------------
Without measurement, every change to the ranker is a guess. This module
provides the standard information-retrieval metrics that let you compare
ranking systems quantitatively:

- precision@k:  fraction of the top-k that are relevant
- recall@k:     fraction of all relevant items that landed in the top-k
- MAP@k:        Mean Average Precision — rewards relevance ranked high
- NDCG@k:       Normalized Discounted Cumulative Gain — log-discounted

All metrics use binary relevance (relevant / not). Graded relevance
(0/1/2/3 levels) can be added later if/when human-rated labels exist.

`synthesize_relevance` provides a deterministic ground-truth label for
the seed dataset by thresholding multiple signals at once. Because this
is structurally different from the ranker's *continuous weighted* score,
the metrics still measure something meaningful — but the labels are a
stand-in until real outcome data (accepts, chats started, retention) is
available.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

from .embeddings import Vector
from .models import UserProfile
from .scoring import rank_candidates

# A ranker is anything that takes (source, candidates) and returns the
# candidate user_ids in ranked order (best first). Thin contract on
# purpose — lets you plug in pure-rules, rules+embeddings, random, or a
# future learned model without changing the harness.
Ranker = Callable[[UserProfile, list[UserProfile]], list[str]]


def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of the top-k retrieved items that are relevant.

    Returns 0.0 for k <= 0 or empty retrieval. Uses min(k, len(retrieved))
    in the denominator so a short retrieval list is not penalized as if
    it returned k items.
    """
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(top_k)


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of all relevant items that appear in the top-k."""
    if not relevant or k <= 0:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def average_precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Average precision over the top-k.

    Sums precision-at-i for every rank i where item i is relevant, then
    divides by `min(len(relevant), k)`. Higher when relevant items are
    pushed toward the top of the ranking.
    """
    if not relevant or k <= 0:
        return 0.0
    top_k = retrieved[:k]
    sum_precision = 0.0
    hits = 0
    for i, item in enumerate(top_k, start=1):
        if item in relevant:
            hits += 1
            sum_precision += hits / i
    if hits == 0:
        return 0.0
    return sum_precision / min(len(relevant), k)


def ndcg_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at k (binary relevance).

    DCG = sum( gain[i] / log2(i + 2) ) for i in 0..k-1
    IDCG = same but with all relevant items packed at the top.
    NDCG = DCG / IDCG (so a perfect ranking scores 1.0).
    """
    if not relevant or k <= 0:
        return 0.0
    top_k = retrieved[:k]
    dcg = sum((1.0 / math.log2(i + 2)) if item in relevant else 0.0 for i, item in enumerate(top_k))
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


@dataclass(frozen=True)
class EvaluationResult:
    """Average metrics across a query set."""

    n_queries: int
    k: int
    precision: float
    recall: float
    map: float
    ndcg: float


# A query is (source, candidates_to_rank, ground_truth_relevant_ids).
Query = tuple[UserProfile, list[UserProfile], set[str]]


def evaluate_ranker(
    ranker: Ranker,
    queries: Iterable[Query],
    k: int = 10,
) -> EvaluationResult:
    """Run `ranker` against each query and return the macro-averaged metrics.

    "Macro-averaged" here means we compute each metric per query and then
    average across queries (every query contributes equally regardless of
    how many relevant items it has). That is the standard IR convention.
    """
    p_total = 0.0
    r_total = 0.0
    map_total = 0.0
    ndcg_total = 0.0
    n = 0
    for source, candidates, relevant in queries:
        retrieved = ranker(source, candidates)
        p_total += precision_at_k(retrieved, relevant, k)
        r_total += recall_at_k(retrieved, relevant, k)
        map_total += average_precision_at_k(retrieved, relevant, k)
        ndcg_total += ndcg_at_k(retrieved, relevant, k)
        n += 1
    if n == 0:
        return EvaluationResult(0, k, 0.0, 0.0, 0.0, 0.0)
    return EvaluationResult(
        n_queries=n,
        k=k,
        precision=p_total / n,
        recall=r_total / n,
        map=map_total / n,
        ndcg=ndcg_total / n,
    )


def synthesize_relevance(source: UserProfile, candidate: UserProfile) -> bool:
    """Heuristic 'would they realistically match?' label.

    The label is True when at least 3 of the following are true:
      1. ≥2 shared interests
      2. ≥2 shared liked_topics
      3. shared hometown
      4. age gap ≤ 5 years
      5. ≥1 mutual friend

    This is a *thresholded multi-signal* rule, structurally different from
    the ranker's continuous weighted score, so the resulting labels still
    let us compare ranking quality fairly. Replace with real outcome data
    (accepts, chats started, retention) as soon as that exists.
    """
    age_close = (
        source.age is not None
        and candidate.age is not None
        and abs(source.age - candidate.age) <= 5
    )
    signals = [
        len(source.interests & candidate.interests) >= 2,
        len(source.liked_topics & candidate.liked_topics) >= 2,
        bool(source.location and source.location == candidate.location),
        age_close,
        bool(source.mutual_friend_ids & candidate.mutual_friend_ids),
    ]
    return sum(signals) >= 3


def build_queries(
    profiles: list[UserProfile],
    n_sources: int,
    seed: int,
    relevance_fn: Callable[[UserProfile, UserProfile], bool] = synthesize_relevance,
) -> list[Query]:
    """Sample `n_sources` random sources and label all other profiles for each.

    Returns a list of (source, candidates, relevant_user_ids) tuples ready
    to feed into `evaluate_ranker` or a learning-to-rank training loop.
    """
    rng = random.Random(seed)
    sources = rng.sample(profiles, min(n_sources, len(profiles)))
    queries: list[Query] = []
    for source in sources:
        candidates = [p for p in profiles if p.user_id != source.user_id]
        relevant = {
            candidate.user_id for candidate in candidates if relevance_fn(source, candidate)
        }
        queries.append((source, candidates, relevant))
    return queries


def split_queries(
    queries: Sequence[Query], train_fraction: float, seed: int
) -> tuple[list[Query], list[Query]]:
    """Shuffle and split queries into train/test by query (no leakage).

    A train/test split must happen at the *query* level — never at the
    candidate level — otherwise the model would see the same source in
    both halves and metrics would be optimistically biased.
    """
    rng = random.Random(seed)
    shuffled = list(queries)
    rng.shuffle(shuffled)
    cut = int(len(shuffled) * train_fraction)
    return shuffled[:cut], shuffled[cut:]


def make_rules_ranker(
    profile_embeddings: Mapping[str, Vector] | None = None,
) -> Ranker:
    """Wrap `rank_candidates` to return user_ids only (Ranker contract)."""

    def ranker(source: UserProfile, candidates: list[UserProfile]) -> list[str]:
        ranked = rank_candidates(source, candidates, profile_embeddings=profile_embeddings)
        return [profile.user_id for profile, _ in ranked]

    return ranker


def make_random_ranker(seed: int = 0) -> Ranker:
    """Random shuffle, deterministic per source — used as a sanity baseline.

    A real ranker should always beat this. If it doesn't, the labels or
    the ranker have a bug.
    """

    def ranker(source: UserProfile, candidates: list[UserProfile]) -> list[str]:
        rng = random.Random(f"{seed}:{source.user_id}")
        shuffled = list(candidates)
        rng.shuffle(shuffled)
        return [profile.user_id for profile in shuffled]

    return ranker
