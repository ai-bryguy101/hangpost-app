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

import hashlib
import math
import random
import struct
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from dataclasses import replace as dataclass_replace

from .embeddings import Vector
from .models import ScoringWeights, UserProfile
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
      4. shared college
      5. age gap ≤ 5 years
      6. ≥1 mutual friend

    Hometown and college are listed as independent signals because they
    are independent friendship cues — you can match on one without the
    other.

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
        bool(source.hometown and source.hometown == candidate.hometown),
        bool(source.college and source.college == candidate.college),
        age_close,
        bool(source.mutual_friend_ids & candidate.mutual_friend_ids),
    ]
    return sum(signals) >= 3


# ---------- realistic label generators ----------
#
# The thresholded `synthesize_relevance` above is structurally different
# from the weighted-sum ranker, but it still uses exactly the same five
# signals the ranker scores on. That makes the rules baseline look
# artificially strong (P@10 near 1.0) and gives the learned ranker
# little headroom.
#
# The two generators below close that gap so the evaluation reflects a
# more realistic ceiling:
#
#   make_noisy_relevance_fn       — flip a fraction of `synthesize_relevance`
#                                    labels at random. Useful for ablation
#                                    studies that ask "how much does my
#                                    model degrade as label noise climbs?"
#   make_simulated_outcome_fn     — model outcomes as a *logistic mixture*
#                                    of (a) continuous observable affinity
#                                    and (b) cosine similarity between
#                                    hidden per-user "personality" vectors
#                                    the ranker can't see. Adds Bernoulli
#                                    noise on top. This is the closest
#                                    stand-in for real interaction data
#                                    until production labels exist.


def _stable_personality_vector(user_id: str, dims: int = 8) -> list[float]:
    """Deterministically map a user_id to a `dims`-length float vector.

    The vector is the same every call, so labels are reproducible across
    train and evaluation. The mapping is unrelated to any observable
    profile field — that is the point: it represents a "hidden trait"
    the ranker has no access to, the way real user behaviour depends on
    factors no feature captures (mood, time-of-day, prior history).
    """
    digest = hashlib.sha256(user_id.encode("utf-8")).digest()
    # Each 4-byte chunk → one float in [-1.0, 1.0).
    needed = dims * 4
    if len(digest) < needed:
        digest = (digest * ((needed // len(digest)) + 1))[:needed]
    values: list[float] = []
    for i in range(dims):
        (raw,) = struct.unpack_from("<i", digest, i * 4)
        # Map int32 → [-1.0, 1.0). 2**31 is the magnitude of int32.
        values.append(raw / 2**31)
    return values


def _continuous_observable_affinity(source: UserProfile, candidate: UserProfile) -> float:
    """Smooth version of the synthesize_relevance signals, in [0.0, 1.0].

    Continuous so the resulting outcome probability isn't a step function
    of the same thresholds the rules ranker uses. This is the part of
    affinity the ranker *can* see — features it has access to.
    """
    interest_overlap = len(source.interests & candidate.interests)
    liked_overlap = len(source.liked_topics & candidate.liked_topics)
    same_hometown = 1.0 if source.hometown and source.hometown == candidate.hometown else 0.0
    same_college = 1.0 if source.college and source.college == candidate.college else 0.0
    if source.age is not None and candidate.age is not None:
        age_closeness = max(0.0, 1.0 - abs(source.age - candidate.age) / 15.0)
    else:
        age_closeness = 0.0
    mutual_friends = len(source.mutual_friend_ids & candidate.mutual_friend_ids)

    # Weighted sum with diminishing returns on count-style signals.
    # Hometown and college are peer-strength friendship cues, weighted the
    # same — they're independent so a candidate can light up one without
    # the other.
    raw = (
        0.20 * min(interest_overlap / 4.0, 1.0)
        + 0.15 * min(liked_overlap / 4.0, 1.0)
        + 0.20 * same_hometown
        + 0.20 * same_college
        + 0.15 * age_closeness
        + 0.10 * min(mutual_friends / 3.0, 1.0)
    )
    return min(max(raw, 0.0), 1.0)


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def make_noisy_relevance_fn(
    noise_level: float = 0.15,
    seed: int = 0,
    base_fn: Callable[[UserProfile, UserProfile], bool] = synthesize_relevance,
) -> Callable[[UserProfile, UserProfile], bool]:
    """Wrap `base_fn` with deterministic per-pair Bernoulli label flips.

    `noise_level` ∈ [0.0, 1.0] is the probability a label flips. The flip
    decision is hashed from `(source_id, candidate_id, seed)` so calling
    the returned function twice gives the same answer (essential for
    reproducible train/test splits).

    Use this for ablation studies: train at one noise level, evaluate at
    a different one to see how robust your ranker is.
    """
    if not 0.0 <= noise_level <= 1.0:
        raise ValueError(f"noise_level must be in [0, 1], got {noise_level}")

    def fn(source: UserProfile, candidate: UserProfile) -> bool:
        truth = base_fn(source, candidate)
        if noise_level == 0.0:
            return truth
        rng = random.Random(f"noise:{seed}:{source.user_id}:{candidate.user_id}")
        if rng.random() < noise_level:
            return not truth
        return truth

    return fn


def make_simulated_outcome_fn(
    seed: int = 0,
    hidden_weight: float = 0.5,
    observable_weight: float = 1.5,
    bias: float = -1.0,
    noise_level: float = 0.10,
) -> Callable[[UserProfile, UserProfile], bool]:
    """Return a relevance fn driven by a logistic mixture + hidden confounders.

    Outcome model:

        observable = _continuous_observable_affinity(source, candidate)  ∈ [0, 1]
        hidden     = cosine(personality(source), personality(candidate)) ∈ [-1, 1]
        logit      = bias + observable_weight * observable
                          + hidden_weight     * hidden
        prob       = sigmoid(logit)
        label      = (deterministic uniform in [0, 1)) < prob
        + Bernoulli flip with probability `noise_level`

    Key property: `hidden` depends on `user_id` only, so the ranker — which
    sees the same observable features either way — cannot perfectly recover
    the label. This gives the learned model real headroom over the rules
    baseline (good), and prevents either from approaching P@10 = 1.0
    against this generator (also good — realistic).
    """
    if not 0.0 <= noise_level <= 1.0:
        raise ValueError(f"noise_level must be in [0, 1], got {noise_level}")

    def fn(source: UserProfile, candidate: UserProfile) -> bool:
        observable = _continuous_observable_affinity(source, candidate)
        src_vec = _stable_personality_vector(source.user_id)
        cand_vec = _stable_personality_vector(candidate.user_id)
        # Cosine similarity, hand-rolled to avoid the embeddings import cycle.
        dot = sum(a * b for a, b in zip(src_vec, cand_vec, strict=True))
        norm_s = math.sqrt(sum(a * a for a in src_vec))
        norm_c = math.sqrt(sum(a * a for a in cand_vec))
        hidden = dot / (norm_s * norm_c) if norm_s > 0 and norm_c > 0 else 0.0

        logit = bias + observable_weight * observable + hidden_weight * hidden
        prob = _sigmoid(logit)

        draw_rng = random.Random(f"draw:{seed}:{source.user_id}:{candidate.user_id}")
        label = draw_rng.random() < prob

        if noise_level > 0.0:
            flip_rng = random.Random(f"flip:{seed}:{source.user_id}:{candidate.user_id}")
            if flip_rng.random() < noise_level:
                label = not label

        return label

    return fn


# Friendly registry so CLI scripts can pick a labeller by name.
RELEVANCE_GENERATORS: dict[str, Callable[[int], Callable[[UserProfile, UserProfile], bool]]] = {
    "rule_based": lambda _seed: synthesize_relevance,
    "noisy": lambda seed: make_noisy_relevance_fn(seed=seed),
    "simulated": lambda seed: make_simulated_outcome_fn(seed=seed),
}


def get_relevance_fn(name: str, seed: int = 0) -> Callable[[UserProfile, UserProfile], bool]:
    """Look up a relevance generator by short name."""
    try:
        factory = RELEVANCE_GENERATORS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(RELEVANCE_GENERATORS))
        raise ValueError(f"Unknown relevance fn {name!r}. Choices: {choices}") from exc
    return factory(seed)


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


# ---------- per-feature ablation ----------
#
# The ablation harness answers the question "which signal is actually
# carrying the ranker?" by zeroing each weight in turn and re-evaluating.
# The metric drop vs. the full-weights baseline is the contribution of
# that signal. Useful both as a sanity check on the weights and as a
# resume-worthy artifact ("we measured every feature's marginal value").


# Weight field names that are safe to zero individually. `friend_common_boost`
# is excluded because it lives in a separate boost lane, not the base score —
# its ablation story is "the two-lane sort still runs but the boost is 0,"
# which is a different question and worth a dedicated experiment.
ABLATABLE_WEIGHT_FIELDS: tuple[str, ...] = (
    "interest_overlap",
    "liked_topic_overlap",
    "mutual_friends",
    "hometown_match",
    "college_match",
    "age_compatibility",
    "semantic_similarity",
)


@dataclass(frozen=True)
class AblationRow:
    """One row of an ablation table: which feature was zeroed and what happened."""

    feature: str  # weight field that was set to 0.0 (or "<full>" for the baseline)
    result: EvaluationResult
    # Drop vs. the full-weights baseline (positive = the ranker got worse).
    delta_precision: float
    delta_recall: float
    delta_map: float
    delta_ndcg: float


def ablate_weights(
    queries: Iterable[Query],
    weights: ScoringWeights | None = None,
    profile_embeddings: Mapping[str, Vector] | None = None,
    k: int = 10,
    features: Sequence[str] | None = None,
) -> list[AblationRow]:
    """Evaluate the rules ranker with each weight zeroed in turn.

    Returns a list of `AblationRow` starting with the full-weights baseline
    (`feature="<full>"`) followed by one row per ablated weight. Each
    `delta_*` is `baseline - ablated`, so positive means "removing this
    feature made the ranker worse" (the usual case).

    Why this isn't a permutation test: zeroing a weight is faster than
    permuting feature values across queries and answers the simpler
    product question "what happens if we just turn this signal off?"
    The two analyses are complementary.
    """
    base_weights = weights or ScoringWeights()
    targets = tuple(features) if features is not None else ABLATABLE_WEIGHT_FIELDS

    # Validate names against the live dataclass so a typo fails loud.
    known = {f.name for f in dataclass_fields(base_weights)}
    for name in targets:
        if name not in known:
            raise ValueError(
                f"Unknown ScoringWeights field {name!r}. Known fields: {sorted(known)}"
            )

    queries = list(queries)

    baseline_ranker = _weighted_rules_ranker(base_weights, profile_embeddings)
    baseline = evaluate_ranker(baseline_ranker, queries, k=k)

    rows: list[AblationRow] = [
        AblationRow(
            feature="<full>",
            result=baseline,
            delta_precision=0.0,
            delta_recall=0.0,
            delta_map=0.0,
            delta_ndcg=0.0,
        )
    ]

    for name in targets:
        ablated = dataclass_replace(base_weights, **{name: 0.0})
        ablated_ranker = _weighted_rules_ranker(ablated, profile_embeddings)
        result = evaluate_ranker(ablated_ranker, queries, k=k)
        rows.append(
            AblationRow(
                feature=name,
                result=result,
                delta_precision=baseline.precision - result.precision,
                delta_recall=baseline.recall - result.recall,
                delta_map=baseline.map - result.map,
                delta_ndcg=baseline.ndcg - result.ndcg,
            )
        )

    return rows


def _weighted_rules_ranker(
    weights: ScoringWeights,
    profile_embeddings: Mapping[str, Vector] | None,
) -> Ranker:
    """Internal: rules ranker bound to a specific ScoringWeights."""

    def ranker(source: UserProfile, candidates: list[UserProfile]) -> list[str]:
        ranked = rank_candidates(
            source, candidates, weights=weights, profile_embeddings=profile_embeddings
        )
        return [profile.user_id for profile, _ in ranked]

    return ranker
