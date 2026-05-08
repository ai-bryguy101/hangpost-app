"""Offline evaluation script.

Loads the seed CSV, generates synthetic relevance labels, and prints
precision@k / recall@k / NDCG@k / MAP@k for three rankers:

    1. random         — sanity baseline
    2. rules_only     — Phase 1 weighted scoring
    3. rules+embed    — Phase 2 (only when --with-embeddings is passed,
                         requires the [ml] extra installed)

Run:
    python scripts/evaluate.py
    python scripts/evaluate.py --queries 100 --k 5
    python scripts/evaluate.py --with-embeddings   # requires [ml] extra
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from collections.abc import Mapping  # noqa: E402

from hangpost_matching import (  # noqa: E402
    Query,
    Ranker,
    UserProfile,
    Vector,
    evaluate_ranker,
    load_profiles_from_csv,
    rank_candidates,
    synthesize_relevance,
)


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


def build_queries(profiles: list[UserProfile], n_sources: int, seed: int) -> list[Query]:
    """Pick `n_sources` random sources and label all other profiles."""
    rng = random.Random(seed)
    sources = rng.sample(profiles, min(n_sources, len(profiles)))
    queries: list[Query] = []
    for source in sources:
        candidates = [p for p in profiles if p.user_id != source.user_id]
        relevant = {
            candidate.user_id for candidate in candidates if synthesize_relevance(source, candidate)
        }
        queries.append((source, candidates, relevant))
    return queries


def _print_row(name: str, result: object, k: int) -> None:
    # `result` is an EvaluationResult; typed as object here to keep the
    # signature short. The dataclass fields are well-known.
    print(
        f"{name:<20} "
        f"{getattr(result, 'precision'):>8.3f} "  # noqa: B009
        f"{getattr(result, 'recall'):>8.3f} "  # noqa: B009
        f"{getattr(result, 'ndcg'):>9.3f} "  # noqa: B009
        f"{getattr(result, 'map'):>9.3f}"  # noqa: B009
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="data/test_profiles.csv")
    parser.add_argument(
        "--queries",
        type=int,
        default=50,
        help="How many random source profiles to evaluate against",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--with-embeddings",
        action="store_true",
        help="Also evaluate rules+embeddings (requires [ml] extra installed)",
    )
    args = parser.parse_args()

    profiles = load_profiles_from_csv(Path(args.csv))
    print(f"Loaded {len(profiles)} profiles from {args.csv}")

    queries = build_queries(profiles, args.queries, args.seed)
    queries_with_relevant = [q for q in queries if q[2]]
    avg_relevant = (
        sum(len(q[2]) for q in queries_with_relevant) / len(queries_with_relevant)
        if queries_with_relevant
        else 0.0
    )
    print(
        f"{len(queries_with_relevant)}/{len(queries)} queries have ≥1 relevant "
        f"candidate (avg {avg_relevant:.1f} relevant per query)"
    )

    if not queries_with_relevant:
        print("No queries with relevant candidates — relax synthesize_relevance.")
        return

    print()
    print(
        f"{'System':<20} {'P@' + str(args.k):>8} {'R@' + str(args.k):>8} "
        f"{'NDCG@' + str(args.k):>9} {'MAP@' + str(args.k):>9}"
    )
    print("-" * 60)

    random_result = evaluate_ranker(make_random_ranker(args.seed), queries_with_relevant, k=args.k)
    _print_row("random", random_result, args.k)

    rules_result = evaluate_ranker(make_rules_ranker(), queries_with_relevant, k=args.k)
    _print_row("rules_only", rules_result, args.k)

    if args.with_embeddings:
        from hangpost_matching import (
            SentenceTransformerEmbedder,
            embed_profiles,
        )

        print("\nLoading sentence-transformer model and embedding all profiles...")
        embedder = SentenceTransformerEmbedder()
        embeddings = embed_profiles(profiles, embedder)
        print(f"Embedded {len(embeddings)} profiles.\n")

        embed_result = evaluate_ranker(
            make_rules_ranker(profile_embeddings=embeddings),
            queries_with_relevant,
            k=args.k,
        )
        _print_row("rules+embeddings", embed_result, args.k)


if __name__ == "__main__":
    main()
