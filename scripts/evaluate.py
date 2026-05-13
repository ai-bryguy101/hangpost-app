"""Offline evaluation script.

Loads the seed CSV, generates synthetic relevance labels, and prints
precision@k / recall@k / NDCG@k / MAP@k for up to four rankers:

    1. random              — sanity baseline
    2. rules_only          — Phase 1 weighted scoring
    3. rules+embeddings    — Phase 2 (--with-embeddings, requires [ml])
    4. learned             — Phase 3 (--learned-model PATH, requires [ml])

Run:
    python scripts/evaluate.py
    python scripts/evaluate.py --queries 100 --k 5
    python scripts/evaluate.py --with-embeddings
    python scripts/evaluate.py --with-embeddings --learned-model models/learned_ranker.joblib
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from hangpost_matching import (  # noqa: E402
    RELEVANCE_GENERATORS,
    AblationRow,
    EvaluationResult,
    Query,
    Ranker,
    Vector,
    ablate_weights,
    build_queries,
    evaluate_ranker,
    get_relevance_fn,
    load_profiles_from_csv,
    make_random_ranker,
    make_rules_ranker,
)


def _print_row(name: str, result: EvaluationResult) -> None:
    print(
        f"{name:<22} "
        f"{result.precision:>8.3f} "
        f"{result.recall:>8.3f} "
        f"{result.ndcg:>9.3f} "
        f"{result.map:>9.3f}"
    )


def _evaluate(name: str, ranker: Ranker, queries: list[Query], k: int) -> None:
    result = evaluate_ranker(ranker, queries, k=k)
    _print_row(name, result)


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
    parser.add_argument(
        "--learned-model",
        type=Path,
        default=None,
        help="Path to a saved LearnedRanker (Phase 3) to evaluate",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help=(
            "After the main comparison, print a per-feature ablation table for "
            "the rules ranker: each weight is zeroed in turn and the metric "
            "drop vs. the full-weights baseline is reported."
        ),
    )
    parser.add_argument(
        "--relevance",
        choices=sorted(RELEVANCE_GENERATORS),
        default="rule_based",
        help=(
            "Which relevance label generator to use. 'rule_based' is the "
            "original thresholded synthesizer; 'noisy' flips a fraction of "
            "those labels; 'simulated' samples outcomes from a logistic "
            "model that mixes observable affinity with a hidden personality "
            "vector the ranker can't see (more realistic ceiling)."
        ),
    )
    args = parser.parse_args()

    profiles = load_profiles_from_csv(Path(args.csv))
    print(f"Loaded {len(profiles)} profiles from {args.csv}")

    relevance_fn = get_relevance_fn(args.relevance, args.seed)
    print(f"Relevance generator: {args.relevance}")
    queries = build_queries(profiles, args.queries, args.seed, relevance_fn=relevance_fn)
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
        f"{'System':<22} {'P@' + str(args.k):>8} {'R@' + str(args.k):>8} "
        f"{'NDCG@' + str(args.k):>9} {'MAP@' + str(args.k):>9}"
    )
    print("-" * 62)

    _evaluate("random", make_random_ranker(args.seed), queries_with_relevant, args.k)
    _evaluate("rules_only", make_rules_ranker(), queries_with_relevant, args.k)

    embeddings = None
    if args.with_embeddings:
        from hangpost_matching import (
            SentenceTransformerEmbedder,
            embed_profiles,
        )

        print("\nLoading sentence-transformer model and embedding all profiles...")
        embedder = SentenceTransformerEmbedder()
        embeddings = embed_profiles(profiles, embedder)
        print(f"Embedded {len(embeddings)} profiles.\n")

        _evaluate(
            "rules+embeddings",
            make_rules_ranker(profile_embeddings=embeddings),
            queries_with_relevant,
            args.k,
        )

    if args.learned_model is not None:
        from hangpost_matching import LearnedRanker

        print(f"\nLoading learned ranker from {args.learned_model}...")
        learned = LearnedRanker.load(args.learned_model)
        _evaluate(
            "learned (Phase 3)",
            learned.as_ranker(),
            queries_with_relevant,
            args.k,
        )

    if args.ablation:
        _print_ablation(queries_with_relevant, embeddings, args.k)


def _print_ablation(
    queries: list[Query],
    profile_embeddings: Mapping[str, Vector] | None,
    k: int,
) -> None:
    """Run per-feature ablation on the rules ranker and print a delta table.

    The baseline row uses default `ScoringWeights`. Each subsequent row sets
    one weight to 0.0 — a positive delta means the metric got worse without
    that signal, i.e., that feature was actually contributing.
    """
    print()
    print(f"Per-feature ablation (rules ranker, k={k})")
    print(
        "Positive deltas mean the ranker got worse when this feature was "
        "zeroed (so the feature was helping)."
    )
    print()
    header = (
        f"{'Feature ablated':<22} {'P@' + str(k):>8} {'R@' + str(k):>8} "
        f"{'NDCG@' + str(k):>9} {'MAP@' + str(k):>9} "
        f"{'ΔP':>8} {'ΔR':>8} {'ΔNDCG':>9} {'ΔMAP':>9}"
    )
    print(header)
    print("-" * len(header))

    rows = ablate_weights(queries, profile_embeddings=profile_embeddings, k=k)
    for row in rows:
        _print_ablation_row(row)


def _print_ablation_row(row: AblationRow) -> None:
    r = row.result
    print(
        f"{row.feature:<22} "
        f"{r.precision:>8.3f} {r.recall:>8.3f} {r.ndcg:>9.3f} {r.map:>9.3f} "
        f"{row.delta_precision:>+8.3f} {row.delta_recall:>+8.3f} "
        f"{row.delta_ndcg:>+9.3f} {row.delta_map:>+9.3f}"
    )


if __name__ == "__main__":
    main()
