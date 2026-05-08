"""Train a Phase 3 LightGBM learning-to-rank model.

This script:
  1. Loads the seed CSV.
  2. Generates synthetic relevance labels via `synthesize_relevance`.
  3. Splits queries into train/test by *source* (no leakage).
  4. Optionally embeds every profile with sentence-transformers so the
     learned ranker has access to the semantic similarity feature.
  5. Fits a LightGBM `LGBMRanker` with the LambdaRank objective.
  6. Evaluates random / rules_only / [rules+embeddings] / learned on the
     held-out test queries and prints a comparison.
  7. Saves the trained model + embedding cache to disk.

Requires the [ml] extra:
    pip install -e ".[ml]"

Run:
    python scripts/train.py
    python scripts/train.py --with-embeddings --queries 200
    python scripts/train.py --with-embeddings --out models/learned.joblib
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from hangpost_matching import (  # noqa: E402
    EvaluationResult,
    LearnedRanker,
    Query,
    Ranker,
    build_queries,
    evaluate_ranker,
    load_profiles_from_csv,
    make_random_ranker,
    make_rules_ranker,
    split_queries,
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
    _print_row(name, evaluate_ranker(ranker, queries, k=k))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="data/test_profiles.csv")
    parser.add_argument(
        "--queries",
        type=int,
        default=200,
        help="Total source profiles to sample before train/test split",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.7,
        help="Fraction of queries used for training (rest is held out)",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--with-embeddings",
        action="store_true",
        help="Embed profiles with sentence-transformers before training",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("models/learned_ranker.joblib"),
        help="Where to save the trained LearnedRanker",
    )
    args = parser.parse_args()

    profiles = load_profiles_from_csv(Path(args.csv))
    print(f"Loaded {len(profiles)} profiles from {args.csv}")

    queries = [q for q in build_queries(profiles, args.queries, args.seed) if q[2]]
    train_queries, test_queries = split_queries(queries, args.train_fraction, args.seed)
    print(
        f"{len(train_queries)} train queries / {len(test_queries)} test queries "
        f"(both filtered to ≥1 relevant)"
    )

    embeddings = None
    if args.with_embeddings:
        from hangpost_matching import SentenceTransformerEmbedder, embed_profiles

        print("\nLoading sentence-transformer model and embedding all profiles...")
        embedder = SentenceTransformerEmbedder()
        embeddings = embed_profiles(profiles, embedder)
        print(f"Embedded {len(embeddings)} profiles.")

    print("\nTraining LightGBM LGBMRanker...")
    learned = LearnedRanker(profile_embeddings=embeddings)
    learned.fit(train_queries)
    print("Done.")

    print(f"\nHeld-out evaluation ({len(test_queries)} queries, k={args.k}):")
    print(
        f"\n{'System':<22} {'P@' + str(args.k):>8} {'R@' + str(args.k):>8} "
        f"{'NDCG@' + str(args.k):>9} {'MAP@' + str(args.k):>9}"
    )
    print("-" * 62)
    _evaluate("random", make_random_ranker(args.seed), test_queries, args.k)
    _evaluate("rules_only", make_rules_ranker(), test_queries, args.k)
    if embeddings is not None:
        _evaluate(
            "rules+embeddings",
            make_rules_ranker(profile_embeddings=embeddings),
            test_queries,
            args.k,
        )
    _evaluate("learned (Phase 3)", learned.as_ranker(), test_queries, args.k)

    learned.save(args.out)
    print(f"\nSaved learned ranker to {args.out}")


if __name__ == "__main__":
    main()
