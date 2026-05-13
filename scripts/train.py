"""Train a Phase 3 LightGBM learning-to-rank model.

This script:
  1. Loads the seed CSV.
  2. Builds relevance labels via the chosen `--relevance` generator.
  3. Splits queries into train/test by *source* (no leakage).
  4. Optionally embeds every profile with sentence-transformers so the
     learned ranker has access to the semantic similarity feature.
  5. Fits a LightGBM `LGBMRanker` with the LambdaRank objective.
  6. Evaluates random / rules_only / [rules+embeddings] / learned on the
     held-out test queries and prints a comparison.
  7. Saves the trained model + embedding cache to disk.
  8. (Optional) Logs every parameter, metric, and the saved model to
     MLflow for reproducible experiment tracking.

Requires the [ml] extra:
    pip install -e ".[ml]"

Run:
    python scripts/train.py
    python scripts/train.py --with-embeddings --queries 200
    python scripts/train.py --relevance simulated --mlflow
    python scripts/train.py --with-embeddings --out models/learned.joblib
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from hangpost_matching import (  # noqa: E402
    DEFAULT_RELEVANCE_THRESHOLD,
    RELEVANCE_GENERATORS,
    EvaluationResult,
    LearnedRanker,
    Query,
    Ranker,
    build_queries,
    evaluate_ranker,
    get_relevance_fn,
    load_profiles_from_csv,
    load_verdicts,
    make_random_ranker,
    make_rules_ranker,
    queries_from_verdicts,
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


def _evaluate_and_log(
    name: str,
    ranker: Ranker,
    queries: list[Query],
    k: int,
    tracker: _Tracker,
) -> EvaluationResult:
    result = evaluate_ranker(ranker, queries, k=k)
    _print_row(name, result)
    tracker.log_metrics(name, result)
    return result


class _Tracker:
    """Thin wrapper that logs to MLflow when enabled, no-ops otherwise.

    Keeping the conditional in one class means the rest of `main` stays
    flat and readable. The actual `mlflow` import is lazy so the [dev]
    extra never needs it.
    """

    def __init__(self, enabled: bool, experiment: str | None = None) -> None:
        self.enabled = enabled
        self._mlflow: Any = None
        if not enabled:
            return
        try:
            import mlflow
        except ImportError as exc:
            raise ImportError(
                'mlflow is required when --mlflow is set. Install with: pip install -e ".[ml]"'
            ) from exc
        self._mlflow = mlflow
        if experiment:
            mlflow.set_experiment(experiment)

    @contextlib.contextmanager
    def run(self, run_name: str | None = None) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        with self._mlflow.start_run(run_name=run_name):
            yield

    def log_params(self, params: dict[str, Any]) -> None:
        if not self.enabled:
            return
        self._mlflow.log_params(params)

    def log_metrics(self, ranker_name: str, result: EvaluationResult) -> None:
        if not self.enabled:
            return
        prefix = (
            ranker_name.replace(" ", "_").replace("+", "_plus_").replace("(", "").replace(")", "")
        )
        self._mlflow.log_metrics(
            {
                f"{prefix}/precision@{result.k}": result.precision,
                f"{prefix}/recall@{result.k}": result.recall,
                f"{prefix}/ndcg@{result.k}": result.ndcg,
                f"{prefix}/map@{result.k}": result.map,
            }
        )

    def log_artifact(self, path: Path) -> None:
        if not self.enabled:
            return
        self._mlflow.log_artifact(str(path))


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
        "--relevance",
        choices=sorted(RELEVANCE_GENERATORS),
        default="rule_based",
        help=(
            "Which relevance label generator to use during training and "
            "evaluation. 'simulated' produces a more realistic ceiling by "
            "introducing hidden confounders and noise. "
            "Ignored when --labels is set."
        ),
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help=(
            "Path to a JSONL file of LLM judge verdicts produced by "
            "scripts/label.py. When set, queries are built directly from "
            "the verdicts and --queries / --relevance are ignored — this is "
            "the teacher→student distillation path."
        ),
    )
    parser.add_argument(
        "--label-threshold",
        type=int,
        default=DEFAULT_RELEVANCE_THRESHOLD,
        help="Minimum judge rating (0-4) to count as relevant. Default 3.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("models/learned_ranker.joblib"),
        help="Where to save the trained LearnedRanker",
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Log parameters, metrics, and model artifact to MLflow.",
    )
    parser.add_argument(
        "--mlflow-experiment",
        default="hangpost-matching",
        help="MLflow experiment name (only used when --mlflow is set).",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="LightGBM n_estimators (boosting rounds).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="LightGBM learning rate.",
    )
    parser.add_argument(
        "--num-leaves",
        type=int,
        default=31,
        help="LightGBM num_leaves.",
    )
    args = parser.parse_args()

    tracker = _Tracker(enabled=args.mlflow, experiment=args.mlflow_experiment)

    label_source = (
        f"labels:{args.labels.name}" if args.labels is not None else args.relevance
    )
    run_name = f"{label_source}-{'emb' if args.with_embeddings else 'no_emb'}"

    with tracker.run(run_name=run_name):
        profiles = load_profiles_from_csv(Path(args.csv))
        print(f"Loaded {len(profiles)} profiles from {args.csv}")

        if args.labels is not None:
            verdicts = load_verdicts(args.labels)
            if not verdicts:
                raise SystemExit(f"No verdicts found at {args.labels}")
            judged = queries_from_verdicts(
                profiles, verdicts, threshold=args.label_threshold
            )
            queries = [q for q in judged if q[2]]
            print(
                f"Loaded {len(verdicts)} verdicts → {len(queries)} queries "
                f"with ≥1 positive (threshold>={args.label_threshold})"
            )
        else:
            relevance_fn = get_relevance_fn(args.relevance, args.seed)
            print(f"Relevance generator: {args.relevance}")
            queries = [
                q
                for q in build_queries(
                    profiles, args.queries, args.seed, relevance_fn=relevance_fn
                )
                if q[2]
            ]
        train_queries, test_queries = split_queries(queries, args.train_fraction, args.seed)
        print(
            f"{len(train_queries)} train queries / {len(test_queries)} test queries "
            f"(both filtered to ≥1 relevant)"
        )

        tracker.log_params(
            {
                "csv": args.csv,
                "queries": args.queries,
                "train_fraction": args.train_fraction,
                "k": args.k,
                "seed": args.seed,
                "with_embeddings": args.with_embeddings,
                "label_source": label_source,
                "label_threshold": args.label_threshold,
                "n_estimators": args.n_estimators,
                "learning_rate": args.learning_rate,
                "num_leaves": args.num_leaves,
                "n_profiles": len(profiles),
                "n_train_queries": len(train_queries),
                "n_test_queries": len(test_queries),
            }
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
        learned.fit(
            train_queries,
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
        )
        print("Done.")

        print(f"\nHeld-out evaluation ({len(test_queries)} queries, k={args.k}):")
        print(
            f"\n{'System':<22} {'P@' + str(args.k):>8} {'R@' + str(args.k):>8} "
            f"{'NDCG@' + str(args.k):>9} {'MAP@' + str(args.k):>9}"
        )
        print("-" * 62)
        _evaluate_and_log("random", make_random_ranker(args.seed), test_queries, args.k, tracker)
        _evaluate_and_log("rules_only", make_rules_ranker(), test_queries, args.k, tracker)
        if embeddings is not None:
            _evaluate_and_log(
                "rules+embeddings",
                make_rules_ranker(profile_embeddings=embeddings),
                test_queries,
                args.k,
                tracker,
            )
        _evaluate_and_log("learned", learned.as_ranker(), test_queries, args.k, tracker)

        learned.save(args.out)
        print(f"\nSaved learned ranker to {args.out}")
        tracker.log_artifact(args.out)


if __name__ == "__main__":
    main()
