"""Latency benchmark for the matching engine.

Measures wall-clock time to rank N candidates with each phase of the
roadmap. The point is to give honest numbers a reviewer can cite:
"rules_only ranks 1,000 candidates in X ms, rules+embeddings in Y ms,
learned in Z ms." Embedding precompute is reported separately because
in production it would be cached in a vector store, not paid per
request.

Usage:
    python scripts/bench.py
    python scripts/bench.py --sizes 100,1000,5000 --repeats 50
    python scripts/bench.py --with-embeddings  # requires [ml] extra
    python scripts/bench.py --with-embeddings --learned-model models/learned_ranker.joblib

The output is intentionally `docs/BENCHMARKS.md`-friendly so it can be
pasted into the README under "Latency" without reformatting.
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from hangpost_matching import (  # noqa: E402
    Ranker,
    UserProfile,
    load_profiles_from_csv,
    make_random_ranker,
    make_rules_ranker,
)


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(values)
    rank = (len(values_sorted) - 1) * (p / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(values_sorted) - 1)
    frac = rank - lower
    return values_sorted[lower] + frac * (values_sorted[upper] - values_sorted[lower])


def _time_ranker(
    name: str,
    ranker: Ranker,
    profiles: list[UserProfile],
    sizes: list[int],
    repeats: int,
    rng: random.Random,
) -> list[tuple[str, int, float, float, float]]:
    """Return one row per (ranker, size) with mean / p50 / p95 in milliseconds."""
    rows: list[tuple[str, int, float, float, float]] = []
    for size in sizes:
        timings_ms: list[float] = []
        # Pre-pick distinct source / candidate slices so we measure the
        # ranker, not the sampling cost.
        for _ in range(repeats):
            source = rng.choice(profiles)
            candidates = rng.sample(
                [p for p in profiles if p.user_id != source.user_id],
                min(size, len(profiles) - 1),
            )
            start = time.perf_counter()
            ranker(source, candidates)
            timings_ms.append((time.perf_counter() - start) * 1000.0)
        rows.append(
            (
                name,
                size,
                statistics.mean(timings_ms),
                _percentile(timings_ms, 50.0),
                _percentile(timings_ms, 95.0),
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="data/test_profiles.csv")
    parser.add_argument(
        "--sizes",
        default="10,100,500,1000",
        help="Comma-separated candidate-pool sizes to benchmark.",
    )
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--with-embeddings",
        action="store_true",
        help="Also benchmark rules+embeddings (requires [ml] extra).",
    )
    parser.add_argument(
        "--learned-model",
        type=Path,
        default=None,
        help="Path to a saved LearnedRanker to benchmark (requires [ml] extra).",
    )
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    rng = random.Random(args.seed)

    profiles = load_profiles_from_csv(Path(args.csv))
    print(f"Loaded {len(profiles)} profiles from {args.csv}\n")

    rows: list[tuple[str, int, float, float, float]] = []

    rows.extend(
        _time_ranker("random", make_random_ranker(args.seed), profiles, sizes, args.repeats, rng)
    )
    rows.extend(_time_ranker("rules_only", make_rules_ranker(), profiles, sizes, args.repeats, rng))

    if args.with_embeddings:
        from hangpost_matching import SentenceTransformerEmbedder, embed_profiles

        print("Loading sentence-transformer model and embedding all profiles...")
        emb_start = time.perf_counter()
        embedder = SentenceTransformerEmbedder()
        embeddings = embed_profiles(profiles, embedder)
        emb_elapsed_ms = (time.perf_counter() - emb_start) * 1000.0
        print(
            f"Embedded {len(embeddings)} profiles in {emb_elapsed_ms:.1f} ms "
            f"({emb_elapsed_ms / max(len(embeddings), 1):.2f} ms/profile)\n"
        )
        rows.extend(
            _time_ranker(
                "rules+embeddings",
                make_rules_ranker(profile_embeddings=embeddings),
                profiles,
                sizes,
                args.repeats,
                rng,
            )
        )

        if args.learned_model is not None:
            from hangpost_matching import LearnedRanker

            learned = LearnedRanker.load(args.learned_model)
            # Override the model's stored embeddings with the freshly-computed
            # ones so we're measuring the model, not stale embeddings.
            learned.profile_embeddings = embeddings
            rows.extend(
                _time_ranker(
                    "learned",
                    learned.as_ranker(),
                    profiles,
                    sizes,
                    args.repeats,
                    rng,
                )
            )

    # Pretty markdown-ready table.
    print(f"{'Ranker':<22} {'N':>6} {'mean (ms)':>12} {'p50 (ms)':>10} {'p95 (ms)':>10}")
    print("-" * 64)
    for name, size, mean_ms, p50_ms, p95_ms in rows:
        print(f"{name:<22} {size:>6d} {mean_ms:>12.3f} {p50_ms:>10.3f} {p95_ms:>10.3f}")


if __name__ == "__main__":
    main()
