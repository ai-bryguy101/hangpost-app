"""Gold-set calibration pass for the LLM-judge labels.

Phase 3.5 of the roadmap labels the bulk pairs with a *cheap* model
(Haiku 4.5 by default) so the LightGBM `LearnedRanker` has a large
training set to distill from. This script does the second half of the
hybrid pattern: it re-judges a small stratified subset of the same
pairs with a *stronger* model (Sonnet 4.6 by default), writes the
verdicts to a separate JSONL, and prints inter-rater agreement metrics
so you can quantify how much the bulk labels deviate from the gold
ones.

Why two models, why these two
-----------------------------
- Haiku is ~30x cheaper than Opus per labelled pair. Use it where
  volume matters (~900 bulk pairs at ~$1-2 total).
- Sonnet is the natural "second opinion": ~5x cheaper than Opus, and
  on a constrained rubric-driven 0-4 rating it stays close to Opus
  quality. Use it on a small calibration subset (~100 pairs at $2-3).
- Sonnet sits closer to Haiku on the capability ladder than Opus does,
  which makes the agreement metric easier to interpret —
  disagreements are more likely to be substantive rating differences
  than "the stronger model overthought it."

Metrics
-------
The script prints four flavours of agreement:

- **exact agreement %** — strict, what you'd put on a slide.
- **adjacent (±1) %** — forgiving of "is this a 2 or a 3" calls.
- **mean |Δrating|** — magnitude of disagreement, not just count.
- **quadratic-weighted Cohen's κ** — chance-corrected, ordinal-aware;
  the standard inter-rater agreement metric for this kind of scale.

Requires the [judge] extra:
    pip install -e ".[judge]"

Run:
    export ANTHROPIC_API_KEY=...
    python scripts/gold_label.py --n 100
    python scripts/gold_label.py --n 200 --bulk-labels data/judge_labels.jsonl \
        --gold-labels data/judge_labels_gold.jsonl --model claude-sonnet-4-6

The gold JSONL is append-only and idempotent (re-runs only call Claude
for new pairs), exactly like the bulk pass.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from hangpost_matching import (  # noqa: E402
    ClaudeJudge,
    JudgeVerdict,
    UserProfile,
    agreement_report,
    judge_pairs,
    load_profiles_from_csv,
    load_verdicts,
    sample_for_gold_pass,
)

# Sonnet 4.6 is the default "gold" model. Stronger than Haiku on
# nuanced rubric calls; an order of magnitude cheaper than Opus per
# labelled pair on the same task.
DEFAULT_GOLD_MODEL = "claude-sonnet-4-6"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        default="data/test_profiles.csv",
        help="Profile CSV (same as scripts/label.py).",
    )
    parser.add_argument(
        "--bulk-labels",
        type=Path,
        default=Path("data/judge_labels.jsonl"),
        help="Bulk-pass JSONL to sample from. Must exist; run scripts/label.py first.",
    )
    parser.add_argument(
        "--gold-labels",
        type=Path,
        default=Path("data/judge_labels_gold.jsonl"),
        help="Output JSONL for the gold verdicts. Acts as cache on re-runs.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=100,
        help="How many pairs to sample for the gold pass.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the (stratified) sample so the gold subset is reproducible.",
    )
    parser.add_argument(
        "--no-stratify",
        action="store_true",
        help="Sample uniformly at random instead of stratifying by bulk rating.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_GOLD_MODEL,
        help=f"Claude model ID for the gold judge (default: {DEFAULT_GOLD_MODEL}).",
    )
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help=(
            "Disable adaptive thinking on the gold judge. Faster and "
            "cheaper; on a small gold pass leaving thinking on is usually "
            "worth the few dollars."
        ),
    )
    args = parser.parse_args()

    bulk = load_verdicts(args.bulk_labels)
    if not bulk:
        raise SystemExit(
            f"No bulk verdicts found at {args.bulk_labels}. Run scripts/label.py first."
        )
    print(f"Loaded {len(bulk)} bulk verdicts from {args.bulk_labels}.")

    profiles = load_profiles_from_csv(Path(args.csv))
    by_id = {p.user_id: p for p in profiles}

    sampled_keys = sample_for_gold_pass(
        bulk,
        n=args.n,
        seed=args.seed,
        stratify_by_rating=not args.no_stratify,
    )
    print(
        f"Sampled {len(sampled_keys)} pairs for the gold pass "
        f"({'stratified by rating' if not args.no_stratify else 'uniform random'})."
    )

    # Resolve each (sid, cid) back to (source_profile, candidate_profile).
    # Any pair whose profiles aren't in the CSV is dropped with a warning —
    # that only happens if the CSV changed since the bulk pass.
    pairs: list[tuple[UserProfile, UserProfile]] = []
    missing = 0
    for sid, cid in sampled_keys:
        source = by_id.get(sid)
        candidate = by_id.get(cid)
        if source is None or candidate is None:
            missing += 1
            continue
        pairs.append((source, candidate))
    if missing:
        print(f"  warning: dropped {missing} pairs whose profiles weren't in the CSV.")

    existing_gold = load_verdicts(args.gold_labels)
    new_pairs = [(s, c) for (s, c) in pairs if (s.user_id, c.user_id) not in existing_gold]
    print(
        f"{len(pairs)} pairs total; {len(new_pairs)} need a fresh call, "
        f"{len(pairs) - len(new_pairs)} already cached in {args.gold_labels}."
    )

    if new_pairs:
        judge = ClaudeJudge(model=args.model, thinking=not args.no_thinking)
        print(
            f"Judging with model={args.model} thinking={not args.no_thinking}. "
            f"This will cost real API tokens — proceed carefully."
        )

        started = time.monotonic()

        def progress(i: int, total: int, verdict: JudgeVerdict) -> None:
            elapsed = time.monotonic() - started
            rate = i / elapsed if elapsed > 0 else 0.0
            print(
                f"  [{i:>4}/{total}] "
                f"({verdict.source_id} -> {verdict.candidate_id}) "
                f"rating={verdict.rating}  "
                f"[{rate:.2f} pair/s]",
                flush=True,
            )

        judge_pairs(judge, pairs, cache_path=args.gold_labels, progress=progress)

    # Recompute the full gold set (including any pre-existing cache) so the
    # agreement report covers every overlap, not just the new pairs.
    gold = load_verdicts(args.gold_labels)
    report = agreement_report(bulk, gold)

    print()
    print("=" * 60)
    print(f"Inter-rater agreement: bulk ({args.bulk_labels.name})")
    print(f"                    vs gold ({args.gold_labels.name}, model={args.model})")
    print("=" * 60)
    print(report.render_table())
    print("=" * 60)
    print(
        "Reading the κ: <0.20 poor · 0.20-0.40 fair · 0.40-0.60 moderate · "
        "0.60-0.80 substantial · >0.80 almost perfect."
    )


if __name__ == "__main__":
    main()
