"""Use Claude as a teacher to label (source, candidate) pairs.

This is Phase 3.5 of the roadmap: the synthetic labels that the rest of
the harness has been graded against don't reflect real friendship
likelihood — they're a thresholded combination of the *same signals* the
ranker already sees. Sending each pair to Claude with a rubric gives us
labels that aren't structurally identical to the inputs, so the metrics
in `scripts/evaluate.py` actually start meaning something, and the
LightGBM `LearnedRanker` in `scripts/train.py` finally has a teacher
worth distilling from.

Labelling strategy
------------------
For each sampled source profile, the script judges:

  - the top `--top-k` candidates returned by the rules ranker (so we
    grade re-ranking quality on the candidates that matter for P@k), and
  - `--random-k` additional candidates picked uniformly at random (so a
    learned ranker that surfaces strong candidates the rules ranker
    missed gets credit for it).

Every verdict is appended to a JSONL file as it's produced. The next
run reads that file, skips pairs that already have a verdict for the
model, and only calls Claude for the gaps. The file is both the cache
and the final labels artifact.

Requires the [judge] extra:
    pip install -e ".[judge]"

Run:
    export ANTHROPIC_API_KEY=...
    python scripts/label.py --queries 30 --top-k 15 --random-k 15
    python scripts/label.py --queries 30 --no-thinking          # faster, cheaper
    python scripts/label.py --model claude-sonnet-4-6 --queries 50

The labels file feeds:
    python scripts/evaluate.py --labels data/judge_labels.jsonl
    python scripts/train.py    --labels data/judge_labels.jsonl
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from hangpost_matching import (  # noqa: E402
    JUDGE_DEFAULT_MODEL,
    ClaudeJudge,
    JudgeVerdict,
    UserProfile,
    judge_pairs,
    load_profiles_from_csv,
    load_verdicts,
    make_rules_ranker,
)


def _pairs_for_source(
    source: UserProfile,
    candidates: list[UserProfile],
    top_k: int,
    random_k: int,
    rng: random.Random,
) -> list[tuple[UserProfile, UserProfile]]:
    """Pick the candidate pool we'll send to the judge for one source.

    `top_k` from the rules ranker + `random_k` extras, deduped by
    `user_id` so the judge isn't asked the same pair twice.
    """
    rules_ranker = make_rules_ranker()
    ranked_ids = rules_ranker(source, candidates)
    by_id = {c.user_id: c for c in candidates}

    chosen_ids: list[str] = []
    seen: set[str] = set()
    for cid in ranked_ids[:top_k]:
        if cid not in seen:
            chosen_ids.append(cid)
            seen.add(cid)

    remaining = [cid for cid in by_id if cid not in seen]
    rng.shuffle(remaining)
    for cid in remaining[:random_k]:
        chosen_ids.append(cid)
        seen.add(cid)

    return [(source, by_id[cid]) for cid in chosen_ids]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="data/test_profiles.csv")
    parser.add_argument(
        "--queries",
        type=int,
        default=30,
        help="Number of source profiles to sample for labelling.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=15,
        help="Per source, judge the top-K candidates from the rules ranker.",
    )
    parser.add_argument(
        "--random-k",
        type=int,
        default=15,
        help=(
            "Per source, judge K additional uniformly-random candidates "
            "(so a learned ranker can be credited for surfacing strong "
            "candidates the rules ranker missed)."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model",
        default=JUDGE_DEFAULT_MODEL,
        help="Claude model ID for the judge (default: claude-opus-4-7).",
    )
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help=(
            "Disable adaptive thinking on the judge. Faster and cheaper "
            "per call; slightly noisier ratings."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/judge_labels.jsonl"),
        help="JSONL file to append verdicts to (also acts as the cache).",
    )
    args = parser.parse_args()

    profiles = load_profiles_from_csv(Path(args.csv))
    print(f"Loaded {len(profiles)} profiles from {args.csv}")

    # Cache hits require a model match — verdicts from a different judge model
    # do not count as already-done for this run.
    existing_all = load_verdicts(args.out)
    existing_for_model = {key: v for key, v in existing_all.items() if v.model == args.model}
    if existing_all:
        print(
            f"Found {len(existing_all)} verdicts in {args.out} "
            f"({len(existing_for_model)} for model={args.model})"
        )

    rng = random.Random(args.seed)
    sources = rng.sample(profiles, min(args.queries, len(profiles)))
    pairs: list[tuple[UserProfile, UserProfile]] = []
    for source in sources:
        candidates = [p for p in profiles if p.user_id != source.user_id]
        pairs.extend(_pairs_for_source(source, candidates, args.top_k, args.random_k, rng))

    new_pairs = [(s, c) for (s, c) in pairs if (s.user_id, c.user_id) not in existing_for_model]
    print(
        f"Sampled {len(pairs)} (source, candidate) pairs "
        f"({len(new_pairs)} new, {len(pairs) - len(new_pairs)} cached)"
    )

    if not new_pairs:
        print("Nothing to do — all pairs already labelled.")
        return

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

    verdicts = judge_pairs(judge, pairs, cache_path=args.out, progress=progress)

    ratings = [v.rating for v in verdicts.values()]
    if ratings:
        dist = {r: ratings.count(r) for r in range(5)}
        print(f"\nLabel distribution: {dist}")
        print(f"Wrote {len(verdicts)} total verdicts to {args.out}")


if __name__ == "__main__":
    main()
