"""Random sample ranking example.

Picks a random source profile and N candidates from the CSV, runs the
matching algorithm, and prints a readable table showing how each
candidate scored.

Usage:
    python examples/random_sample_ranking.py --csv data/test_profiles_10k.csv --seed 42
    python examples/random_sample_ranking.py --sample-size 11 --seed 7

WHAT CHANGED AND WHY (v0.2.0):
- The table now shows the new scoring signals: hobby_overlap, interest_overlap,
  fan_of_overlap, college_match, faith_match, travel_overlap.
  WHY: The scoring engine now computes 9 signals instead of the old 5. Showing
  them all lets you see what's driving each match at a glance.

- Location column now shows city + state from the CSV (was just "hometown").
  WHY: Location scoring is now tiered (city > state), so seeing both fields
  helps you understand why two people in the same state but different cities
  get partial credit.

- Default CSV path changed to test_profiles_10k.csv (the standard dataset).
"""

import argparse
import csv
import random
import sys
from pathlib import Path

# Allow running from repo root without installing the package first.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from hangpost_matching import rank_candidates
from hangpost_matching.loader import profile_from_row


def run_sample(csv_path: Path, sample_size: int, seed: int | None = None) -> None:
    """Sample random profiles, rank them, and print a readable table.

    HOW IT WORKS:
    1. Load all rows from the CSV.
    2. Randomly pick `sample_size` rows (first one becomes "source").
    3. Run `rank_candidates` to score+sort the remaining candidates.
    4. Print a human-readable table with scores and breakdowns.

    WHY random sampling: You can't eyeball 10,000 profiles. Sampling a small
    group lets you manually verify the algorithm's behavior on specific cases.
    The --seed flag makes it reproducible so you can compare runs.
    """
    if seed is not None:
        random.seed(seed)

    with csv_path.open() as file_handle:
        rows = list(csv.DictReader(file_handle))

    if sample_size < 2:
        raise ValueError("sample_size must be at least 2")
    if sample_size > len(rows):
        raise ValueError(f"sample_size={sample_size} exceeds row count={len(rows)}")

    sampled_rows = random.sample(rows, sample_size)
    profiles = [profile_from_row(i, row) for i, row in enumerate(sampled_rows)]
    profile_name_by_id = {profile.user_id: row['name'] for profile, row in zip(profiles, sampled_rows)}
    row_by_id = {profile.user_id: row for profile, row in zip(profiles, sampled_rows)}

    source = profiles[0]
    candidates = profiles[1:]
    ranked = rank_candidates(source, candidates)

    # ── Print source profile summary ──
    source_row = sampled_rows[0]
    source_name = source_row["name"]
    print(f"Source profile: {source_name} ({source.user_id})")
    print(f"  Age: {source.age}")
    print(f"  Location: {source_row.get('city', '?')}, {source_row.get('state', '?')}")
    print(f"  College: {source_row.get('college', '?')}")
    print(f"  Faith: {source_row.get('faith', '?')}")

    # ── Print ranked candidates table ──
    # WHY this format: Wide tables are hard to read in terminals. Instead of
    # cramming everything into one row, we show the most important columns
    # in the table and let the evaluate_matches script handle full details.
    print(f"\n{'-' * 130}")
    print(
        f"{'Rank':>4} | {'Name':<22} | {'Age':>3} | {'Gap':>3} | "
        f"{'Score':>5} | {'Mutual':>6} | {'Boost':>5} | "
        f"{'Hobby':>5} | {'Intr':>5} | {'Fan':>5} | "
        f"{'Loc':>5} | {'AgeCmp':>6} | {'Coll':>4} | {'Faith':>5} | {'Trvl':>4}"
    )
    print(f"{'-' * 130}")

    for rank, (candidate, breakdown) in enumerate(ranked, start=1):
        candidate_name = profile_name_by_id[candidate.user_id]
        candidate_age = int(row_by_id[candidate.user_id]["age"])
        age_gap = abs((source.age or 0) - candidate_age)

        # Each column maps to a field on MatchBreakdown from compute_match_score().
        print(
            f"{rank:>4} | {candidate_name:<22} | {candidate_age:>3} | {age_gap:>3} | "
            f"{breakdown.total_score:>5.3f} | {str(breakdown.has_mutual_friends):>6} | {breakdown.social_boost:>5.3f} | "
            f"{breakdown.hobby_overlap:>5.3f} | {breakdown.interest_overlap:>5.3f} | {breakdown.fan_of_overlap:>5.3f} | "
            f"{breakdown.location_match:>5.3f} | {breakdown.age_compatibility:>6.3f} | "
            f"{breakdown.college_match:>4.2f} | {breakdown.faith_match:>5.2f} | {breakdown.travel_overlap:>4.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank a random sample of CSV profiles.")
    parser.add_argument("--csv", default="data/test_profiles_10k.csv", help="Path to CSV dataset")
    parser.add_argument("--sample-size", type=int, default=21, help="How many random profiles to sample (1 source + N candidates)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible sampling")
    args = parser.parse_args()

    run_sample(Path(args.csv), args.sample_size, args.seed)


if __name__ == "__main__":
    main()
