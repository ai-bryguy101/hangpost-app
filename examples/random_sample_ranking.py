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
    """Sample random profiles, rank them, and print a readable table."""
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

    source_name = sampled_rows[0]["name"]
    print(f"Source profile: {source_name} ({source.user_id})")
    print(f"Source age: {source.age}")
    print("-" * 120)
    print("Rank | Name | CandAge | AgeGap | AgeComp | Score | Mutual? | SocialBoost | Location | Interests")
    print("-" * 120)

    for rank, (candidate, breakdown) in enumerate(ranked, start=1):
        candidate_name = profile_name_by_id[candidate.user_id]
        candidate_age = int(row_by_id[candidate.user_id]["age"])
        age_gap = abs((source.age or 0) - candidate_age)
        print(
            f"{rank:>4} | {candidate_name:<22} | {candidate_age:>7} | {age_gap:>6} | "
            f"{breakdown.age_compatibility:>7.3f} | {breakdown.total_score:>5.3f} | "
            f"{str(breakdown.has_mutual_friends):<7} | {breakdown.social_boost:>10.3f} | "
            f"{breakdown.location_match:>8.3f} | {breakdown.interest_overlap:>9.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank a random sample of CSV profiles.")
    parser.add_argument("--csv", default="data/test_profiles.csv", help="Path to CSV dataset")
    parser.add_argument("--sample-size", type=int, default=21, help="How many random profiles to sample (1 source + N candidates)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible sampling")
    args = parser.parse_args()

    run_sample(Path(args.csv), args.sample_size, args.seed)


if __name__ == "__main__":
    main()
