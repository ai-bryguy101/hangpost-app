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

from hangpost_matching import UserProfile, rank_candidates  # noqa: E402


def _tokenize(cell: str) -> set[str]:
    """Split semicolon-separated text into lowercase tokens."""
    return {token.strip().lower() for token in cell.split(";") if token.strip()}


def _profile_from_row(index: int, row: dict[str, str]) -> UserProfile:
    """Convert one CSV row into the internal UserProfile model.

    Beginner note:
    `friends_in_common` in the CSV is a count, but the scoring model expects a
    set of IDs so it can compute intersections. We synthesize fake IDs like
    common_friend_1, common_friend_2, ... based on that count.
    """
    friend_count = int(row["friends_in_common"])
    mutual_friend_ids = {f"common_friend_{i}" for i in range(1, friend_count + 1)}

    return UserProfile(
        user_id=f"csv_{index}_{row['name'].lower().replace(' ', '_')}",
        interests=_tokenize(row["hobbies_activities_sports_games_skills_certifications"]),
        liked_topics=_tokenize(row["interests_likes"]),
        location=row["hometown"].strip().lower() or None,
        age=int(row["age"]),
        mutual_friend_ids=mutual_friend_ids,
    )


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
    profiles = [_profile_from_row(i, row) for i, row in enumerate(sampled_rows)]
    paired = list(zip(profiles, sampled_rows, strict=True))
    profile_name_by_id = {profile.user_id: row["name"] for profile, row in paired}
    row_by_id = {profile.user_id: row for profile, row in paired}

    source = profiles[0]
    candidates = profiles[1:]
    ranked = rank_candidates(source, candidates)

    source_name = sampled_rows[0]["name"]
    print(f"Source profile: {source_name} ({source.user_id})")
    print(f"Source age: {source.age}")
    header = (
        "Rank | Name | CandAge | AgeGap | AgeComp | Score | "
        "Mutual? | SocialBoost | Location | Interests"
    )
    print("-" * 120)
    print(header)
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
    parser.add_argument(
        "--csv",
        default="data/test_profiles.csv",
        help="Path to CSV dataset",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10,
        help="How many random profiles to sample",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling",
    )
    args = parser.parse_args()

    run_sample(Path(args.csv), args.sample_size, args.seed)


if __name__ == "__main__":
    main()
