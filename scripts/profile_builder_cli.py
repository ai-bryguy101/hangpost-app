#!/usr/bin/env python3
"""Terminal-based profile builder.

Quick way to build a custom profile and match it against the database
without needing a browser. Uses numbered menus for selection.

Usage:
    python scripts/profile_builder_cli.py
    python scripts/profile_builder_cli.py --csv data/test_profiles_10k.csv --top 10
"""

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from hangpost_matching import UserProfile, rank_candidates
from hangpost_matching.loader import load_profiles
from hangpost_matching.options import (
    COLLEGES, DEGREES, FAITHS, FAN_OF, HOBBIES, HOMESTATES, HOMETOWNS,
    INTERESTS_LIKES, JOBS, SKILLS_CERTS, TRAVEL_DESTINATIONS,
)


def _pick_one(label: str, options: list[str]) -> str:
    """Present numbered options and return the user's single choice."""
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"{'─' * 60}")
    sorted_opts = sorted(options)
    for i, opt in enumerate(sorted_opts, 1):
        print(f"  {i:>3}. {opt}")
    while True:
        raw = input(f"\n  Enter number (1-{len(sorted_opts)}): ").strip()
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(sorted_opts):
                return sorted_opts[idx]
        except ValueError:
            pass
        print("  Invalid choice, try again.")


def _pick_many(label: str, options: list[str], min_count: int, max_count: int) -> list[str]:
    """Present numbered options and return multiple choices."""
    print(f"\n{'─' * 60}")
    print(f"  {label}  (pick {min_count}-{max_count})")
    print(f"{'─' * 60}")
    sorted_opts = sorted(options)
    # Print in columns for readability
    col_width = 35
    per_row = 2
    for i, opt in enumerate(sorted_opts, 1):
        end = "\n" if i % per_row == 0 else ""
        print(f"  {i:>3}. {opt:<{col_width}}", end=end)
    if len(sorted_opts) % per_row != 0:
        print()

    while True:
        raw = input(f"\n  Enter numbers separated by commas (e.g. 1,3,7): ").strip()
        try:
            indices = [int(x.strip()) - 1 for x in raw.split(",") if x.strip()]
            if all(0 <= idx < len(sorted_opts) for idx in indices):
                if min_count <= len(indices) <= max_count:
                    return [sorted_opts[idx] for idx in indices]
                print(f"  Please pick between {min_count} and {max_count} options.")
                continue
        except ValueError:
            pass
        print("  Invalid input, try again.")


def build_profile_interactive() -> tuple[str, UserProfile]:
    """Walk the user through building a profile via terminal prompts."""
    print("\n" + "=" * 60)
    print("  HANGPOST PROFILE BUILDER")
    print("=" * 60)

    name = input("\n  Your name: ").strip() or "Test User"
    while True:
        age_raw = input("  Your age (18-65): ").strip()
        try:
            age = int(age_raw)
            if 18 <= age <= 65:
                break
        except ValueError:
            pass
        print("  Enter a number between 18 and 65.")

    hometown = _pick_one("Hometown", HOMETOWNS)
    _pick_one("Home State", HOMESTATES)  # collected for display but not used in scoring
    _pick_one("College", COLLEGES)
    _pick_one("Degree", DEGREES)
    _pick_one("Job", JOBS)
    _pick_one("Faith", FAITHS)

    hobbies = _pick_many("Hobbies & Activities", HOBBIES, 2, 8)
    skills = _pick_many("Skills & Certifications", SKILLS_CERTS, 0, 4)
    interests = _pick_many("Interests & Likes", INTERESTS_LIKES, 3, 7)
    _pick_many("Fan Of", FAN_OF, 2, 5)
    _pick_many("Travel Wishlist", TRAVEL_DESTINATIONS, 2, 4)

    hobbies_combined = hobbies + skills
    profile = UserProfile(
        user_id="custom_profile",
        interests={h.lower() for h in hobbies_combined},
        liked_topics={i.lower() for i in interests},
        location=hometown.lower(),
        age=age,
        mutual_friend_ids=set(),
    )

    return name, profile


def display_results(
    name: str,
    profile: UserProfile,
    ranked: list,
    database_rows: dict[str, dict],
    top_n: int,
) -> None:
    """Print the top N matches with full profile details."""
    print(f"\n{'=' * 70}")
    print(f"  TOP {top_n} MATCHES FOR: {name}")
    print(f"  (age {profile.age}, {profile.location})")
    print(f"{'=' * 70}")

    for rank, (candidate, breakdown) in enumerate(ranked[:top_n], start=1):
        row = database_rows.get(candidate.user_id, {})
        cand_age = int(row.get("age", 0))
        age_gap = abs((profile.age or 0) - cand_age)

        print(f"\n  #{rank} — {row.get('name', '?')}  (Score: {breakdown.total_score:.3f})")
        print(f"  {'─' * 60}")
        print(f"  Age: {cand_age} (gap: {age_gap})  |  {row.get('hometown', '?')}, {row.get('homestate', '?')}")
        print(f"  College: {row.get('college', '?')}  |  Degree: {row.get('degree', '?')}")
        print(f"  Job: {row.get('job', '?')}  |  Faith: {row.get('faith_religion', '?')}")
        print(f"  Hobbies:   {row.get('hobbies_activities_sports_games_skills_certifications', '?')}")
        print(f"  Interests: {row.get('interests_likes', '?')}")
        print(f"  Fan of:    {row.get('fan_of', '?')}")
        print(f"  Travel:    {row.get('travel', '?')}")
        print(f"  Mutual friends: {row.get('friends_in_common', '0')}")
        print(f"  --- Score Breakdown ---")
        print(f"    Interest overlap:  {breakdown.interest_overlap:.3f}")
        print(f"    Topic overlap:     {breakdown.liked_topic_overlap:.3f}")
        print(f"    Mutual friends:    {breakdown.mutual_friends:.3f}  (boost: {breakdown.social_boost:.3f})")
        print(f"    Location match:    {breakdown.location_match:.3f}")
        print(f"    Age compatibility: {breakdown.age_compatibility:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Terminal-based profile builder.")
    parser.add_argument("--csv", default="data/test_profiles_10k.csv", help="CSV database to match against")
    parser.add_argument("--top", type=int, default=20, help="Number of top matches to show")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    print(f"Loading profiles from {csv_path}...")
    profiles = load_profiles(csv_path)

    with csv_path.open() as fh:
        rows = list(csv.DictReader(fh))
    row_by_id = {prof.user_id: row for prof, row in zip(profiles, rows)}

    print(f"Loaded {len(profiles):,} profiles.")

    name, custom_profile = build_profile_interactive()

    print(f"\nRunning matching algorithm against {len(profiles):,} profiles...")
    ranked = rank_candidates(custom_profile, profiles)

    display_results(name, custom_profile, ranked, row_by_id, args.top)


if __name__ == "__main__":
    main()
