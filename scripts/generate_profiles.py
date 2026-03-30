#!/usr/bin/env python3
"""Generate a large synthetic CSV dataset of user profiles.

Usage:
    python scripts/generate_profiles.py                   # default 10,000 profiles
    python scripts/generate_profiles.py --count 5000 --seed 123
    python scripts/generate_profiles.py --output data/custom.csv

The distributions are designed to be realistic:
- Most users (90%) have 0 mutual friends; ~8% have 1; ~2% have 2-5
- Age follows a roughly normal distribution centered around 28
- Interests, topics, and locations are drawn from curated pools
"""

import argparse
import csv
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from hangpost_matching.options import (
    COLLEGES, DEGREES, FAITHS, FAN_OF, FIRST_NAMES, HOBBIES, HOMESTATES,
    HOMETOWNS, INTERESTS_LIKES, JOBS, LAST_NAMES, SKILLS_CERTS,
    TRAVEL_DESTINATIONS,
)


# ---------------------------------------------------------------------------
# Generation logic
# ---------------------------------------------------------------------------

def _pick_semicolon_list(pool: list[str], min_count: int, max_count: int) -> str:
    """Pick a random subset and join with '; '."""
    k = random.randint(min_count, max_count)
    return "; ".join(random.sample(pool, min(k, len(pool))))


def _generate_friends_in_common() -> int:
    """Realistic distribution: 90% have 0, ~8% have 1, ~2% have 2-5."""
    roll = random.random()
    if roll < 0.90:
        return 0
    elif roll < 0.98:
        return 1
    else:
        return random.randint(2, 5)


def _generate_age() -> int:
    """Age distribution roughly normal, centered at 28, range 18-65."""
    age = int(random.gauss(28, 6))
    return max(18, min(65, age))


def generate_row() -> dict[str, str]:
    """Generate a single synthetic profile row."""
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    name = f"{first} {last}"

    hobbies = random.sample(HOBBIES, random.randint(2, 5))
    skills = random.sample(SKILLS_CERTS, random.randint(0, 3))
    hobbies_combined = "; ".join(hobbies + skills)

    return {
        "name": name,
        "friends_in_common": str(_generate_friends_in_common()),
        "age": str(_generate_age()),
        "college": random.choice(COLLEGES),
        "hometown": random.choice(HOMETOWNS),
        "degree": random.choice(DEGREES),
        "job": random.choice(JOBS),
        "homestate": random.choice(HOMESTATES),
        "hobbies_activities_sports_games_skills_certifications": hobbies_combined,
        "interests_likes": _pick_semicolon_list(INTERESTS_LIKES, 3, 7),
        "fan_of": _pick_semicolon_list(FAN_OF, 2, 5),
        "faith_religion": random.choice(FAITHS),
        "travel": _pick_semicolon_list(TRAVEL_DESTINATIONS, 2, 4),
    }


def generate_csv(output_path: Path, count: int, seed: int | None = None) -> None:
    """Write *count* synthetic profiles to a CSV file."""
    if seed is not None:
        random.seed(seed)

    fieldnames = [
        "name", "friends_in_common", "age", "college", "hometown",
        "degree", "job", "homestate",
        "hobbies_activities_sports_games_skills_certifications",
        "interests_likes", "fan_of", "faith_religion", "travel",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for _ in range(count):
            writer.writerow(generate_row())

    print(f"Wrote {count:,} profiles to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic test profiles.")
    parser.add_argument("--count", type=int, default=10_000, help="Number of profiles to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output", default="data/test_profiles_10k.csv", help="Output CSV path")
    args = parser.parse_args()

    generate_csv(Path(args.output), args.count, args.seed)


if __name__ == "__main__":
    main()
