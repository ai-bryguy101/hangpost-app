#!/usr/bin/env python3
"""Generate a large synthetic CSV dataset of user profiles.

Usage:
    python scripts/generate_profiles.py                   # default 10,000 profiles
    python scripts/generate_profiles.py --count 5000 --seed 123
    python scripts/generate_profiles.py --output data/custom.csv

WHAT CHANGED AND WHY (v0.2.0):
- CSV columns now match the new taxonomy: separate `hobbies`, `interests`,
  `fan_of` columns instead of the old mega-column.
- City and state are generated as linked pairs from CITIES (no more
  mismatched "Austin, Florida" combinations).
- College and faith columns are included (they're now scored).

The distributions are designed to be realistic:
- Most users (90%) have 0 mutual friends; ~8% have 1; ~2% have 2-5
- Age follows a roughly normal distribution centered around 28
- Hobbies, interests, and fan_of are drawn from curated pools
"""

import argparse
import csv
import random
import sys
from pathlib import Path

# Allow running from repo root without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from hangpost_matching.options import (
    CITIES, COLLEGES, DEGREES, FAITHS, FAN_OF, FIRST_NAMES, HOBBIES,
    INTERESTS, JOBS, LAST_NAMES, TRAVEL_DESTINATIONS,
)


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def _pick_semicolon_list(pool: list[str], min_count: int, max_count: int) -> str:
    """Pick a random subset from the pool and join with '; '.

    WHY semicolons: Our tokenizer splits on ';' when loading profiles.
    This format is human-readable in a CSV while supporting multiple values.
    """
    k = random.randint(min_count, max_count)
    return "; ".join(random.sample(pool, min(k, len(pool))))


def _generate_friends_in_common() -> int:
    """Generate a friend count with realistic distribution.

    Distribution: 90% have 0, ~8% have 1, ~2% have 2-5.
    WHY this distribution: In a real social network, most random pairs
    of people share zero friends. Only a small fraction are socially
    connected. This makes mutual-friend matches rare and meaningful.
    """
    roll = random.random()
    if roll < 0.90:
        return 0
    elif roll < 0.98:
        return 1
    else:
        return random.randint(2, 5)


def _generate_age() -> int:
    """Generate an age from a roughly normal distribution centered at 28.

    WHY normal distribution: Real social apps skew young-adult. The Gaussian
    gives us a realistic bell curve with most users 22-34, some 18-21 and
    35-45, and rare outliers beyond that. Clamped to 18-65.
    """
    age = int(random.gauss(28, 6))
    return max(18, min(65, age))


def generate_row() -> dict[str, str]:
    """Generate a single synthetic profile row.

    Each row maps directly to what profile_from_row() expects when loading.
    """
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)

    # Pick a city+state pair (always linked — no mismatches).
    city, state = random.choice(CITIES)

    return {
        "name": f"{first} {last}",
        "friends_in_common": str(_generate_friends_in_common()),
        "age": str(_generate_age()),
        "city": city,
        "state": state,
        "college": random.choice(COLLEGES),
        "degree": random.choice(DEGREES),
        "job": random.choice(JOBS),
        "faith": random.choice(FAITHS),
        # The three taxonomy fields: hobbies, interests, fan_of.
        "hobbies": _pick_semicolon_list(HOBBIES, 2, 6),
        "interests": _pick_semicolon_list(INTERESTS, 3, 7),
        "fan_of": _pick_semicolon_list(FAN_OF, 3, 8),
        "travel": _pick_semicolon_list(TRAVEL_DESTINATIONS, 2, 4),
    }


def generate_csv(output_path: Path, count: int, seed: int | None = None) -> None:
    """Write *count* synthetic profiles to a CSV file."""
    if seed is not None:
        random.seed(seed)

    # Column order in the output CSV. This defines the "schema" for our data.
    fieldnames = [
        "name", "friends_in_common", "age", "city", "state",
        "college", "degree", "job", "faith",
        "hobbies", "interests", "fan_of", "travel",
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
