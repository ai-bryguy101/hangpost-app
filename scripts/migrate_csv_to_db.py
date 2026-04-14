#!/usr/bin/env python3
"""Migrate the CSV dataset into the SQLite database.

Reads the existing test_profiles_10k.csv, creates the DB schema, and inserts
all profiles. For synthetic data, it also assigns random current-location
coordinates so we can test the geo-radius filtering.

Usage:
    python scripts/migrate_csv_to_db.py
    python scripts/migrate_csv_to_db.py --csv data/test_profiles_10k.csv --db data/hangpost.db
    python scripts/migrate_csv_to_db.py --seed 42    # reproducible coordinates

WHAT THIS DOES:
1. Creates the database and schema (safe to re-run).
2. Reads every row from the CSV.
3. For each row:
   a. Generates a user_id (same logic as loader.py for consistency).
   b. Assigns a random current_lat/current_lng near a major U.S. city.
   c. Inserts the profile into the DB.
4. For profiles with friends_in_common > 0, creates synthetic friendship rows.
5. Prints summary statistics.

WHY RANDOM CURRENT LOCATIONS:
In a real app, current_lat/current_lng come from the user's phone GPS.
For our synthetic dataset, we simulate this by randomly placing each user
near one of ~15 major U.S. cities. This gives us realistic clusters of
users so the radius filter has something to work with.

The "near" part is important: we add ±0.15° of jitter (~10 miles) around
each city center. This means a 20-mile radius search from downtown Denver
will find ~1/15th of the dataset, which is a realistic density.

WHY SYNTHETIC FRIENDSHIPS:
The CSV has a `friends_in_common` count (e.g., "2") but no actual friend
relationships. We create synthetic friendships using the same convention
as loader.py: a profile with friends_in_common=2 gets friends
"common_friend_1" and "common_friend_2". These synthetic friend profiles
don't exist as real profiles — they're just IDs used to create overlapping
friend sets between profiles so the mutual-friend scoring works correctly.
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

from hangpost_matching.db import get_connection, init_schema, insert_friendship, insert_profile


# ---------------------------------------------------------------------------
# City coordinates for assigning random current locations
# ---------------------------------------------------------------------------
# These are approximate lat/lng for major U.S. cities. Each synthetic profile
# will be randomly assigned to one of these cities, then jittered ±0.15°
# to simulate being "somewhere in the metro area."
#
# WHY these specific cities: They're geographically spread across the U.S.
# and represent the kinds of cities where a social app would launch.
# Having 15 clusters means a 20-mile radius search from any one of them
# returns a manageable number of profiles (~600-700 out of 10k).

CITY_COORDINATES: list[tuple[str, float, float]] = [
    # (city_name, latitude, longitude)
    ("New York",      40.7128, -74.0060),
    ("Los Angeles",   34.0522, -118.2437),
    ("Chicago",       41.8781, -87.6298),
    ("Houston",       29.7604, -95.3698),
    ("Phoenix",       33.4484, -112.0740),
    ("Philadelphia",  39.9526, -75.1652),
    ("San Antonio",   29.4241, -98.4936),
    ("San Diego",     32.7157, -117.1611),
    ("Dallas",        32.7767, -96.7970),
    ("Austin",        30.2672, -97.7431),
    ("Denver",        39.7392, -104.9903),
    ("Seattle",       47.6062, -122.3321),
    ("Boston",        42.3601, -71.0589),
    ("Nashville",     36.1627, -86.7816),
    ("Miami",         25.7617, -80.1918),
    ("Atlanta",       33.7490, -84.3880),
    ("Portland",      45.5152, -122.6784),
]


def _make_user_id(index: int, name: str) -> str:
    """Generate a user_id matching the convention in loader.py.

    WHY match loader.py: So that profiles loaded from CSV and profiles loaded
    from DB have the same user_ids. This makes it easy to compare results
    between the two loading methods during development.
    """
    return f"csv_{index}_{name.lower().replace(' ', '_')}"


def _random_current_location(rng: random.Random) -> tuple[float, float]:
    """Pick a random city and add jitter for a realistic current location.

    Returns (latitude, longitude) near a random major U.S. city.

    WHY ±0.15° jitter: 0.15° of latitude ≈ 10.4 miles. This means two users
    "in the same city" will be 0-15 miles apart, which is realistic for a
    metro area. A 20-mile radius search will catch most of them.
    """
    _, base_lat, base_lng = rng.choice(CITY_COORDINATES)

    # Add random jitter to simulate being somewhere in the metro area.
    lat = base_lat + rng.uniform(-0.15, 0.15)
    lng = base_lng + rng.uniform(-0.15, 0.15)

    return round(lat, 6), round(lng, 6)


def migrate(csv_path: Path, db_path: Path, seed: int | None = None) -> None:
    """Import the CSV dataset into the SQLite database."""
    rng = random.Random(seed)

    # ── Step 1: Create DB and schema ──
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = get_connection(db_path)
    init_schema(conn)

    # ── Step 2: Read CSV ──
    with csv_path.open() as fh:
        rows = list(csv.DictReader(fh))

    print(f"Read {len(rows):,} profiles from {csv_path}")

    # ── Step 3: Insert profiles ──
    friendship_pairs: list[tuple[str, str]] = []

    for i, row in enumerate(rows):
        user_id = _make_user_id(i, row["name"])
        lat, lng = _random_current_location(rng)

        profile_data = {
            "user_id": user_id,
            "name": row["name"],
            "age": int(row["age"]) if row.get("age") else None,
            # Hometown (for scoring) comes from the CSV's city/state columns.
            "hometown_city": row.get("city", ""),
            "hometown_state": row.get("state", ""),
            "college": row.get("college", ""),
            "degree": row.get("degree", ""),
            "job": row.get("job", ""),
            "faith": row.get("faith", ""),
            "hobbies": row.get("hobbies", ""),
            "interests": row.get("interests", ""),
            "fan_of": row.get("fan_of", ""),
            "travel": row.get("travel", ""),
            # Current location (for geo filtering) is randomly assigned.
            "current_lat": lat,
            "current_lng": lng,
        }
        insert_profile(conn, profile_data)

        # ── Step 3a: Track friendships for this profile ──
        # Same convention as loader.py: friends_in_common=2 means this user
        # is friends with synthetic users "common_friend_1" and "common_friend_2".
        friend_count = int(row.get("friends_in_common", "0"))
        for fi in range(1, friend_count + 1):
            friendship_pairs.append((user_id, f"common_friend_{fi}"))

    conn.commit()
    print(f"Inserted {len(rows):,} profiles into {db_path}")

    # ── Step 4: Insert friendships ──
    # The synthetic friend IDs ("common_friend_1", etc.) don't exist as real
    # profiles in the profiles table. This means foreign key constraints would
    # fail. We temporarily disable FK enforcement for the synthetic data import.
    # WHY not just remove the FK constraint: In production with real users,
    # FK enforcement prevents orphaned friendships. We only skip it here
    # because synthetic data is inherently "fake."
    conn.execute("PRAGMA foreign_keys=OFF")
    for user_id, friend_id in friendship_pairs:
        insert_friendship(conn, user_id, friend_id)

    conn.execute("PRAGMA foreign_keys=ON")
    conn.commit()
    print(f"Inserted {len(friendship_pairs):,} friendship edges")

    # ── Step 5: Summary stats ──
    profile_count = conn.execute("SELECT COUNT(*) FROM profiles").fetchone()[0]
    friendship_count = conn.execute("SELECT COUNT(*) FROM friendships").fetchone()[0]
    with_location = conn.execute(
        "SELECT COUNT(*) FROM profiles WHERE current_lat IS NOT NULL"
    ).fetchone()[0]

    print(f"\nDatabase summary:")
    print(f"  Profiles:    {profile_count:,}")
    print(f"  Friendships: {friendship_count:,}")
    print(f"  With geo:    {with_location:,} ({100*with_location/max(profile_count,1):.0f}%)")

    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate CSV profiles into SQLite database.")
    parser.add_argument("--csv", default="data/test_profiles_10k.csv", help="Source CSV file")
    parser.add_argument("--db", default="data/hangpost.db", help="Target SQLite database")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for current-location assignment")
    args = parser.parse_args()

    migrate(Path(args.csv), Path(args.db), args.seed)


if __name__ == "__main__":
    main()
