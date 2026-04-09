"""CSV profile loader.

Provides reusable utilities for loading UserProfile instances from CSV files.
This was extracted from examples/random_sample_ranking.py so that any script,
test, or future API layer can load profiles without duplicating parsing logic.

WHAT CHANGED AND WHY (v0.2.0):
- profile_from_row now maps to the new 3-field taxonomy (hobbies, interests,
  fan_of) instead of the old 2-field model (interests, liked_topics).
- Location is now loaded as a Location(city, state) pair instead of a bare string.
- New fields (college, faith, travel) are now loaded from the CSV.
- The CSV column names changed to match the new taxonomy:
  OLD: hobbies_activities_sports_games_skills_certifications, interests_likes
  NEW: hobbies, interests, fan_of  (cleaner, matches the data model)
"""

import csv
from pathlib import Path

from .models import Location, UserProfile


def tokenize(cell: str) -> set[str]:
    """Split semicolon-separated text into lowercase tokens.

    WHY lowercase: So "Hiking" and "hiking" are treated as the same thing.
    Without this, Jaccard similarity would miss obvious matches.

    WHY semicolons: Our CSV data uses semicolons to separate multiple values
    within a single cell (e.g., "Hiking; Chess; Guitar").

    Example:
        >>> tokenize("Hiking; Cooking; yoga")
        {'hiking', 'cooking', 'yoga'}
    """
    return {token.strip().lower() for token in cell.split(";") if token.strip()}


def profile_from_row(index: int, row: dict[str, str]) -> UserProfile:
    """Convert one CSV row into a UserProfile.

    HOW FRIEND IDS WORK:
    The CSV has a `friends_in_common` count (e.g., "2"), but our scoring model
    needs a *set of IDs* so it can compute intersections between two profiles.
    We synthesize deterministic IDs like `common_friend_1`, `common_friend_2`.

    WHY synthetic IDs: In a real app, these would be actual user IDs from the
    social graph. For testing with synthetic data, we use a convention where
    profiles with `friends_in_common=2` get IDs {common_friend_1, common_friend_2}.
    This means any two profiles with overlapping friend counts will share some
    synthetic friend IDs, which correctly triggers the mutual-friends scoring.

    HOW LOCATION WORKS:
    The CSV has separate `city` and `state` columns. We combine them into a
    Location(city, state) pair. If city is missing, we set location to None
    (no location signal for scoring).
    """
    # Parse friend count → synthetic friend ID set.
    friend_count = int(row.get("friends_in_common", "0"))
    mutual_friend_ids = {f"common_friend_{i}" for i in range(1, friend_count + 1)}

    # Parse age, handling missing/empty values gracefully.
    raw_age = row.get("age", "").strip()
    age = int(raw_age) if raw_age else None

    # Build location from city+state columns.
    # If city is missing, we treat the whole location as unknown.
    city = row.get("city", "").strip()
    state = row.get("state", "").strip()
    location = Location(city=city, state=state) if city else None

    # College and faith: empty strings → None (unknown).
    college_raw = row.get("college", "").strip()
    college = college_raw if college_raw else None

    faith_raw = row.get("faith", "").strip()
    faith = faith_raw if faith_raw else None

    return UserProfile(
        user_id=f"csv_{index}_{row.get('name', 'unknown').lower().replace(' ', '_')}",

        # The three taxonomy fields, each tokenized from their own CSV column.
        hobbies=tokenize(row.get("hobbies", "")),
        interests=tokenize(row.get("interests", "")),
        fan_of=tokenize(row.get("fan_of", "")),

        location=location,
        age=age,
        mutual_friend_ids=mutual_friend_ids,
        college=college,
        faith=faith,
        travel_wishlist=tokenize(row.get("travel", "")),
    )


def load_profiles(csv_path: str | Path) -> list[UserProfile]:
    """Load all profiles from a CSV file.

    Args:
        csv_path: Path to the CSV file. Must have a header row with column
                  names matching what profile_from_row expects.

    Returns:
        A list of UserProfile instances, one per CSV row.
    """
    csv_path = Path(csv_path)
    with csv_path.open() as fh:
        rows = list(csv.DictReader(fh))
    return [profile_from_row(i, row) for i, row in enumerate(rows)]
