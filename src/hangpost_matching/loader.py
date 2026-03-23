"""CSV profile loader.

Provides reusable utilities for loading UserProfile instances from CSV files.
This was extracted from examples/random_sample_ranking.py so that any script,
test, or future API layer can load profiles without duplicating parsing logic.
"""

import csv
from pathlib import Path

from .models import UserProfile


def tokenize(cell: str) -> set[str]:
    """Split semicolon-separated text into lowercase tokens.

    Example:
        >>> tokenize("Hiking; Cooking; yoga")
        {'hiking', 'cooking', 'yoga'}
    """
    return {token.strip().lower() for token in cell.split(";") if token.strip()}


def profile_from_row(index: int, row: dict[str, str]) -> UserProfile:
    """Convert one CSV row into a UserProfile.

    The CSV's ``friends_in_common`` column is a count, but the scoring model
    expects a set of IDs for intersection logic.  We synthesize deterministic
    IDs like ``common_friend_1``, ``common_friend_2``, etc.
    """
    friend_count = int(row["friends_in_common"])
    mutual_friend_ids = {f"common_friend_{i}" for i in range(1, friend_count + 1)}

    raw_age = row.get("age", "").strip()
    age = int(raw_age) if raw_age else None

    location_raw = row.get("hometown", "").strip().lower()
    location = location_raw if location_raw else None

    return UserProfile(
        user_id=f"csv_{index}_{row['name'].lower().replace(' ', '_')}",
        interests=tokenize(row.get("hobbies_activities_sports_games_skills_certifications", "")),
        liked_topics=tokenize(row.get("interests_likes", "")),
        location=location,
        age=age,
        mutual_friend_ids=mutual_friend_ids,
    )


def load_profiles(csv_path: str | Path) -> list[UserProfile]:
    """Load all profiles from a CSV file.

    Args:
        csv_path: Path to the CSV file (must have a header row matching
                  the expected column names).

    Returns:
        A list of UserProfile instances, one per CSV row.
    """
    csv_path = Path(csv_path)
    with csv_path.open() as fh:
        rows = list(csv.DictReader(fh))
    return [profile_from_row(i, row) for i, row in enumerate(rows)]
