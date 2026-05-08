"""CSV → UserProfile loading utilities.

Centralizes the parsing logic so the demo, evaluation harness, and any
future scripts share the same row-to-profile mapping. If the on-disk
schema ever changes, only this module needs to change.
"""

from __future__ import annotations

import csv
from pathlib import Path

from .models import UserProfile


def _tokenize(cell: str) -> set[str]:
    """Split semicolon-separated text into lowercase tokens."""
    return {token.strip().lower() for token in cell.split(";") if token.strip()}


def load_profiles_from_csv(path: Path) -> list[UserProfile]:
    """Read profiles from a Hangpost-format CSV.

    Expected columns (others are tolerated and ignored):
    - name
    - friends_in_common (an integer count)
    - age
    - hometown
    - hobbies_activities_sports_games_skills_certifications
    - interests_likes

    Note on `friends_in_common`:
    The CSV stores a count per row, but the scoring model expects a set of
    IDs so that intersections work. We synthesize fake shared-friend IDs
    (`common_friend_1`, `common_friend_2`, ...) sized by that count, which
    is enough to drive mutual-friend overlap signals in evaluation.
    """
    profiles: list[UserProfile] = []
    with path.open() as fh:
        for index, row in enumerate(csv.DictReader(fh)):
            friend_count = int(row["friends_in_common"])
            mutual_friend_ids = {f"common_friend_{i}" for i in range(1, friend_count + 1)}
            profiles.append(
                UserProfile(
                    user_id=f"csv_{index}_{row['name'].lower().replace(' ', '_')}",
                    interests=_tokenize(
                        row["hobbies_activities_sports_games_skills_certifications"]
                    ),
                    liked_topics=_tokenize(row["interests_likes"]),
                    location=row["hometown"].strip().lower() or None,
                    age=int(row["age"]),
                    mutual_friend_ids=mutual_friend_ids,
                )
            )
    return profiles
