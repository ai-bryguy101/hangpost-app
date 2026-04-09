#!/usr/bin/env python3
"""Terminal-based profile builder.

Quick way to build a custom profile and match it against the database
without needing a browser. Uses numbered menus for selection.

Usage:
    python scripts/profile_builder_cli.py
    python scripts/profile_builder_cli.py --csv data/test_profiles_10k.csv --top 10

WHAT CHANGED AND WHY (v0.2.0):
- City and state are now picked as a single linked pair from CITIES.
  WHY: The old version had separate Hometown and Home State dropdowns, which
  let you pick "Austin" + "Florida" — a nonsensical combination. Now you pick
  "Austin, Texas" as one choice, guaranteeing a valid city+state pair.

- The old "Hobbies & Activities" + "Skills & Certifications" + "Interests & Likes"
  are now three semantically distinct fields: Hobbies, Interests, Fan Of.
  WHY: The old model mixed activities (hiking), categories (tech), and specific
  fandoms (Kendrick Lamar) into overlapping buckets. The new taxonomy gives
  each type its own field so the algorithm can score them independently.

- Score breakdown now shows all 9 component signals + social boost.
  WHY: With college_match, faith_match, and travel_overlap now feeding into
  the score, users need to see what's driving each match.

- Imports updated: HOMETOWNS/HOMESTATES/INTERESTS_LIKES/SKILLS_CERTS are gone,
  replaced by CITIES/INTERESTS/FAN_OF from the new options module.
"""

import argparse
import csv
import sys
from pathlib import Path

# Allow running from repo root without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from hangpost_matching import Location, UserProfile, rank_candidates
from hangpost_matching.loader import load_profiles
from hangpost_matching.options import (
    CITIES, COLLEGES, DEGREES, FAITHS, FAN_OF, HOBBIES,
    INTERESTS, JOBS, TRAVEL_DESTINATIONS,
)


# ---------------------------------------------------------------------------
# Interactive selection helpers
# ---------------------------------------------------------------------------
# These are reusable TUI building blocks. _pick_one for single-choice fields
# (city, college, etc.) and _pick_many for multi-choice fields (hobbies, etc.).


def _pick_one(label: str, options: list[str]) -> str:
    """Present numbered options and return the user's single choice.

    WHY sorted: Alphabetical order makes it easier for users to scan through
    a long list and find what they're looking for.
    """
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


def _pick_city() -> tuple[str, str]:
    """Present city+state pairs as a single dropdown and return (city, state).

    WHY a combined picker instead of separate city and state:
    Cities only make sense paired with their state — "Portland" could be
    Oregon or Maine. By presenting "Portland, Oregon" as one choice, we
    guarantee a valid pair and prevent mismatches like "Austin, Florida."

    The list is sorted by state first, then city within each state, so
    all Texas cities are grouped together, etc.
    """
    print(f"\n{'─' * 60}")
    print(f"  Location (City, State)")
    print(f"{'─' * 60}")

    # Sort by (state, city) so cities in the same state are grouped together.
    # WHY this sort order: Users often think "I'm in Texas" first, then scan
    # for their specific city. Grouping by state makes that flow natural.
    sorted_cities = sorted(CITIES, key=lambda pair: (pair[1], pair[0]))

    for i, (city, state) in enumerate(sorted_cities, 1):
        print(f"  {i:>3}. {city}, {state}")

    while True:
        raw = input(f"\n  Enter number (1-{len(sorted_cities)}): ").strip()
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(sorted_cities):
                return sorted_cities[idx]
        except ValueError:
            pass
        print("  Invalid choice, try again.")


def _pick_many(label: str, options: list[str], min_count: int, max_count: int) -> list[str]:
    """Present numbered options and return multiple choices.

    WHY min and max counts: Each field type has an expected cardinality.
    Hobbies should have at least 2 (to enable overlap), fan_of at least 2
    for the same reason. The max prevents unrealistic profiles (nobody has
    30 hobbies). These match what generate_profiles.py uses for synthetic data.
    """
    print(f"\n{'─' * 60}")
    print(f"  {label}  (pick {min_count}-{max_count})")
    print(f"{'─' * 60}")
    sorted_opts = sorted(options)

    # Print in 2 columns for readability with long lists.
    col_width = 35
    per_row = 2
    for i, opt in enumerate(sorted_opts, 1):
        end = "\n" if i % per_row == 0 else ""
        print(f"  {i:>3}. {opt:<{col_width}}", end=end)
    if len(sorted_opts) % per_row != 0:
        print()  # Close the last incomplete row.

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


# ---------------------------------------------------------------------------
# Profile builder
# ---------------------------------------------------------------------------

def build_profile_interactive() -> tuple[str, UserProfile]:
    """Walk the user through building a profile via terminal prompts.

    HOW THIS MAPS TO THE DATA MODEL:
    Every choice the user makes here maps to a field on UserProfile. The flow
    follows the same order as the scoring engine processes signals, so you can
    see how each choice will affect matching:
    - Name: for display only (not scored)
    - Age: feeds into age_compatibility (step-down ladder)
    - Location: feeds into location_match (tiered city > state > none)
    - College: feeds into college_match (exact match)
    - Degree/Job: for display only (not yet scored, future signal)
    - Faith: feeds into faith_match (exact match)
    - Hobbies: feeds into hobby_overlap (Jaccard similarity)
    - Interests: feeds into interest_overlap (Jaccard similarity)
    - Fan Of: feeds into fan_of_overlap (Jaccard similarity)
    - Travel: feeds into travel_overlap (Jaccard similarity)
    """
    print("\n" + "=" * 60)
    print("  HANGPOST PROFILE BUILDER")
    print("=" * 60)

    # ── Name (display only) ──
    name = input("\n  Your name: ").strip() or "Test User"

    # ── Age (scored: step-down ladder, 10% per year of gap) ──
    while True:
        age_raw = input("  Your age (18-65): ").strip()
        try:
            age = int(age_raw)
            if 18 <= age <= 65:
                break
        except ValueError:
            pass
        print("  Enter a number between 18 and 65.")

    # ── Location (scored: tiered city+state matching) ──
    # WHY combined: Prevents "Austin, Florida" mismatches. See _pick_city docs.
    city, state = _pick_city()

    # ── College (scored: exact match with other users' college) ──
    college = _pick_one("College", COLLEGES)

    # ── Degree and Job (display only — not yet used in scoring) ──
    # WHY still collected: These will likely become scoring signals in a future
    # version (e.g., people in the same field might connect). For now they
    # enrich the profile display.
    _pick_one("Degree", DEGREES)
    _pick_one("Job", JOBS)

    # ── Faith (scored: exact match) ──
    faith = _pick_one("Faith", FAITHS)

    # ── Hobbies (scored: Jaccard similarity) ──
    # Things you actively DO — hiking, chess, cooking, etc.
    hobbies = _pick_many("Hobbies (activities you do)", HOBBIES, 2, 8)

    # ── Interest categories (scored: Jaccard similarity) ──
    # Broad types/genres — Hip Hop, Japanese Food, Tech, etc.
    interests = _pick_many("Interest Categories (broad types you enjoy)", INTERESTS, 3, 7)

    # ── Fan Of (scored: Jaccard similarity) ──
    # Specific named things — Kendrick Lamar, The Bear, NFL, etc.
    fan_of = _pick_many("Fan Of (specific things you love)", FAN_OF, 2, 8)

    # ── Travel (scored: Jaccard similarity) ──
    travel = _pick_many("Travel Wishlist", TRAVEL_DESTINATIONS, 2, 4)

    # Build the UserProfile with all fields mapped to the new data model.
    # WHY lowercase: All set fields are lowercased so that "Hiking" in the
    # CLI matches "hiking" in the CSV data. Without this, Jaccard similarity
    # would miss obvious matches.
    profile = UserProfile(
        user_id="custom_profile",
        hobbies={h.lower() for h in hobbies},
        interests={i.lower() for i in interests},
        fan_of={f.lower() for f in fan_of},
        location=Location(city=city, state=state),
        age=age,
        mutual_friend_ids=set(),  # Custom profiles have no mutual friends yet.
        college=college,
        faith=faith,
        travel_wishlist={t.lower() for t in travel},
    )

    return name, profile


# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------

def display_results(
    name: str,
    profile: UserProfile,
    ranked: list,
    database_rows: dict[str, dict],
    top_n: int,
) -> None:
    """Print the top N matches with full profile details and score breakdown.

    WHY show the full breakdown: Users need to understand WHY each person
    was matched. A raw score of "0.65" means nothing — but seeing that
    hobby_overlap=0.8 and location_match=1.0 tells a story.
    """
    # Display the user's own location for context.
    loc_display = (
        f"{profile.location.city}, {profile.location.state}"
        if profile.location else "Unknown"
    )

    print(f"\n{'=' * 70}")
    print(f"  TOP {top_n} MATCHES FOR: {name}")
    print(f"  (age {profile.age}, {loc_display})")
    print(f"{'=' * 70}")

    for rank, (candidate, breakdown) in enumerate(ranked[:top_n], start=1):
        row = database_rows.get(candidate.user_id, {})
        cand_age = int(row.get("age", 0))
        age_gap = abs((profile.age or 0) - cand_age)

        print(f"\n  #{rank} — {row.get('name', '?')}  (Score: {breakdown.total_score:.3f})")
        print(f"  {'─' * 60}")
        # Location now shows city + state from the new CSV columns.
        print(f"  Age: {cand_age} (gap: {age_gap})  |  {row.get('city', '?')}, {row.get('state', '?')}")
        print(f"  College: {row.get('college', '?')}  |  Degree: {row.get('degree', '?')}")
        print(f"  Job: {row.get('job', '?')}  |  Faith: {row.get('faith', '?')}")
        print(f"  Hobbies:   {row.get('hobbies', '?')}")
        print(f"  Interests: {row.get('interests', '?')}")
        print(f"  Fan of:    {row.get('fan_of', '?')}")
        print(f"  Travel:    {row.get('travel', '?')}")
        print(f"  Mutual friends: {row.get('friends_in_common', '0')}")

        # Full score breakdown — all 9 signals + social boost.
        # This matches what compute_match_score() produces in scoring.py.
        print(f"  --- Score Breakdown ---")
        print(f"    Hobby overlap:     {breakdown.hobby_overlap:.3f}")
        print(f"    Interest overlap:  {breakdown.interest_overlap:.3f}")
        print(f"    Fan-of overlap:    {breakdown.fan_of_overlap:.3f}")
        print(f"    Mutual friends:    {breakdown.mutual_friends:.3f}  (boost: {breakdown.social_boost:.3f})")
        print(f"    Location match:    {breakdown.location_match:.3f}")
        print(f"    Age compatibility: {breakdown.age_compatibility:.3f}")
        print(f"    College match:     {breakdown.college_match:.3f}")
        print(f"    Faith match:       {breakdown.faith_match:.3f}")
        print(f"    Travel overlap:    {breakdown.travel_overlap:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Terminal-based profile builder.")
    parser.add_argument("--csv", default="data/test_profiles_10k.csv", help="CSV database to match against")
    parser.add_argument("--top", type=int, default=20, help="Number of top matches to show")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    print(f"Loading profiles from {csv_path}...")
    profiles = load_profiles(csv_path)

    # Keep the raw CSV rows so we can display full profile details in results.
    # WHY: UserProfile stores normalized scoring data (lowercase sets), but
    # for display we want the original formatted data from the CSV.
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
