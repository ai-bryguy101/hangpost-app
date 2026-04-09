"""Minimal demo of the hangpost matching engine.

Creates two hand-crafted profiles and runs the scoring algorithm.
Useful for quick sanity checks and understanding how the API works.

Usage:
    python examples/demo.py

WHAT CHANGED AND WHY (v0.2.0):
- Profiles now use the new 3-field taxonomy (hobbies, interests, fan_of)
  instead of the old 2-field model (interests, liked_topics).
  WHY: The old model mixed activities, categories, and specific fandoms into
  two vague buckets. The new model gives each type its own field so the
  algorithm can score them independently.

- Location is now a Location(city, state) pair instead of a bare string.
  WHY: Enables tiered scoring — same city is worth more than same state,
  which is worth more than different states.

- New fields: college, faith, travel_wishlist are now included.
  WHY: These were in the CSV data but never scored. Now they contribute
  to the match score (college and faith as exact match, travel as Jaccard).
"""

from hangpost_matching import Location, UserProfile, rank_candidates


def main() -> None:
    # ── Source profile ──
    # This is the person we're finding matches FOR.
    source = UserProfile(
        user_id="u0",

        # Hobbies: things this person actively DOES.
        hobbies={"hiking", "coding", "cooking"},

        # Interests: broad categories/genres they enjoy.
        interests={"tech", "outdoor adventure", "japanese food"},

        # Fan Of: specific things they love.
        fan_of={"kendrick lamar", "the bear", "zelda"},

        # Location: city+state pair for tiered scoring.
        location=Location(city="Denver", state="Colorado"),

        age=28,

        # Mutual friends: IDs of shared connections.
        # In a real app, these come from the social graph.
        mutual_friend_ids={"a", "b", "c", "d"},

        # New fields that now feed into scoring.
        college="University of Colorado Boulder",
        faith="Agnostic",
        travel_wishlist={"japan", "iceland", "portugal"},
    )

    candidates = [
        # ── Candidate 1: Strong match ──
        # Same city, similar age, overlapping hobbies and fandoms,
        # shares mutual friends. Should score high.
        UserProfile(
            user_id="u1",
            hobbies={"hiking", "coding", "photography"},
            interests={"tech", "outdoor adventure", "film/tv"},
            fan_of={"kendrick lamar", "the bear", "marvel"},
            location=Location(city="Denver", state="Colorado"),
            age=27,
            mutual_friend_ids={"b", "c", "x"},
            college="University of Colorado Boulder",
            faith="Agnostic",
            travel_wishlist={"japan", "iceland", "spain"},
        ),

        # ── Candidate 2: Weak match ──
        # Different city/state, big age gap, no overlap on hobbies or fandoms,
        # only 1 mutual friend. Should score low but still get mutual-friend
        # lane priority because has_mutual_friends=True.
        UserProfile(
            user_id="u2",
            hobbies={"basketball", "video games"},
            interests={"hip hop", "team sports"},
            fan_of={"nba", "drake", "formula 1"},
            location=Location(city="Miami", state="Florida"),
            age=37,
            mutual_friend_ids={"z"},
            college="University of Miami",
            faith="Christian",
            travel_wishlist={"brazil", "colombia"},
        ),
    ]

    # Run the matching algorithm and print results.
    # rank_candidates returns (UserProfile, MatchBreakdown) pairs sorted
    # by the two-lane strategy: mutual friends first, then by score.
    for profile, breakdown in rank_candidates(source, candidates):
        print(f"\n{'=' * 50}")
        print(f"  {profile.user_id}  — Total Score: {breakdown.total_score:.3f}")
        print(f"{'=' * 50}")
        print(f"  Hobby overlap:     {breakdown.hobby_overlap:.3f}")
        print(f"  Interest overlap:  {breakdown.interest_overlap:.3f}")
        print(f"  Fan-of overlap:    {breakdown.fan_of_overlap:.3f}")
        print(f"  Mutual friends:    {breakdown.mutual_friends:.3f}")
        print(f"  Social boost:      {breakdown.social_boost:.3f}")
        print(f"  Location match:    {breakdown.location_match:.3f}")
        print(f"  Age compatibility: {breakdown.age_compatibility:.3f}")
        print(f"  College match:     {breakdown.college_match:.3f}")
        print(f"  Faith match:       {breakdown.faith_match:.3f}")
        print(f"  Travel overlap:    {breakdown.travel_overlap:.3f}")
        print(f"  Has mutual friends: {breakdown.has_mutual_friends}")


if __name__ == "__main__":
    main()
