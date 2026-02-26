# Session Context — Hangpost Matching Engine

_Last updated: 2026-02-26_

## What we built

We created and iterated on a **Phase 1 deterministic matching engine** for friend recommendations.

Core package: `src/hangpost_matching`

- Data models:
  - `UserProfile`
  - `ScoringWeights`
  - `MatchBreakdown`
- Scoring/ranking functions:
  - `compute_match_score`
  - `rank_candidates`

## Current ranking behavior

The model uses a two-lane ranking policy:

1. Candidates **with mutual friends** are prioritized first.
2. Within each lane, candidates are sorted by `total_score` descending.

### Score components

- Interest overlap (Jaccard)
- Liked-topic overlap (Jaccard)
- Mutual-friend ratio (bounded)
- Location exact match
- Age compatibility
- Separate social boost (`friend_common_boost`) when mutual friends exist

### Age compatibility rule (important)

Age compatibility is currently a **step-down ladder**:

- Same age gap (0): `1.0`
- 1 year gap: `0.9`
- 2 year gap: `0.8`
- ...
- 10+ year gap: `0.0`

So it is expected to see `AgeComp = 0.000` when age difference is 10 or more years.

## Dataset and examples

- Seed dataset: `data/test_profiles.csv`
  - Expanded to 1,000 profiles
  - `friends_in_common` is intentionally uncommon (~5.8%)
- Example scripts:
  - `examples/demo.py`
  - `examples/random_sample_ranking.py`

The random sample script now prints debugging-friendly columns:

- source age
- candidate age (`CandAge`)
- age gap (`AgeGap`)
- age compatibility (`AgeComp`)

This helps quickly explain why some candidates show low/zero age contribution.

## Tests

Current test suite is in `tests/test_scoring.py` and verifies:

- Strong overlap scores higher than weak overlap
- Ranking order behavior
- Default weight priority shape
- Social boost behavior
- Age step-ladder values

## Documentation style preference

You asked for beginner-friendly code with lots of comments.

We updated core files with detailed comments/docstrings and added a README section reinforcing this style for future contributions.

## Git/push environment notes

In this environment, direct push to GitHub has been inconsistent/blocked depending on network and remote setup.

Working fallback we discussed:

- Use browser-accessible environments (e.g., Codespaces), or
- Use your Slack-connected VPS agent (clawdbot) as a relay to apply patches, run tests, commit, and push.

## Recommended next steps when resuming

1. Decide deployment workflow (Codespaces vs clawdbot relay) for smoother Git pushes.
2. Add configurable knobs for age ladder (e.g., decrement per year, floor threshold) in `ScoringWeights`.
3. Add a CSV-to-profile loader utility module so examples and future APIs share parsing logic.
4. Start collecting interaction outcomes to prepare for Phase 2/3 improvements.

