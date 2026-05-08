# Hangpost Matching Engine (MVP)

This repository starts with a **transparent profile matching and ranking engine**.

## Recommendation on your 3-phase approach

Your phased plan is strong and practical:

1. **Phase 1 (weighted scoring) is the right MVP**
   - Fast to ship.
   - Fully explainable.
   - Easy to tune with product/domain input.
2. **Phase 2 (text embeddings) is the best first AI upgrade**
   - Captures semantic similarity in bios/interests.
   - Integrates as one additional score in the same weighted framework.
3. **Phase 3 (supervised ML) should wait until you have outcome data**
   - Use accepted requests, chat starts, retention, etc. as labels.

In short: start with rules + math, then add AI as a feature, then let ML optimize once data exists.

---

## Project goals for this repo

- Build a deterministic ranking engine for friend recommendations.
- Keep scoring explainable through component-level breakdowns.
- Support hard constraints (dealbreakers) and soft preferences (weights).

## How location works in Hangpost (important)

Hangpost is a location-based app, but **physical distance is not a ranking signal**.

- **Current location (real-time)** is a *hard pre-filter*: the app only ever
  shows users profiles within a small radius of where they are right now.
  Profiles outside the radius are removed before the matching engine runs.
  By the time `rank_candidates` is called, every candidate is already
  in-radius, so the ranker does **not** know or care about physical distance.
- **Hometown** is a *soft matching signal*: two users from the same hometown
  rank higher because shared origin is a friendship cue. The `location` field
  on `UserProfile` represents hometown today, not current location.

This separation keeps the matching engine focused on compatibility, while the
upstream candidate-retrieval layer (database / geo-index) enforces the radius.

## Current implementation

- `UserProfile` model with structured features:
  - interests
  - liked_topics
  - location
  - age
  - mutual friend IDs
- `ScoringWeights` for configurable component weights.
- `compute_match_score` that combines:
  - interest overlap (Jaccard)
  - liked-topic overlap (Jaccard)
  - mutual-friend score
  - location score
  - age-gap score
- `rank_candidates` returning sorted recommendations with full score breakdown.
- Unit tests for deterministic behavior.

---

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest
python examples/demo.py
```

---



### Run a random 10-profile ranking sample

In Codespaces (or any shell), you can quickly run a random ranking experiment from the CSV:

```bash
python examples/random_sample_ranking.py --sample-size 10
```

Use a seed for reproducible results:

```bash
python examples/random_sample_ranking.py --sample-size 10 --seed 42
```


If AgeComp looks low/zero in a run, use a fixed seed and inspect the printed `CandAge` + `AgeGap` columns.
A gap of 10+ years now intentionally maps to `AgeComp = 0.0` in the current ladder rule.

## Suggested next steps

1. Add profile text embeddings (`bio_embedding_similarity`) to the score breakdown.
2. Log recommendation outcomes (`shown`, `clicked`, `friend_request_sent`, `accepted`).
3. Add online/offline evaluation metrics:
   - precision@k
   - acceptance rate@k
4. Later: train a learning-to-rank model once label volume is sufficient.



## Ranking behavior update

To align with your product direction:

- Most profiles will have **no mutual friends** and are ranked by the standard weighted criteria.
- Profiles that **do** have mutual friends receive a distinct `friend_common_boost` and are prioritized in final sorting.
- Age compatibility now uses a sequential step-down ladder: same age = 1.0, 1 year apart = 0.9, 2 years = 0.8, ... 10+ years = 0.0.
- This gives you a two-lane ranking system: (1) socially connected candidates first, then (2) everyone else ranked by compatibility signals.

## Test data

A seed CSV with profile records for ranking experiments is available at:

- `data/test_profiles.csv`

Columns are ordered by your stated priority: friends in common, age closeness, college, hometown, degree, job, homestate, hobbies/activities/sports/games/skills/certifications, interests/likes, fan-of categories, faith/religion, and travel.

Mutual friends now act as a separate high-priority ranking signal: profiles with friend overlap receive a social boost and are sorted ahead of profiles with no friend overlap.


## Learning-friendly code style

This project is intentionally written with beginner-friendly comments and explicit naming so you can learn as you read.

When adding new code, prefer:
- clear docstrings
- step-by-step comments for non-obvious logic
- explainable score breakdowns over hidden “magic” behavior

