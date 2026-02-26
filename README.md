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
- This gives you a two-lane ranking system: (1) socially connected candidates first, then (2) everyone else ranked by compatibility signals.

## Test data

A seed CSV with profile records for ranking experiments is available at:

- `data/test_profiles.csv`

Columns are ordered by your stated priority: friends in common, age closeness, college, hometown, degree, job, homestate, hobbies/activities/sports/games/skills/certifications, interests/likes, fan-of categories, faith/religion, and travel.

Mutual friends now act as a separate high-priority ranking signal: profiles with friend overlap receive a social boost and are sorted ahead of profiles with no friend overlap.
