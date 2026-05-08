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
  - location (hometown — see "How location works" above)
  - age
  - mutual friend IDs
- `ScoringWeights` for configurable component weights.
- `compute_match_score` that combines:
  - interest overlap (Jaccard)
  - liked-topic overlap (Jaccard)
  - mutual-friend score
  - location score
  - age-gap score
  - **semantic similarity** (cosine similarity between sentence-transformer
    embeddings, computed from auto-synthesized profile text)
- `rank_candidates` returning sorted recommendations with full score breakdown.
- Unit tests for deterministic behavior, including embedding math and tie-breaking.

### Phase 2: semantic profile embeddings

Hangpost users do **not** write a free-text bio. Instead, every profile's
"semantic representation" is auto-built from the structured fields they
already provide (interests, liked topics, hometown, age) by
`profile_to_text`. That synthesized string is what gets embedded.

The ranker itself stays pure — it accepts a precomputed `{user_id: vector}`
map and performs cosine similarity in pure Python — so the core package has
no heavy dependencies. To produce real embeddings, install the `[ml]` extra
and use `SentenceTransformerEmbedder`:

```bash
pip install -e ".[ml]"
python examples/embeddings_demo.py
```

You can swap in any embedder (OpenAI, Cohere, a local model, etc.) by
implementing the small `Embedder` Protocol in `hangpost_matching.embeddings`.

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

1. ~~Add profile text embeddings (`bio_similarity`) to the score breakdown.~~ ✅ done in Phase 2.
2. Build an offline evaluation harness with synthetic relevance labels
   (precision@k, recall@k, NDCG@k) so future changes can be measured rather than guessed.
3. Add EDA notebooks (`notebooks/`) exploring the seed dataset:
   distributions, correlations between signals, embedding visualizations (UMAP/t-SNE).
4. Log recommendation outcomes (`shown`, `clicked`, `friend_request_sent`, `accepted`).
5. Phase 3: train a learning-to-rank model (e.g., LightGBM `LGBMRanker`) once label volume is sufficient.



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

