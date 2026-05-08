# CLAUDE.md — Persistent Context for the Hangpost Matching Engine

This file is loaded automatically by Claude Code at the start of every session.
It captures non-obvious product context that must NOT be re-derived or guessed.

---

## Product premise (read this before changing anything in `scoring.py`)

Hangpost is a **location-based social media app for making new friends within a
small radius**. The radius rule is a **hard pre-filter**, not a ranking signal.

### Two distinct concepts — do not conflate them

| Concept | What it is | Where it lives |
|---|---|---|
| **Current location** (real-time GPS / device location) | A **hard boundary**. Users only ever see profiles from people physically within a small radius of where they are right now. Profiles outside the radius are filtered out *before* the matching engine runs. | The radius filter is **upstream** of the matching engine. By the time `rank_candidates` runs, every candidate is already in-radius. **Distance is not a feature in the score.** Do not add Haversine distance, lat/lon decay, or any current-location signal to `compute_match_score`. |
| **Hometown** (where the user grew up) | A **soft matching signal**. Two users from the same hometown should rank higher because shared origin is a friendship cue. | This is what the `UserProfile.location` field represents today — it is hometown, **not** current location. The field name is misleading and should eventually be renamed to `hometown`. |

### Implications for any future work

- Do not propose, implement, or accept PRs that add real-time geographic
  distance to the matching score. That belongs in the candidate-retrieval layer
  (the database query / geo-index), not the ranker.
- When discussing "location-based" features, default to assuming the user means
  the **radius pre-filter** unless they explicitly say "hometown."
- If the matching engine ever needs to *know* the radius (e.g., for analytics),
  pass the already-filtered candidate list in. Never let the ranker re-check.

---

## The 3-phase ML roadmap

1. **Phase 1 — Deterministic weighted scoring** *(current state)*
   Rules + Jaccard overlaps + step-down age ladder + mutual-friend social boost.
2. **Phase 2 — Text embeddings**
   Add `bio_similarity` from a sentence-transformer model as one more signal in
   the same weighted framework.
3. **Phase 3 — Supervised learning-to-rank**
   Once outcome labels (accepts, chats started, retention) exist, train a
   LightGBM or similar ranker that learns the weights from data.

---

## Code style preferences

- Beginner-friendly comments and docstrings (this is also a learning project).
- Explicit naming over clever shortcuts.
- Explainable score breakdowns (`MatchBreakdown`) over hidden "magic."
- Default to no comments unless WHY is non-obvious; never narrate WHAT.

## Resume context

This repo is being prepared as a portfolio piece for AI engineer roles.
Prioritize work that demonstrates ML/AI engineering practice:
real models, evaluation metrics (precision@k, NDCG@k), notebooks for EDA,
experiment tracking, model/data cards, and CI.

## Branch policy

- Active development branch: `claude/cool-cori-RgQiQ`
- Never push to a different branch without explicit user permission.
