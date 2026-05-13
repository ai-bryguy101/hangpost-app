# CLAUDE.md — Persistent Context for the Hangpost Matching Engine

This file is loaded automatically by Claude Code at the start of every session.
It captures non-obvious product context that must NOT be re-derived or guessed.

> **Before touching anything, read `PRODUCT_VISION.md`** in the repo root —
> that file describes what the *app* is (a location-scoped "make new
> friends in your city" social network) and grounds every design choice
> below. The rest of this file assumes you've read it.

---

## Product premise (read this before changing anything in `scoring.py`)

Hangpost is a **location-based social media app for making new friends within a
small radius**. The radius rule is a **hard pre-filter**, not a ranking signal.

### Two distinct concepts — do not conflate them

| Concept | What it is | Where it lives |
|---|---|---|
| **Current location** (real-time GPS / device location) | A **hard boundary**. Users only ever see profiles from people physically within a small radius of where they are right now. Profiles outside the radius are filtered out *before* the matching engine runs. | The radius filter is **upstream** of the matching engine. By the time `rank_candidates` runs, every candidate is already in-radius. **Distance is not a feature in the score.** Do not add Haversine distance, lat/lon decay, or any current-location signal to `compute_match_score`. |
| **Hometown** (where the user grew up) | A **soft matching signal**. Two users from the same hometown should rank higher because shared origin is a friendship cue. | Stored on `UserProfile.hometown`. Paired with `UserProfile.college` as a peer-strength signal — same college and same hometown are independent friendship cues with equal default weight, and a candidate can light up either, both, or neither. |

### Implications for any future work

- Do not propose, implement, or accept PRs that add real-time geographic
  distance to the matching score. That belongs in the candidate-retrieval layer
  (the database query / geo-index), not the ranker.
- When discussing "location-based" features, default to assuming the user means
  the **radius pre-filter** unless they explicitly say "hometown."
- If the matching engine ever needs to *know* the radius (e.g., for analytics),
  pass the already-filtered candidate list in. Never let the ranker re-check.

---

## Profile semantic representation — auto-synthesized, NOT user-written

Hangpost users do **not** write a free-text bio. Every profile's "semantic
representation" is built deterministically from the structured fields they
already provide (interests, liked topics, hometown, college, age, etc.).
That synthesized string is what gets embedded for the `semantic_similarity`
ranking signal.

- Implementation: `hangpost_matching.embeddings.profile_to_text(profile)`
  returns the natural-language string the embedder consumes. The matching
  engine never reads a user-authored bio.
- Do **not** add a `bio` field to `UserProfile`, do not assume users will
  hand-write bios, and do not surface "write your bio" UI in any spec.
- If a future feature needs more text per user (e.g., "what I'm looking for"),
  add it as a new structured prompt and extend `profile_to_text` — never as
  a generic `bio` blob.

---

## The 3-phase ML roadmap

1. **Phase 1 — Deterministic weighted scoring** *(done)*
   Rules + Jaccard overlaps + step-down age ladder + mutual-friend social boost.
2. **Phase 2 — Text embeddings** *(done)*
   `bio_similarity` via cosine similarity between sentence-transformer
   embeddings, integrated into the same weighted framework. The ranker
   itself is model-free — it accepts a precomputed `{user_id: vector}`
   map. The optional `[ml]` extra wires up `SentenceTransformerEmbedder`.
3. **Phase 3 — Supervised learning-to-rank** *(scaffold done; needs real outcome labels)*
   `hangpost_matching.learning.LearnedRanker` wraps a LightGBM `LGBMRanker`
   (LambdaRank objective). Features are the same components Phases 1+2
   produce. `scripts/train.py` fits the model on synthetic labels for now;
   swap in real outcome labels (accepts, chats started, retention) as soon
   as those become available.

### Evaluation harness *(done)*

`hangpost_matching.evaluation` implements precision@k, recall@k, MAP@k,
and NDCG@k, plus an `evaluate_ranker` that runs any `Ranker` over a
query set, plus `build_queries` / `split_queries` / `make_rules_ranker`
/ `make_random_ranker` so both `scripts/evaluate.py` and `scripts/train.py`
share one source of truth. `synthesize_relevance` provides a
deterministic stand-in ground truth from the structured fields until
real outcome data exists.

`ablate_weights` runs a per-feature ablation on the rules ranker —
each weight is zeroed in turn and the metric drop vs. the full-weights
baseline is reported. `scripts/evaluate.py --ablation` is the CLI entry
point. Use it to answer "which signal is actually carrying the ranker?"

### Test discipline

Heavy dependencies (`lightgbm`, `sentence-transformers`, `numpy`,
`joblib`, `fastapi`, `pydantic`, `uvicorn`) are confined to optional
extras (`[ml]`, `[serve]`) and imported lazily. CI installs only
`[dev]` and runs the full test suite against stubs that satisfy the
`Embedder` and `Predictor` Protocols; server tests skip themselves
when `fastapi` is not present. Real model + service behaviour is
covered by `scripts/train.py`, `scripts/evaluate.py`, the notebooks,
and the Docker image.

### Documentation artifacts

- `docs/MODEL_CARD.md` — intended use, factors, metrics, ethical caveats.
- `docs/DATA_CARD.md` — dataset schema, provenance, sensitive fields.
- `notebooks/01_eda.ipynb` — dataset exploration with plots.
- `notebooks/02_evaluation.ipynb` — Phase 1/2/3 head-to-head with plots.
- `Dockerfile` + `[serve]` extra — FastAPI deployment story.

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
