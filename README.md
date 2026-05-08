# Hangpost Matching Engine

A friend-recommendation engine for a location-based social app. Three rankers
behind one `Ranker` Protocol — each phase is measured against a held-out query
set, and the strongest one ships:

1. **Phase 1 — Rules.** Deterministic weighted scoring (Jaccard interest/topic
   overlap, mutual-friend ratio, hometown match, age-compatibility ladder)
   plus a separate "social-boost" lane for candidates with mutual friends.
2. **Phase 2 — Embeddings.** Adds `semantic_similarity` from
   `sentence-transformers/all-MiniLM-L6-v2`. The text that gets embedded is
   *auto-synthesized* from each user's structured fields by
   `profile_to_text` — Hangpost users never write a bio.
3. **Phase 3 — Learned.** A LightGBM `LGBMRanker` (LambdaRank objective)
   trained on the same features as Phase 1+2, learning the weights from
   labeled query data.

All three are evaluated by the same harness (`hangpost_matching.evaluation`)
which implements precision@k / recall@k / NDCG@k / MAP@k from scratch.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest                       # run the full test suite
python examples/demo.py      # rules-only ranking on a hand-built example
python scripts/evaluate.py   # rules vs random baseline on the seed CSV
```

To use the ML phases (sentence-transformers + LightGBM):

```bash
pip install -e ".[ml]"
python examples/embeddings_demo.py        # Phase 2: semantic similarity
python scripts/train.py --with-embeddings # Phase 3: train + held-out comparison
python scripts/evaluate.py --with-embeddings \
    --learned-model models/learned_ranker.joblib
```

To deploy as an HTTP service:

```bash
pip install -e ".[ml,serve]"
HANGPOST_MODE=embeddings uvicorn hangpost_matching.server:app --host 0.0.0.0 --port 8000
# or, with Docker:
docker build -t hangpost-matching .
docker run --rm -p 8000:8000 -e HANGPOST_MODE=embeddings hangpost-matching
```

## Architecture

```
                          ┌────────────────────────────────────┐
                          │   Candidate retrieval (upstream)   │
                          │   Hard radius pre-filter — geo     │
                          │   index, NOT a ranking signal.     │
                          └───────────────┬────────────────────┘
                                          │
                                          ▼  list[UserProfile]
                          ┌────────────────────────────────────┐
       profile_to_text ──▶│           Embedder (Phase 2)       │──▶ {user_id: vector}
       (structured →      │  SentenceTransformerEmbedder       │
        natural-lang)     │  / OpenAI / Cohere / your own      │
                          └───────────────┬────────────────────┘
                                          │
                                          ▼
   ┌────────────────────────────────────────────────────────────────────────┐
   │                            Ranker contract                             │
   │   def __call__(source, candidates) -> list[user_id]                    │
   ├──────────────────────────┬───────────────────────────┬─────────────────┤
   │  rules (Phase 1+2)       │   learned (Phase 3)       │  random base    │
   │  rank_candidates(...)    │   LearnedRanker.rank(...) │  shuffle(seed)  │
   └──────────────────────────┴───────────────────────────┴─────────────────┘
                                          │
                                          ▼
                          ┌────────────────────────────────────┐
                          │  evaluate_ranker — P@k R@k NDCG MAP│
                          └────────────────────────────────────┘
```

## How "location" works in Hangpost (important)

Hangpost is a location-based app, but **physical distance is not a ranking
signal**.

- **Current location (real-time)** is a *hard pre-filter*: the app only ever
  shows users profiles within a small radius of where they are right now.
  Profiles outside the radius are removed before the matching engine runs.
- **Hometown** is a *soft matching signal*: two users from the same hometown
  rank higher because shared origin is a friendship cue.

This separation keeps the matching engine focused on compatibility while the
upstream candidate-retrieval layer (database / geo-index) enforces the radius.

## Repository layout

```
src/hangpost_matching/
    models.py        UserProfile / ScoringWeights / MatchBreakdown
    scoring.py       compute_match_score, rank_candidates, two-lane sort
    embeddings.py    profile_to_text, cosine, Embedder Protocol
    evaluation.py    P@k, R@k, NDCG@k, MAP@k, query/ranker helpers
    learning.py      extract_features, LearnedRanker (LightGBM)
    data.py          shared CSV → UserProfile loader
    server.py        FastAPI deployment (gated on [serve] extra)
scripts/
    evaluate.py      compare random / rules / rules+embeddings / learned
    train.py         fit + evaluate + persist a LearnedRanker
examples/
    demo.py                 minimal rules-only ranking example
    embeddings_demo.py      Phase 2 with a real sentence-transformer
    random_sample_ranking.py  ad-hoc ranking on a CSV sample
notebooks/
    01_eda.ipynb            dataset exploration with plots
    02_evaluation.ipynb     phase 1/2/3 head-to-head comparison
docs/
    MODEL_CARD.md           intended use, factors, ethical considerations
    DATA_CARD.md            schema, provenance, sensitive attributes
tests/                      ruff + mypy + 50+ pytest tests
```

## Offline evaluation

`scripts/evaluate.py` compares random / rules-only / rules+embeddings /
learned on a query set drawn from the seed CSV. Synthetic relevance labels
come from `synthesize_relevance` (≥3 of 5 multi-signal thresholds) and are an
admitted stand-in until real outcome data exists — see `docs/MODEL_CARD.md`.

Sample output (50 queries, k=10, no `[ml]` extra):

```
System                     P@10     R@10   NDCG@10    MAP@10
--------------------------------------------------------------
random                    0.196    0.013     0.188     0.079
rules_only                0.982    0.067     0.988     0.979
```

The full Phase 1 / 2 / 3 numbers (and accompanying plots) live in
`notebooks/02_evaluation.ipynb` — run with the `[ml]` extra installed.

## Test discipline

Heavy ML dependencies (`numpy`, `sentence-transformers`, `lightgbm`,
`joblib`, `fastapi`) are confined to optional extras and imported lazily.
CI installs only `[dev]` and runs the full test suite against stubs
satisfying the `Embedder` and `Predictor` Protocols. Real model behaviour is
exercised by `scripts/train.py`, `scripts/evaluate.py`, the notebooks, and
the Docker image — none of which the test suite touches.

## License

MIT — see `LICENSE`.
