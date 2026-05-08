# Model Card — Hangpost Matching Engine

This card follows the structure of Mitchell et al., *"Model Cards for Model
Reporting"* (FAccT 2019).

## Model details

| | |
|---|---|
| **Name** | Hangpost Matching Engine |
| **Version** | 0.1.0 |
| **Model types** | Three rankers behind one `Ranker` Protocol: (1) deterministic weighted scoring, (2) rules + sentence-transformer semantic similarity, (3) supervised learning-to-rank with LightGBM `LGBMRanker` (LambdaRank objective) |
| **Embedding backbone** | `sentence-transformers/all-MiniLM-L6-v2` (384-dim, 90MB) |
| **License** | MIT |
| **Repository** | `github.com/ai-bryguy101/hangpost-app` |
| **Citation** | See `LICENSE` and the README's authorship section |

## Intended use

### Primary intended use

Recommend friend candidates for a Hangpost user from a **pre-filtered**
pool of profiles already known to be physically near the user (the radius
filter is enforced upstream by the candidate-retrieval layer — see
`CLAUDE.md`).

### Primary intended users

- Hangpost mobile/web clients calling a recommendation service.
- Internal dashboards inspecting the breakdown of why two users rank where
  they do (the engine emits component-level explanations via
  `MatchBreakdown`).

### Out-of-scope use

- **Real-time geographic distance** is intentionally excluded from
  scoring. Distance is a hard pre-filter, not a ranking signal. Code that
  re-introduces lat/lon decay into the ranker should be rejected at PR
  review.
- **Romantic/dating matching.** This engine targets platonic friendship
  cues (mutual friends, hobbies, hometown). It has not been designed,
  evaluated, or de-risked for romantic compatibility.
- **Decisions with material consequences** (lending, employment, housing,
  etc.). The model has no fairness audit and no calibrated probabilities.

## Factors

Subgroups likely to influence model behaviour:

- **Age.** The age-compatibility component is a 10%-per-year ladder that
  zeroes at gaps ≥10 years. Older users in mostly-younger geographies
  will see fewer in-lane candidates.
- **Hometown rarity.** The `location_match` signal rewards exact hometown
  matches. Users from common hometowns ("Boston", "New York") will see
  this signal fire more often than users from rare hometowns.
- **Mutual-friend density.** New users with no graph edges can never
  enter the social-boost lane and will rank below anyone with even one
  mutual friend, regardless of compatibility.
- **Interest vocabulary.** Jaccard overlap is sensitive to how interests
  are tokenized. Users with verbose lists ("hiking, mountain biking,
  rock climbing") have a different prior than users with terse ones
  ("outdoors").

## Metrics

The evaluation harness in `hangpost_matching.evaluation` reports:

- **precision@k** — fraction of top-k that are relevant
- **recall@k** — fraction of all relevant items in top-k
- **MAP@k** — Mean Average Precision over top-k
- **NDCG@k** — Normalized Discounted Cumulative Gain (binary relevance)

Macro-averaged across queries (each query weighted equally).

## Evaluation data

- **Source.** A 1000-profile synthetic CSV at `data/test_profiles.csv`
  with hometown, age, hobbies, interests, mutual-friend counts, etc.
- **Labels.** Synthetic — `synthesize_relevance` flags a candidate as
  relevant when ≥3 of 5 multi-signal thresholds fire (≥2 shared
  interests, ≥2 shared liked topics, same hometown, age gap ≤ 5,
  ≥1 mutual friend). The thresholding rule is *structurally different*
  from the ranker's continuous weighted score, so the metrics still
  measure ranking quality — but they are an admitted stand-in until
  real outcome data exists.
- **Splitting.** Train/test split by **source profile** (no leakage).
  Default 70/30 in `scripts/train.py`.

## Training data

Same as evaluation data above. The seed CSV is small (N=1000) and
synthetic, so the learned ranker can over-fit to the
`synthesize_relevance` rule. **Do not interpret training-set metrics
as upper bounds on production performance.**

## Quantitative analyses

Run `scripts/train.py --with-embeddings` to produce the
random / rules / rules+embeddings / learned comparison on the held-out
split. Numbers should be pasted into the README each time the harness
or features change.

The repo's CI does not run this comparison (it would require the `[ml]`
extra and a model download); reproduction is the responsibility of
whoever updates the README.

## Ethical considerations

- **Minors.** Hangpost's product premise is location-based discovery
  inside a small radius. Minor-safety controls (age gating, parental
  controls, separate matching pools) must be enforced **upstream** of
  this ranker. The ranker accepts whatever profiles it is given and
  does not, by itself, prevent under-age users being recommended to
  adult users.
- **Harassment / blocking.** Mutual-friend boosting can amplify
  pre-existing social cliques. The product layer should expose
  block/report controls and feed those signals into hard filters
  upstream of the ranker.
- **Transparency.** Every ranking decision is accompanied by a
  `MatchBreakdown` exposing every component score, so ranking
  decisions are individually explainable to an end user.
- **Fairness.** No subgroup-fairness audit has been performed. The
  synthetic evaluation set is not representative of any real population.

## Caveats and recommendations

- Replace synthetic labels with real outcome labels (accepts, chats
  started, retention) as soon as production data is available.
- Re-evaluate every release on a held-out query set. Track NDCG@10 and
  MAP@10 over time as a regression-detection signal.
- Re-train the LightGBM ranker on a cadence proportional to traffic
  growth (a static model decays as user behaviour shifts).
- Add a fairness audit when the user base is large enough to support
  subgroup splits (age bands, hometown rarity bands, mutual-friend
  density bands).
