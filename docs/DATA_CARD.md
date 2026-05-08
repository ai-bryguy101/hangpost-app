# Data Card — `data/test_profiles.csv`

This card follows the structure of Pushkarna et al., *"Data Cards"*
(FAccT 2022), abbreviated for a small synthetic dataset.

## Dataset overview

| | |
|---|---|
| **Name** | `test_profiles.csv` |
| **Version** | 0.1.0 |
| **Records** | 1,000 profiles |
| **Format** | UTF-8 CSV with header row |
| **Origin** | Fully synthetic — generated for prototype/testing only |
| **License** | MIT (same as the repository) |

## Why this dataset exists

To exercise the matching engine end-to-end without exposing real users.
It is *not* a benchmark, a held-out test set for production claims, or a
representative sample of any real population.

## Schema

| Column | Type | Description | Used by ranker? |
|---|---|---|---|
| `name` | string | Display name (synthetic) | No (only for `user_id` synthesis in `data.py`) |
| `friends_in_common` | int | Count of synthetic mutual friends | Yes — converted to `mutual_friend_ids` set |
| `age` | int | Profile age in years | Yes — `age_compatibility` |
| `college` | string | College attended | No (reserved for future) |
| `hometown` | string | Hometown city | Yes — `location_match` (treated as hometown, not current location) |
| `degree` | string | Academic degree | No (reserved) |
| `job` | string | Job title | No (reserved) |
| `homestate` | string | Home US state | No (reserved) |
| `hobbies_activities_sports_games_skills_certifications` | string | Semicolon-separated tokens | Yes — `interests` Jaccard |
| `interests_likes` | string | Semicolon-separated tokens | Yes — `liked_topics` Jaccard |
| `fan_of` | string | Sports / shows / artists | No (reserved) |
| `faith_religion` | string | Religious affiliation | No (sensitive, deliberately not used) |
| `travel` | string | Travel destinations | No (reserved) |

## Provenance

- **Generation method.** Hand-designed templates over a fixed token
  vocabulary (e.g., a list of plausible hobbies, hometowns, jobs).
- **No real personal data.** Every name, biography, and friend count is
  synthetic.
- **No web scraping.** Nothing was ingested from social networks or
  third-party APIs.

## Distribution highlights

These figures are produced and visualised in `notebooks/01_eda.ipynb`.
Re-run the notebook after dataset changes to refresh.

- Age range covers young adults through middle age; mean roughly in the
  late 20s / early 30s (varies by sampling).
- `friends_in_common` is intentionally rare (~5–6% of profiles have ≥1)
  so the social-boost lane stays a *high-precision, low-volume* signal.
- Hometown column has a long-tailed distribution — a handful of common
  cities account for many profiles, with a long tail of rarer ones.

## Sensitive attributes

The CSV includes `faith_religion`. The matching engine **does not** use
this field today. Adding it to the ranker should require:
1. A fairness review.
2. An explicit user-facing opt-in.
3. A test that holding it out does not silently regress.

## Known limitations

- **Synthetic = mis-calibrated.** Token frequencies and signal
  correlations are whatever the generator produced; they may not match
  any real population.
- **No interaction history.** There are no `shown / clicked / accepted`
  events. Synthetic relevance labels stand in until production data
  exists (see `synthesize_relevance` in `hangpost_matching.evaluation`).
- **Static.** The CSV is not time-stamped and does not model user growth
  or behaviour drift.
- **Small.** N=1000 is fine for unit tests and EDA but small for
  fitting a LightGBM ranker without over-fitting.

## Recommended use

- Demos, examples, smoke tests, EDA notebooks, CI for the parsing path.
- A scaffolded training run to verify `scripts/train.py` works
  end-to-end before pointing it at production data.

## Discouraged use

- Reporting any quality claim (precision, NDCG, etc.) without
  explicitly noting that the labels are synthetic.
- Any production decision making.
- Any fairness audit (the demographic distribution is whatever the
  generator produced and is not representative).
