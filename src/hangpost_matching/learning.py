"""Phase 3: supervised learning-to-rank.

This module trains a LightGBM `LGBMRanker` on the same component features
the deterministic ranker uses, but learns the *weights* from labeled
query data instead of having a human pick them.

Design principles
-----------------
- The ranker contract is the same `Ranker` Protocol used by Phases 1 and 2,
  so the offline evaluation harness in `evaluation.py` works unchanged.
- Heavy dependencies (`lightgbm`, `joblib`) are imported lazily so the
  core package and the test suite stay light. They live in the `[ml]`
  extra alongside `numpy` and `sentence-transformers`.
- The `Predictor` Protocol lets tests inject a stub model without ever
  touching LightGBM. CI runs with the stub; real training happens in
  `scripts/train.py` against the `[ml]` install.

Why these features
------------------
We reuse the components already produced by `compute_match_score` so a
hiring manager can read both rankers side-by-side and see exactly what
each gets to learn from. Adding more features later (e.g., interaction
terms, profile age in app, time-of-day) is a matter of extending
`extract_features` — the rest of the harness doesn't change.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from .embeddings import Vector
from .evaluation import Query, Ranker
from .models import UserProfile
from .scoring import compute_match_score

# Feature order is fixed so saved models are stable across runs.
FEATURE_NAMES: tuple[str, ...] = (
    "interest_overlap",
    "liked_topic_overlap",
    "mutual_friends",
    "hometown_match",
    "college_match",
    "age_compatibility",
    "semantic_similarity",
    "has_mutual_friends",
)


def extract_features(
    source: UserProfile,
    candidate: UserProfile,
    profile_embeddings: Mapping[str, Vector] | None = None,
) -> list[float]:
    """Build a feature vector for a (source, candidate) pair.

    The features are exactly the components that go into the deterministic
    weighted score, plus a binary "has any mutual friends" flag. That's a
    deliberate choice: the learned ranker should be able to *match* the
    deterministic ranker by recovering its weights, and only beat it once
    it discovers signal the human-tuned weights miss.
    """
    breakdown = compute_match_score(source, candidate, profile_embeddings=profile_embeddings)
    return [
        breakdown.interest_overlap,
        breakdown.liked_topic_overlap,
        breakdown.mutual_friends,
        breakdown.hometown_match,
        breakdown.college_match,
        breakdown.age_compatibility,
        breakdown.semantic_similarity,
        1.0 if breakdown.has_mutual_friends else 0.0,
    ]


ModelArrayLike = Sequence[Sequence[float]] | Any


class Predictor(Protocol):
    """Anything with a `predict(X) -> Sequence[float]` method.

    LightGBM's `LGBMRanker` satisfies this naturally. Tests inject stubs
    that satisfy the same shape so they don't need lightgbm installed.
    """

    def predict(self, x: ModelArrayLike) -> Sequence[float]: ...


class LearnedRanker:
    """Phase 3: a learning-to-rank wrapper around an LGBMRanker.

    The model is trained per-query (LambdaRank objective by default) so it
    learns ordering, not absolute relevance. Ties between candidates are
    broken by the model's continuous score.
    """

    # Tags stamped into the saved payload so `load` can confirm it is reading
    # a file this class produced (a sanity check against wrong/old files —
    # NOT a security boundary; see `load`).
    MODEL_SIGNATURE = "hangpost_learned_ranker"
    MODEL_PAYLOAD_VERSION = 1

    def __init__(
        self,
        profile_embeddings: Mapping[str, Vector] | None = None,
        model: Predictor | None = None,
    ) -> None:
        self.profile_embeddings = profile_embeddings
        self._model = model

    @property
    def model(self) -> Predictor | None:
        return self._model

    def fit(
        self,
        queries: Iterable[Query],
        **lgbm_params: object,
    ) -> None:
        """Train an LGBMRanker on the given queries.

        Each query contributes one "group" — LightGBM's LambdaRank objective
        only compares pairs *within* a group, which is exactly the semantics
        of "rank candidates for this user." Queries with zero positives are
        skipped because they carry no learning signal.
        """
        try:
            from lightgbm import LGBMRanker
        except ImportError as exc:
            raise ImportError(
                'lightgbm is required to fit a LearnedRanker. Install with: pip install -e ".[ml]"'
            ) from exc

        feature_matrix: list[list[float]] = []
        labels: list[int] = []
        groups: list[int] = []

        for source, candidates, relevant in queries:
            query_features = [
                extract_features(source, candidate, self.profile_embeddings)
                for candidate in candidates
            ]
            query_labels = [1 if candidate.user_id in relevant else 0 for candidate in candidates]
            if not any(query_labels):
                continue
            feature_matrix.extend(query_features)
            labels.extend(query_labels)
            groups.append(len(query_features))

        if not groups:
            raise ValueError(
                "No queries with at least one positive label. "
                "Check synthesize_relevance / your labels."
            )

        defaults = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "n_estimators": 200,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 5,
            "verbosity": -1,
        }
        defaults.update(lgbm_params)
        model = LGBMRanker(**defaults)
        # Wrap features in a DataFrame so the model is fit with named columns;
        # predict() then receives DataFrames with the same columns and sklearn
        # stops complaining about feature-name mismatches.
        import pandas as pd

        x_frame: Any = pd.DataFrame(feature_matrix, columns=list(FEATURE_NAMES))
        model.fit(x_frame, labels, group=groups)
        self._model = model

    def rank(self, source: UserProfile, candidates: list[UserProfile]) -> list[str]:
        """Score candidates with the learned model and return user_ids in order."""
        if self._model is None:
            raise RuntimeError(
                "LearnedRanker has not been fit yet. Call .fit() or pass a model in."
            )
        if not candidates:
            return []
        feature_matrix = [
            extract_features(source, candidate, self.profile_embeddings) for candidate in candidates
        ]
        # Match the named-DataFrame shape used during fit so sklearn's feature
        # name check passes silently. Importing pandas lazily keeps the [dev]
        # install untouched — only [ml] consumers ever hit this code path.
        x_frame: Any
        x_for_predict: ModelArrayLike = feature_matrix
        try:
            import pandas as pd

            x_frame = pd.DataFrame(feature_matrix, columns=list(FEATURE_NAMES))
            if hasattr(self._model, "feature_name_"):
                x_for_predict = x_frame
        except ImportError:
            pass
        scores = self._model.predict(x_for_predict)
        if len(scores) != len(candidates):
            raise ValueError(
                "Predictor returned a score count that does not match candidates: "
                f"{len(scores)} != {len(candidates)}"
            )
        ranked = sorted(
            zip(candidates, scores, strict=True),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return [candidate.user_id for candidate, _ in ranked]

    def as_ranker(self) -> Ranker:
        """Return the bound `rank` method as a plain `Ranker` callable."""
        return self.rank

    def save(self, path: str | Path) -> None:
        """Persist the model + embedding map with joblib."""
        try:
            import joblib
        except ImportError as exc:
            raise ImportError(
                'joblib is required to save a LearnedRanker. Install with: pip install -e ".[ml]"'
            ) from exc
        if self._model is None:
            raise RuntimeError("Cannot save a LearnedRanker that has not been fit.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "signature": self.MODEL_SIGNATURE,
            "payload_version": self.MODEL_PAYLOAD_VERSION,
            "model": self._model,
            "profile_embeddings": self.profile_embeddings,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        allowed_dir: str | Path | None = None,
    ) -> LearnedRanker:
        """Load a model + embedding map saved by `save`.

        SECURITY: this uses ``joblib`` which unpickles arbitrary Python —
        loading an untrusted file can execute arbitrary code. Only load
        models you produced or otherwise trust. ``allowed_dir`` is opt-in
        defense-in-depth (confine where models may live); it does not make
        loading a malicious file safe, since the signature/version tags
        below are read *after* unpickling has already run.
        """
        try:
            import joblib
        except ImportError as exc:
            raise ImportError(
                'joblib is required to load a LearnedRanker. Install with: pip install -e ".[ml]"'
            ) from exc
        model_path = Path(path).resolve()
        if allowed_dir is not None:
            allowed_root = Path(allowed_dir).resolve()
            if allowed_root not in model_path.parents and model_path != allowed_root:
                raise ValueError(
                    f"Refusing to load model outside allowlisted directory: {allowed_root}"
                )
        payload = joblib.load(model_path)
        if payload.get("signature") != cls.MODEL_SIGNATURE:
            raise ValueError("Model payload signature mismatch.")
        if payload.get("payload_version") != cls.MODEL_PAYLOAD_VERSION:
            raise ValueError("Model payload version mismatch.")
        return cls(
            profile_embeddings=payload["profile_embeddings"],
            model=payload["model"],
        )
