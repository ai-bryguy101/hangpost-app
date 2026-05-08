"""Optional FastAPI deployment of the matching engine.

This module is gated behind the `[serve]` extra so the core package and
its tests have zero web framework dependencies. Install and run with:

    pip install -e ".[ml,serve]"
    uvicorn hangpost_matching.server:app --host 0.0.0.0 --port 8000

Three modes selectable at startup via env vars:

    HANGPOST_MODE                — "rules" | "embeddings" | "learned"
                                   (default "rules")
    HANGPOST_LEARNED_MODEL_PATH  — path to a saved LearnedRanker
                                   (only used when mode="learned";
                                   default "models/learned_ranker.joblib")

The endpoint signature is intentionally tiny:

    POST /rank      — rank a list of candidates for one source profile
    GET  /healthz   — liveness probe
    GET  /          — describe the active mode
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .embeddings import Embedder, Vector
from .learning import Predictor
from .models import UserProfile
from .scoring import compute_match_score, rank_candidates

Mode = Literal["rules", "embeddings", "learned"]


# ---------- wire types ----------


class UserProfileIn(BaseModel):
    """JSON-friendly mirror of `UserProfile` (sets become lists on the wire)."""

    user_id: str
    interests: list[str] = Field(default_factory=list)
    liked_topics: list[str] = Field(default_factory=list)
    location: str | None = None
    age: int | None = None
    mutual_friend_ids: list[str] = Field(default_factory=list)

    def to_profile(self) -> UserProfile:
        return UserProfile(
            user_id=self.user_id,
            interests=set(self.interests),
            liked_topics=set(self.liked_topics),
            location=self.location,
            age=self.age,
            mutual_friend_ids=set(self.mutual_friend_ids),
        )


class RankRequest(BaseModel):
    source: UserProfileIn
    candidates: list[UserProfileIn]


class RankedItem(BaseModel):
    """One candidate's full explainable breakdown, ordered field-by-field."""

    user_id: str
    total_score: float
    has_mutual_friends: bool
    interest_overlap: float
    liked_topic_overlap: float
    mutual_friends: float
    location_match: float
    age_compatibility: float
    semantic_similarity: float


class RankResponse(BaseModel):
    mode: Mode
    results: list[RankedItem]


# ---------- runtime state ----------


@dataclass
class _State:
    """Resources loaded once per process at startup."""

    mode: Mode = "rules"
    embedder: Embedder | None = None
    learned_model: Predictor | None = None

    candidates_seen: int = field(default=0)
    requests_seen: int = field(default=0)


def _resolve_mode(value: str) -> Mode:
    if value not in ("rules", "embeddings", "learned"):
        raise ValueError(f"HANGPOST_MODE must be one of rules|embeddings|learned, got {value!r}")
    return value  # type: ignore[return-value]


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    state = _State(mode=_resolve_mode(os.environ.get("HANGPOST_MODE", "rules")))

    if state.mode in ("embeddings", "learned"):
        from .embeddings import SentenceTransformerEmbedder

        state.embedder = SentenceTransformerEmbedder()

    if state.mode == "learned":
        from .learning import LearnedRanker

        path = Path(os.environ.get("HANGPOST_LEARNED_MODEL_PATH", "models/learned_ranker.joblib"))
        if not path.exists():
            raise RuntimeError(
                f"HANGPOST_MODE=learned but no model at {path}. Run scripts/train.py first."
            )
        learned = LearnedRanker.load(path)
        if learned.model is None:
            raise RuntimeError(f"Loaded LearnedRanker at {path} has no fitted model")
        state.learned_model = learned.model

    app.state.matching = state
    try:
        yield
    finally:
        # Nothing to clean up — keeps the lifespan symmetric/explicit.
        pass


# ---------- FastAPI app ----------

app = FastAPI(
    title="Hangpost Matching Engine",
    version="0.1.0",
    description=(
        "Rank friend-recommendation candidates for one source profile. "
        "Supports rules-only, rules+embeddings, and a learned LightGBM ranker."
    ),
    lifespan=_lifespan,
)


@app.get("/")
def describe() -> dict[str, str]:
    state: _State = app.state.matching
    return {
        "service": "hangpost-matching",
        "version": app.version,
        "mode": state.mode,
    }


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


def _breakdown_to_item(user_id: str, breakdown: object) -> RankedItem:
    # `breakdown` is a `MatchBreakdown` dataclass; pull fields by name for
    # forward-compatibility with future component additions.
    return RankedItem(
        user_id=user_id,
        total_score=getattr(breakdown, "total_score"),  # noqa: B009
        has_mutual_friends=getattr(breakdown, "has_mutual_friends"),  # noqa: B009
        interest_overlap=getattr(breakdown, "interest_overlap"),  # noqa: B009
        liked_topic_overlap=getattr(breakdown, "liked_topic_overlap"),  # noqa: B009
        mutual_friends=getattr(breakdown, "mutual_friends"),  # noqa: B009
        location_match=getattr(breakdown, "location_match"),  # noqa: B009
        age_compatibility=getattr(breakdown, "age_compatibility"),  # noqa: B009
        semantic_similarity=getattr(breakdown, "semantic_similarity"),  # noqa: B009
    )


@app.post("/rank")
def rank(req: RankRequest) -> RankResponse:
    state: _State = app.state.matching
    state.requests_seen += 1
    state.candidates_seen += len(req.candidates)

    if not req.candidates:
        return RankResponse(mode=state.mode, results=[])

    source = req.source.to_profile()
    candidates = [c.to_profile() for c in req.candidates]

    # Phase 2 / 3 modes: embed the source + candidates on the fly.
    # Production deployments would precompute and cache these in a vector
    # store; the per-request embedding here is for demo simplicity.
    embeddings: Mapping[str, Vector] | None = None
    if state.embedder is not None:
        from .embeddings import embed_profiles

        embeddings = embed_profiles([source, *candidates], state.embedder)

    if state.mode == "learned":
        if state.learned_model is None:
            raise HTTPException(status_code=500, detail="learned model not loaded")
        # Rebuild a LearnedRanker bound to this request's embeddings.
        from .learning import LearnedRanker

        runtime = LearnedRanker(profile_embeddings=embeddings, model=state.learned_model)
        ordered_ids = runtime.rank(source, candidates)
        breakdowns = {
            candidate.user_id: compute_match_score(source, candidate, profile_embeddings=embeddings)
            for candidate in candidates
        }
        results = [_breakdown_to_item(uid, breakdowns[uid]) for uid in ordered_ids]
    else:
        ranked = rank_candidates(source, candidates, profile_embeddings=embeddings)
        results = [_breakdown_to_item(p.user_id, b) for p, b in ranked]

    return RankResponse(mode=state.mode, results=results)
