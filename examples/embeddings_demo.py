"""Phase 2 demo: rank candidates using semantic profile similarity.

This script actually loads `sentence-transformers/all-MiniLM-L6-v2` and
embeds each profile's auto-synthesized text. It is intentionally separate
from the lightweight `demo.py` so that the latter (and the test suite)
stay model-free and fast.

Hangpost users do NOT write a free-text bio. The string that gets
embedded is built deterministically from each user's structured fields
(interests, liked topics, hometown, age) by `profile_to_text`.

Run:
    pip install -e ".[ml]"
    python examples/embeddings_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from hangpost_matching import (  # noqa: E402
    SentenceTransformerEmbedder,
    UserProfile,
    embed_profiles,
    profile_to_text,
    rank_candidates,
)


def main() -> None:
    source = UserProfile(
        user_id="u0",
        interests={"hiking", "coding", "trail running", "espresso"},
        liked_topics={"tech", "travel", "mountains"},
        hometown="denver",
        age=28,
    )

    candidates = [
        UserProfile(
            user_id="u1",
            interests={"backend dev", "trail running", "coffee"},
            liked_topics={"tech", "travel"},
            hometown="denver",
            age=29,
        ),
        UserProfile(
            user_id="u2",
            interests={"poker", "casinos", "nightlife"},
            liked_topics={"vegas", "gambling"},
            hometown="denver",
            age=28,
        ),
        UserProfile(
            user_id="u3",
            interests={"mountain biking", "climbing", "alpinism", "coffee"},
            liked_topics={"mountains", "outdoors"},
            hometown="denver",
            age=27,
        ),
    ]

    print("Synthesized profile text (this is what gets embedded):")
    print("-" * 80)
    for profile in [source, *candidates]:
        print(f"  {profile.user_id}: {profile_to_text(profile)}")
    print()

    print("Loading sentence-transformer model (first run may download weights)...")
    embedder = SentenceTransformerEmbedder()

    print("Computing profile embeddings...")
    embeddings = embed_profiles([source, *candidates], embedder)

    print("\nRanking with semantic_similarity enabled:")
    print("-" * 80)
    ranked = rank_candidates(source, candidates, profile_embeddings=embeddings)
    for rank, (profile, breakdown) in enumerate(ranked, start=1):
        print(
            f"{rank}. {profile.user_id} | total={breakdown.total_score:.3f} | "
            f"semantic_sim={breakdown.semantic_similarity:.3f}"
        )


if __name__ == "__main__":
    main()
