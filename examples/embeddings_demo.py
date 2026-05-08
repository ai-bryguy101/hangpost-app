"""Phase 2 demo: rank candidates using semantic bio similarity.

This script actually loads `sentence-transformers/all-MiniLM-L6-v2` and
embeds each profile's bio. It is intentionally separate from the lightweight
`demo.py` so that the latter (and the test suite) stay model-free and fast.

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
    rank_candidates,
)


def main() -> None:
    source = UserProfile(
        user_id="u0",
        interests={"hiking", "coding"},
        liked_topics={"tech", "travel"},
        location="denver",
        age=28,
        bio="Software engineer who unwinds on long mountain hikes and weekend espresso runs.",
    )

    candidates = [
        UserProfile(
            user_id="u1",
            interests={"hiking", "coding"},
            liked_topics={"tech", "travel"},
            location="denver",
            age=29,
            bio="Backend developer, trail runner, and amateur barista.",
        ),
        UserProfile(
            user_id="u2",
            interests={"hiking", "coding"},
            liked_topics={"tech", "travel"},
            location="denver",
            age=28,
            bio="Casino regular who plays poker every weekend and loves Vegas nightlife.",
        ),
        UserProfile(
            user_id="u3",
            interests={"hiking", "coding"},
            liked_topics={"tech", "travel"},
            location="denver",
            age=27,
            bio="Mountain biker and climber chasing alpine summits and good coffee.",
        ),
    ]

    print("Loading sentence-transformer model (first run may download weights)...")
    embedder = SentenceTransformerEmbedder()

    print("Computing bio embeddings...")
    embeddings = embed_profiles([source, *candidates], embedder)

    print("\nRanking with bio_similarity enabled:")
    print("-" * 80)
    ranked = rank_candidates(source, candidates, bio_embeddings=embeddings)
    for rank, (profile, breakdown) in enumerate(ranked, start=1):
        print(
            f"{rank}. {profile.user_id} | total={breakdown.total_score:.3f} | "
            f"bio_sim={breakdown.bio_similarity:.3f} | bio: {profile.bio}"
        )


if __name__ == "__main__":
    main()
