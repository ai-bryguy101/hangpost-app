"""Pair-feature extraction for ML ranking.

Planned contents:
- extract_pair_features(a: UserProfile, b: UserProfile) -> dict[str, float]
  Deterministic features reused from scoring.py (hobby/interest/fan_of/travel
  Jaccard, Haversine, tiered location, age gap, mutual-friend count, college/
  faith exact match) plus future embedding cosine similarity.
"""
