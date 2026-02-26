from hangpost_matching import ScoringWeights, UserProfile, compute_match_score, rank_candidates


def test_compute_match_score_prefers_overlap() -> None:
    source = UserProfile(
        user_id="source",
        interests={"hiking", "coding"},
        liked_topics={"tech", "travel"},
        location="austin",
        age=30,
        mutual_friend_ids={"f1", "f2", "f3"},
    )
    strong = UserProfile(
        user_id="strong",
        interests={"hiking", "coding", "reading"},
        liked_topics={"tech", "travel"},
        location="austin",
        age=29,
        mutual_friend_ids={"f2", "f3"},
    )
    weak = UserProfile(
        user_id="weak",
        interests={"chess"},
        liked_topics={"finance"},
        location="seattle",
        age=50,
        mutual_friend_ids=set(),
    )

    strong_score = compute_match_score(source, strong).total_score
    weak_score = compute_match_score(source, weak).total_score

    assert strong_score > weak_score


def test_rank_candidates_descending() -> None:
    source = UserProfile(user_id="source", interests={"a"}, liked_topics={"x"}, location="nyc", age=24)
    c1 = UserProfile(user_id="c1", interests={"a"}, liked_topics={"x"}, location="nyc", age=24)
    c2 = UserProfile(user_id="c2", interests={"b"}, liked_topics={"y"}, location="la", age=40)

    ranked = rank_candidates(source, [c2, c1])

    assert ranked[0][0].user_id == "c1"
    assert ranked[1][0].user_id == "c2"


def test_default_weights_prioritize_mutual_friends_then_age() -> None:
    weights = ScoringWeights()

    assert weights.mutual_friends > weights.age_compatibility
    assert weights.age_compatibility > weights.interest_overlap
