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


def test_mutual_friends_get_social_boost_and_priority() -> None:
    source = UserProfile(
        user_id="source",
        interests={"hiking", "coding"},
        liked_topics={"tech", "travel"},
        location="austin",
        age=30,
        mutual_friend_ids={"f1", "f2", "f3"},
    )

    no_mutual_but_compatible = UserProfile(
        user_id="no_mutual",
        interests={"hiking", "coding"},
        liked_topics={"tech", "travel"},
        location="austin",
        age=30,
        mutual_friend_ids=set(),
    )

    has_mutual_less_compatible = UserProfile(
        user_id="with_mutual",
        interests={"gaming"},
        liked_topics={"esports"},
        location="seattle",
        age=39,
        mutual_friend_ids={"f2"},
    )

    scored_no_mutual = compute_match_score(source, no_mutual_but_compatible)
    scored_mutual = compute_match_score(source, has_mutual_less_compatible)

    assert scored_no_mutual.has_mutual_friends is False
    assert scored_mutual.has_mutual_friends is True
    assert scored_mutual.social_boost > 0.0

    ranked = rank_candidates(source, [no_mutual_but_compatible, has_mutual_less_compatible])
    assert ranked[0][0].user_id == "with_mutual"


def test_age_compatibility_uses_10_percent_step_down_per_year() -> None:
    source = UserProfile(user_id="source", age=30)

    same_age = UserProfile(user_id="same", age=30)
    one_year_apart = UserProfile(user_id="one", age=29)
    two_years_apart = UserProfile(user_id="two", age=28)
    ten_years_apart = UserProfile(user_id="ten", age=20)

    assert compute_match_score(source, same_age).age_compatibility == 1.0
    assert compute_match_score(source, one_year_apart).age_compatibility == 0.9
    assert compute_match_score(source, two_years_apart).age_compatibility == 0.8
    assert compute_match_score(source, ten_years_apart).age_compatibility == 0.0
