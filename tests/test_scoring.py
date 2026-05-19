from hangpost_matching import ScoringWeights, UserProfile, compute_match_score, rank_candidates


def test_compute_match_score_prefers_overlap() -> None:
    source = UserProfile(
        user_id="source",
        interests={"hiking", "coding"},
        liked_topics={"tech", "travel"},
        hometown="austin",
        age=30,
        mutual_friend_ids={"f1", "f2", "f3"},
    )
    strong = UserProfile(
        user_id="strong",
        interests={"hiking", "coding", "reading"},
        liked_topics={"tech", "travel"},
        hometown="austin",
        age=29,
        mutual_friend_ids={"f2", "f3"},
    )
    weak = UserProfile(
        user_id="weak",
        interests={"chess"},
        liked_topics={"finance"},
        hometown="seattle",
        age=50,
        mutual_friend_ids=set(),
    )

    strong_score = compute_match_score(source, strong).total_score
    weak_score = compute_match_score(source, weak).total_score

    assert strong_score > weak_score


def test_rank_candidates_descending() -> None:
    source = UserProfile(
        user_id="source", interests={"a"}, liked_topics={"x"}, hometown="nyc", age=24
    )
    c1 = UserProfile(user_id="c1", interests={"a"}, liked_topics={"x"}, hometown="nyc", age=24)
    c2 = UserProfile(user_id="c2", interests={"b"}, liked_topics={"y"}, hometown="la", age=40)

    ranked = rank_candidates(source, [c2, c1])

    assert ranked[0][0].user_id == "c1"
    assert ranked[1][0].user_id == "c2"


def test_default_weights_prioritize_mutual_friends_then_age() -> None:
    weights = ScoringWeights()

    assert weights.mutual_friends > weights.age_compatibility
    assert weights.age_compatibility > weights.interest_overlap


def test_hometown_and_college_are_peer_strength_signals() -> None:
    """Same-hometown and same-college should contribute equally and independently."""
    weights = ScoringWeights()
    assert weights.hometown_match == weights.college_match

    source = UserProfile(user_id="source", hometown="boston", college="bu")
    hometown_only = UserProfile(user_id="ht", hometown="boston", college="harvard")
    college_only = UserProfile(user_id="co", hometown="atlanta", college="bu")
    both = UserProfile(user_id="both", hometown="boston", college="bu")
    neither = UserProfile(user_id="none", hometown="nyc", college="nyu")

    hometown_score = compute_match_score(source, hometown_only)
    college_score = compute_match_score(source, college_only)
    both_score = compute_match_score(source, both)
    none_score = compute_match_score(source, neither)

    assert hometown_score.hometown_match == 1.0
    assert hometown_score.college_match == 0.0
    assert college_score.hometown_match == 0.0
    assert college_score.college_match == 1.0
    assert hometown_score.total_score == college_score.total_score
    assert both_score.total_score > hometown_score.total_score
    assert hometown_score.total_score > none_score.total_score


def test_mutual_friends_get_social_boost_and_priority() -> None:
    source = UserProfile(
        user_id="source",
        interests={"hiking", "coding"},
        liked_topics={"tech", "travel"},
        hometown="austin",
        age=30,
        mutual_friend_ids={"f1", "f2", "f3"},
    )

    no_mutual_but_compatible = UserProfile(
        user_id="no_mutual",
        interests={"hiking", "coding"},
        liked_topics={"tech", "travel"},
        hometown="austin",
        age=30,
        mutual_friend_ids=set(),
    )

    has_mutual_less_compatible = UserProfile(
        user_id="with_mutual",
        interests={"gaming"},
        liked_topics={"esports"},
        hometown="seattle",
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


def test_shared_background_lane_beats_tier_3_regardless_of_score() -> None:
    """Tier-2 candidate (shared hometown OR college) must outrank tier-3 candidate
    with much higher hobby/age compatibility — the lane is a hard invariant.
    """
    source = UserProfile(
        user_id="source",
        interests={"hiking", "coding", "cooking", "yoga"},
        liked_topics={"tech", "travel", "music"},
        hometown="boston",
        college="bu",
        age=30,
        mutual_friend_ids=set(),
    )

    # Tier 2: no mutuals, same hometown only, but completely mismatched on
    # hobbies and age.
    shared_background = UserProfile(
        user_id="shared_bg",
        interests={"opera"},
        liked_topics={"taxidermy"},
        hometown="boston",
        college="ucla",  # different college on purpose
        age=55,  # large age gap on purpose
        mutual_friend_ids=set(),
    )

    # Tier 3: no mutuals, different hometown AND college, but perfect on
    # every other signal.
    perfect_tier3 = UserProfile(
        user_id="perfect_tier3",
        interests={"hiking", "coding", "cooking", "yoga"},
        liked_topics={"tech", "travel", "music"},
        hometown="seattle",
        college="uw",
        age=30,
        mutual_friend_ids=set(),
    )

    ranked = rank_candidates(source, [perfect_tier3, shared_background])

    assert ranked[0][0].user_id == "shared_bg"
    assert ranked[0][1].has_shared_background is True
    assert ranked[1][0].user_id == "perfect_tier3"
    assert ranked[1][1].has_shared_background is False
    # Sanity: tier-3 candidate has a strictly higher weighted score, the
    # lane invariant is what's putting tier-2 on top.
    assert ranked[1][1].total_score > ranked[0][1].total_score


def test_mutual_friends_lane_beats_shared_background_lane() -> None:
    """Tier 1 still wins over tier 2: a candidate with a mutual friend but
    nothing else in common outranks a candidate with shared background and
    high compatibility.
    """
    source = UserProfile(
        user_id="source",
        interests={"hiking"},
        hometown="boston",
        college="bu",
        age=30,
        mutual_friend_ids={"f1"},
    )

    tier1_only = UserProfile(
        user_id="tier1",
        interests={"opera"},
        hometown="atlanta",
        college="emory",
        age=55,
        mutual_friend_ids={"f1"},
    )
    tier2_high_score = UserProfile(
        user_id="tier2",
        interests={"hiking"},
        hometown="boston",
        college="bu",
        age=30,
        mutual_friend_ids=set(),
    )

    ranked = rank_candidates(source, [tier2_high_score, tier1_only])

    assert ranked[0][0].user_id == "tier1"
    assert ranked[0][1].has_mutual_friends is True
    assert ranked[1][0].user_id == "tier2"
    assert ranked[1][1].has_mutual_friends is False
    assert ranked[1][1].has_shared_background is True


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


# ---------- six-tier sort: hometown+college is a hard tier over either-alone ----------


def test_both_background_outranks_single_background_even_with_weaker_hobbies() -> None:
    """Same hometown AND same college must beat same hometown alone — even
    when the hometown-only candidate has dramatically stronger hobby
    overlap. The both-match is a HARD tier, not a score nudge."""
    source = UserProfile(
        user_id="source",
        interests={"hiking", "coding", "music"},
        liked_topics={"travel"},
        hometown="austin",
        college="ut",
        age=30,
    )
    both_match = UserProfile(
        user_id="both",
        interests=set(),  # zero hobby overlap
        liked_topics=set(),
        hometown="austin",  # matches
        college="ut",  # matches
        age=99,  # 0 age compatibility
    )
    hometown_only_with_great_hobbies = UserProfile(
        user_id="hometown_only",
        interests={"hiking", "coding", "music"},  # perfect hobby overlap
        liked_topics={"travel"},
        hometown="austin",
        college="harvard",  # doesn't match
        age=30,  # perfect age
    )
    ranked = rank_candidates(source, [hometown_only_with_great_hobbies, both_match])
    assert ranked[0][0].user_id == "both"
    assert ranked[1][0].user_id == "hometown_only"


def test_breakdown_flags_both_shared_background_only_when_both_match() -> None:
    source = UserProfile(user_id="src", hometown="austin", college="ut")
    both = UserProfile(user_id="both", hometown="austin", college="ut")
    hometown_only = UserProfile(user_id="ho", hometown="austin", college="harvard")
    college_only = UserProfile(user_id="co", hometown="seattle", college="ut")
    neither = UserProfile(user_id="ne", hometown="seattle", college="harvard")

    assert compute_match_score(source, both).has_both_shared_background is True
    assert compute_match_score(source, hometown_only).has_both_shared_background is False
    assert compute_match_score(source, college_only).has_both_shared_background is False
    assert compute_match_score(source, neither).has_both_shared_background is False

    # Sanity: has_shared_background still True for single-signal matches.
    assert compute_match_score(source, hometown_only).has_shared_background is True
    assert compute_match_score(source, college_only).has_shared_background is True
    assert compute_match_score(source, neither).has_shared_background is False


def test_mutual_friend_plus_both_backgrounds_plus_same_age_is_absolute_top() -> None:
    """The 'perfect match' invariant from the product spec: a candidate
    with a mutual friend, same hometown, same college, and the same age
    is the highest-ranked result. Other Lane A candidates without the
    background match — even with many mutual friends and great hobbies —
    must rank below."""
    source = UserProfile(
        user_id="source",
        interests={"hiking", "coding"},
        liked_topics={"travel"},
        hometown="austin",
        college="ut",
        age=30,
        mutual_friend_ids={"f1"},
    )
    perfect = UserProfile(
        user_id="perfect",
        interests=set(),  # no hobby overlap
        liked_topics=set(),
        hometown="austin",  # match
        college="ut",  # match
        age=30,  # match
        mutual_friend_ids={"f1"},  # 1 mutual
    )
    many_mutual_no_bg = UserProfile(
        user_id="many_mutual_no_bg",
        interests={"hiking", "coding"},  # perfect hobby overlap
        liked_topics={"travel"},
        hometown="seattle",  # no match
        college="harvard",  # no match
        age=30,
        mutual_friend_ids={f"f{i}" for i in range(1, 11)},  # 10 mutuals
    )
    one_mutual_one_bg = UserProfile(
        user_id="one_mutual_one_bg",
        interests={"hiking", "coding"},
        liked_topics={"travel"},
        hometown="austin",  # match
        college="harvard",  # no match
        age=30,
        mutual_friend_ids={"f1"},
    )
    no_mutual_both_bg = UserProfile(
        user_id="no_mutual_both_bg",
        interests={"hiking", "coding"},
        liked_topics={"travel"},
        hometown="austin",
        college="ut",
        age=30,
        mutual_friend_ids=set(),  # no mutual
    )

    ranked = rank_candidates(
        source, [many_mutual_no_bg, one_mutual_one_bg, no_mutual_both_bg, perfect]
    )
    order = [c.user_id for c, _ in ranked]
    # Tier 1: perfect (mutual + both bg)
    # Tier 2: one_mutual_one_bg (mutual + one bg)
    # Tier 3: many_mutual_no_bg (mutual only)
    # Tier 4: no_mutual_both_bg (no mutual + both bg)
    assert order == [
        "perfect",
        "one_mutual_one_bg",
        "many_mutual_no_bg",
        "no_mutual_both_bg",
    ]


def test_no_mutual_both_background_outranks_no_mutual_either_background() -> None:
    """Lane B+ (both bg, no mutual) must rank above Lane B (one bg, no mutual)."""
    source = UserProfile(user_id="src", hometown="austin", college="ut", age=30)
    both_bg = UserProfile(
        user_id="both_bg",
        hometown="austin",
        college="ut",
        age=99,  # weak age
    )
    one_bg_strong = UserProfile(
        user_id="one_bg",
        hometown="austin",
        college="harvard",
        age=30,  # perfect age
        interests={"hiking", "coding", "music"},
    )
    ranked = rank_candidates(source, [one_bg_strong, both_bg])
    assert ranked[0][0].user_id == "both_bg"
