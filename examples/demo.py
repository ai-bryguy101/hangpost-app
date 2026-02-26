from hangpost_matching import UserProfile, rank_candidates


def main() -> None:
    source = UserProfile(
        user_id="u0",
        interests={"hiking", "coding", "cooking"},
        liked_topics={"tech", "travel", "fitness"},
        location="denver",
        age=28,
        mutual_friend_ids={"a", "b", "c", "d"},
    )

    candidates = [
        UserProfile(
            user_id="u1",
            interests={"hiking", "coding", "photography"},
            liked_topics={"tech", "travel", "music"},
            location="denver",
            age=27,
            mutual_friend_ids={"b", "c", "x"},
        ),
        UserProfile(
            user_id="u2",
            interests={"basketball", "gaming"},
            liked_topics={"sports", "esports"},
            location="miami",
            age=37,
            mutual_friend_ids={"z"},
        ),
    ]

    for profile, breakdown in rank_candidates(source, candidates):
        print(profile.user_id, breakdown)


if __name__ == "__main__":
    main()
