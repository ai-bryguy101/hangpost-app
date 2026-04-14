"""Tests for the database layer (db.py).

Covers:
- Schema creation and table existence
- Profile insertion and retrieval
- Friendship insertion and friend_ids lookup
- Geolocation math (Haversine, bounding box)
- Radius-based nearby-profile queries
- The row_to_profile bridge (DB row → UserProfile)
- find_and_rank_candidates end-to-end flow

WHY THESE TESTS:
The database layer is the bridge between stored data and the scoring engine.
If it silently drops a field, miscomputes a distance, or returns profiles
outside the radius, the entire matching experience breaks. Every function
in db.py has at least one test.

All tests use in-memory SQLite (:memory:) so they're fast and leave no
artifacts. Each test gets a fresh database via the `db` fixture.
"""

import math

import pytest

from hangpost_matching.db import (
    bounding_box,
    find_and_rank_candidates,
    find_nearby_profiles,
    get_friend_ids,
    haversine_miles,
    init_schema,
    insert_friendship,
    insert_profile,
    row_to_profile,
)
from hangpost_matching.models import Location

import sqlite3


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db():
    """Create a fresh in-memory database with the schema applied.

    WHY in-memory: Each test gets an isolated database that disappears when
    the test ends. No leftover files, no test pollution, fast execution.
    """
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=OFF")  # synthetic data has no real FK targets
    init_schema(conn)
    return conn


def _insert_test_profile(
    conn, user_id, name="Test User", age=25,
    city="Denver", state="Colorado",
    college="UCLA", faith="Agnostic",
    hobbies="Hiking; Coding", interests="Tech",
    fan_of="The Bear", travel="Japan",
    lat=39.7392, lng=-104.9903,
):
    """Helper to insert a test profile with sensible defaults.

    WHY a helper: Every DB test needs profiles. Repeating all 15 fields
    in every test would be noisy. This lets tests specify only the fields
    they care about and get defaults for the rest.
    """
    insert_profile(conn, {
        "user_id": user_id,
        "name": name,
        "age": age,
        "hometown_city": city,
        "hometown_state": state,
        "college": college,
        "degree": "BS CS",
        "job": "Engineer",
        "faith": faith,
        "hobbies": hobbies,
        "interests": interests,
        "fan_of": fan_of,
        "travel": travel,
        "current_lat": lat,
        "current_lng": lng,
    })
    conn.commit()


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TestSchema:
    """Verify that init_schema creates the expected tables and indexes."""

    def test_profiles_table_exists(self, db):
        tables = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='profiles'"
        ).fetchall()
        assert len(tables) == 1

    def test_friendships_table_exists(self, db):
        tables = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='friendships'"
        ).fetchall()
        assert len(tables) == 1

    def test_geo_index_exists(self, db):
        """The geo index on (current_lat, current_lng) is critical for query speed."""
        indexes = db.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_profiles_geo'"
        ).fetchall()
        assert len(indexes) == 1

    def test_schema_idempotent(self, db):
        """Running init_schema twice should not error (IF NOT EXISTS)."""
        init_schema(db)  # second call
        count = db.execute("SELECT COUNT(*) FROM profiles").fetchone()[0]
        assert count == 0  # still empty, no crash


# ---------------------------------------------------------------------------
# Profile insertion and retrieval tests
# ---------------------------------------------------------------------------

class TestProfileInsertRetrieve:
    """Test that profiles are correctly stored and retrieved."""

    def test_insert_and_select(self, db):
        _insert_test_profile(db, "user1", name="Alice", age=28, city="Austin", state="Texas")
        row = db.execute("SELECT * FROM profiles WHERE user_id = 'user1'").fetchone()
        assert row["name"] == "Alice"
        assert row["age"] == 28
        assert row["hometown_city"] == "Austin"
        assert row["hometown_state"] == "Texas"

    def test_insert_or_replace(self, db):
        """Re-inserting the same user_id should update, not duplicate."""
        _insert_test_profile(db, "user1", name="Alice", age=28)
        _insert_test_profile(db, "user1", name="Alice Updated", age=29)
        count = db.execute("SELECT COUNT(*) FROM profiles WHERE user_id = 'user1'").fetchone()[0]
        assert count == 1
        row = db.execute("SELECT * FROM profiles WHERE user_id = 'user1'").fetchone()
        assert row["name"] == "Alice Updated"
        assert row["age"] == 29

    def test_null_fields(self, db):
        """Profiles with missing fields should store NULLs gracefully."""
        insert_profile(db, {
            "user_id": "minimal",
            "name": "Minimal User",
            "age": None,
            "hometown_city": None,
            "hometown_state": None,
            "college": None,
            "degree": None,
            "job": None,
            "faith": None,
            "hobbies": None,
            "interests": None,
            "fan_of": None,
            "travel": None,
            "current_lat": None,
            "current_lng": None,
        })
        db.commit()
        row = db.execute("SELECT * FROM profiles WHERE user_id = 'minimal'").fetchone()
        assert row["age"] is None
        assert row["hometown_city"] is None


# ---------------------------------------------------------------------------
# Friendship tests
# ---------------------------------------------------------------------------

class TestFriendships:
    """Test the social graph (friendships table)."""

    def test_insert_and_lookup(self, db):
        """Adding a friendship and looking it up should return the friend."""
        _insert_test_profile(db, "alice")
        _insert_test_profile(db, "bob")
        insert_friendship(db, "alice", "bob")
        db.commit()

        alice_friends = get_friend_ids(db, "alice")
        bob_friends = get_friend_ids(db, "bob")

        assert "bob" in alice_friends
        assert "alice" in bob_friends

    def test_friendship_deduplication(self, db):
        """Adding the same friendship twice should not create duplicates.

        WHY: insert_friendship sorts the IDs and uses INSERT OR IGNORE,
        so (alice, bob) and (bob, alice) both map to the same row.
        """
        _insert_test_profile(db, "alice")
        _insert_test_profile(db, "bob")
        insert_friendship(db, "alice", "bob")
        insert_friendship(db, "bob", "alice")  # same friendship, reversed
        db.commit()

        count = db.execute("SELECT COUNT(*) FROM friendships").fetchone()[0]
        assert count == 1

    def test_multiple_friends(self, db):
        """A user can have multiple friends."""
        for name in ["alice", "bob", "carol", "dave"]:
            _insert_test_profile(db, name)
        insert_friendship(db, "alice", "bob")
        insert_friendship(db, "alice", "carol")
        insert_friendship(db, "alice", "dave")
        db.commit()

        friends = get_friend_ids(db, "alice")
        assert friends == {"bob", "carol", "dave"}

    def test_no_friends(self, db):
        """A user with no friendships should return an empty set."""
        _insert_test_profile(db, "loner")
        friends = get_friend_ids(db, "loner")
        assert friends == set()


# ---------------------------------------------------------------------------
# Geolocation math tests
# ---------------------------------------------------------------------------

class TestHaversine:
    """Test the Haversine distance formula.

    WHY test this: A bug here means radius filtering is wrong — you'd either
    show profiles that are too far away or miss nearby ones.
    """

    def test_same_point(self):
        """Distance from a point to itself should be 0."""
        assert haversine_miles(39.7392, -104.9903, 39.7392, -104.9903) == 0.0

    def test_known_distance_denver_to_boulder(self):
        """Denver to Boulder is ~25 miles. Verify Haversine is in the right ballpark."""
        dist = haversine_miles(39.7392, -104.9903, 40.0150, -105.2705)
        assert 24 < dist < 26  # ~25 miles

    def test_known_distance_nyc_to_la(self):
        """NYC to LA is ~2,451 miles. Verify Haversine handles large distances."""
        dist = haversine_miles(40.7128, -74.0060, 34.0522, -118.2437)
        assert 2400 < dist < 2500

    def test_symmetry(self):
        """Distance A→B should equal distance B→A."""
        ab = haversine_miles(39.7392, -104.9903, 40.0150, -105.2705)
        ba = haversine_miles(40.0150, -105.2705, 39.7392, -104.9903)
        assert ab == pytest.approx(ba)


class TestBoundingBox:
    """Test the bounding-box computation for geo filtering."""

    def test_box_contains_center(self):
        """The center point should be inside its own bounding box."""
        min_lat, max_lat, min_lng, max_lng = bounding_box(39.7392, -104.9903, 20)
        assert min_lat < 39.7392 < max_lat
        assert min_lng < -104.9903 < max_lng

    def test_larger_radius_larger_box(self):
        """A bigger radius should produce a bigger box."""
        small = bounding_box(39.7392, -104.9903, 10)
        large = bounding_box(39.7392, -104.9903, 50)
        # Large box should be wider in both lat and lng.
        assert (large[1] - large[0]) > (small[1] - small[0])  # lat range
        assert (large[3] - large[2]) > (small[3] - small[2])  # lng range

    def test_box_roughly_correct_size(self):
        """A 20-mile radius should produce ~0.58° lat range (20/69 × 2)."""
        min_lat, max_lat, _, _ = bounding_box(39.7392, -104.9903, 20)
        lat_range = max_lat - min_lat
        expected = 2 * 20 / 69.0  # ~0.58°
        assert lat_range == pytest.approx(expected, abs=0.01)


# ---------------------------------------------------------------------------
# Nearby profile query tests
# ---------------------------------------------------------------------------

class TestFindNearbyProfiles:
    """Test the radius-based profile query."""

    def test_finds_profiles_within_radius(self, db):
        """Profiles within 20 miles should be returned."""
        # Place two profiles near Denver (within ~5 miles of each other).
        _insert_test_profile(db, "nearby", lat=39.7392, lng=-104.9903)
        _insert_test_profile(db, "also_nearby", lat=39.7500, lng=-105.0000)
        # Place one profile in Miami (far away).
        _insert_test_profile(db, "far_away", lat=25.7617, lng=-80.1918)

        results = find_nearby_profiles(db, 39.7392, -104.9903, radius_miles=20.0)
        user_ids = {row["user_id"] for row in results}

        assert "nearby" in user_ids
        assert "also_nearby" in user_ids
        assert "far_away" not in user_ids

    def test_excludes_self(self, db):
        """The source user should not appear in their own results."""
        _insert_test_profile(db, "me", lat=39.7392, lng=-104.9903)
        _insert_test_profile(db, "friend", lat=39.7400, lng=-104.9900)

        results = find_nearby_profiles(
            db, 39.7392, -104.9903, radius_miles=20.0, exclude_user_id="me"
        )
        user_ids = {row["user_id"] for row in results}

        assert "me" not in user_ids
        assert "friend" in user_ids

    def test_empty_when_nobody_nearby(self, db):
        """If no profiles are nearby, return an empty list."""
        _insert_test_profile(db, "far_away", lat=25.7617, lng=-80.1918)  # Miami
        results = find_nearby_profiles(db, 39.7392, -104.9903, radius_miles=20.0)
        assert results == []

    def test_respects_radius_size(self, db):
        """A profile 15 miles away should be found with a 20mi radius but
        not with a 10mi radius."""
        # Denver center
        _insert_test_profile(db, "center", lat=39.7392, lng=-104.9903)
        # ~15 miles north of Denver (about 0.22° lat)
        _insert_test_profile(db, "fifteen_mi", lat=39.96, lng=-104.9903)

        wide = find_nearby_profiles(db, 39.7392, -104.9903, radius_miles=20.0)
        narrow = find_nearby_profiles(db, 39.7392, -104.9903, radius_miles=10.0)

        wide_ids = {row["user_id"] for row in wide}
        narrow_ids = {row["user_id"] for row in narrow}

        assert "fifteen_mi" in wide_ids
        assert "fifteen_mi" not in narrow_ids


# ---------------------------------------------------------------------------
# row_to_profile bridge tests
# ---------------------------------------------------------------------------

class TestRowToProfile:
    """Test the DB row → UserProfile conversion."""

    def test_basic_conversion(self, db):
        """A DB row should convert to a UserProfile with all fields mapped."""
        _insert_test_profile(
            db, "user1", name="Alice", age=28,
            city="Austin", state="Texas", college="UT Austin",
            faith="Agnostic", hobbies="Hiking; Coding",
            interests="Tech; Outdoor Adventure",
            fan_of="The Bear; Zelda", travel="Japan; Iceland",
        )
        row = db.execute("SELECT * FROM profiles WHERE user_id = 'user1'").fetchone()
        profile = row_to_profile(row)

        assert profile.user_id == "user1"
        assert profile.age == 28
        assert profile.location == Location(city="Austin", state="Texas")
        assert profile.college == "UT Austin"
        assert profile.faith == "Agnostic"
        assert profile.hobbies == {"hiking", "coding"}
        assert profile.interests == {"tech", "outdoor adventure"}
        assert profile.fan_of == {"the bear", "zelda"}
        assert profile.travel_wishlist == {"japan", "iceland"}

    def test_friend_ids_passed_through(self, db):
        """Friend IDs should be set from the friend_ids parameter."""
        _insert_test_profile(db, "user1")
        row = db.execute("SELECT * FROM profiles WHERE user_id = 'user1'").fetchone()

        profile_no_friends = row_to_profile(row, friend_ids=None)
        profile_with_friends = row_to_profile(row, friend_ids={"alice", "bob"})

        assert profile_no_friends.mutual_friend_ids == set()
        assert profile_with_friends.mutual_friend_ids == {"alice", "bob"}

    def test_missing_hometown_gives_none_location(self, db):
        """If hometown_city is empty, location should be None."""
        _insert_test_profile(db, "user1", city="", state="")
        row = db.execute("SELECT * FROM profiles WHERE user_id = 'user1'").fetchone()
        profile = row_to_profile(row)
        assert profile.location is None


# ---------------------------------------------------------------------------
# End-to-end: find_and_rank_candidates
# ---------------------------------------------------------------------------

class TestFindAndRankCandidates:
    """Test the full flow: geo filter → load profiles → score → rank."""

    def test_returns_ranked_results(self, db):
        """Should return matches ranked by compatibility score."""
        # Source user in Denver.
        _insert_test_profile(
            db, "source", name="Source", city="Austin", state="Texas",
            college="UCLA", hobbies="Hiking; Coding",
            lat=39.7392, lng=-104.9903,
        )
        # Good match nearby: same college, same hobbies.
        _insert_test_profile(
            db, "good", name="Good Match", city="Austin", state="Texas",
            college="UCLA", hobbies="Hiking; Coding",
            lat=39.7400, lng=-104.9900,
        )
        # Weak match nearby: nothing in common.
        _insert_test_profile(
            db, "weak", name="Weak Match", city="Miami", state="Florida",
            college="NYU", hobbies="Basketball",
            lat=39.7410, lng=-104.9910,
        )

        results = find_and_rank_candidates(
            db, "source", 39.7392, -104.9903, radius_miles=20.0, top_n=10,
        )

        assert len(results) == 2
        # Good match should rank first.
        assert results[0][0].user_id == "good"
        assert results[1][0].user_id == "weak"
        # Good match should have a higher score.
        assert results[0][1].total_score > results[1][1].total_score

    def test_excludes_profiles_outside_radius(self, db):
        """Profiles outside the radius should not appear in results."""
        _insert_test_profile(db, "source", lat=39.7392, lng=-104.9903)
        _insert_test_profile(db, "far_away", lat=25.7617, lng=-80.1918)  # Miami

        results = find_and_rank_candidates(
            db, "source", 39.7392, -104.9903, radius_miles=20.0,
        )
        assert len(results) == 0

    def test_respects_top_n(self, db):
        """Should return at most top_n results."""
        _insert_test_profile(db, "source", lat=39.7392, lng=-104.9903)
        for i in range(10):
            _insert_test_profile(db, f"c{i}", lat=39.7392 + i * 0.001, lng=-104.9903)

        results = find_and_rank_candidates(
            db, "source", 39.7392, -104.9903, radius_miles=20.0, top_n=3,
        )
        assert len(results) == 3

    def test_nonexistent_source_raises(self, db):
        """Querying for a non-existent source user should raise ValueError."""
        with pytest.raises(ValueError, match="not found"):
            find_and_rank_candidates(db, "ghost", 39.7, -104.9)
