"""SQLite database layer for the matching engine.

This module handles everything between the database and the scoring engine:
creating the schema, inserting profiles, and querying candidates.

ARCHITECTURE — TWO DIFFERENT "LOCATIONS":
Hangpost is a location-based social app for making new friends. There are
two separate location concepts, and mixing them up would break everything:

1. CURRENT LOCATION (latitude/longitude — WHERE YOU ARE RIGHT NOW)
   - Used for FILTERING: "show me people within 20 miles"
   - Stored as current_lat/current_lng on each profile
   - Queried with a bounding-box approximation for speed
   - This is NOT a scoring signal — it's a prerequisite. If someone is
     outside your radius, you never see them, period.

2. HOMETOWN (city + state — WHERE YOU GREW UP)
   - Used for SCORING: "oh you're from Austin too? +0.22 match weight"
   - Stored as hometown_city/hometown_state on each profile
   - Fed into the Location(city, state) dataclass for tiered scoring
   - This IS a scoring signal — one of the strongest (Tier 2, weight 0.22)

THE QUERY FLOW:
   User opens app at (39.75, -105.0) with 20-mile radius
   → Step 1: DB query filters to profiles within bounding box (fast, uses index)
   → Step 2: Haversine formula refines to exact radius (eliminates box corners)
   → Step 3: Scoring engine ranks the ~200-2000 nearby profiles by compatibility
   → Step 4: User sees the top 20 matches

WHY SQLite FIRST:
- Zero infrastructure: no server to run, no Docker, no credentials
- Python stdlib supports it (no pip install)
- Perfect for development and single-server deployment
- The schema is identical to what PostgreSQL/PostGIS would use later
- When you need concurrent writes from multiple users, swap to Postgres

WHY NOT POSTGIS YET:
PostGIS would give us real geospatial indexes (R-tree) for radius queries.
But SQLite's bounding-box approximation + Haversine is plenty fast for
<100k profiles. We'd switch to PostGIS when:
- Profiles exceed ~500k (bounding box scan gets slow)
- We need complex geo queries (polygons, drive-time zones)
- We need concurrent writes from multiple app servers
"""

import math
import sqlite3
from pathlib import Path
from typing import Iterator

from .loader import tokenize
from .models import Location, UserProfile


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Earth's radius in miles, used for Haversine distance calculation.
# WHY miles: U.S.-based app, users think in miles. Convert to km if needed.
EARTH_RADIUS_MILES = 3959.0

# Default database path (relative to repo root).
DEFAULT_DB_PATH = Path("data/hangpost.db")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
# WHY a single profiles table instead of normalized hobbies/interests tables:
# At this stage, the scoring engine needs to load full profiles into memory
# as UserProfile objects anyway. Normalizing hobbies into a separate table
# would add JOIN complexity without a performance benefit — we're not querying
# "find everyone who likes hiking" in SQL, we're loading profiles and doing
# Jaccard similarity in Python. When we move to Postgres with full-text search,
# we can revisit normalization.

SCHEMA_SQL = """
-- ─── Profiles table ──────────────────────────────────────────────────
-- One row per user. Mirrors the CSV columns plus geolocation fields.
CREATE TABLE IF NOT EXISTS profiles (
    -- Unique identifier. In a real app this would be a UUID from your auth system.
    user_id         TEXT PRIMARY KEY,

    -- Display fields (not used in scoring, but needed for UI).
    name            TEXT NOT NULL,
    degree          TEXT,
    job             TEXT,

    -- ── Scoring fields ──
    -- These map directly to UserProfile fields and feed into compute_match_score.

    -- Age → age_compatibility signal (step-down ladder, 10% per year).
    age             INTEGER,

    -- Hometown → location_match signal (tiered: same city=1.0, same state=0.4).
    -- This is WHERE YOU GREW UP, not where you are right now.
    hometown_city   TEXT,
    hometown_state  TEXT,

    -- College → college_match signal (exact match, 0.18 weight).
    college         TEXT,

    -- Faith → faith_match signal (exact match, 0.03 weight).
    faith           TEXT,

    -- Semicolon-separated strings. Tokenized into sets when loaded into UserProfile.
    -- WHY not normalized: See comment above SCHEMA_SQL.
    hobbies         TEXT,    -- → hobby_overlap (Jaccard, 0.15 weight)
    interests       TEXT,    -- → interest_overlap (Jaccard, 0.08 weight)
    fan_of          TEXT,    -- → fan_of_overlap (Jaccard, 0.05 weight)
    travel          TEXT,    -- → travel_overlap (Jaccard, 0.02 weight)

    -- ── Current geolocation (for radius filtering, NOT for scoring) ──
    -- These are the user's CURRENT coordinates, updated when they open the app.
    -- Used to answer: "who is near me right now?"
    current_lat     REAL,
    current_lng     REAL,

    -- Metadata.
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ─── Friendships table ───────────────────────────────────────────────
-- The social graph: who is friends with whom.
-- WHY a separate table: A user can have hundreds of friends. Storing them
-- as a comma-separated string in the profiles table would be un-queryable.
-- A junction table lets us efficiently find mutual friends with SQL.
--
-- CONVENTION: user_id_a < user_id_b (alphabetically). This prevents
-- duplicate rows for the same friendship (A→B and B→A). The query layer
-- handles this by checking both directions.
CREATE TABLE IF NOT EXISTS friendships (
    user_id_a   TEXT NOT NULL REFERENCES profiles(user_id),
    user_id_b   TEXT NOT NULL REFERENCES profiles(user_id),
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id_a, user_id_b)
);

-- ─── Indexes ─────────────────────────────────────────────────────────
-- These speed up the two most common query patterns.

-- Geolocation index: finding profiles within a lat/lng bounding box.
-- WHY a compound index: The bounding-box query filters on BOTH lat AND lng,
-- so a compound index lets SQLite satisfy both conditions in one index scan.
CREATE INDEX IF NOT EXISTS idx_profiles_geo
    ON profiles(current_lat, current_lng);

-- State index: for pre-filtering candidates by hometown state.
-- Used in the "find potential matches" query to prioritize same-state profiles.
CREATE INDEX IF NOT EXISTS idx_profiles_state
    ON profiles(hometown_state);

-- College index: for pre-filtering candidates by college.
CREATE INDEX IF NOT EXISTS idx_profiles_college
    ON profiles(college);

-- Friendship lookup: given a user_id, find all their friends quickly.
-- WHY two indexes: friendships are stored with user_id_a < user_id_b,
-- so to find all friends of user X we need to check both columns.
CREATE INDEX IF NOT EXISTS idx_friendships_a ON friendships(user_id_a);
CREATE INDEX IF NOT EXISTS idx_friendships_b ON friendships(user_id_b);
"""


# ---------------------------------------------------------------------------
# Database connection helpers
# ---------------------------------------------------------------------------

def get_connection(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open a connection to the SQLite database.

    WHY Row factory: sqlite3.Row lets us access columns by name (row["city"])
    instead of by index (row[3]). This makes the code self-documenting and
    prevents bugs when column order changes.

    WHY WAL mode: Write-Ahead Logging allows concurrent reads while a write
    is in progress. Without WAL, any write blocks all reads. For a web app
    serving multiple users, this is essential. (SQLite still only allows one
    writer at a time — for true concurrent writes, upgrade to Postgres.)
    """
    db_path = Path(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """Create all tables and indexes if they don't exist.

    Safe to call multiple times — all CREATE statements use IF NOT EXISTS.
    """
    conn.executescript(SCHEMA_SQL)


# ---------------------------------------------------------------------------
# Insert helpers
# ---------------------------------------------------------------------------

def insert_profile(conn: sqlite3.Connection, profile_data: dict) -> None:
    """Insert a single profile into the database.

    Args:
        profile_data: A dict with keys matching the profiles table columns.
            Required: user_id, name
            Optional: everything else (defaults to NULL in the DB)

    WHY INSERT OR REPLACE: During development, we often re-import the same
    synthetic data. REPLACE prevents "UNIQUE constraint failed" errors
    without needing a separate DELETE step. In production, you'd use plain
    INSERT and handle conflicts explicitly.
    """
    conn.execute("""
        INSERT OR REPLACE INTO profiles (
            user_id, name, age, hometown_city, hometown_state,
            college, degree, job, faith,
            hobbies, interests, fan_of, travel,
            current_lat, current_lng
        ) VALUES (
            :user_id, :name, :age, :hometown_city, :hometown_state,
            :college, :degree, :job, :faith,
            :hobbies, :interests, :fan_of, :travel,
            :current_lat, :current_lng
        )
    """, profile_data)


def insert_friendship(conn: sqlite3.Connection, user_a: str, user_b: str) -> None:
    """Record a friendship between two users.

    WHY the sort: We always store (smaller_id, larger_id) to avoid duplicate
    rows. Without this, adding A→B and then B→A would create two rows for
    the same friendship.
    """
    a, b = sorted([user_a, user_b])
    conn.execute(
        "INSERT OR IGNORE INTO friendships (user_id_a, user_id_b) VALUES (?, ?)",
        (a, b),
    )


# ---------------------------------------------------------------------------
# Geolocation math
# ---------------------------------------------------------------------------

def haversine_miles(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Calculate great-circle distance between two points in miles.

    WHY Haversine instead of Euclidean distance:
    The Earth is a sphere (roughly). At high latitudes, one degree of longitude
    covers far fewer miles than at the equator. Haversine accounts for this
    curvature. Euclidean distance on lat/lng would give wildly wrong results
    for any non-equatorial location.

    Accuracy: within ~0.3% for distances under 1000 miles. Good enough for
    a social app — we're not navigating ships.
    """
    lat1, lng1, lat2, lng2 = map(math.radians, [lat1, lng1, lat2, lng2])
    dlat = lat2 - lat1
    dlng = lng2 - lng1

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return EARTH_RADIUS_MILES * c


def bounding_box(lat: float, lng: float, radius_miles: float) -> tuple[float, float, float, float]:
    """Compute a lat/lng bounding box for a given radius.

    Returns: (min_lat, max_lat, min_lng, max_lng)

    WHY a bounding box instead of computing Haversine for every row:
    Haversine is expensive. If we calculated it for all 100k profiles, the
    query would be slow. Instead, we first filter with a cheap bounding-box
    check (just >= and <= on indexed columns), which might return ~2x the
    actual results, then refine with Haversine on the smaller set.

    This is the standard approach used by every geolocation system before
    upgrading to a proper spatial index (PostGIS R-tree, etc.).

    The lng calculation accounts for latitude: at the poles, one degree of
    longitude covers fewer miles than at the equator. We use cos(lat) to
    adjust the longitude range accordingly.
    """
    # One degree of latitude ≈ 69 miles everywhere on Earth.
    lat_delta = radius_miles / 69.0

    # One degree of longitude varies by latitude.
    # At the equator: ~69 miles. At 45°N: ~49 miles. At 60°N: ~34.5 miles.
    lng_delta = radius_miles / (69.0 * math.cos(math.radians(lat)))

    return (
        lat - lat_delta,   # min_lat
        lat + lat_delta,   # max_lat
        lng - lng_delta,   # min_lng
        lng + lng_delta,   # max_lng
    )


# ---------------------------------------------------------------------------
# Query: find nearby candidates
# ---------------------------------------------------------------------------

def find_nearby_profiles(
    conn: sqlite3.Connection,
    user_lat: float,
    user_lng: float,
    radius_miles: float = 20.0,
    exclude_user_id: str | None = None,
) -> list[sqlite3.Row]:
    """Find all profiles within radius_miles of the given coordinates.

    TWO-STEP FILTERING:
    1. Bounding box (fast, uses the geo index): returns a rough superset
       of profiles that MIGHT be within the radius.
    2. Haversine (accurate, in Python): filters the bounding-box results
       down to only those truly within the circle.

    WHY two steps: The bounding box is a rectangle around a circle. The
    corners of the rectangle extend beyond the radius, so the box returns
    some false positives. Haversine eliminates those. This is ~100x faster
    than running Haversine on every profile in the database.

    Returns sqlite3.Row objects (dict-like) with all profile columns.
    """
    min_lat, max_lat, min_lng, max_lng = bounding_box(user_lat, user_lng, radius_miles)

    # Step 1: Bounding-box filter (uses idx_profiles_geo index).
    query = """
        SELECT * FROM profiles
        WHERE current_lat BETWEEN ? AND ?
          AND current_lng BETWEEN ? AND ?
    """
    params: list = [min_lat, max_lat, min_lng, max_lng]

    if exclude_user_id:
        query += " AND user_id != ?"
        params.append(exclude_user_id)

    rows = conn.execute(query, params).fetchall()

    # Step 2: Haversine refinement (eliminates bounding-box corner false positives).
    nearby = []
    for row in rows:
        dist = haversine_miles(user_lat, user_lng, row["current_lat"], row["current_lng"])
        if dist <= radius_miles:
            nearby.append(row)

    return nearby


# ---------------------------------------------------------------------------
# Query: get mutual friend IDs for a user
# ---------------------------------------------------------------------------

def get_friend_ids(conn: sqlite3.Connection, user_id: str) -> set[str]:
    """Return the set of user IDs that are friends with the given user.

    WHY check both columns: Friendships are stored with user_id_a < user_id_b.
    So if user "alice" is friends with "bob", the row is (alice, bob).
    To find all of alice's friends, we need to check BOTH:
    - rows where alice is user_id_a → friend is in user_id_b column
    - rows where alice is user_id_b → friend is in user_id_a column
    """
    # Friends where this user is the "a" side.
    cursor_a = conn.execute(
        "SELECT user_id_b FROM friendships WHERE user_id_a = ?", (user_id,)
    )
    # Friends where this user is the "b" side.
    cursor_b = conn.execute(
        "SELECT user_id_a FROM friendships WHERE user_id_b = ?", (user_id,)
    )
    return {row[0] for row in cursor_a} | {row[0] for row in cursor_b}


# ---------------------------------------------------------------------------
# Convert DB row → UserProfile (bridges DB layer to scoring engine)
# ---------------------------------------------------------------------------

def row_to_profile(
    row: sqlite3.Row,
    friend_ids: set[str] | None = None,
) -> UserProfile:
    """Convert a database row into a UserProfile for scoring.

    This is the bridge between the database and the scoring engine. The
    scoring engine only sees UserProfile objects — it never touches the DB.

    HOW THIS DIFFERS FROM profile_from_row() in loader.py:
    - loader.py reads from CSV dicts with keys like "city", "state", "friends_in_common"
    - This function reads from DB rows with keys like "hometown_city", "hometown_state"
    - loader.py synthesizes fake friend IDs from a count; this uses real IDs from the
      friendships table

    Args:
        row: A sqlite3.Row from the profiles table.
        friend_ids: Pre-fetched set of friend user_ids for this profile.
            If None, mutual_friend_ids will be empty (no friend data available).
    """
    # Build hometown Location from DB columns.
    city = (row["hometown_city"] or "").strip()
    state = (row["hometown_state"] or "").strip()
    location = Location(city=city, state=state) if city else None

    return UserProfile(
        user_id=row["user_id"],
        hobbies=tokenize(row["hobbies"] or ""),
        interests=tokenize(row["interests"] or ""),
        fan_of=tokenize(row["fan_of"] or ""),
        location=location,
        age=row["age"],
        mutual_friend_ids=friend_ids or set(),
        college=row["college"],
        faith=row["faith"],
        travel_wishlist=tokenize(row["travel"] or ""),
    )


# ---------------------------------------------------------------------------
# High-level: find and rank candidates for a user
# ---------------------------------------------------------------------------

def find_and_rank_candidates(
    conn: sqlite3.Connection,
    source_user_id: str,
    source_lat: float,
    source_lng: float,
    radius_miles: float = 20.0,
    top_n: int = 20,
) -> list[tuple[UserProfile, object]]:
    """The main entry point for the app: find nearby people and rank them.

    This is the complete flow from "user opens the app" to "here are your
    top matches":

    1. Query DB for profiles within the radius (geo filter)
    2. Load the source user's friend list from the friendships table
    3. Convert all nearby profiles to UserProfile objects
    4. Run the scoring engine to rank by compatibility
    5. Return the top N results

    Returns:
        List of (UserProfile, MatchBreakdown) tuples, sorted best-first.
    """
    # Avoid circular import — scoring imports models, we import scoring here.
    from .scoring import rank_candidates

    # Step 1: Validate the source user exists FIRST.
    # WHY before the geo query: If the source doesn't exist, we should fail
    # fast with a clear error rather than returning an empty list that looks
    # like "no nearby profiles found."
    source_row = conn.execute(
        "SELECT * FROM profiles WHERE user_id = ?", (source_user_id,)
    ).fetchone()

    if source_row is None:
        raise ValueError(f"Source user '{source_user_id}' not found in database")

    # Step 2: Geo filter — find everyone nearby.
    nearby_rows = find_nearby_profiles(
        conn, source_lat, source_lng, radius_miles, exclude_user_id=source_user_id
    )

    if not nearby_rows:
        return []

    # Step 3: Load the source user's friend list and build their UserProfile.
    source_friends = get_friend_ids(conn, source_user_id)
    source_profile = row_to_profile(source_row, friend_ids=source_friends)

    # Step 4: Convert nearby rows to UserProfile objects.
    # For each candidate, we need their friend list to compute mutual friends.
    # WHY load friend lists here: The scoring engine computes mutual friends
    # as |source.mutual_friend_ids & candidate.mutual_friend_ids|. We need
    # both sets loaded before scoring.
    candidate_profiles = []
    for row in nearby_rows:
        cand_friends = get_friend_ids(conn, row["user_id"])
        candidate_profiles.append(row_to_profile(row, friend_ids=cand_friends))

    # Step 5: Rank by compatibility (the scoring engine handles this).
    ranked = rank_candidates(source_profile, candidate_profiles)

    # Step 6: Return top N.
    return ranked[:top_n]
