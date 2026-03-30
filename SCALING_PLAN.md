# Hangpost Scaling Plan

## 1. Location Hierarchy (City + State)

### Current Problem
- Location is a single string ("Austin") with binary exact-match scoring
- No state-level matching — two people in different Texas cities get 0.0
- Hometown and state are separate, unlinked dropdowns

### Proposed Data Model
```python
@dataclass(frozen=True)
class Location:
    city: str       # e.g. "Austin"
    state: str      # e.g. "Texas"

class UserProfile:
    location: Location | None = None
```

### Proposed Scoring (replaces binary `_location_score`)
```
Same city + same state  → 1.0   (full match)
Different city, same state → 0.4  (regional match)
Different state          → 0.0
```

### U.S. City Data
- Use a static JSON/CSV of all U.S. cities mapped to states
- Source: U.S. Census Bureau or simplemaps.com (free dataset, ~30k cities)
- Cities are pre-linked to states — no way to pick "Austin, Florida" unless it exists
- Web UI becomes a searchable/filterable dropdown (type-ahead)
- Store as `data/us_cities.json`: `[{"city": "Austin", "state": "Texas"}, ...]`

---

## 2. Restructured Interest Taxonomy

### Current Problem
- `interests` field mixes hobbies ("Hiking") with skills ("Python")
- `liked_topics` mixes categories ("Tech") with specific items ("Kendrick Lamar")
- `fan_of` data exists in CSV but is never scored

### Proposed Taxonomy

**interests** → Broad categories / types of things you enjoy:
- Music genres: Hip Hop, R&B, Rock, Pop, Country, EDM, Indie, Jazz, Latin, K-Pop, Classical
- Food types: Japanese, Mexican, Italian, Thai, BBQ, Vegan, Brunch, Street Food, Fine Dining
- Activity types: Outdoor Sports, Team Sports, Individual Sports, Board/Card Games, Video Games, Fitness, Creative Arts, Performing Arts, Reading/Writing, Cooking/Baking, DIY/Making
- Lifestyle: Tech, Travel, Fashion, Film/TV, Anime/Manga, True Crime, Philosophy, Psychology, Sustainability, Wellness/Mindfulness, Astrology

**fan_of** → Specific named entities you're a fan of:
- Artists: Kendrick Lamar, Taylor Swift, Bad Bunny, SZA, Coldplay, etc.
- Shows/Movies: The Bear, Succession, Stranger Things, Dune, Marvel, etc.
- Sports: NFL, NBA, Premier League, etc.
- Books/Podcasts: Atomic Habits, Joe Rogan, Lex Fridman, etc.
- Games: Zelda, Minecraft, Smash Bros, etc.

**hobbies** → Stays as-is (activities you actively do):
- Hiking, Cycling, Photography, Cooking, Chess, Guitar, etc.

### Proposed UserProfile Fields
```python
class UserProfile:
    hobbies: set[str]        # things you DO (Jaccard similarity)
    interests: set[str]      # broad categories (Jaccard similarity)
    fan_of: set[str]         # specific entities (Jaccard similarity)
    ...
```

### Proposed Scoring Weights
```python
class ScoringWeights:
    hobby_overlap: float = 0.15       # shared activities
    interest_overlap: float = 0.15    # shared taste categories
    fan_of_overlap: float = 0.10      # specific shared fandoms
    mutual_friends: float = 0.25
    location_match: float = 0.10
    age_compatibility: float = 0.25
    friend_common_boost: float = 0.35
```

---

## 3. Database Architecture (Production Path)

### Phase 1 → Phase 2 Migration Path

```
Current (Phase 1):          Future (Phase 2+):
CSV files on disk    →      PostgreSQL database
In-memory matching   →      Pre-computed indexes + query
10k profiles         →      100k+ profiles
```

### Database Schema (PostgreSQL)

```sql
-- Core profile table
CREATE TABLE profiles (
    id            UUID PRIMARY KEY,
    name          TEXT NOT NULL,
    age           INT,
    city          TEXT,
    state         TEXT,
    college       TEXT,
    degree        TEXT,
    job           TEXT,
    faith         TEXT,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

-- Hobbies/interests/fan_of as normalized tag tables
CREATE TABLE tags (
    id        SERIAL PRIMARY KEY,
    category  TEXT NOT NULL,         -- 'hobby', 'interest', 'fan_of'
    value     TEXT NOT NULL,
    UNIQUE(category, value)
);

CREATE TABLE profile_tags (
    profile_id  UUID REFERENCES profiles(id),
    tag_id      INT REFERENCES tags(id),
    PRIMARY KEY (profile_id, tag_id)
);

-- Social graph
CREATE TABLE friendships (
    user_a  UUID REFERENCES profiles(id),
    user_b  UUID REFERENCES profiles(id),
    status  TEXT DEFAULT 'accepted',
    PRIMARY KEY (user_a, user_b)
);

-- Outcome logging (for future ML)
CREATE TABLE match_events (
    id          SERIAL PRIMARY KEY,
    source_id   UUID REFERENCES profiles(id),
    candidate_id UUID REFERENCES profiles(id),
    event_type  TEXT,   -- 'shown', 'clicked', 'request_sent', 'accepted'
    score       FLOAT,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
```

### Where It Lives in Production
```
                    ┌──────────────┐
  New user signup → │  API Server  │ ← FastAPI / Django
                    │  (Python)    │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
       ┌────────────┐ ┌────────┐ ┌──────────┐
       │ PostgreSQL  │ │ Redis  │ │ S3/Blob  │
       │ (profiles,  │ │ (cache,│ │ (photos, │
       │  tags,      │ │ online │ │  media)  │
       │  events)    │ │ status)│ │          │
       └────────────┘ └────────┘ └──────────┘
```

### How New Users Get Matched
1. User signs up → profile saved to `profiles` + `profile_tags` tables
2. On first load of "People You May Know":
   - Query pulls candidates (same state, age within 10 years — pre-filter)
   - Matching engine scores the filtered set (not all 100k+)
   - Top N returned with explanations
3. Results cached in Redis for ~1 hour
4. Every interaction logged to `match_events` for future ML training

### Scaling Strategy for Matching
- **< 50k profiles:** Score all candidates in-memory (current approach works)
- **50k–500k:** Pre-filter by state + age range, then score the reduced set
- **500k+:** Approximate nearest neighbor (ANN) index for embeddings + rule-based re-ranking
- **1M+:** Candidate generation pipeline (retrieval → scoring → ranking)

---

## 4. Expanding to All U.S. Cities + Open-Ended Fan-Of

### Cities
- Load ~30k U.S. cities from a static dataset
- Web UI: searchable type-ahead input (not a dropdown with 30k items)
- Stored as city + state pair, validated against the dataset
- Enables proper state-level partial matching

### Interests (Categories)
- Curated list of ~50-80 broad categories (manageable for checkboxes)
- Organized into groups: "Music", "Food", "Activities", "Lifestyle"
- Relatively static — new categories added occasionally

### Fan-Of (Specific Entities)
- Starts as a curated list of ~200-500 popular items
- Future: searchable with auto-complete (like Spotify/IMDB lookup)
- Could integrate external APIs (Spotify for artists, TMDB for shows)
- Users can also type custom entries not in the list

---

## 5. Implementation Order

### Step A: Data model refactor (do now)
- [ ] Split location into city + state with tiered scoring
- [ ] Restructure into hobbies / interests / fan_of (3 scored fields)
- [ ] Update UserProfile, ScoringWeights, scoring.py
- [ ] Update loader, generator, and builder scripts
- [ ] Update tests

### Step B: Expand option pools (do now)
- [ ] Curate proper interest categories (~60 items in groups)
- [ ] Curate fan_of list (~200 specific entities)
- [ ] Add U.S. cities dataset (data/us_cities.json)
- [ ] Update web UI with grouped sections and type-ahead for cities

### Step C: Database migration (next phase)
- [ ] Add SQLite support for local dev (drop-in for CSV)
- [ ] Design PostgreSQL schema for production
- [ ] Build FastAPI endpoints wrapping the matching engine
- [ ] Add match event logging

### Step D: Scale matching (later)
- [ ] Pre-filtering by state + age range
- [ ] Caching layer
- [ ] Embedding-based retrieval for Phase 3
