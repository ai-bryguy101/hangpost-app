"""Shared option pools for profile generation and interactive builders.

All scripts that need the list of valid hobbies, interests, locations, etc.
should import from here so the options stay consistent across:
- The profile generator (scripts/generate_profiles.py)
- The web UI (scripts/profile_builder.py)
- The CLI builder (scripts/profile_builder_cli.py)

WHAT CHANGED AND WHY (v0.2.0):
- INTERESTS_LIKES was split into INTERESTS (broad categories) and FAN_OF
  (specific entities). The old list mixed "Tech" (a category) with
  "Kendrick Lamar" (a specific artist) — that's like putting "fruit" and
  "apple" in the same bucket. Now they're separate and the algorithm can
  score them independently.

- HOMETOWNS and HOMESTATES were merged into CITIES — a list of (city, state)
  tuples. WHY: A city only makes sense paired with its state. The old design
  let you pick "Austin" from one dropdown and "Florida" from another, which
  is nonsensical. Now every city is pre-linked to its correct state.

- FAN_OF was expanded from ~22 items to ~80+ specific entities organized by
  type (artists, shows, sports, etc.) so there's enough variety for 10k
  synthetic profiles to have meaningful overlap patterns.
"""

# ---------------------------------------------------------------------------
# Cities: (city, state) tuples
# ---------------------------------------------------------------------------
# Each city is permanently linked to its state. This prevents mismatches
# and enables the tiered location scoring (same city > same state > neither).
#
# For now this is a curated list of ~50 major U.S. cities. In production,
# you'd load all ~30k U.S. cities from a census dataset. The scoring logic
# doesn't care how many cities there are — it just compares city+state pairs.

CITIES: list[tuple[str, str]] = [
    # Texas — multiple cities to test same-state matching
    ("Austin", "Texas"),
    ("Dallas", "Texas"),
    ("Houston", "Texas"),
    ("San Antonio", "Texas"),
    ("Fort Worth", "Texas"),
    # California
    ("Los Angeles", "California"),
    ("San Francisco", "California"),
    ("San Diego", "California"),
    ("Sacramento", "California"),
    # Florida
    ("Miami", "Florida"),
    ("Tampa", "Florida"),
    ("Orlando", "Florida"),
    ("Jacksonville", "Florida"),
    # New York
    ("New York", "New York"),
    ("Buffalo", "New York"),
    # Georgia
    ("Atlanta", "Georgia"),
    ("Savannah", "Georgia"),
    # Massachusetts
    ("Boston", "Massachusetts"),
    ("Cambridge", "Massachusetts"),
    # Illinois
    ("Chicago", "Illinois"),
    # Pennsylvania
    ("Philadelphia", "Pennsylvania"),
    ("Pittsburgh", "Pennsylvania"),
    # Ohio
    ("Columbus", "Ohio"),
    ("Cleveland", "Ohio"),
    # Arizona
    ("Phoenix", "Arizona"),
    ("Tucson", "Arizona"),
    # Colorado
    ("Denver", "Colorado"),
    ("Boulder", "Colorado"),
    # Washington
    ("Seattle", "Washington"),
    # Oregon
    ("Portland", "Oregon"),
    # North Carolina
    ("Charlotte", "North Carolina"),
    ("Raleigh", "North Carolina"),
    # Virginia
    ("Richmond", "Virginia"),
    # Tennessee
    ("Nashville", "Tennessee"),
    ("Memphis", "Tennessee"),
    # Michigan
    ("Detroit", "Michigan"),
    ("Ann Arbor", "Michigan"),
    # Minnesota
    ("Minneapolis", "Minnesota"),
    # Indiana
    ("Indianapolis", "Indiana"),
    # Missouri
    ("Kansas City", "Missouri"),
    ("St. Louis", "Missouri"),
    # Wisconsin
    ("Milwaukee", "Wisconsin"),
    # Maryland
    ("Baltimore", "Maryland"),
    # Connecticut
    ("Hartford", "Connecticut"),
    # Utah
    ("Salt Lake City", "Utah"),
    # Washington DC (treated as its own "state" for matching purposes)
    ("Washington DC", "District of Columbia"),
    # Oklahoma
    ("Oklahoma City", "Oklahoma"),
    # Louisiana
    ("New Orleans", "Louisiana"),
]

# ---------------------------------------------------------------------------
# Colleges
# ---------------------------------------------------------------------------

COLLEGES = [
    "Boston University", "University of Texas at Austin", "Penn State",
    "UCLA", "NYU", "Georgia Tech", "University of Michigan", "Stanford",
    "UC Berkeley", "University of Florida", "Ohio State", "ASU",
    "University of Washington", "USC", "UT Dallas", "Purdue",
    "University of Colorado Boulder", "Virginia Tech", "UIUC",
    "University of North Carolina", "Cornell", "Emory", "Tulane",
    "Northeastern", "Rice University", "Vanderbilt", "Texas A&M",
    "University of Georgia", "Florida State", "University of Miami",
]

# ---------------------------------------------------------------------------
# Degrees
# ---------------------------------------------------------------------------

DEGREES = [
    "BS Computer Science", "BS Marketing", "BS Information Systems",
    "BS Psychology", "BS Biology", "BS Economics", "BA English",
    "BA Communications", "BS Mechanical Engineering", "BS Finance",
    "MS Data Science", "MS Public Health", "MBA", "MS Computer Science",
    "BS Nursing", "BS Kinesiology", "BA Sociology", "BS Chemistry",
    "BS Electrical Engineering", "BA Political Science", "BS Mathematics",
]

# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

JOBS = [
    "Software Engineer", "Data Analyst", "Product Manager", "DevOps Engineer",
    "Community Manager", "Product Designer", "Marketing Manager",
    "UX Researcher", "Frontend Developer", "Backend Developer",
    "Data Scientist", "Project Manager", "Business Analyst",
    "Sales Representative", "HR Specialist", "Content Strategist",
    "Financial Analyst", "Graphic Designer", "QA Engineer",
    "Technical Writer", "Account Manager", "Operations Manager",
    "Teacher", "Nurse", "Consultant", "Freelancer",
]

# ---------------------------------------------------------------------------
# Hobbies: things you actively DO
# ---------------------------------------------------------------------------
# These are *activities* — you spend time doing them. They answer the
# question "what do you do for fun?" Shared hobbies mean shared time.

HOBBIES = [
    # Outdoor / sports
    "Hiking", "Cycling", "Running", "Swimming", "Yoga", "Basketball",
    "Soccer", "Tennis", "Volleyball", "Pickleball", "Rock Climbing",
    "Surfing", "Skiing", "Snowboarding", "Skateboarding", "Golf",
    # Creative
    "Photography", "Painting", "Drawing", "Cooking", "Baking",
    "Gardening", "Woodworking", "Knitting", "Pottery",
    # Mental / social
    "Reading", "Writing", "Blogging", "Podcasting",
    "Chess", "Board Games", "Video Games", "D&D", "Trivia",
    # Music / performance
    "Singing", "Guitar", "Piano", "Drums", "Dancing",
    # Outdoors
    "Camping", "Fishing", "Kayaking",
    # Lifestyle
    "Meditation", "Journaling", "Volunteering", "Thrifting",
]

# ---------------------------------------------------------------------------
# Interests: broad CATEGORIES of things you enjoy
# ---------------------------------------------------------------------------
# These are *types/genres/categories* — not specific items. They answer
# the question "what kind of stuff are you into?" This captures taste
# alignment at a higher level than specific fandoms.
#
# WHY separate from fan_of: Two people who both like "Hip Hop" have
# compatible taste even if one listens to Kendrick and the other to Drake.
# The interest captures the genre; fan_of captures the specific artist.

INTERESTS = [
    # Music genres
    "Hip Hop", "R&B", "Rock", "Pop", "Country", "EDM",
    "Indie", "Jazz", "Latin Music", "K-Pop", "Classical Music",
    # Food types
    "Japanese Food", "Mexican Food", "Italian Food", "Thai Food",
    "BBQ", "Vegan/Plant-Based", "Brunch Culture", "Street Food",
    "Fine Dining", "Coffee Culture", "Craft Cocktails", "Wine",
    # Activity types
    "Outdoor Adventure", "Team Sports", "Solo Fitness", "Martial Arts",
    "Board/Card Games", "Video Gaming", "Creative Arts", "Live Music",
    # Lifestyle / mindset
    "Tech", "Travel", "Fashion", "Film/TV", "Anime/Manga",
    "True Crime", "Philosophy", "Psychology", "Sustainability",
    "Wellness/Mindfulness", "Astrology", "Interior Design",
    "Architecture", "Street Art", "Nutrition/Health",
    "Entrepreneurship", "Personal Finance",
]

# ---------------------------------------------------------------------------
# Fan Of: SPECIFIC named things you love
# ---------------------------------------------------------------------------
# These are *particular artists, shows, teams, books, podcasts, games*.
# They answer "what specific things are you a fan of?" Sharing a specific
# fandom is a powerful ice-breaker ("Oh you watch The Bear too?!").
#
# WHY separate from interests: Interests capture broad taste. Fan_of
# captures the specific things. You might both like "Hip Hop" (interest)
# but if you're BOTH specifically fans of Kendrick Lamar, that's a
# much stronger connection signal.

FAN_OF = [
    # Music artists
    "Kendrick Lamar", "Taylor Swift", "Coldplay", "SZA", "Bad Bunny",
    "Frank Ocean", "Phoebe Bridgers", "Tyler the Creator", "Doja Cat",
    "Drake", "The Weeknd", "Billie Eilish", "Harry Styles",
    "Olivia Rodrigo", "Post Malone", "Rihanna", "BTS",
    # TV shows
    "Ted Lasso", "The Bear", "Succession", "Stranger Things",
    "The Office", "Breaking Bad", "Game of Thrones", "Abbott Elementary",
    "Severance", "White Lotus", "Yellowjackets", "Shogun",
    # Movies / franchises
    "Dune", "Marvel", "Star Wars", "Lord of the Rings",
    "A24 Films", "Studio Ghibli", "Christopher Nolan",
    # Sports
    "NFL", "NBA", "MLB", "MLS", "Premier League",
    "Formula 1", "UFC", "College Football",
    # Books
    "Atomic Habits", "How to Win Friends", "Sapiens",
    "The Alchemist", "Thinking Fast and Slow",
    # Podcasts / media
    "Joe Rogan", "Lex Fridman", "Call Her Daddy",
    "Huberman Lab", "Conan O'Brien",
    # Games
    "Zelda", "Mario Kart", "Smash Bros", "Minecraft",
    "Elden Ring", "Animal Crossing", "Pokemon",
]

# ---------------------------------------------------------------------------
# Faith / religion
# ---------------------------------------------------------------------------

FAITHS = [
    "Christian", "Catholic", "Jewish", "Muslim", "Hindu", "Buddhist",
    "Agnostic", "Atheist", "Spiritual", "None", "Prefer not to say",
]

# ---------------------------------------------------------------------------
# Travel destinations
# ---------------------------------------------------------------------------

TRAVEL_DESTINATIONS = [
    "Japan", "Spain", "Italy", "France", "Thailand", "Mexico",
    "Portugal", "South Korea", "England", "Netherlands", "Greece",
    "Colombia", "Peru", "Australia", "New Zealand", "Iceland",
    "Morocco", "Bali", "Costa Rica", "Germany", "Ireland",
    "Argentina", "Vietnam", "Croatia", "Switzerland", "Norway",
    "Turkey", "Egypt", "Brazil", "Scotland",
]

# ---------------------------------------------------------------------------
# Names (for synthetic profile generation only)
# ---------------------------------------------------------------------------

FIRST_NAMES = [
    "Aiden", "Aria", "Blake", "Brooke", "Caleb", "Cameron", "Chloe", "Connor",
    "Dakota", "Dana", "Dylan", "Elena", "Eli", "Emma", "Ethan", "Fiona",
    "Gabriel", "Grace", "Harper", "Hudson", "Isaac", "Ivy", "Jack", "Jade",
    "Jordan", "Julia", "Kai", "Kira", "Leo", "Lily", "Logan", "Luna",
    "Mason", "Maya", "Miles", "Mila", "Nathan", "Nora", "Oliver", "Olivia",
    "Owen", "Piper", "Quinn", "Riley", "Ryan", "Sage", "Sam", "Sophia",
    "Theo", "Tyler", "Violet", "Wesley", "Wyatt", "Xander", "Yara", "Zoe",
    "Aisha", "Andre", "Carmen", "Darius", "Emeka", "Fatima", "Gia", "Hector",
    "Ines", "Jamal", "Kenji", "Layla", "Marco", "Nia", "Omar", "Priya",
    "Ravi", "Sana", "Tariq", "Uma", "Valentina", "Wei", "Xiomara", "Yusuf",
]

LAST_NAMES = [
    "Adams", "Bennett", "Carter", "Diaz", "Evans", "Foster", "Garcia", "Hayes",
    "Ishikawa", "Johnson", "Kim", "Lee", "Martinez", "Nguyen", "Owens", "Patel",
    "Quinn", "Ramirez", "Singh", "Torres", "Uribe", "Vargas", "Williams", "Xu",
    "Yamamoto", "Zhang", "Anderson", "Brown", "Clark", "Davis", "Edwards",
    "Franklin", "Gonzalez", "Harris", "Ibrahim", "Jackson", "Khan", "Lopez",
    "Mitchell", "Nelson", "Ortiz", "Phillips", "Reed", "Scott", "Thomas",
    "Walker", "Young", "Chen", "Park", "Nakamura",
]
