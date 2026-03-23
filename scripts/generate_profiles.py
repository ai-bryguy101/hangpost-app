#!/usr/bin/env python3
"""Generate a large synthetic CSV dataset of user profiles.

Usage:
    python scripts/generate_profiles.py                   # default 10,000 profiles
    python scripts/generate_profiles.py --count 5000 --seed 123
    python scripts/generate_profiles.py --output data/custom.csv

The distributions are designed to be realistic:
- Most users (90%) have 0 mutual friends; ~8% have 1; ~2% have 2-5
- Age follows a roughly normal distribution centered around 28
- Interests, topics, and locations are drawn from curated pools
"""

import argparse
import csv
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Data pools (curated for realistic variety)
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

COLLEGES = [
    "Boston University", "University of Texas at Austin", "Penn State",
    "UCLA", "NYU", "Georgia Tech", "University of Michigan", "Stanford",
    "UC Berkeley", "University of Florida", "Ohio State", "ASU",
    "University of Washington", "USC", "UT Dallas", "Purdue",
    "University of Colorado Boulder", "Virginia Tech", "UIUC",
    "University of North Carolina", "Cornell", "Emory", "Tulane",
    "Northeastern", "Rice University", "Vanderbilt",
]

HOMETOWNS = [
    "Austin", "Atlanta", "Boston", "Chicago", "Dallas", "Denver",
    "Houston", "Los Angeles", "Miami", "Nashville", "New York",
    "Philadelphia", "Phoenix", "Portland", "San Diego", "San Francisco",
    "Seattle", "Tampa", "Washington DC", "Minneapolis", "Detroit",
    "Charlotte", "Raleigh", "Salt Lake City", "Columbus", "Indianapolis",
    "San Antonio", "Jacksonville", "Fort Worth", "Oklahoma City",
]

HOMESTATES = [
    "Texas", "California", "Florida", "New York", "Georgia",
    "Massachusetts", "Illinois", "Pennsylvania", "Ohio", "Arizona",
    "Colorado", "Washington", "Oregon", "North Carolina", "Virginia",
    "Tennessee", "Michigan", "Minnesota", "Indiana", "Missouri",
    "Wisconsin", "Maryland", "New Jersey", "Connecticut", "Utah",
]

DEGREES = [
    "BS Computer Science", "BS Marketing", "BS Information Systems",
    "BS Psychology", "BS Biology", "BS Economics", "BA English",
    "BA Communications", "BS Mechanical Engineering", "BS Finance",
    "MS Data Science", "MS Public Health", "MBA", "MS Computer Science",
    "BS Nursing", "BS Kinesiology", "BA Sociology", "BS Chemistry",
    "BS Electrical Engineering", "BA Political Science", "BS Mathematics",
]

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

HOBBIES = [
    "Hiking", "Cycling", "Running", "Swimming", "Yoga", "Basketball",
    "Soccer", "Tennis", "Volleyball", "Pickleball", "Rock Climbing",
    "Surfing", "Skiing", "Snowboarding", "Skateboarding", "Golf",
    "Photography", "Painting", "Drawing", "Cooking", "Baking",
    "Gardening", "Reading", "Writing", "Blogging", "Podcasting",
    "Chess", "Board Games", "Video Games", "D&D", "Trivia",
    "Singing", "Guitar", "Piano", "Drums", "Dancing",
    "Camping", "Fishing", "Kayaking", "Woodworking", "Knitting",
    "Meditation", "Journaling", "Volunteering", "Thrifting",
]

SKILLS_CERTS = [
    "Python", "JavaScript", "React", "SQL", "AWS Practitioner",
    "Tableau", "Public Speaking", "CPR Certified", "Excel",
    "Figma", "Adobe Suite", "Scrum Master", "Google Analytics",
    "First Aid", "Project Management", "Data Visualization",
]

INTERESTS_LIKES = [
    "Sushi", "Ramen", "BBQ", "Tacos", "Pizza", "Pasta", "Thai Food",
    "Craft Cocktails", "Wine Bars", "Coffee Culture", "Boba Tea",
    "Kendrick Lamar", "Taylor Swift", "Coldplay", "SZA", "Bad Bunny",
    "Frank Ocean", "Phoebe Bridgers", "Tyler the Creator", "Doja Cat",
    "Tech", "Travel", "Fashion", "Film", "Anime", "True Crime",
    "Stoicism", "Mindfulness", "Philosophy", "Psychology",
    "Astrology", "Sustainability", "Fitness", "Nutrition",
    "Interior Design", "Architecture", "Street Art",
    "Zelda", "Mario Kart", "Smash Bros", "Minecraft",
]

FAN_OF = [
    "NFL", "NBA", "MLB", "MLS", "Premier League",
    "Ted Lasso", "The Bear", "Succession", "Stranger Things",
    "Dune", "Marvel", "Star Wars", "Lord of the Rings",
    "Atomic Habits", "How to Win Friends", "Sapiens",
    "Joe Rogan", "Lex Fridman", "Call Her Daddy",
    "Coldplay", "Phoebe Bridgers", "Bad Bunny",
]

FAITHS = [
    "Christian", "Catholic", "Jewish", "Muslim", "Hindu", "Buddhist",
    "Agnostic", "Atheist", "Spiritual", "None", "Prefer not to say",
]

TRAVEL_DESTINATIONS = [
    "Japan", "Spain", "Italy", "France", "Thailand", "Mexico",
    "Portugal", "Lisbon", "Seoul", "London", "Amsterdam", "Greece",
    "Colombia", "Peru", "Australia", "New Zealand", "Iceland",
    "Morocco", "Bali", "Costa Rica", "Austin", "NYC", "Denver",
    "Nashville", "Vancouver", "Berlin", "Barcelona", "Tokyo",
    "Buenos Aires", "Dublin",
]


# ---------------------------------------------------------------------------
# Generation logic
# ---------------------------------------------------------------------------

def _pick_semicolon_list(pool: list[str], min_count: int, max_count: int) -> str:
    """Pick a random subset and join with '; '."""
    k = random.randint(min_count, max_count)
    return "; ".join(random.sample(pool, min(k, len(pool))))


def _generate_friends_in_common() -> int:
    """Realistic distribution: 90% have 0, ~8% have 1, ~2% have 2-5."""
    roll = random.random()
    if roll < 0.90:
        return 0
    elif roll < 0.98:
        return 1
    else:
        return random.randint(2, 5)


def _generate_age() -> int:
    """Age distribution roughly normal, centered at 28, range 18-65."""
    age = int(random.gauss(28, 6))
    return max(18, min(65, age))


def generate_row() -> dict[str, str]:
    """Generate a single synthetic profile row."""
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    name = f"{first} {last}"

    hobbies = random.sample(HOBBIES, random.randint(2, 5))
    skills = random.sample(SKILLS_CERTS, random.randint(0, 3))
    hobbies_combined = "; ".join(hobbies + skills)

    return {
        "name": name,
        "friends_in_common": str(_generate_friends_in_common()),
        "age": str(_generate_age()),
        "college": random.choice(COLLEGES),
        "hometown": random.choice(HOMETOWNS),
        "degree": random.choice(DEGREES),
        "job": random.choice(JOBS),
        "homestate": random.choice(HOMESTATES),
        "hobbies_activities_sports_games_skills_certifications": hobbies_combined,
        "interests_likes": _pick_semicolon_list(INTERESTS_LIKES, 3, 7),
        "fan_of": _pick_semicolon_list(FAN_OF, 2, 5),
        "faith_religion": random.choice(FAITHS),
        "travel": _pick_semicolon_list(TRAVEL_DESTINATIONS, 2, 4),
    }


def generate_csv(output_path: Path, count: int, seed: int | None = None) -> None:
    """Write *count* synthetic profiles to a CSV file."""
    if seed is not None:
        random.seed(seed)

    fieldnames = [
        "name", "friends_in_common", "age", "college", "hometown",
        "degree", "job", "homestate",
        "hobbies_activities_sports_games_skills_certifications",
        "interests_likes", "fan_of", "faith_religion", "travel",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for _ in range(count):
            writer.writerow(generate_row())

    print(f"Wrote {count:,} profiles to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic test profiles.")
    parser.add_argument("--count", type=int, default=10_000, help="Number of profiles to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output", default="data/test_profiles_10k.csv", help="Output CSV path")
    args = parser.parse_args()

    generate_csv(Path(args.output), args.count, args.seed)


if __name__ == "__main__":
    main()
