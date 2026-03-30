"""Shared option pools for profile generation and interactive builders.

All scripts that need the list of valid hobbies, interests, locations, etc.
should import from here so the options stay consistent.
"""

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
