#!/usr/bin/env python3
"""Interactive web-based profile builder.

Start the server and open the URL in your browser to build a custom profile,
then run it against the 10k test database to see your top matches.

Usage:
    python scripts/profile_builder.py
    python scripts/profile_builder.py --port 8080
    python scripts/profile_builder.py --csv data/test_profiles_10k.csv

WHAT CHANGED (v0.2.0):
- Location is now a single dropdown of "City, State" pairs (no more separate
  unlinked dropdowns that let you pick "Austin, Florida").
- Interests section replaced with the new 3-field taxonomy: Hobbies,
  Interest Categories, and Fan Of (specific entities).
- Score breakdown now shows all 9 component signals including college,
  faith, and travel matches.
"""

import argparse
import csv
import json
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from hangpost_matching import Location, UserProfile, rank_candidates
from hangpost_matching.loader import load_profiles
from hangpost_matching.options import (
    CITIES, COLLEGES, DEGREES, FAITHS, FAN_OF, HOBBIES,
    INTERESTS, JOBS, TRAVEL_DESTINATIONS,
)

# ---------------------------------------------------------------------------
# Load profiles once at startup
# ---------------------------------------------------------------------------

DATABASE_PROFILES: list[UserProfile] = []
DATABASE_ROWS: dict[str, dict] = {}


def load_database(csv_path: Path) -> None:
    global DATABASE_PROFILES, DATABASE_ROWS
    print(f"Loading profiles from {csv_path}...")
    with csv_path.open() as fh:
        rows = list(csv.DictReader(fh))
    DATABASE_PROFILES = load_profiles(csv_path)
    DATABASE_ROWS = {
        prof.user_id: row
        for prof, row in zip(DATABASE_PROFILES, rows)
    }
    print(f"Loaded {len(DATABASE_PROFILES):,} profiles into memory.")


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def _build_options_html(items: list[str], field_id: str, multi: bool = False) -> str:
    input_type = "checkbox" if multi else "radio"
    html_parts = []
    for item in sorted(items):
        safe = item.replace('"', '&quot;').replace("'", "&#39;")
        html_parts.append(
            f'<label class="option-chip">'
            f'<input type="{input_type}" name="{field_id}" value="{safe}"> '
            f'{safe}</label>'
        )
    return "\n".join(html_parts)


def _build_select_html(items: list[str], field_id: str, placeholder: str = "Select...") -> str:
    opts = [f'<option value="">{placeholder}</option>']
    for item in sorted(items):
        safe = item.replace('"', '&quot;')
        opts.append(f'<option value="{safe}">{safe}</option>')
    return f'<select name="{field_id}" id="{field_id}">{"".join(opts)}</select>'


def _build_city_select() -> str:
    """Build a single dropdown of "City, State" pairs.

    WHY a single dropdown instead of two separate ones:
    Cities are inherently linked to states. Two separate dropdowns allowed
    nonsensical combinations like "Austin, Florida". This dropdown only
    shows valid city+state pairs, stored as "City|State" in the value
    so we can split them apart on the server side.
    """
    opts = ['<option value="">Select city...</option>']
    # Sort by state first, then city, so cities are grouped by state.
    for city, state in sorted(CITIES, key=lambda x: (x[1], x[0])):
        display = f"{city}, {state}"
        value = f"{city}|{state}"
        opts.append(f'<option value="{value}">{display}</option>')
    return f'<select name="location" id="location">{"".join(opts)}</select>'


def build_form_page() -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Hangpost Profile Builder</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #0f1117; color: #e0e0e0; padding: 20px; max-width: 960px; margin: 0 auto; }}
  h1 {{ color: #7c5cfc; margin-bottom: 8px; font-size: 28px; }}
  .subtitle {{ color: #888; margin-bottom: 30px; font-size: 14px; }}
  .section {{ background: #1a1d27; border-radius: 12px; padding: 24px; margin-bottom: 20px;
              border: 1px solid #2a2d3a; }}
  .section h2 {{ color: #a78bfa; font-size: 16px; margin-bottom: 12px; text-transform: uppercase;
                 letter-spacing: 1px; }}
  .row {{ display: flex; gap: 16px; flex-wrap: wrap; }}
  .field {{ flex: 1; min-width: 200px; }}
  .field label {{ display: block; color: #aaa; font-size: 13px; margin-bottom: 4px; }}
  input[type="text"], input[type="number"], select {{
    width: 100%; padding: 10px 12px; background: #12141c; border: 1px solid #333;
    border-radius: 8px; color: #e0e0e0; font-size: 14px; outline: none;
  }}
  input:focus, select:focus {{ border-color: #7c5cfc; }}
  .chips {{ display: flex; flex-wrap: wrap; gap: 6px; }}
  .option-chip {{
    display: inline-flex; align-items: center; padding: 6px 12px;
    background: #12141c; border: 1px solid #333; border-radius: 20px;
    font-size: 13px; cursor: pointer; transition: all 0.15s; user-select: none;
  }}
  .option-chip:hover {{ border-color: #7c5cfc; background: #1e1e30; }}
  .option-chip input {{ display: none; }}
  .option-chip:has(input:checked) {{ background: #7c5cfc22; border-color: #7c5cfc; color: #c4b5fd; }}
  .hint {{ color: #666; font-size: 12px; margin-top: 4px; margin-bottom: 8px; }}
  button[type="submit"] {{
    background: #7c5cfc; color: white; border: none; padding: 14px 40px;
    border-radius: 10px; font-size: 16px; font-weight: 600; cursor: pointer;
    display: block; margin: 30px auto 0; transition: background 0.2s;
  }}
  button[type="submit"]:hover {{ background: #6a4de0; }}
  .results {{ margin-top: 30px; }}
  .match-card {{ background: #1a1d27; border: 1px solid #2a2d3a; border-radius: 12px;
                 padding: 20px; margin-bottom: 16px; position: relative; }}
  .match-card.source-card {{ border-color: #7c5cfc; border-width: 2px; }}
  .rank-badge {{ position: absolute; top: -10px; left: 16px; background: #7c5cfc;
                 color: white; padding: 2px 12px; border-radius: 10px; font-size: 13px; font-weight: 600; }}
  .match-card .name {{ font-size: 18px; font-weight: 600; color: #e0e0e0; }}
  .match-card .score {{ color: #7c5cfc; font-size: 14px; font-weight: 600; float: right; }}
  .match-card .details {{ margin-top: 10px; font-size: 13px; color: #aaa; line-height: 1.8; }}
  .match-card .details span {{ color: #ccc; }}
  .breakdown {{ display: flex; gap: 8px; flex-wrap: wrap; margin-top: 12px; }}
  .breakdown .stat {{ background: #12141c; padding: 6px 10px; border-radius: 8px; font-size: 11px; }}
  .breakdown .stat .val {{ color: #a78bfa; font-weight: 600; }}
  #loading {{ display: none; text-align: center; padding: 40px; color: #888; }}
  .spinner {{ display: inline-block; width: 30px; height: 30px; border: 3px solid #333;
              border-top-color: #7c5cfc; border-radius: 50%;
              animation: spin 0.8s linear infinite; margin-bottom: 12px; }}
  @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
  .tag {{ display: inline-block; background: #7c5cfc22; color: #c4b5fd; padding: 2px 8px;
          border-radius: 4px; font-size: 12px; margin: 2px; }}
</style>
</head>
<body>

<h1>Hangpost Profile Builder</h1>
<p class="subtitle">Build a profile and match it against {len(DATABASE_PROFILES):,} profiles in the database</p>

<form id="profileForm">

<div class="section">
  <h2>Basic Info</h2>
  <div class="row">
    <div class="field">
      <label for="name">Name</label>
      <input type="text" name="name" id="name" placeholder="Your name" required>
    </div>
    <div class="field">
      <label for="age">Age</label>
      <input type="number" name="age" id="age" min="18" max="65" placeholder="25" required>
    </div>
    <div class="field">
      <label for="location">Location</label>
      {_build_city_select()}
    </div>
  </div>
</div>

<div class="section">
  <h2>Education & Work</h2>
  <div class="row">
    <div class="field">
      <label for="college">College</label>
      {_build_select_html(COLLEGES, "college", "Select college...")}
    </div>
    <div class="field">
      <label for="degree">Degree</label>
      {_build_select_html(DEGREES, "degree", "Select degree...")}
    </div>
    <div class="field">
      <label for="job">Job</label>
      {_build_select_html(JOBS, "job", "Select job...")}
    </div>
  </div>
</div>

<div class="section">
  <h2>Faith</h2>
  <div class="chips">
    {_build_options_html(FAITHS, "faith", multi=False)}
  </div>
</div>

<div class="section">
  <h2>Hobbies & Activities</h2>
  <p class="hint">Things you actively DO — select 2-8</p>
  <div class="chips">
    {_build_options_html(HOBBIES, "hobbies", multi=True)}
  </div>
</div>

<div class="section">
  <h2>Interest Categories</h2>
  <p class="hint">Broad types of things you're into — select 3-7</p>
  <div class="chips">
    {_build_options_html(INTERESTS, "interests", multi=True)}
  </div>
</div>

<div class="section">
  <h2>Fan Of</h2>
  <p class="hint">Specific artists, shows, teams, games you love — select 3-8</p>
  <div class="chips">
    {_build_options_html(FAN_OF, "fan_of", multi=True)}
  </div>
</div>

<div class="section">
  <h2>Travel Wishlist</h2>
  <p class="hint">Places you want to visit — select 2-4</p>
  <div class="chips">
    {_build_options_html(TRAVEL_DESTINATIONS, "travel", multi=True)}
  </div>
</div>

<button type="submit">Find My Top 20 Matches</button>
</form>

<div id="loading">
  <div class="spinner"></div>
  <p>Running matching algorithm against {len(DATABASE_PROFILES):,} profiles...</p>
</div>

<div id="results" class="results"></div>

<script>
document.getElementById('profileForm').addEventListener('submit', async (e) => {{
  e.preventDefault();
  const form = e.target;
  const data = new FormData(form);

  const payload = {{
    name: data.get('name'),
    age: parseInt(data.get('age')),
    location: data.get('location'),
    college: data.get('college'),
    degree: data.get('degree'),
    job: data.get('job'),
    faith: data.get('faith') || '',
    hobbies: data.getAll('hobbies'),
    interests: data.getAll('interests'),
    fan_of: data.getAll('fan_of'),
    travel: data.getAll('travel'),
  }};

  form.style.display = 'none';
  document.getElementById('loading').style.display = 'block';
  document.getElementById('results').innerHTML = '';

  try {{
    const resp = await fetch('/api/match', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify(payload),
    }});
    const result = await resp.json();
    document.getElementById('loading').style.display = 'none';
    renderResults(result, payload);
  }} catch (err) {{
    document.getElementById('loading').style.display = 'none';
    document.getElementById('results').innerHTML =
      '<p style="color:red;">Error: ' + err.message + '</p>';
    form.style.display = 'block';
  }}
}});

function makeTags(items) {{
  if (!items || items.length === 0) return '<span style="color:#666;">\u2014</span>';
  return items.map(i => '<span class="tag">' + i + '</span>').join(' ');
}}

function renderResults(result, payload) {{
  const el = document.getElementById('results');
  let html = '';
  const loc = payload.location ? payload.location.replace('|', ', ') : '\u2014';

  // Source card
  html += '<div class="match-card source-card">';
  html += '<div class="rank-badge">YOUR PROFILE</div>';
  html += '<div class="name">' + payload.name + '</div>';
  html += '<div class="details">';
  html += '<strong>Age:</strong> <span>' + payload.age + '</span> &nbsp;|&nbsp; ';
  html += '<strong>Location:</strong> <span>' + loc + '</span> &nbsp;|&nbsp; ';
  html += '<strong>College:</strong> <span>' + (payload.college || '\u2014') + '</span> &nbsp;|&nbsp; ';
  html += '<strong>Job:</strong> <span>' + (payload.job || '\u2014') + '</span><br>';
  html += '<strong>Faith:</strong> <span>' + (payload.faith || '\u2014') + '</span><br>';
  html += '<strong>Hobbies:</strong> ' + makeTags(payload.hobbies) + '<br>';
  html += '<strong>Interests:</strong> ' + makeTags(payload.interests) + '<br>';
  html += '<strong>Fan of:</strong> ' + makeTags(payload.fan_of) + '<br>';
  html += '<strong>Travel:</strong> ' + makeTags(payload.travel);
  html += '</div></div>';

  // Match cards
  for (const m of result.matches) {{
    html += '<div class="match-card">';
    html += '<div class="rank-badge">#' + m.rank + '</div>';
    html += '<div class="score">Score: ' + m.score.toFixed(3) + '</div>';
    html += '<div class="name">' + m.name + '</div>';
    html += '<div class="details">';
    html += '<strong>Age:</strong> <span>' + m.age + ' (gap: ' + m.age_gap + ')</span> &nbsp;|&nbsp; ';
    html += '<strong>Location:</strong> <span>' + m.city + ', ' + m.state + '</span> &nbsp;|&nbsp; ';
    html += '<strong>College:</strong> <span>' + m.college + '</span> &nbsp;|&nbsp; ';
    html += '<strong>Job:</strong> <span>' + m.job + '</span><br>';
    html += '<strong>Faith:</strong> <span>' + m.faith + '</span><br>';
    html += '<strong>Hobbies:</strong> ' + makeTags(m.hobbies.split('; ')) + '<br>';
    html += '<strong>Interests:</strong> ' + makeTags(m.interests.split('; ')) + '<br>';
    html += '<strong>Fan of:</strong> ' + makeTags(m.fan_of.split('; ')) + '<br>';
    html += '<strong>Travel:</strong> ' + makeTags(m.travel.split('; ')) + '<br>';
    html += '<strong>Mutual friends:</strong> <span>' + m.friends_in_common + '</span>';
    html += '</div>';
    html += '<div class="breakdown">';
    html += '<div class="stat">Hobbies <span class="val">' + m.bd.hobby.toFixed(3) + '</span></div>';
    html += '<div class="stat">Interests <span class="val">' + m.bd.interest.toFixed(3) + '</span></div>';
    html += '<div class="stat">Fan Of <span class="val">' + m.bd.fan_of.toFixed(3) + '</span></div>';
    html += '<div class="stat">Friends <span class="val">' + m.bd.mutual.toFixed(3) + '</span></div>';
    html += '<div class="stat">Boost <span class="val">' + m.bd.boost.toFixed(3) + '</span></div>';
    html += '<div class="stat">Location <span class="val">' + m.bd.location.toFixed(3) + '</span></div>';
    html += '<div class="stat">Age <span class="val">' + m.bd.age.toFixed(3) + '</span></div>';
    html += '<div class="stat">College <span class="val">' + m.bd.college.toFixed(3) + '</span></div>';
    html += '<div class="stat">Faith <span class="val">' + m.bd.faith.toFixed(3) + '</span></div>';
    html += '<div class="stat">Travel <span class="val">' + m.bd.travel.toFixed(3) + '</span></div>';
    html += '</div></div>';
  }}

  html += '<button onclick="location.reload()" style="display:block;margin:30px auto;padding:12px 32px;' +
          'background:#333;color:#ccc;border:1px solid #555;border-radius:8px;font-size:14px;cursor:pointer;">' +
          'Build Another Profile</button>';

  el.innerHTML = html;
}}
</script>

</body>
</html>"""


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class ProfileBuilderHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            page = build_form_page()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(page.encode())
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/api/match":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            payload = json.loads(body)

            # Parse the "City|State" location value back into a Location object.
            location = None
            loc_raw = payload.get("location", "")
            if loc_raw and "|" in loc_raw:
                city, state = loc_raw.split("|", 1)
                location = Location(city=city.strip(), state=state.strip())

            # Build UserProfile from the submitted form data.
            source = UserProfile(
                user_id="custom_profile",
                hobbies={h.lower() for h in payload.get("hobbies", [])},
                interests={i.lower() for i in payload.get("interests", [])},
                fan_of={f.lower() for f in payload.get("fan_of", [])},
                location=location,
                age=payload.get("age"),
                mutual_friend_ids=set(),
                college=payload.get("college") or None,
                faith=payload.get("faith") or None,
                travel_wishlist={t.lower() for t in payload.get("travel", [])},
            )

            # Rank against the full database.
            ranked = rank_candidates(source, DATABASE_PROFILES)
            top_20 = ranked[:20]

            # Build JSON response with all profile details + score breakdown.
            matches = []
            for rank, (candidate, bd) in enumerate(top_20, start=1):
                row = DATABASE_ROWS.get(candidate.user_id, {})
                cand_age = int(row.get("age", 0))
                matches.append({
                    "rank": rank,
                    "score": bd.total_score,
                    "name": row.get("name", ""),
                    "age": cand_age,
                    "age_gap": abs((payload.get("age", 0) or 0) - cand_age),
                    "city": row.get("city", ""),
                    "state": row.get("state", ""),
                    "college": row.get("college", ""),
                    "degree": row.get("degree", ""),
                    "job": row.get("job", ""),
                    "faith": row.get("faith", ""),
                    "hobbies": row.get("hobbies", ""),
                    "interests": row.get("interests", ""),
                    "fan_of": row.get("fan_of", ""),
                    "travel": row.get("travel", ""),
                    "friends_in_common": row.get("friends_in_common", "0"),
                    # Shortened keys for the JS breakdown rendering.
                    "bd": {
                        "hobby": bd.hobby_overlap,
                        "interest": bd.interest_overlap,
                        "fan_of": bd.fan_of_overlap,
                        "mutual": bd.mutual_friends,
                        "boost": bd.social_boost,
                        "location": bd.location_match,
                        "age": bd.age_compatibility,
                        "college": bd.college_match,
                        "faith": bd.faith_match,
                        "travel": bd.travel_overlap,
                    },
                })

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"matches": matches}).encode())
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Interactive profile builder web UI.")
    parser.add_argument("--csv", default="data/test_profiles_10k.csv", help="CSV database to match against")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the server on")
    args = parser.parse_args()

    load_database(Path(args.csv))

    server = HTTPServer(("0.0.0.0", args.port), ProfileBuilderHandler)
    print(f"\n  Profile Builder running at: http://localhost:{args.port}")
    print(f"  Open this URL in your browser to build a profile.\n")
    print(f"  Press Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
