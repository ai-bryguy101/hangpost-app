"""Hangpost Matching Engine — Gradio demo for HuggingFace Spaces.

Pick a source profile, see the top-10 candidates the rules-based ranker
recommends, with the full `MatchBreakdown` for every candidate so the
user can see *why* each one ranked where it did. That transparency is
the entire product premise of Hangpost — surfacing it in the demo is
the point of the demo.

The Space installs `hangpost-matching` directly from the GitHub repo
(see `requirements.txt`) so this app stays in sync with the matching
engine without vendoring code.
"""

from __future__ import annotations

import csv
from pathlib import Path

import gradio as gr
from hangpost_matching import (
    MatchBreakdown,
    UserProfile,
    load_profiles_from_csv,
    rank_candidates,
)

REPO_ROOT = Path(__file__).parent
CSV_PATH = REPO_ROOT / "data" / "test_profiles.csv"
TOP_K = 10


# ---------- data loading ----------


def _build_display_names(csv_path: Path) -> dict[str, str]:
    """Map the package's synthetic user_ids back to the CSV `name` field.

    The package's loader hashes user_ids as `csv_<idx>_<slug>`, which is
    great for stability but ugly in a dropdown. This rebuilds a friendly
    display name from the same CSV.
    """
    names: dict[str, str] = {}
    with csv_path.open() as fh:
        for index, row in enumerate(csv.DictReader(fh)):
            user_id = f"csv_{index}_{row['name'].lower().replace(' ', '_')}"
            names[user_id] = row["name"]
    return names


PROFILES: list[UserProfile] = load_profiles_from_csv(CSV_PATH)
DISPLAY_NAMES: dict[str, str] = _build_display_names(CSV_PATH)
PROFILE_BY_ID: dict[str, UserProfile] = {p.user_id: p for p in PROFILES}

# Dropdown choices: (label_shown_to_user, value_returned_to_callback).
DROPDOWN_CHOICES: list[tuple[str, str]] = sorted(
    (
        (f"{DISPLAY_NAMES[p.user_id]} — {p.age or '?'}, {p.hometown or '?'}", p.user_id)
        for p in PROFILES
    ),
    key=lambda pair: pair[0],
)


# ---------- rendering ----------


def _lane_badge(breakdown: MatchBreakdown) -> str:
    """Six-tier badge reflecting the lexicographic sort in rank_candidates.

    Both-background candidates outrank either-background as a HARD tier,
    so the badge surfaces that — otherwise the user can't tell from
    score numbers alone why a candidate ranked higher.
    """
    if breakdown.has_mutual_friends and breakdown.has_both_shared_background:
        return "🌟 **Tier 1 — mutual friend + same hometown AND college**"
    if breakdown.has_mutual_friends and breakdown.has_shared_background:
        return "🟢 **Tier 2 — mutual friend + shared background**"
    if breakdown.has_mutual_friends:
        return "🟢 **Tier 3 — mutual friend**"
    if breakdown.has_both_shared_background:
        return "🔵 **Tier 4 — same hometown AND same college**"
    if breakdown.has_shared_background:
        return "🔵 **Tier 5 — shared hometown OR college**"
    return "⚪ Tier 6 — general compatibility"


def _profile_summary(profile: UserProfile) -> str:
    name = DISPLAY_NAMES.get(profile.user_id, profile.user_id)
    bits = [f"**{name}**"]
    if profile.age is not None:
        bits.append(f"age {profile.age}")
    if profile.hometown:
        bits.append(f"from {profile.hometown}")
    if profile.college:
        bits.append(f"went to {profile.college}")
    return " · ".join(bits)


def _format_component(value: float, weight: float) -> str:
    """Tiny inline bar so the breakdown is scannable, not a wall of decimals."""
    pct = max(0.0, min(1.0, value))
    bars = "█" * int(round(pct * 10))
    pad = "░" * (10 - len(bars))
    return f"`{bars}{pad}` {value:.2f} (weight {weight:.2f})"


def _render_candidate(rank: int, candidate: UserProfile, breakdown: MatchBreakdown) -> str:
    # Weights are the package defaults — kept inline here so the demo is
    # self-contained and recruiters can see exactly what each row weighs.
    parts = [
        f"### #{rank} — {_profile_summary(candidate)}",
        f"{_lane_badge(breakdown)} · **total score {breakdown.total_score:.3f}**",
        "",
        f"- mutual friends ratio: {_format_component(breakdown.mutual_friends, 0.30)}",
        f"- hometown match: {_format_component(breakdown.hometown_match, 0.10)}",
        f"- college match: {_format_component(breakdown.college_match, 0.10)}",
        f"- interest overlap: {_format_component(breakdown.interest_overlap, 0.20)}",
        f"- liked-topic overlap: {_format_component(breakdown.liked_topic_overlap, 0.15)}",
        f"- age compatibility: {_format_component(breakdown.age_compatibility, 0.25)}",
    ]
    if breakdown.social_boost > 0:
        parts.append(f"- 🚀 social boost applied: +{breakdown.social_boost:.2f}")
    return "\n".join(parts)


def rank_for_source(source_id: str) -> str:
    if not source_id:
        return "_Pick a source profile to see recommendations._"
    source = PROFILE_BY_ID[source_id]
    candidates = [p for p in PROFILES if p.user_id != source_id]
    ranked = rank_candidates(source, candidates)[:TOP_K]

    header = (
        f"## Source: {_profile_summary(source)}\n\n"
        f"Top {TOP_K} of {len(candidates)} candidates, ranked by the three-lane "
        "sort (mutual friends → shared hometown/college → everyone else), with "
        "the weighted score breaking ties *within* each lane."
    )
    body = "\n\n---\n\n".join(
        _render_candidate(i + 1, c, b) for i, (c, b) in enumerate(ranked)
    )
    return f"{header}\n\n---\n\n{body}"


# ---------- UI ----------


with gr.Blocks(title="Hangpost Matching Engine") as demo:
    gr.Markdown(
        "# Hangpost Matching Engine — Live Demo\n\n"
        "Pick a source profile and see who the rules-based ranker recommends, "
        "with a full breakdown of *why* each candidate ranked where it did.\n\n"
        "**Tiering policy** (top to bottom, hard invariant — no weight tuning "
        "can break it):\n\n"
        "- 🌟 Tier 1: mutual friend **and** same hometown **and** same college\n"
        "- 🟢 Tier 2: mutual friend + shared hometown OR college\n"
        "- 🟢 Tier 3: mutual friend, no background overlap\n"
        "- 🔵 Tier 4: no mutual + same hometown **and** same college\n"
        "- 🔵 Tier 5: no mutual + same hometown OR same college\n"
        "- ⚪ Tier 6: everyone else\n\n"
        "Within each tier, candidates are ordered by the weighted compatibility "
        "score (hobbies, age closeness, semantic similarity) so the soft "
        "signals still decide who's surfaced first among peers.\n\n"
        "[Source on GitHub →](https://github.com/ai-bryguy101/hangpost-app)"
    )

    with gr.Row():
        source = gr.Dropdown(
            choices=DROPDOWN_CHOICES,
            label="Source profile",
            info="The user we're recommending matches for.",
            value=DROPDOWN_CHOICES[0][1],
        )
    output = gr.Markdown()
    source.change(rank_for_source, inputs=source, outputs=output)
    demo.load(rank_for_source, inputs=source, outputs=output)


if __name__ == "__main__":
    demo.launch()
