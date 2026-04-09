#!/usr/bin/env python3
"""Evaluate match quality: pick 1 source vs 20 candidates, show full profiles,
export a reviewable CSV, and optionally ask an LLM whether the top matches
would actually become friends.

Usage:
    # Basic: terminal output + CSV export
    python scripts/evaluate_matches.py --seed 7

    # With AI evaluation of top 5 (requires ANTHROPIC_API_KEY or OPENAI_API_KEY)
    python scripts/evaluate_matches.py --seed 7 --ai-eval --ai-top 5

    # Use a specific provider / model
    python scripts/evaluate_matches.py --seed 7 --ai-eval --ai-provider openai --ai-model gpt-4o

WHAT CHANGED (v0.2.0):
- Updated to display the new taxonomy fields (hobbies, interests, fan_of)
  and new scoring signals (college_match, faith_match, travel_overlap).
- Location now shows city + state together.
- Score breakdown includes all 9 component signals.
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from hangpost_matching import rank_candidates
from hangpost_matching.loader import profile_from_row

# CSV columns we care about for the "full profile" view.
PROFILE_DISPLAY_FIELDS = [
    "name", "age", "city", "state", "college", "degree", "job", "faith",
    "hobbies", "interests", "fan_of", "travel", "friends_in_common",
]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _print_full_profile(label: str, row: dict[str, str]) -> None:
    """Pretty-print a single profile to the terminal."""
    print(f"\n{'=' * 80}")
    print(f"  {label}: {row['name']}")
    print(f"{'=' * 80}")
    print(f"  Age:        {row['age']}")
    print(f"  Location:   {row['city']}, {row['state']}")
    print(f"  College:    {row['college']}")
    print(f"  Degree:     {row['degree']}")
    print(f"  Job:        {row['job']}")
    print(f"  Faith:      {row['faith']}")
    print(f"  Mutual friends in common: {row['friends_in_common']}")
    print(f"  Hobbies:    {row['hobbies']}")
    print(f"  Interests:  {row['interests']}")
    print(f"  Fan of:     {row['fan_of']}")
    print(f"  Travel:     {row['travel']}")


def _print_ranked_match(rank: int, row: dict[str, str], breakdown, source_age: int | None) -> None:
    """Pretty-print a ranked candidate with score breakdown."""
    cand_age = int(row["age"])
    age_gap = abs((source_age or 0) - cand_age)
    print(f"\n  #{rank} — {row['name']}  (Score: {breakdown.total_score:.3f})")
    print(f"  {'─' * 60}")
    print(f"  Age: {cand_age}  (gap: {age_gap})  |  {row['city']}, {row['state']}")
    print(f"  College: {row['college']}  |  Degree: {row['degree']}")
    print(f"  Job: {row['job']}  |  Faith: {row['faith']}")
    print(f"  Hobbies:   {row['hobbies']}")
    print(f"  Interests: {row['interests']}")
    print(f"  Fan of:    {row['fan_of']}")
    print(f"  Travel:    {row['travel']}")
    print(f"  Mutual friends in common: {row['friends_in_common']}")
    print(f"  --- Score Breakdown ---")
    print(f"    Hobby overlap:     {breakdown.hobby_overlap:.3f}")
    print(f"    Interest overlap:  {breakdown.interest_overlap:.3f}")
    print(f"    Fan-of overlap:    {breakdown.fan_of_overlap:.3f}")
    print(f"    Mutual friends:    {breakdown.mutual_friends:.3f}  (boost: {breakdown.social_boost:.3f})")
    print(f"    Location match:    {breakdown.location_match:.3f}")
    print(f"    Age compatibility: {breakdown.age_compatibility:.3f}")
    print(f"    College match:     {breakdown.college_match:.3f}")
    print(f"    Faith match:       {breakdown.faith_match:.3f}")
    print(f"    Travel overlap:    {breakdown.travel_overlap:.3f}")


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_evaluation_csv(
    output_path: Path,
    source_row: dict[str, str],
    ranked_rows: list[tuple[int, dict[str, str], object]],
) -> None:
    """Write a reviewable CSV with the source profile + all ranked candidates."""
    fieldnames = [
        "role", "rank", "score",
        "hobby_overlap", "interest_overlap", "fan_of_overlap",
        "mutual_friends_score", "social_boost",
        "location_match", "age_compatibility",
        "college_match", "faith_match", "travel_overlap",
    ] + PROFILE_DISPLAY_FIELDS

    with output_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        # Source row (no scores — it's the person we're matching FOR).
        source_out = {f: source_row.get(f, "") for f in PROFILE_DISPLAY_FIELDS}
        source_out["role"] = "SOURCE"
        for score_col in fieldnames[:13]:
            if score_col not in ("role",):
                source_out.setdefault(score_col, "")
        writer.writerow(source_out)

        # Candidate rows with full score breakdowns.
        for rank, row, breakdown in ranked_rows:
            cand_out = {f: row.get(f, "") for f in PROFILE_DISPLAY_FIELDS}
            cand_out["role"] = "CANDIDATE"
            cand_out["rank"] = rank
            cand_out["score"] = f"{breakdown.total_score:.4f}"
            cand_out["hobby_overlap"] = f"{breakdown.hobby_overlap:.4f}"
            cand_out["interest_overlap"] = f"{breakdown.interest_overlap:.4f}"
            cand_out["fan_of_overlap"] = f"{breakdown.fan_of_overlap:.4f}"
            cand_out["mutual_friends_score"] = f"{breakdown.mutual_friends:.4f}"
            cand_out["social_boost"] = f"{breakdown.social_boost:.4f}"
            cand_out["location_match"] = f"{breakdown.location_match:.4f}"
            cand_out["age_compatibility"] = f"{breakdown.age_compatibility:.4f}"
            cand_out["college_match"] = f"{breakdown.college_match:.4f}"
            cand_out["faith_match"] = f"{breakdown.faith_match:.4f}"
            cand_out["travel_overlap"] = f"{breakdown.travel_overlap:.4f}"
            writer.writerow(cand_out)

    print(f"\nExported evaluation CSV -> {output_path}")


# ---------------------------------------------------------------------------
# AI evaluation
# ---------------------------------------------------------------------------

def _build_profile_text(row: dict[str, str]) -> str:
    """Format a profile as readable text for an LLM prompt."""
    return (
        f"Name: {row['name']}\n"
        f"Age: {row['age']}\n"
        f"Location: {row['city']}, {row['state']}\n"
        f"College: {row['college']}\n"
        f"Degree: {row['degree']}\n"
        f"Job: {row['job']}\n"
        f"Faith: {row['faith']}\n"
        f"Hobbies: {row['hobbies']}\n"
        f"Interests: {row['interests']}\n"
        f"Fan of: {row['fan_of']}\n"
        f"Travel: {row['travel']}\n"
        f"Friends in common: {row['friends_in_common']}"
    )


def _build_eval_prompt(source_row: dict[str, str], candidates: list[tuple[int, dict[str, str], object]]) -> str:
    """Build the evaluation prompt for the LLM."""
    source_text = _build_profile_text(source_row)

    candidate_sections = []
    for rank, row, breakdown in candidates:
        candidate_sections.append(
            f"--- Candidate #{rank} (Algorithm Score: {breakdown.total_score:.3f}) ---\n"
            f"{_build_profile_text(row)}"
        )

    candidates_text = "\n\n".join(candidate_sections)

    return f"""You are evaluating a friend-matching algorithm. Below is a SOURCE person's profile, followed by the algorithm's top {len(candidates)} recommended matches (ranked by score).

For each candidate, assess:
1. Would these two people realistically become friends? (Yes / Maybe / Unlikely)
2. What specific things connect them? (shared hobbies, similar age, same city, etc.)
3. What might work against a friendship? (big age gap, nothing in common, etc.)
4. Rate the match quality: Strong / Decent / Weak / Poor

After evaluating each candidate, give an overall assessment:
- Did the algorithm get the ranking order roughly right?
- Which candidates would YOU rank in the top 3 and why?
- Any candidates the algorithm ranked too high or too low?

Be honest and use common sense about real-world friendships.

============================
SOURCE PROFILE
============================
{source_text}

============================
ALGORITHM'S TOP {len(candidates)} MATCHES
============================

{candidates_text}"""


def _call_anthropic(prompt: str, model: str) -> str:
    """Call the Anthropic API."""
    try:
        import anthropic
    except ImportError:
        sys.exit("ERROR: pip install anthropic  (or set --ai-provider openai)")

    client = anthropic.Anthropic()
    message = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def _call_openai(prompt: str, model: str) -> str:
    """Call any OpenAI-compatible API."""
    try:
        import openai
    except ImportError:
        sys.exit("ERROR: pip install openai  (or set --ai-provider anthropic)")

    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def run_ai_evaluation(
    source_row: dict[str, str],
    top_candidates: list[tuple[int, dict[str, str], object]],
    provider: str,
    model: str,
) -> str:
    """Send profiles to an LLM and get a friendship-plausibility assessment."""
    prompt = _build_eval_prompt(source_row, top_candidates)

    print(f"\nSending top {len(top_candidates)} matches to {provider} ({model}) for evaluation...")

    if provider == "anthropic":
        return _call_anthropic(prompt, model)
    elif provider == "openai":
        return _call_openai(prompt, model)
    else:
        sys.exit(f"Unknown AI provider: {provider}. Use 'anthropic' or 'openai'.")


# ---------------------------------------------------------------------------
# Main evaluation flow
# ---------------------------------------------------------------------------

def run_evaluation(
    csv_path: Path,
    sample_size: int,
    seed: int | None,
    output_csv: Path,
    ai_eval: bool,
    ai_top: int,
    ai_provider: str,
    ai_model: str,
) -> None:
    if seed is not None:
        random.seed(seed)

    with csv_path.open() as fh:
        rows = list(csv.DictReader(fh))

    if sample_size < 2:
        raise ValueError("sample_size must be at least 2")
    if sample_size > len(rows):
        raise ValueError(f"sample_size={sample_size} exceeds row count={len(rows)}")

    sampled_rows = random.sample(rows, sample_size)
    profiles = [profile_from_row(i, row) for i, row in enumerate(sampled_rows)]
    row_by_id = {profile.user_id: row for profile, row in zip(profiles, sampled_rows)}

    source = profiles[0]
    source_row = sampled_rows[0]
    candidates = profiles[1:]
    ranked = rank_candidates(source, candidates)

    # --- Terminal output ---
    _print_full_profile("SOURCE PROFILE", source_row)

    print(f"\n{'=' * 80}")
    print(f"  RANKED MATCHES (1 to {len(ranked)})")
    print(f"{'=' * 80}")

    ranked_rows = []
    for rank, (candidate, breakdown) in enumerate(ranked, start=1):
        row = row_by_id[candidate.user_id]
        _print_ranked_match(rank, row, breakdown, source.age)
        ranked_rows.append((rank, row, breakdown))

    # --- CSV export ---
    export_evaluation_csv(output_csv, source_row, ranked_rows)

    # --- AI evaluation ---
    if ai_eval:
        top_n = ranked_rows[:ai_top]
        result = run_ai_evaluation(source_row, top_n, ai_provider, ai_model)
        print(f"\n{'=' * 80}")
        print(f"  AI EVALUATION ({ai_provider} / {ai_model})")
        print(f"{'=' * 80}")
        print(result)

        ai_output_path = output_csv.with_suffix(".ai_eval.txt")
        with ai_output_path.open("w") as fh:
            fh.write(f"Provider: {ai_provider}\nModel: {ai_model}\n")
            fh.write(f"Source: {source_row['name']}\n")
            fh.write(f"Top {ai_top} candidates evaluated\n")
            fh.write(f"{'=' * 60}\n\n")
            fh.write(result)
        print(f"\nAI evaluation saved -> {ai_output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate match quality: full profiles, CSV export, optional AI review.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--csv", default="data/test_profiles_10k.csv", help="Path to CSV dataset")
    parser.add_argument("--sample-size", type=int, default=21, help="1 source + N candidates (default: 21 = 1+20)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible sampling")
    parser.add_argument("--output", default="data/evaluation_output.csv", help="Output CSV path")

    ai_group = parser.add_argument_group("AI evaluation")
    ai_group.add_argument("--ai-eval", action="store_true", help="Enable AI evaluation of top matches")
    ai_group.add_argument("--ai-top", type=int, default=5, help="How many top matches to send to AI (default: 5)")
    ai_group.add_argument(
        "--ai-provider", default="anthropic", choices=["anthropic", "openai"],
        help="LLM provider (default: anthropic). 'openai' works with any OpenAI-compatible API.",
    )
    ai_group.add_argument("--ai-model", default=None, help="Model name override")

    args = parser.parse_args()

    if args.ai_model is None:
        args.ai_model = {
            "anthropic": "claude-sonnet-4-20250514",
            "openai": "gpt-4o",
        }[args.ai_provider]

    run_evaluation(
        csv_path=Path(args.csv),
        sample_size=args.sample_size,
        seed=args.seed,
        output_csv=Path(args.output),
        ai_eval=args.ai_eval,
        ai_top=args.ai_top,
        ai_provider=args.ai_provider,
        ai_model=args.ai_model,
    )


if __name__ == "__main__":
    main()
