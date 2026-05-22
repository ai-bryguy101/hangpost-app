"""Generate the headline SVG plots used in README.md.

Why SVG and not matplotlib?
---------------------------
SVGs render crisp at any zoom level on GitHub, weigh a few kilobytes
instead of a multi-hundred-KB PNG, and don't require matplotlib (which
is a heavy install just to draw four bar charts). The plotting helpers
here are deliberately tiny — they exist to ship the README, not to be a
general charting library.

Output
------
    docs/img/ndcg_by_label_set.svg
    docs/img/ablation.svg
    docs/img/ndcg_vs_k.svg
    docs/img/phase_comparison.svg     (requires [ml] extra + --learned-model)

Run
---
    python scripts/make_plots.py
    python scripts/make_plots.py --with-embeddings \
        --learned-model models/learned_ranker.joblib
    python scripts/make_plots.py --labels data/judge_labels.jsonl \
        --with-embeddings --learned-model models/learned_ranker.joblib

When `--labels` is set, the phase-comparison chart is evaluated against
the LLM-judge labels instead of `simulated`. That's the honest "Phase 3
beats Phase 1 by X" number you want in the README hero.

The script regenerates plots from scratch every run, so the README is
always in sync with the current ranker code.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from hangpost_matching import (  # noqa: E402
    UserProfile,
    ablate_weights,
    build_queries,
    evaluate_ranker,
    get_relevance_fn,
    load_profiles_from_csv,
    load_verdicts,
    make_random_ranker,
    make_rules_ranker,
    queries_from_verdicts,
)

OUT_DIR = REPO_ROOT / "docs" / "img"

# Brand-ish palette. Picked for contrast on both light and dark GitHub themes.
COLOR_RANDOM = "#9CA3AF"  # neutral gray — baseline
COLOR_RULES = "#2563EB"  # blue — Phase 1
COLOR_RULES_EMB = "#7C3AED"  # purple — Phase 2 (rules + embeddings)
COLOR_LEARNED = "#059669"  # emerald — Phase 3 (LightGBM)
COLOR_POSITIVE_DELTA = "#DC2626"  # red — feature *helps* (worse without it)
COLOR_NEGATIVE_DELTA = "#10B981"  # green — feature *hurts* (better without it)
COLOR_AXIS = "#374151"
COLOR_GRID = "#E5E7EB"


# ---------- tiny SVG helpers ----------


def _svg_open(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'role="img" aria-labelledby="title" '
        f'font-family="-apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif">',
        f'<title id="title">{title}</title>',
        f'<rect width="{width}" height="{height}" fill="white"/>',
    ]


def _text(
    x: float,
    y: float,
    s: str,
    *,
    size: int = 12,
    anchor: str = "start",
    fill: str = COLOR_AXIS,
    weight: str = "normal",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
        f'text-anchor="{anchor}" fill="{fill}" font-weight="{weight}">{s}</text>'
    )


def _rect(x: float, y: float, w: float, h: float, fill: str) -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{fill}"/>'


def _line(x1: float, y1: float, x2: float, y2: float, stroke: str, width: float = 1.0) -> str:
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{width}"/>'
    )


def _save(parts: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n</svg>\n")


# ---------- chart 1: NDCG@10 by label set ----------


def _collect_ndcg_by_label_set(
    profiles: list[UserProfile], queries_n: int, k: int, seed: int
) -> dict[str, dict[str, float]]:
    """For each label generator, run both rankers and collect NDCG@k."""
    out: dict[str, dict[str, float]] = {}
    for label_set in ("rule_based", "noisy", "simulated"):
        relevance_fn = get_relevance_fn(label_set, seed)
        queries = build_queries(profiles, queries_n, seed, relevance_fn=relevance_fn)
        queries = [q for q in queries if q[2]]
        random_ndcg = evaluate_ranker(make_random_ranker(seed), queries, k=k).ndcg
        rules_ndcg = evaluate_ranker(make_rules_ranker(), queries, k=k).ndcg
        out[label_set] = {"random": random_ndcg, "rules_only": rules_ndcg}
    return out


def plot_ndcg_by_label_set(data: dict[str, dict[str, float]], k: int, path: Path) -> None:
    width, height = 720, 360
    margin_top, margin_left, margin_right, margin_bottom = 60, 56, 24, 80

    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    label_sets = list(data.keys())
    group_w = plot_w / len(label_sets)
    bar_w = group_w * 0.32
    bar_gap = group_w * 0.06

    parts = _svg_open(width, height, f"NDCG@{k} by label set")
    parts.append(
        _text(
            width / 2,
            28,
            f"NDCG@{k}: rules vs random across three label sets",
            size=16,
            anchor="middle",
            weight="bold",
        )
    )

    # y-axis (0 to 1)
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        ty = margin_top + plot_h - tick * plot_h
        parts.append(_line(margin_left, ty, margin_left + plot_w, ty, COLOR_GRID))
        parts.append(_text(margin_left - 8, ty + 4, f"{tick:.2f}", size=11, anchor="end"))

    # bars
    for gi, label_set in enumerate(label_sets):
        gx = margin_left + gi * group_w + (group_w - 2 * bar_w - bar_gap) / 2
        for bi, (name, color) in enumerate([("random", COLOR_RANDOM), ("rules_only", COLOR_RULES)]):
            v = data[label_set][name]
            bh = v * plot_h
            bx = gx + bi * (bar_w + bar_gap)
            by = margin_top + plot_h - bh
            parts.append(_rect(bx, by, bar_w, bh, color))
            parts.append(_text(bx + bar_w / 2, by - 4, f"{v:.3f}", size=10, anchor="middle"))
        # group label
        parts.append(
            _text(
                margin_left + gi * group_w + group_w / 2,
                margin_top + plot_h + 22,
                label_set,
                size=12,
                anchor="middle",
                weight="bold",
            )
        )

    # legend
    lx = margin_left
    ly = height - 18
    parts.append(_rect(lx, ly - 10, 12, 12, COLOR_RANDOM))
    parts.append(_text(lx + 18, ly, "random baseline", size=11))
    parts.append(_rect(lx + 150, ly - 10, 12, 12, COLOR_RULES))
    parts.append(_text(lx + 168, ly, "rules (Phase 1)", size=11))

    parts.append(
        _text(
            width / 2,
            margin_top + plot_h + 56,
            "rule_based and noisy share signals with the ranker (inflated). "
            "simulated holds out a hidden personality vector → realistic ceiling.",
            size=10,
            anchor="middle",
            fill="#6B7280",
        )
    )

    _save(parts, path)


# ---------- chart 2: ablation ----------


def plot_ablation(
    profiles: list[UserProfile], queries_n: int, k: int, seed: int, path: Path
) -> None:
    relevance_fn = get_relevance_fn("simulated", seed)
    queries = build_queries(profiles, queries_n, seed, relevance_fn=relevance_fn)
    queries = [q for q in queries if q[2]]
    rows = ablate_weights(queries, profile_embeddings=None, k=k)
    # First row is the full-weights baseline (feature == "<full>"). Skip it.
    ablated = [r for r in rows if r.feature != "<full>"]

    width, height = 720, max(280, 60 + 28 * len(ablated) + 40)
    margin_top, margin_left, margin_right, margin_bottom = 56, 180, 24, 40

    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    # delta range — symmetric around 0
    deltas = [r.delta_ndcg for r in ablated]
    max_abs = max(0.001, max(abs(d) for d in deltas))
    # round up to a nice tick
    tick = 0.005
    while tick < max_abs:
        tick *= 2
    max_abs = tick

    parts = _svg_open(width, height, f"Per-feature ablation (ΔNDCG@{k})")
    parts.append(
        _text(
            width / 2,
            24,
            f"Per-feature ablation — ΔNDCG@{k} when each weight is zeroed",
            size=15,
            anchor="middle",
            weight="bold",
        )
    )
    parts.append(
        _text(
            width / 2,
            42,
            "Red bars to the right: removing the feature made the ranker worse "
            "(so it was helping).",
            size=10,
            anchor="middle",
            fill="#6B7280",
        )
    )

    zero_x = margin_left + plot_w / 2

    # grid ticks
    for frac in (-1.0, -0.5, 0.0, 0.5, 1.0):
        tx = zero_x + frac * (plot_w / 2)
        parts.append(_line(tx, margin_top, tx, margin_top + plot_h, COLOR_GRID))
        v = frac * max_abs
        parts.append(_text(tx, margin_top + plot_h + 16, f"{v:+.3f}", size=10, anchor="middle"))

    # bars
    row_h = plot_h / len(ablated)
    bar_h = row_h * 0.6
    for i, r in enumerate(ablated):
        cy = margin_top + i * row_h + row_h / 2
        bx = zero_x
        bw = (r.delta_ndcg / max_abs) * (plot_w / 2)
        if bw < 0:
            bx = zero_x + bw
            bw = -bw
            color = COLOR_NEGATIVE_DELTA
        else:
            color = COLOR_POSITIVE_DELTA
        parts.append(_rect(bx, cy - bar_h / 2, max(bw, 1.5), bar_h, color))
        parts.append(_text(margin_left - 8, cy + 4, r.feature, size=11, anchor="end"))
        label_x = bx + bw + 6 if r.delta_ndcg >= 0 else bx - 6
        anchor = "start" if r.delta_ndcg >= 0 else "end"
        parts.append(_text(label_x, cy + 4, f"{r.delta_ndcg:+.3f}", size=10, anchor=anchor))

    # zero axis
    parts.append(_line(zero_x, margin_top, zero_x, margin_top + plot_h, COLOR_AXIS, 1.5))

    _save(parts, path)


# ---------- chart 3: NDCG vs k ----------


def plot_ndcg_vs_k(
    profiles: list[UserProfile],
    queries_n: int,
    ks: Sequence[int],
    seed: int,
    path: Path,
) -> None:
    # Use simulated labels for the honest ceiling story.
    relevance_fn = get_relevance_fn("simulated", seed)
    queries = build_queries(profiles, queries_n, seed, relevance_fn=relevance_fn)
    queries = [q for q in queries if q[2]]

    series: dict[str, list[float]] = {"random": [], "rules_only": []}
    for k in ks:
        series["random"].append(evaluate_ranker(make_random_ranker(seed), queries, k=k).ndcg)
        series["rules_only"].append(evaluate_ranker(make_rules_ranker(), queries, k=k).ndcg)

    width, height = 720, 360
    margin_top, margin_left, margin_right, margin_bottom = 56, 56, 24, 64

    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    parts = _svg_open(width, height, "NDCG vs k (simulated labels)")
    parts.append(
        _text(
            width / 2,
            24,
            "NDCG vs k — rules vs random (simulated labels)",
            size=16,
            anchor="middle",
            weight="bold",
        )
    )
    parts.append(
        _text(
            width / 2,
            42,
            "Simulated labels mix observable affinity with a hidden personality "
            "vector — a realistic ceiling.",
            size=10,
            anchor="middle",
            fill="#6B7280",
        )
    )

    # y-axis (auto-scaled around the data — clipped to 0..1)
    y_max = min(1.0, max(max(series["random"]), max(series["rules_only"])) * 1.15)
    y_max = round(y_max * 10 + 0.5) / 10  # round up to nearest 0.1
    for tick in [t / 10 for t in range(0, int(y_max * 10) + 1, 1)]:
        ty = margin_top + plot_h - (tick / y_max) * plot_h
        parts.append(_line(margin_left, ty, margin_left + plot_w, ty, COLOR_GRID))
        parts.append(_text(margin_left - 8, ty + 4, f"{tick:.1f}", size=11, anchor="end"))

    # x-axis ticks at the actual k values
    n = len(ks)
    for i, k in enumerate(ks):
        x = margin_left + (i / (n - 1)) * plot_w
        parts.append(_text(x, margin_top + plot_h + 18, f"k={k}", size=11, anchor="middle"))

    def point(i: int, v: float) -> tuple[float, float]:
        x = margin_left + (i / (n - 1)) * plot_w
        y = margin_top + plot_h - (v / y_max) * plot_h
        return x, y

    for name, color in [("random", COLOR_RANDOM), ("rules_only", COLOR_RULES)]:
        vals = series[name]
        for i in range(n - 1):
            x1, y1 = point(i, vals[i])
            x2, y2 = point(i + 1, vals[i + 1])
            parts.append(_line(x1, y1, x2, y2, color, 2.5))
        for i, v in enumerate(vals):
            x, y = point(i, v)
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
            parts.append(_text(x, y - 10, f"{v:.2f}", size=10, anchor="middle", fill=color))

    # legend
    lx = margin_left
    ly = height - 14
    parts.append(_rect(lx, ly - 10, 12, 12, COLOR_RANDOM))
    parts.append(_text(lx + 18, ly, "random baseline", size=11))
    parts.append(_rect(lx + 150, ly - 10, 12, 12, COLOR_RULES))
    parts.append(_text(lx + 168, ly, "rules (Phase 1)", size=11))

    _save(parts, path)


# ---------- chart 4: phase comparison (random / rules / +embeddings / learned) ----------


def _collect_phase_comparison(
    profiles: list[UserProfile],
    queries_n: int,
    k: int,
    seed: int,
    with_embeddings: bool,
    learned_model_path: Path | None,
    labels_path: Path | None,
    label_threshold: int | None = None,
) -> tuple[dict[str, float], str]:
    """Run every available ranker on the chosen label set and collect NDCG@k.

    Returns the score map plus a human-readable label-set name for the
    chart subtitle.

    Phase 2/3 rows are only added when the [ml] extra is importable. If
    the import fails, the chart still renders with just random + rules.
    """
    if labels_path is not None:
        verdicts = load_verdicts(labels_path)
        if not verdicts:
            raise SystemExit(
                f"--labels {labels_path} contains no verdicts. Run scripts/label.py first."
            )
        from hangpost_matching import DEFAULT_RELEVANCE_THRESHOLD

        threshold = label_threshold if label_threshold is not None else DEFAULT_RELEVANCE_THRESHOLD
        queries = queries_from_verdicts(profiles, verdicts, threshold=threshold)
        queries = [q for q in queries if q[2]]
        label_name = f"LLM-judge labels ({labels_path.name}, rating≥{threshold})"
    else:
        relevance_fn = get_relevance_fn("simulated", seed)
        queries = build_queries(profiles, queries_n, seed, relevance_fn=relevance_fn)
        queries = [q for q in queries if q[2]]
        label_name = "simulated labels (logistic mixture + hidden personality)"

    if not queries:
        raise SystemExit(
            "No queries with positive labels — nothing to plot. Check the label set / threshold."
        )

    scores: dict[str, float] = {
        "random": evaluate_ranker(make_random_ranker(seed), queries, k=k).ndcg,
        "rules (Phase 1)": evaluate_ranker(make_rules_ranker(), queries, k=k).ndcg,
    }

    if with_embeddings or learned_model_path is not None:
        try:
            from hangpost_matching import SentenceTransformerEmbedder, embed_profiles
        except ImportError:
            print(
                "  [skip] sentence-transformers not installed — "
                "phase comparison will only show Phase 1.",
                file=sys.stderr,
            )
            return scores, label_name

        print("  embedding profiles for Phase 2/3 (one-time, ~seconds-to-minutes)...")
        embedder = SentenceTransformerEmbedder()
        embeddings = embed_profiles(profiles, embedder)

        scores["+ embeddings (Phase 2)"] = evaluate_ranker(
            make_rules_ranker(profile_embeddings=embeddings), queries, k=k
        ).ndcg

        if learned_model_path is not None:
            try:
                from hangpost_matching import LearnedRanker
            except ImportError:
                print(
                    "  [skip] lightgbm not installed — Phase 3 omitted.",
                    file=sys.stderr,
                )
            else:
                if not learned_model_path.exists():
                    raise SystemExit(
                        f"--learned-model {learned_model_path} does not exist. "
                        "Run scripts/train.py first."
                    )
                learned = LearnedRanker.load(learned_model_path)
                learned.profile_embeddings = embeddings
                scores["learned (Phase 3)"] = evaluate_ranker(
                    learned.as_ranker(), queries, k=k
                ).ndcg

    return scores, label_name


def plot_phase_comparison(
    scores: dict[str, float],
    k: int,
    label_name: str,
    path: Path,
) -> None:
    """Vertical bar chart of NDCG@k for every phase that was evaluated.

    The whole point of the project is "I built three phases and measured
    each one." This chart makes that headline visible at the top of the
    README without anyone having to open a notebook.
    """
    palette = {
        "random": COLOR_RANDOM,
        "rules (Phase 1)": COLOR_RULES,
        "+ embeddings (Phase 2)": COLOR_RULES_EMB,
        "learned (Phase 3)": COLOR_LEARNED,
    }

    names = list(scores.keys())
    width, height = 720, 380
    margin_top, margin_left, margin_right, margin_bottom = 70, 56, 24, 86

    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    parts = _svg_open(width, height, f"NDCG@{k} by ranker phase")
    parts.append(
        _text(
            width / 2,
            28,
            f"NDCG@{k} by ranker phase",
            size=16,
            anchor="middle",
            weight="bold",
        )
    )
    parts.append(
        _text(
            width / 2,
            48,
            f"Evaluated on {label_name}.",
            size=10,
            anchor="middle",
            fill="#6B7280",
        )
    )

    # y-axis (0..1)
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        ty = margin_top + plot_h - tick * plot_h
        parts.append(_line(margin_left, ty, margin_left + plot_w, ty, COLOR_GRID))
        parts.append(_text(margin_left - 8, ty + 4, f"{tick:.2f}", size=11, anchor="end"))

    # bars
    n = len(names)
    slot_w = plot_w / max(n, 1)
    bar_w = slot_w * 0.55
    for i, name in enumerate(names):
        v = scores[name]
        color = palette.get(name, COLOR_RULES)
        bx = margin_left + i * slot_w + (slot_w - bar_w) / 2
        bh = v * plot_h
        by = margin_top + plot_h - bh
        parts.append(_rect(bx, by, bar_w, bh, color))
        parts.append(_text(bx + bar_w / 2, by - 6, f"{v:.3f}", size=11, anchor="middle"))
        parts.append(
            _text(
                margin_left + i * slot_w + slot_w / 2,
                margin_top + plot_h + 22,
                name,
                size=11,
                anchor="middle",
                weight="bold",
            )
        )

    parts.append(
        _text(
            width / 2,
            margin_top + plot_h + 60,
            "Higher is better. Each phase adds a layer on top of the previous one.",
            size=10,
            anchor="middle",
            fill="#6B7280",
        )
    )

    _save(parts, path)


# ---------- main ----------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=REPO_ROOT / "data" / "test_profiles.csv",
        help="Seed profile CSV.",
    )
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--with-embeddings",
        action="store_true",
        help="Generate the phase-comparison chart with Phase 2 (requires [ml] extra).",
    )
    parser.add_argument(
        "--learned-model",
        type=Path,
        default=None,
        help="Path to a saved LearnedRanker to include as Phase 3 (requires [ml] extra).",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help=(
            "Path to LLM-judge JSONL labels. When set, the phase-comparison "
            "chart is evaluated against these labels instead of `simulated`."
        ),
    )
    parser.add_argument(
        "--label-threshold",
        type=int,
        default=None,
        help=(
            "Minimum judge rating (0-4) to count as relevant when --labels is "
            "set. Defaults to the package default (3). Use 4 for strict "
            "'strong match only' grading."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    profiles = load_profiles_from_csv(args.csv)
    print(f"Loaded {len(profiles)} profiles.")

    print("Generating NDCG-by-label-set chart...")
    data = _collect_ndcg_by_label_set(profiles, args.queries, args.k, args.seed)
    plot_ndcg_by_label_set(data, args.k, OUT_DIR / "ndcg_by_label_set.svg")

    print("Generating ablation chart...")
    plot_ablation(profiles, args.queries, args.k, args.seed, OUT_DIR / "ablation.svg")

    print("Generating NDCG-vs-k chart...")
    plot_ndcg_vs_k(profiles, args.queries, [5, 10, 20, 50], args.seed, OUT_DIR / "ndcg_vs_k.svg")

    print("Generating phase-comparison chart...")
    scores, label_name = _collect_phase_comparison(
        profiles,
        args.queries,
        args.k,
        args.seed,
        with_embeddings=args.with_embeddings,
        learned_model_path=args.learned_model,
        labels_path=args.labels,
        label_threshold=args.label_threshold,
    )
    plot_phase_comparison(scores, args.k, label_name, OUT_DIR / "phase_comparison.svg")

    print(f"Done — wrote 4 SVGs to {OUT_DIR.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
