"""Phase 3.5: LLM-as-judge labels for the matching engine.

Why this module exists
----------------------
Every metric in `evaluation.py` is currently scored against
`synthesize_relevance` (or one of its noisy/simulated variants) — a
deterministic rule built from the *same signals* the ranker sees. That
makes the rules baseline look artificially strong and gives any learned
ranker very little headroom. Until real outcome data (accepts, chats
started, retention) is available, the next best thing is a *teacher
model*: send every (source, candidate) pair to Claude with a clear
rubric and use Claude's 0-4 verdict as the label.

What this gives us
------------------
1. **Realistic evaluation.** Every metric in the repo (precision@k,
   NDCG@k, MAP@k, ablation deltas) becomes meaningful — judged against
   labels that aren't structurally identical to the inputs.
2. **A teacher for distillation.** `scripts/label.py` writes the
   verdicts to JSONL; `scripts/train.py` can then fit the existing
   LightGBM `LearnedRanker` on those labels, producing a cheap
   deterministic student that approximates the LLM's judgment.

Design principles
-----------------
- **Heavy import is lazy.** `anthropic` lives in the optional `[judge]`
  extra and is imported inside `ClaudeJudge.__init__`. The rest of this
  module — `JudgeVerdict`, JSONL I/O, `queries_from_verdicts` — works
  with `[dev]` only.
- **Disk-backed cache.** Every verdict is appended to a JSONL file the
  moment it's produced. Re-running the labeller skips pairs that already
  have a verdict for the chosen model. The file is both the cache and
  the final labels artifact.
- **Stub-friendly.** The `LLMJudge` Protocol is one method
  (`judge(source, candidate) -> JudgeVerdict`), so tests inject a
  deterministic stub instead of hitting the real API.
- **Structured outputs, not regex parsing.** The judge prompt uses
  `output_config.format` with a strict JSON schema that pins the rating
  to an integer in [0, 4]. There is no free-text parsing of the model's
  reply.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from .embeddings import profile_to_text
from .evaluation import Query
from .models import UserProfile

# Default model for the judge. Sonnet 4.6 is the best balance of judgment
# quality and cost for high-volume offline labelling — strong enough to be a
# trustworthy teacher without Opus pricing. Override via the CLI to use Opus
# for a higher-quality gold pass or Haiku for the cheapest bulk runs.
DEFAULT_MODEL = "claude-sonnet-4-6"

# Verdicts >= this rating are treated as "yes, would become friends".
# 0-4 scale: 0 = no chance, 4 = strong match. Threshold at 3 means we
# only call ratings that lean positive "relevant".
DEFAULT_RELEVANCE_THRESHOLD = 3


# The rubric the judge sees in its system prompt. Kept here as a module
# constant so swapping prompts is a one-line code change and so the
# prompt version is git-tracked alongside the verdicts that came from
# it. Note: Sonnet 4.6's prompt-cache minimum is 2048 tokens, so this
# rubric is below the cache threshold today. Caching will start firing
# automatically if the rubric is fleshed out further.
JUDGE_SYSTEM_PROMPT = """You are an expert grader for Hangpost, a location-based social media app
for making new friends within a small radius of where the user is right now.

KEY PRODUCT FACTS:
- Hangpost is about making FRIENDS, not dating. Romance and physical
  attraction are NOT relevant signals. Two users from different
  demographics with overlapping interests are a strong match.
- Every candidate the model sees has already passed a hard geographic
  pre-filter (they're physically nearby). Distance is not a signal in
  the rating.
- Users do NOT write free-text bios. Each profile is summarized
  deterministically from structured fields (age, hometown, college,
  interests, liked topics). You see that summary.

THE TIER ORDER (use this to calibrate your ratings):
1. **Mutual friends** — friends-of-friends is the strongest real-world
   path to new friendship. Pairs with ≥1 mutual friend should rate
   highly (typically 3 or 4) almost regardless of other signals.
2. **Shared background** — same hometown OR same college (independently)
   is a meaningful friendship cue, even with weak interest overlap.
3. **Compatibility** — for everyone else, similar age + overlapping
   interests + overlapping liked topics is what carries the rating.

THE 0-4 RATING SCALE:
- 0: No realistic path to friendship. Nothing in common, no shared
     context, big life-stage gap.
- 1: Weak. One mild overlap (e.g., they both like "coffee" but nothing
     else); a meeting would feel awkward.
- 2: Plausible acquaintance. Some shared ground but not enough to seek
     them out (e.g., similar age, one shared interest, no other context).
- 3: Promising match. Either (a) ≥1 mutual friend, OR (b) shared
     hometown/college, OR (c) ≥2 shared interests + close in age.
     A meeting would have something obvious to talk about.
- 4: Strong match. Multiple signals stack — e.g., mutual friend AND
     shared interests, or same college AND multiple shared liked topics
     AND close in age. Both parties would likely want to meet again.

EVALUATION INSTRUCTIONS:
- Read both profiles and the pair-level signals carefully.
- Ignore signals that don't apply (an absent hometown is not a negative).
- Use the tier order above as your primary calibration — do not let
  surface-level interest overlap outrank a mutual-friend signal.
- Output ONE JSON object with two fields:
    rating: integer in [0, 4]
    reasoning: short string (≤ 200 chars) explaining the decisive
               factor in your rating.
- Do not return any text outside the JSON object.
"""


# ---------- data type ----------


@dataclass(frozen=True)
class JudgeVerdict:
    """One labelled (source, candidate) pair.

    Serialized as a single JSON object per line in the labels file so
    incremental writes are append-only and cheap to dedup.
    """

    source_id: str
    candidate_id: str
    rating: int  # 0..4 inclusive
    reasoning: str
    model: str

    def to_json_line(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json_line(cls, line: str) -> JudgeVerdict:
        data = json.loads(line)
        return cls(
            source_id=str(data["source_id"]),
            candidate_id=str(data["candidate_id"]),
            rating=int(data["rating"]),
            reasoning=str(data["reasoning"]),
            model=str(data["model"]),
        )


VerdictMap = dict[tuple[str, str], JudgeVerdict]


# ---------- prompt assembly ----------


def pair_to_prompt(source: UserProfile, candidate: UserProfile) -> str:
    """Render one (source, candidate) pair into a user-message prompt.

    Reuses `profile_to_text` for the per-profile summaries (so the judge
    sees the exact same text the embedding model would) and adds an
    explicit pair-level signal line — mutual friends, shared hometown,
    shared college, age gap — that the per-profile text cannot carry.
    """
    shared_friends = len(source.mutual_friend_ids & candidate.mutual_friend_ids)
    same_hometown = bool(source.hometown and source.hometown == candidate.hometown)
    same_college = bool(source.college and source.college == candidate.college)
    age_gap: int | None = (
        abs(source.age - candidate.age)
        if source.age is not None and candidate.age is not None
        else None
    )
    pair_facts: list[str] = []
    if shared_friends:
        pair_facts.append(f"{shared_friends} mutual friend(s)")
    if same_hometown:
        pair_facts.append(f"both from {source.hometown}")
    if same_college:
        pair_facts.append(f"both attended {source.college}")
    if age_gap is not None:
        pair_facts.append(f"age gap {age_gap} year(s)")
    pair_line = "; ".join(pair_facts) if pair_facts else "no obvious shared signals"

    source_text = profile_to_text(source) or "(no profile information)"
    candidate_text = profile_to_text(candidate) or "(no profile information)"

    return (
        f"PROFILE A: {source_text}\n"
        f"PROFILE B: {candidate_text}\n"
        f"PAIR SIGNALS: {pair_line}\n\n"
        "Rate how likely PROFILE A and PROFILE B would become friends on "
        "Hangpost, using the 0-4 scale from the rubric. Return JSON only."
    )


# ---------- judge implementations ----------


class LLMJudge(Protocol):
    """Anything that can rate one pair.

    Implementations are responsible for any retry / timeout policy
    appropriate to their backend. The disk cache in this module retries
    nothing — it caches results, not failures.
    """

    def judge(self, source: UserProfile, candidate: UserProfile) -> JudgeVerdict: ...


# The output schema is module-level so the test suite can import it and
# so the prompt is unambiguously declared once.
_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "rating": {"type": "integer", "enum": [0, 1, 2, 3, 4]},
        "reasoning": {"type": "string"},
    },
    "required": ["rating", "reasoning"],
    "additionalProperties": False,
}


class ClaudeJudge:
    """Concrete `LLMJudge` backed by the Anthropic Messages API.

    `anthropic` is imported inside `__init__` so this class only matters
    when the `[judge]` extra is installed. Tests use stub judges that
    satisfy the `LLMJudge` Protocol without ever touching the network.

    The system prompt (rubric) carries a `cache_control` marker so
    Anthropic can serve it from prompt cache on repeated calls — useful
    once the rubric grows past the 4096-token cache minimum.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        thinking: bool = True,
        api_key: str | None = None,
        rubric: str = JUDGE_SYSTEM_PROMPT,
        max_tokens: int = 1024,
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                'anthropic is required for ClaudeJudge. Install with: pip install -e ".[judge]"'
            ) from exc
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._thinking = thinking
        self._rubric = rubric
        self._max_tokens = max_tokens

    @property
    def model(self) -> str:
        return self._model

    def judge(self, source: UserProfile, candidate: UserProfile) -> JudgeVerdict:
        user_prompt = pair_to_prompt(source, candidate)
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "system": [
                {
                    "type": "text",
                    "text": self._rubric,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [{"role": "user", "content": user_prompt}],
            "output_config": {"format": {"type": "json_schema", "schema": _OUTPUT_SCHEMA}},
        }
        if self._thinking:
            kwargs["thinking"] = {"type": "adaptive"}

        response = self._client.messages.create(**kwargs)
        text_block = next((b for b in response.content if b.type == "text"), None)
        if text_block is None:
            raise RuntimeError(
                f"Judge returned no text block for ({source.user_id}, {candidate.user_id}). "
                f"stop_reason={response.stop_reason}"
            )
        data = json.loads(text_block.text)
        return JudgeVerdict(
            source_id=source.user_id,
            candidate_id=candidate.user_id,
            rating=int(data["rating"]),
            reasoning=str(data["reasoning"]),
            model=self._model,
        )


# ---------- JSONL I/O ----------


def load_verdicts(path: Path) -> VerdictMap:
    """Read all verdicts from a JSONL file into a `(source_id, candidate_id)` map.

    Returns an empty dict if the file does not exist. Later lines for
    the same pair overwrite earlier ones — useful when re-labelling with
    a different model and appending to the same file.
    """
    verdicts: VerdictMap = {}
    if not path.exists():
        return verdicts
    with path.open() as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            verdict = JudgeVerdict.from_json_line(stripped)
            verdicts[(verdict.source_id, verdict.candidate_id)] = verdict
    return verdicts


def append_verdict(path: Path, verdict: JudgeVerdict) -> None:
    """Append one verdict to the JSONL file, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(verdict.to_json_line() + "\n")


def judge_pairs(
    judge: LLMJudge,
    pairs: Iterable[tuple[UserProfile, UserProfile]],
    cache_path: Path,
    progress: Callable[[int, int, JudgeVerdict], None] | None = None,
) -> VerdictMap:
    """Judge each pair, caching to `cache_path` on the fly.

    Pairs already present in `cache_path` are skipped — the judge is
    never called twice for the same (source_id, candidate_id) within a
    single run, and re-running the labeller is idempotent.
    """
    verdicts = load_verdicts(cache_path)
    pair_list = list(pairs)
    for i, (source, candidate) in enumerate(pair_list, start=1):
        key = (source.user_id, candidate.user_id)
        if key in verdicts:
            verdict = verdicts[key]
        else:
            verdict = judge.judge(source, candidate)
            append_verdict(cache_path, verdict)
            verdicts[key] = verdict
        if progress is not None:
            progress(i, len(pair_list), verdict)
    return verdicts


# ---------- consumer hooks ----------


def queries_from_verdicts(
    profiles: Iterable[UserProfile],
    verdicts: Mapping[tuple[str, str], JudgeVerdict],
    threshold: int = DEFAULT_RELEVANCE_THRESHOLD,
) -> list[Query]:
    """Build a `Query` list from judge verdicts.

    Each unique `source_id` in `verdicts` becomes one query. The
    candidate pool for that source is *only* the candidates the judge
    has rated — evaluation against this query set measures "given the
    candidates the judge actually saw, did the ranker put the
    high-rated ones at the top?" This is honest: we cannot ask a metric
    to grade rankings on candidates we have no label for.

    Candidates with `rating >= threshold` are marked relevant.
    """
    by_id = {p.user_id: p for p in profiles}
    grouped: dict[str, dict[str, int]] = {}
    for (sid, cid), verdict in verdicts.items():
        grouped.setdefault(sid, {})[cid] = verdict.rating

    queries: list[Query] = []
    for sid, ratings in grouped.items():
        source = by_id.get(sid)
        if source is None:
            continue
        candidates: list[UserProfile] = []
        for cid in ratings:
            candidate = by_id.get(cid)
            if candidate is not None:
                candidates.append(candidate)
        relevant = {cid for cid, rating in ratings.items() if rating >= threshold and cid in by_id}
        queries.append((source, candidates, relevant))
    return queries


def graded_gains_from_verdicts(
    verdicts: Mapping[tuple[str, str], JudgeVerdict],
) -> dict[str, dict[str, float]]:
    """Build per-source graded gain maps from the judge's 0-4 ratings.

    Returns `{source_id: {candidate_id: gain}}` where `gain = 2**rating - 1`.
    That's the textbook NDCG gain formulation: a rating-4 candidate is
    worth ~16x a rating-1 candidate at the same rank, so the metric
    actually punishes the ranker for surfacing borderline matches above
    obvious matches — something the binary threshold throws away.
    """
    out: dict[str, dict[str, float]] = {}
    for (sid, cid), verdict in verdicts.items():
        out.setdefault(sid, {})[cid] = float(2**verdict.rating - 1)
    return out


# ---------- inter-rater agreement (hybrid Haiku + Sonnet calibration) ----------
#
# In the two-model labelling pattern, Haiku rates the bulk pairs cheaply
# and Sonnet re-rates a small "gold" subset of the same pairs. These
# helpers quantify how well the two raters agree on the overlap, so a
# disagreement isn't just a hunch — it's a number you can put in a
# README. We track four flavours of agreement because no single one
# tells the whole story for ordinal ratings:
#
#   - exact agreement %        — strict; useful headline
#   - adjacent agreement %     — forgiving of "is this a 2 or a 3" calls
#   - mean absolute difference — magnitude, not just count, of disagreements
#   - quadratic-weighted κ     — chance-corrected; the standard for ordinal data


# Rating scale is fixed by JUDGE_SYSTEM_PROMPT: integers in [0, 4].
RATING_LEVELS: tuple[int, ...] = (0, 1, 2, 3, 4)


@dataclass(frozen=True)
class AgreementReport:
    """Pairwise inter-rater agreement summary.

    `confusion[i][j]` is the count of pairs the *bulk* labeller rated `i`
    and the *gold* labeller rated `j`. The off-diagonal mass is the
    disagreement.
    """

    n_overlap: int
    exact_agreement: float  # fraction in [0, 1]
    adjacent_agreement: float  # fraction within ±1
    mean_absolute_difference: float
    quadratic_weighted_kappa: float
    confusion: list[list[int]]  # [bulk_rating][gold_rating]

    def render_table(self) -> str:
        """Return a markdown-ready confusion matrix + headline metrics."""
        header = "       gold→  " + "  ".join(f"{j:>3d}" for j in RATING_LEVELS)
        rows = [header, "  bulk↓  " + "-" * (5 * len(RATING_LEVELS) + 4)]
        for i in RATING_LEVELS:
            row_cells = "  ".join(f"{self.confusion[i][j]:>3d}" for j in RATING_LEVELS)
            rows.append(f"     {i:>2d}     {row_cells}")
        rows.append("")
        rows.append(f"  n (overlap)         : {self.n_overlap}")
        rows.append(f"  exact agreement     : {self.exact_agreement:.1%}")
        rows.append(f"  adjacent (±1) agree : {self.adjacent_agreement:.1%}")
        rows.append(f"  mean |Δrating|      : {self.mean_absolute_difference:.2f}")
        rows.append(f"  weighted κ (quad.)  : {self.quadratic_weighted_kappa:.3f}")
        return "\n".join(rows)


def _quadratic_weighted_kappa(confusion: list[list[int]]) -> float:
    """Quadratic-weighted Cohen's κ on a 5x5 confusion matrix.

    Formula:
        w_ij = (i - j)^2 / (k - 1)^2          (k = number of categories)
        κ    = 1 - (Σ w_ij O_ij) / (Σ w_ij E_ij)

    Where O_ij is the observed count and E_ij is the expected count under
    independence of the two raters' marginals. Quadratic weights penalise
    "1 vs 4" disagreements much more harshly than "2 vs 3", which is the
    right shape for an ordinal friendship-rating scale.

    Returns 1.0 when both raters agree exactly on every pair, 0.0 when
    agreement is at chance, and a negative value when the raters disagree
    *more* than chance would predict.
    """
    k = len(RATING_LEVELS)
    total = sum(sum(row) for row in confusion)
    if total == 0:
        return 0.0

    # Marginals.
    row_totals = [sum(confusion[i]) for i in range(k)]
    col_totals = [sum(confusion[i][j] for i in range(k)) for j in range(k)]

    # Weighted observed and expected disagreement.
    denom = (k - 1) ** 2
    num = 0.0
    den = 0.0
    for i in range(k):
        for j in range(k):
            w = (i - j) ** 2 / denom
            observed = confusion[i][j]
            expected = row_totals[i] * col_totals[j] / total
            num += w * observed
            den += w * expected

    if den == 0.0:
        # Degenerate case: one rater put every pair in the same bucket.
        # κ is undefined; report 0.0 (no information about agreement).
        return 0.0
    return 1.0 - num / den


def agreement_report(
    bulk: Mapping[tuple[str, str], JudgeVerdict],
    gold: Mapping[tuple[str, str], JudgeVerdict],
) -> AgreementReport:
    """Compute inter-rater agreement on the pairs labelled by *both* sets.

    Only pairs present in both `bulk` and `gold` contribute. Pairs unique
    to either map are silently ignored — they carry no agreement signal.
    """
    k = len(RATING_LEVELS)
    confusion = [[0 for _ in range(k)] for _ in range(k)]
    abs_diffs: list[int] = []

    for key, bulk_verdict in bulk.items():
        gold_verdict = gold.get(key)
        if gold_verdict is None:
            continue
        b = bulk_verdict.rating
        g = gold_verdict.rating
        # Defensive — JudgeVerdict.rating is typed int but a malformed
        # JSONL could carry an out-of-range value.
        if b not in RATING_LEVELS or g not in RATING_LEVELS:
            continue
        confusion[b][g] += 1
        abs_diffs.append(abs(b - g))

    n = len(abs_diffs)
    if n == 0:
        return AgreementReport(
            n_overlap=0,
            exact_agreement=0.0,
            adjacent_agreement=0.0,
            mean_absolute_difference=0.0,
            quadratic_weighted_kappa=0.0,
            confusion=confusion,
        )

    exact = sum(1 for d in abs_diffs if d == 0) / n
    adjacent = sum(1 for d in abs_diffs if d <= 1) / n
    mad = sum(abs_diffs) / n
    qwk = _quadratic_weighted_kappa(confusion)
    return AgreementReport(
        n_overlap=n,
        exact_agreement=exact,
        adjacent_agreement=adjacent,
        mean_absolute_difference=mad,
        quadratic_weighted_kappa=qwk,
        confusion=confusion,
    )


def sample_for_gold_pass(
    verdicts: Mapping[tuple[str, str], JudgeVerdict],
    n: int,
    seed: int = 42,
    stratify_by_rating: bool = True,
) -> list[tuple[str, str]]:
    """Pick `n` (source_id, candidate_id) keys to re-judge in the gold pass.

    With `stratify_by_rating=True` (the default), the sample is balanced
    across the five rating buckets so the agreement metrics aren't
    dominated by whatever rating Haiku happened to assign most often
    (typically 0s and 1s on a sparse-friendship dataset). With it off,
    the sample is uniform random over all keys.

    Deterministic given `seed`. Returns at most `min(n, len(verdicts))`
    keys.
    """
    import random as _random

    rng = _random.Random(seed)
    all_keys = list(verdicts.keys())
    if n >= len(all_keys):
        return sorted(all_keys)  # take everything, deterministic order

    if not stratify_by_rating:
        rng.shuffle(all_keys)
        return all_keys[:n]

    by_rating: dict[int, list[tuple[str, str]]] = {r: [] for r in RATING_LEVELS}
    for key, verdict in verdicts.items():
        if verdict.rating in by_rating:
            by_rating[verdict.rating].append(key)

    # How many slots per bucket? Distribute evenly, then top up from the
    # largest buckets to handle remainders.
    per_bucket, remainder = divmod(n, len(RATING_LEVELS))
    chosen: list[tuple[str, str]] = []
    leftovers: list[tuple[str, str]] = []
    for r in RATING_LEVELS:
        bucket = list(by_rating[r])
        rng.shuffle(bucket)
        take = min(per_bucket, len(bucket))
        chosen.extend(bucket[:take])
        leftovers.extend(bucket[take:])

    rng.shuffle(leftovers)
    chosen.extend(leftovers[:remainder])
    # Sort for stable ordering across re-runs at the same seed.
    return sorted(chosen)
