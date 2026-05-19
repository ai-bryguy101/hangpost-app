"""Tests for the LLM-as-judge module.

These tests never touch the real Anthropic API. A deterministic
`StubJudge` satisfies the `LLMJudge` Protocol and is used everywhere a
real judge would be needed, so the test suite runs against the `[dev]`
extra only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from hangpost_matching import (
    DEFAULT_RELEVANCE_THRESHOLD,
    JudgeVerdict,
    LLMJudge,
    UserProfile,
    append_verdict,
    judge_pairs,
    load_verdicts,
    pair_to_prompt,
    queries_from_verdicts,
)


def _profile(
    user_id: str,
    *,
    age: int | None = 30,
    hometown: str | None = "boston",
    college: str | None = "bu",
    interests: set[str] | None = None,
    liked_topics: set[str] | None = None,
    mutual_friend_ids: set[str] | None = None,
) -> UserProfile:
    return UserProfile(
        user_id=user_id,
        age=age,
        hometown=hometown,
        college=college,
        interests=interests or set(),
        liked_topics=liked_topics or set(),
        mutual_friend_ids=mutual_friend_ids or set(),
    )


class StubJudge:
    """Deterministic judge for tests.

    The rating is decided from the structured signals so the tests can
    assert on behavior without mocking. The reasoning is just the
    rating description.
    """

    model = "stub-judge"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def judge(self, source: UserProfile, candidate: UserProfile) -> JudgeVerdict:
        self.calls.append((source.user_id, candidate.user_id))
        score = 0
        if source.mutual_friend_ids & candidate.mutual_friend_ids:
            score += 2
        if source.hometown and source.hometown == candidate.hometown:
            score += 1
        if source.college and source.college == candidate.college:
            score += 1
        rating = min(score, 4)
        return JudgeVerdict(
            source_id=source.user_id,
            candidate_id=candidate.user_id,
            rating=rating,
            reasoning=f"score={rating}",
            model=self.model,
        )


# Sanity: StubJudge is a structural LLMJudge.
def test_stub_judge_satisfies_protocol() -> None:
    judge: LLMJudge = StubJudge()
    verdict = judge.judge(_profile("a"), _profile("b"))
    assert verdict.source_id == "a"
    assert verdict.candidate_id == "b"


# ---------- JSONL round-trip ----------


def test_verdict_json_line_roundtrip() -> None:
    original = JudgeVerdict(
        source_id="alice",
        candidate_id="bob",
        rating=3,
        reasoning="shared hometown, ≥2 interests",
        model="claude-opus-4-7",
    )
    line = original.to_json_line()
    restored = JudgeVerdict.from_json_line(line)
    assert restored == original


def test_load_verdicts_missing_file(tmp_path: Path) -> None:
    assert load_verdicts(tmp_path / "does-not-exist.jsonl") == {}


def test_load_verdicts_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    v = JudgeVerdict("a", "b", 2, "ok", "stub")
    path.write_text("\n" + v.to_json_line() + "\n\n")
    loaded = load_verdicts(path)
    assert loaded == {("a", "b"): v}


def test_append_and_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "nested/labels.jsonl"
    v1 = JudgeVerdict("s1", "c1", 3, "good", "stub")
    v2 = JudgeVerdict("s1", "c2", 1, "weak", "stub")
    append_verdict(path, v1)
    append_verdict(path, v2)
    loaded = load_verdicts(path)
    assert loaded == {("s1", "c1"): v1, ("s1", "c2"): v2}


def test_load_verdicts_later_line_wins_for_same_pair(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    append_verdict(path, JudgeVerdict("s", "c", 1, "first", "stub"))
    append_verdict(path, JudgeVerdict("s", "c", 4, "rejudged", "stub"))
    loaded = load_verdicts(path)
    assert loaded[("s", "c")].rating == 4


# ---------- judge_pairs caching ----------


def test_judge_pairs_skips_cached_pairs(tmp_path: Path) -> None:
    """judge_pairs must not call the judge for pairs already in the cache."""
    cache = tmp_path / "labels.jsonl"
    cached = JudgeVerdict("alice", "bob", 4, "already done", "stub")
    append_verdict(cache, cached)

    judge = StubJudge()
    alice = _profile("alice")
    bob = _profile("bob")
    carol = _profile("carol", college="ucla", hometown="seattle")

    verdicts = judge_pairs(judge, [(alice, bob), (alice, carol)], cache)

    # Cached pair returned verbatim; only the new pair hit the judge.
    assert verdicts[("alice", "bob")] == cached
    assert judge.calls == [("alice", "carol")]
    # Newly-judged pair is now in the file.
    on_disk = load_verdicts(cache)
    assert ("alice", "carol") in on_disk


def test_judge_pairs_progress_callback_runs(tmp_path: Path) -> None:
    cache = tmp_path / "labels.jsonl"
    seen: list[int] = []

    def progress(i: int, total: int, verdict: JudgeVerdict) -> None:
        seen.append(i)
        assert total == 2
        assert isinstance(verdict, JudgeVerdict)

    judge_pairs(
        StubJudge(),
        [(_profile("a"), _profile("b")), (_profile("a"), _profile("c"))],
        cache,
        progress=progress,
    )
    assert seen == [1, 2]


# ---------- pair_to_prompt ----------


def test_pair_to_prompt_includes_pair_signals() -> None:
    source = _profile(
        "alice",
        age=30,
        hometown="boston",
        college="bu",
        interests={"hiking", "coffee"},
        mutual_friend_ids={"f1"},
    )
    candidate = _profile(
        "bob",
        age=33,
        hometown="boston",
        college="ucla",
        interests={"hiking"},
        mutual_friend_ids={"f1"},
    )
    prompt = pair_to_prompt(source, candidate)
    assert "PROFILE A:" in prompt
    assert "PROFILE B:" in prompt
    assert "1 mutual friend" in prompt
    assert "both from boston" in prompt
    assert "age gap 3" in prompt
    # Different colleges — must NOT claim "both attended"
    assert "both attended" not in prompt


def test_pair_to_prompt_handles_empty_profile() -> None:
    source = _profile("a", age=None, hometown=None, college=None)
    candidate = _profile("b", age=None, hometown=None, college=None)
    prompt = pair_to_prompt(source, candidate)
    assert "no obvious shared signals" in prompt
    assert "no profile information" in prompt


# ---------- queries_from_verdicts ----------


def test_queries_from_verdicts_groups_by_source() -> None:
    profiles = [_profile("s"), _profile("c1"), _profile("c2"), _profile("c3")]
    verdicts = {
        ("s", "c1"): JudgeVerdict("s", "c1", 4, "", "stub"),
        ("s", "c2"): JudgeVerdict("s", "c2", 1, "", "stub"),
        ("s", "c3"): JudgeVerdict("s", "c3", 3, "", "stub"),
    }
    queries = queries_from_verdicts(profiles, verdicts, threshold=3)
    assert len(queries) == 1
    source, candidates, relevant = queries[0]
    assert source.user_id == "s"
    assert {c.user_id for c in candidates} == {"c1", "c2", "c3"}
    assert relevant == {"c1", "c3"}


def test_queries_from_verdicts_default_threshold_is_3() -> None:
    profiles = [_profile("s"), _profile("a"), _profile("b")]
    verdicts = {
        ("s", "a"): JudgeVerdict("s", "a", 2, "", "stub"),
        ("s", "b"): JudgeVerdict("s", "b", 3, "", "stub"),
    }
    queries = queries_from_verdicts(profiles, verdicts)
    assert DEFAULT_RELEVANCE_THRESHOLD == 3
    _, _, relevant = queries[0]
    assert relevant == {"b"}


def test_queries_from_verdicts_drops_unknown_profile_ids() -> None:
    """A verdict referencing a profile that's not in the dataset is skipped silently."""
    profiles = [_profile("s"), _profile("c1")]
    verdicts = {
        ("s", "c1"): JudgeVerdict("s", "c1", 4, "", "stub"),
        ("s", "ghost"): JudgeVerdict("s", "ghost", 4, "", "stub"),
        ("missing", "c1"): JudgeVerdict("missing", "c1", 4, "", "stub"),
    }
    queries = queries_from_verdicts(profiles, verdicts)
    assert len(queries) == 1
    _, candidates, relevant = queries[0]
    assert [c.user_id for c in candidates] == ["c1"]
    assert relevant == {"c1"}


def test_graded_gains_from_verdicts_uses_2_to_the_rating() -> None:
    from hangpost_matching import graded_gains_from_verdicts

    verdicts = {
        ("s", "c0"): JudgeVerdict("s", "c0", 0, "", "stub"),
        ("s", "c1"): JudgeVerdict("s", "c1", 1, "", "stub"),
        ("s", "c2"): JudgeVerdict("s", "c2", 2, "", "stub"),
        ("s", "c3"): JudgeVerdict("s", "c3", 3, "", "stub"),
        ("s", "c4"): JudgeVerdict("s", "c4", 4, "", "stub"),
    }
    gains = graded_gains_from_verdicts(verdicts)
    assert gains["s"] == {"c0": 0.0, "c1": 1.0, "c2": 3.0, "c3": 7.0, "c4": 15.0}


def test_claude_judge_import_error_when_extra_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """ClaudeJudge must raise a helpful ImportError if anthropic is not installed."""
    import builtins
    import sys

    # If anthropic is already imported, simulate it being missing.
    monkeypatch.delitem(sys.modules, "anthropic", raising=False)

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "anthropic":
            raise ImportError("simulated missing anthropic")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    from hangpost_matching import ClaudeJudge

    with pytest.raises(ImportError, match='".\\[judge]"'):
        ClaudeJudge()
