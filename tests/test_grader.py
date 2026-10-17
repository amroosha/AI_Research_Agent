"""Tests for the LLM grader and its calibration (Stage 2, T5).

No network: a fake LLM returns scripted structured outputs. The one ``slow``
test grades three chunks with the real Ollama model.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from research_agent.config import DEFAULT_GRADER, DEFAULT_LLM, GraderSettings
from research_agent.eval.grader_eval import (
    disagreements,
    position_bias,
    precision_recall_f1,
    proxy_labels,
    question_outcome,
    score_distribution,
    threshold_sweep,
)
from research_agent.golden import GoldenItem
from research_agent.grader import (
    GradeBatch,
    build_batches,
    build_prompt,
    estimate_tokens,
    fits_budget,
    grade_chunks,
    load_prompt,
    validate_grades,
)
from research_agent.state import Chunk

BOOK = "synthetic-book"


def _load_script_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "eval_grader.py"
    spec = importlib.util.spec_from_file_location("eval_grader_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


script = _load_script_module()


def _chunk(index: int, start: int | None = None, text: str | None = None) -> Chunk:
    page = start if start is not None else index + 1
    return Chunk(
        id=f"{BOOK}-{index:05d}",
        text=text or f"chunk text {index}",
        source="book",
        book_slug=BOOK,
        book_title="Synthetic",
        section_path=["Chapter 1"],
        pdf_page_start=page,
        pdf_page_end=page,
    )


def _item(
    query: str = "q",
    book: str | None = BOOK,
    ranges: tuple[tuple[int, int], ...] = ((10, 12),),
    is_negative: bool = False,
    item_id: str = "fix:000",
) -> GoldenItem:
    return GoldenItem(
        id=item_id,
        query=query,
        book=book,
        ranges=ranges,
        is_negative=is_negative,
        source_file="fix.yaml",
    )


class FakeStructured:
    """Mimics ``llm.with_structured_output(GradeBatch)`` with scripted replies.

    ``script`` is a list of items consumed one per invoke: an ``Exception``
    (raised) or a list of (ref, score) tuples (returned as a raw object with a
    ``grades`` attribute, deliberately bypassing pydantic so that
    ``validate_grades`` is the only validation gate under test).
    """

    def __init__(self, script: list) -> None:
        self.script = list(script)
        self.prompts: list[str] = []

    def invoke(self, prompt: str):
        self.prompts.append(prompt)
        if not self.script:
            raise AssertionError("fake LLM script exhausted")
        reply = self.script.pop(0)
        if isinstance(reply, Exception):
            raise reply
        refs = [ref for ref, score in reply]
        scores = [score for ref, score in reply]
        return SimpleNamespace(refs=refs, scores=scores)


class FakeLLM:
    """Fake chat model: each call to with_structured_output pops one reply."""

    def __init__(self, script: list) -> None:
        self._structured = FakeStructured(script)
        self.calls = 0

    @property
    def prompts(self) -> list[str]:
        return self._structured.prompts

    def with_structured_output(self, schema, **_kwargs):
        self.calls += 1
        return self._structured


def _batch(*, refs_scores: list[tuple[int, int]]) -> GradeBatch:
    refs, scores = zip(*refs_scores)
    return GradeBatch(refs=list(refs), scores=list(scores))


# --- prompt + batching --------------------------------------------------------


def test_prompt_labels_chunks_in_order() -> None:
    template = load_prompt(DEFAULT_GRADER.prompt_version)
    chunks = [_chunk(0), _chunk(1)]
    prompt = build_prompt("what is X?", chunks, template)
    assert "[1]" in prompt and "[2]" in prompt
    assert "chunk text 0" in prompt and "chunk text 1" in prompt
    assert "what is X?" in prompt
    assert "90-100" in prompt  # rubric present


def test_estimate_tokens() -> None:
    assert estimate_tokens("a" * 35) == 10
    assert estimate_tokens("") == 0


def test_build_batches_respects_batch_size() -> None:
    chunks = [_chunk(i, text="x" * 35) for i in range(10)]
    batches = build_batches("q", chunks, DEFAULT_GRADER, DEFAULT_LLM)
    assert [len(batch) for batch in batches] == [6, 4]


def test_build_batches_splits_by_token_budget() -> None:
    # ~2000 tokens per chunk: 4 chunks + overhead exceed the 6668-token budget
    # (8192 - 1024 - 500), forcing a split; halves fit.
    chunks = [_chunk(i, text="x" * 7000) for i in range(4)]
    cfg = GraderSettings(grader_batch=4, token_margin=500)
    llm_cfg = DEFAULT_LLM  # budget = 8192 - 1024 - 500 = 6668
    batches = build_batches("q", chunks, cfg, llm_cfg)
    assert len(batches) > 1
    assert sum(len(batch) for batch in batches) == 4
    for batch in batches:
        assert fits_budget("q", batch, llm_cfg, cfg.token_margin) or len(batch) == 1


def test_fits_budget_boundary() -> None:
    cfg = DEFAULT_GRADER
    llm_cfg = DEFAULT_LLM  # budget 6668
    small = [_chunk(0, text="x" * 35)]  # ~1 token
    assert fits_budget("q", small, llm_cfg, cfg.token_margin)
    huge = [_chunk(0, text="x" * 35_000)]  # ~10k tokens
    assert not fits_budget("q", huge, llm_cfg, cfg.token_margin)


# --- validate_grades ----------------------------------------------------------


def test_validate_grades_valid() -> None:
    mapping = validate_grades([1, 2], [90, 10], {1, 2})
    assert mapping == {1: 90, 2: 10}


@pytest.mark.parametrize("score", [101, -1])
def test_validate_grades_out_of_range(score: int) -> None:
    with pytest.raises(ValueError, match="0..100"):
        validate_grades([1], [score], {1})


def test_validate_grades_non_int_score() -> None:
    with pytest.raises(ValueError, match="0..100"):
        validate_grades([1], ["50"], {1})
    boolean = True  # bool is not an int here
    with pytest.raises(ValueError, match="0..100"):
        validate_grades([1], [boolean], {1})


def test_validate_grades_missing_and_duplicate_refs() -> None:
    with pytest.raises(ValueError, match="refs mismatch"):
        validate_grades([1], [50], {1, 2})
    with pytest.raises(ValueError, match="more than once"):
        validate_grades([1, 1], [50, 60], {1})
    with pytest.raises(ValueError, match="refs mismatch"):
        validate_grades([1, 3], [50, 60], {1, 2})


# --- grade_chunks -------------------------------------------------------------


def test_valid_batch_maps_refs_to_chunks_in_order() -> None:
    llm = FakeLLM([[(1, 95), (2, 40), (3, 70)]])
    chunks = [_chunk(0), _chunk(1), _chunk(2)]
    graded = grade_chunks(llm, "q?", chunks)
    assert [chunk.score for chunk in graded] == [95, 40, 70]
    assert [chunk.id for chunk in graded] == [chunk.id for chunk in chunks]
    # Inputs untouched (frozen model, new instances returned).
    assert all(chunk.score is None for chunk in chunks)


def test_missing_ref_retry_succeeds() -> None:
    # First reply misses ref 2; the retry (same batch) is complete.
    llm = FakeLLM([[(1, 90)], [(1, 90), (2, 30)]])
    graded = grade_chunks(llm, "q?", [_chunk(0), _chunk(1)])
    assert [chunk.score for chunk in graded] == [90, 30]
    assert len(llm.prompts) == 2  # first reply + retry


def test_retry_fails_then_individual_fallback() -> None:
    # Batch replies invalid twice -> per-chunk grading (each succeeds).
    llm = FakeLLM(
        [
            ValueError("bad"),  # attempt 1
            ValueError("bad"),  # attempt 2
            [(1, 80)],  # chunk 1 individually
            [(1, 20)],  # chunk 2 individually
        ]
    )
    graded = grade_chunks(llm, "q?", [_chunk(0), _chunk(1)])
    assert [chunk.score for chunk in graded] == [80, 20]


def test_individual_failure_scores_zero_and_counted() -> None:
    # Batch fails twice; chunk 1 grades fine; chunk 1's twin fails twice alone.
    llm = FakeLLM(
        [
            RuntimeError("down"),
            RuntimeError("down"),
            [(1, 80)],
            RuntimeError("down"),
            RuntimeError("down"),
        ]
    )
    chunks = [_chunk(0), _chunk(1)]
    failures: list[str] = []
    graded = grade_chunks(llm, "q?", chunks, parse_failures=failures)
    assert [chunk.score for chunk in graded] == [80, 0]
    assert failures == [chunks[1].id]


def test_empty_chunks_returns_empty() -> None:
    llm = FakeLLM([])
    assert grade_chunks(llm, "q?", []) == []


def test_latencies_recorded_per_call() -> None:
    llm = FakeLLM([[(1, 50)]])
    latencies: list[float] = []
    grade_chunks(llm, "q?", [_chunk(0)], latencies=latencies)
    assert len(latencies) == 1
    assert latencies[0] >= 0.0


# --- proxy labels + metrics ---------------------------------------------------


def test_proxy_labels_from_ranges() -> None:
    item = _item(ranges=((10, 12),))
    chunks = [_chunk(0, start=11), _chunk(1, start=50)]
    assert proxy_labels(chunks, item) == [True, False]


def test_proxy_labels_negative_all_false() -> None:
    item = _item(book=None, ranges=(), is_negative=True)
    chunks = [_chunk(0, start=11), _chunk(1)]
    assert proxy_labels(chunks, item) == [False, False]


def test_precision_recall_f1_hand_computed() -> None:
    # predicted: T F T T F; actual: T T F T F -> TP=2, FP=1, FN=1
    metrics = precision_recall_f1(
        [True, False, True, True, False], [True, True, False, True, False]
    )
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["f1"] == pytest.approx(2 / 3)


def test_precision_recall_f1_empty_and_no_positives() -> None:
    assert precision_recall_f1([], []) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    metrics = precision_recall_f1([False, False], [False, False])
    assert metrics == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_score_distribution_quartiles() -> None:
    dist = score_distribution([10, 20, 30, 40, 50])
    assert dist["median"] == 30
    assert dist["q1"] == pytest.approx(20)
    assert dist["q3"] == pytest.approx(40)
    assert score_distribution([])["median"] is None


# --- question-level outcomes --------------------------------------------------


def _graded_with_scores(scores: list[int]) -> list[Chunk]:
    return [_chunk(i).model_copy(update={"score": score}) for i, score in enumerate(scores)]


def test_outcome_positive_correct() -> None:
    item = _item(ranges=((10, 12),))
    graded = _graded_with_scores([80, 30])
    labels = [True, False]
    assert question_outcome(item, graded, labels, 75) == "correct"


def test_outcome_positive_wrong_generate() -> None:
    item = _item(ranges=((10, 12),))
    graded = _graded_with_scores([80, 90])
    labels = [False, False]  # both above 75 but irrelevant
    assert question_outcome(item, graded, labels, 75) == "wrong_generate"


def test_outcome_positive_needless_retry() -> None:
    item = _item(ranges=((10, 12),))
    graded = _graded_with_scores([60, 40])
    labels = [True, False]  # relevant chunk exists but scores below threshold
    assert question_outcome(item, graded, labels, 75) == "needless_retry"


def test_outcome_positive_retrieval_miss() -> None:
    item = _item(ranges=((10, 12),))
    graded = _graded_with_scores([60, 40])
    labels = [False, False]
    assert question_outcome(item, graded, labels, 75) == "retrieval_miss"


def test_outcome_negative_correct_and_false_generate() -> None:
    item = _item(book=None, ranges=(), is_negative=True)
    labels = [False, False]
    assert question_outcome(item, _graded_with_scores([70, 60]), labels, 75) == "correct"
    assert question_outcome(item, _graded_with_scores([70, 90]), labels, 75) == "false_generate"


def test_threshold_sweep_counts() -> None:
    items = [
        _item(query="a", ranges=((10, 12),), item_id="fix:a"),
        _item(query="b", book=None, ranges=(), is_negative=True, item_id="fix:b"),
    ]
    gradeds = [_graded_with_scores([80]), _graded_with_scores([60])]
    labelss = [[True], [False]]
    sweep = threshold_sweep(items, gradeds, labelss, thresholds=(50, 65, 80))
    assert sweep[50] == {"correct": 1, "false_generate": 1}
    assert sweep[65] == {"correct": 2}
    assert sweep[80] == {"correct": 2}


def test_recommended_threshold_picks_safest() -> None:
    sweep = {
        50: {"false_generate": 3, "correct": 2},
        75: {"correct": 5},
        90: {"needless_retry": 3, "correct": 2},
    }
    assert script.recommend_threshold(sweep) == 75


# --- position bias ------------------------------------------------------------


def test_position_bias_metric() -> None:
    # diffs: 5, 10, 30; threshold-75 flips: (70,75) and (60,90) -> 2 of 3.
    bias = position_bias([70, 50, 60], [75, 60, 90], threshold=75)
    assert bias["mean_abs_diff"] == pytest.approx(15.0)
    assert bias["flip_fraction"] == pytest.approx(2 / 3)
    assert bias["n"] == 3


def test_position_bias_identical_runs() -> None:
    bias = position_bias([80, 70], [80, 70])
    assert bias["mean_abs_diff"] == 0.0
    assert bias["flip_fraction"] == 0.0


def test_position_bias_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        position_bias([1, 2], [1])


# --- disagreements ------------------------------------------------------------


def test_disagreements_sorted_and_limited() -> None:
    item = _item()
    triples = [
        (item, _chunk(0).model_copy(update={"score": 10}), True),  # disagreement 90
        (item, _chunk(1).model_copy(update={"score": 95}), False),  # disagreement 95
        (item, _chunk(2).model_copy(update={"score": 50}), False),  # 50
    ]
    rows = disagreements(triples, limit=2)
    assert len(rows) == 2
    assert rows[0]["score"] == 95 and rows[0]["proxy_label"] == "not_relevant"
    assert rows[1]["score"] == 10 and rows[1]["proxy_label"] == "relevant"
    assert rows[0]["text_preview"]


# --- protection flag ----------------------------------------------------------


def test_protection_refuses_test_file(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        script.check_test_protection(tmp_path / "test_set.yaml", allow_test=False)
    assert excinfo.value.code == 2


def test_protection_dev_runs(tmp_path: Path) -> None:
    assert script.check_test_protection(tmp_path / "dev_set.yaml", allow_test=False) == "dev"


# --- slow: real Ollama --------------------------------------------------------


@pytest.mark.slow
def test_real_grader_ranks_relevant_chunk_highest() -> None:
    import os

    if not os.environ.get("OLLAMA_MODEL"):
        pytest.skip("OLLAMA_MODEL not set")
    from research_agent.llm import get_llm

    question = "What is the boiling point of water at sea level?"
    relevant = _chunk(
        0,
        text="Water boils at 100 degrees Celsius at sea level, where atmospheric "
        "pressure is exactly 1 atmosphere. At higher altitudes the boiling point "
        "is lower because pressure decreases with elevation.",
    )
    unrelated = [
        _chunk(
            1,
            text="Gradient descent updates model parameters by moving them "
            "proportional to the negative gradient of the loss function.",
        ),
        _chunk(
            2,
            text="The recipe requires two cups of flour, a pinch of salt, "
            "and slow kneading for ten minutes before the first rise.",
        ),
    ]
    graded = grade_chunks(get_llm(), question, [unrelated[0], relevant, unrelated[1]])
    scores = {chunk.id: chunk.score for chunk in graded}
    assert scores[relevant.id] == max(scores.values())
    assert scores[relevant.id] is not None and scores[relevant.id] >= 75
