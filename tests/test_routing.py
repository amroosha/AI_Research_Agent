"""Tests for pure routing logic and its config thresholds."""

from __future__ import annotations

import pathlib

import pytest

from research_agent.config import Settings
from research_agent.routing import classify, next_strategy, route, usable_chunks
from research_agent.state import AgentState, Chunk


def make_book(chunk_id: str, score: int | None) -> Chunk:
    return Chunk(
        id=chunk_id,
        text="t",
        source="book",
        book_slug="book",
        pdf_page_start=1,
        score=score,
    )


def make_web(chunk_id: str, score: int | None = None) -> Chunk:
    return Chunk(
        id=chunk_id,
        text="t",
        source="web",
        url="https://example.com",
        score=score,
    )


def make_state(
    chunks: list[Chunk],
    strategies_tried: list[str] | None = None,
) -> AgentState:
    return {
        "question": "q",
        "current_query": "q",
        "chunks": chunks,
        "strategies_tried": strategies_tried or [],
        "answer": None,
        "cited_chunk_ids": [],
    }


# --- classify -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (100, "relevant"),
        (75, "relevant"),
        (74, "ambiguous"),
        (50, "ambiguous"),
        (49, "irrelevant"),
        (0, "irrelevant"),
    ],
)
def test_classify_boundaries(score: int, expected: str) -> None:
    assert classify(score) == expected


def test_classify_custom_cfg() -> None:
    cfg = Settings(relevant_threshold=90, ambiguous_threshold=60)
    assert classify(90, cfg) == "relevant"
    assert classify(89, cfg) == "ambiguous"
    assert classify(60, cfg) == "ambiguous"
    assert classify(59, cfg) == "irrelevant"


# --- route --------------------------------------------------------------------


def test_route_empty_chunks_zero_tried_rewrites() -> None:
    assert route(make_state([])) == "rewrite"


def test_route_relevant_generates() -> None:
    assert route(make_state([make_book("a", 75)])) == "generate"


def test_route_ambiguous_zero_tried_rewrites() -> None:
    assert route(make_state([make_book("a", 74)])) == "rewrite"


def test_route_no_relevant_within_retries_rewrites() -> None:
    assert route(make_state([make_book("a", 49), make_book("b", 60)], ["step_back"])) == "rewrite"


def test_route_no_relevant_at_max_retries_web_search() -> None:
    state = make_state(
        [make_book("a", 49), make_book("b", 60)], ["step_back", "specific", "fusion"]
    )
    assert route(state) == "web_search"


def test_route_relevant_beats_max_retries() -> None:
    state = make_state([make_book("a", 75)], ["step_back", "specific", "fusion"])
    assert route(state) == "generate"


def test_route_ungraded_book_chunk_raises() -> None:
    with pytest.raises(ValueError):
        route(make_state([make_book("a", None)]))


def test_route_ungraded_web_chunk_ignored() -> None:
    # Web chunk with None score must not trigger the ungraded error.
    assert route(make_state([make_web("w")], ["step_back"])) == "rewrite"


def test_route_custom_max_retries() -> None:
    cfg = Settings(max_retries=1)
    assert route(make_state([make_book("a", 40)], []), cfg) == "rewrite"
    assert route(make_state([make_book("a", 40)], ["step_back"]), cfg) == "web_search"


# --- usable_chunks ------------------------------------------------------------


def test_usable_chunks_threshold_boundaries() -> None:
    chunks = [make_book("in", 50), make_book("out", 49)]
    result = usable_chunks(make_state(chunks))
    assert [c.id for c in result] == ["in"]


def test_usable_chunks_sorted_by_score_desc_stable() -> None:
    chunks = [make_book("a", 60), make_book("b", 90), make_book("c", 90)]
    result = usable_chunks(make_state(chunks))
    # Descending score; equal scores keep original order (b before c).
    assert [c.id for c in result] == ["b", "c", "a"]


def test_usable_chunks_web_always_included_after_books() -> None:
    chunks = [make_web("w1"), make_book("a", 80), make_web("w2"), make_book("b", 49)]
    result = usable_chunks(make_state(chunks))
    assert [c.id for c in result] == ["a", "w1", "w2"]


# --- next_strategy ------------------------------------------------------------


def test_next_strategy_default_order() -> None:
    assert next_strategy(make_state([])) == "step_back"


def test_next_strategy_skips_tried() -> None:
    assert next_strategy(make_state([], ["step_back"])) == "specific"
    assert next_strategy(make_state([], ["step_back", "specific"])) == "fusion"


def test_next_strategy_fusion_tried_returns_step_back() -> None:
    assert next_strategy(make_state([], ["fusion"])) == "step_back"


def test_next_strategy_all_tried_raises() -> None:
    with pytest.raises(ValueError):
        next_strategy(make_state([], ["step_back", "specific", "fusion"]))


# --- purity guard -------------------------------------------------------------


def test_routing_imports_no_langgraph_or_langchain() -> None:
    source = (
        pathlib.Path(__file__).resolve().parents[1] / "src" / "research_agent" / "routing.py"
    ).read_text(encoding="utf-8")
    assert "langgraph" not in source
    assert "langchain" not in source
