"""Tests for the frozen Chunk contract, merge_chunks reducer, and AgentState."""

from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from research_agent.state import AgentState, Chunk, merge_chunks


def make_book_chunk(
    chunk_id: str,
    score: int | None = None,
    *,
    text: str = "text",
    pdf_page_start: int | None = 1,
    book_slug: str | None = "book",
) -> Chunk:
    """Helper to build a valid book chunk."""
    return Chunk(
        id=chunk_id,
        text=text,
        source="book",
        book_slug=book_slug,
        pdf_page_start=pdf_page_start,
        score=score,
    )


# --- Chunk validation ---------------------------------------------------------


def test_book_chunk_valid() -> None:
    chunk = make_book_chunk("b-1")
    assert chunk.source == "book"
    assert chunk.section_path == []
    assert chunk.score is None


def test_book_chunk_without_slug_fails() -> None:
    with pytest.raises(ValidationError):
        Chunk(id="b", text="t", source="book", pdf_page_start=1)


def test_book_chunk_without_pdf_page_start_fails() -> None:
    with pytest.raises(ValidationError):
        Chunk(id="b", text="t", source="book", book_slug="book")


def test_web_chunk_valid() -> None:
    chunk = Chunk(id="w", text="t", source="web", url="https://example.com")
    assert chunk.url == "https://example.com"


def test_web_chunk_without_url_fails() -> None:
    with pytest.raises(ValidationError):
        Chunk(id="w", text="t", source="web")


@pytest.mark.parametrize("score", [-1, 101])
def test_score_out_of_range_fails(score: int) -> None:
    with pytest.raises(ValidationError):
        make_book_chunk("b", score=score)


@pytest.mark.parametrize("score", [0, 50, 100])
def test_score_boundaries_ok(score: int) -> None:
    assert make_book_chunk("b", score=score).score == score


def test_chunk_is_frozen() -> None:
    chunk = make_book_chunk("b")
    with pytest.raises(ValidationError):
        chunk.score = 42  # type: ignore[misc]


def test_score_updated_via_model_copy() -> None:
    chunk = make_book_chunk("b", score=10)
    updated = chunk.model_copy(update={"score": 90})
    assert updated.score == 90
    assert chunk.score == 10  # original unchanged


# --- merge_chunks -------------------------------------------------------------


def test_merge_dedupes_by_id() -> None:
    old = [make_book_chunk("a", 10), make_book_chunk("b", 20)]
    new = [make_book_chunk("b", 30), make_book_chunk("c", 40)]
    result = merge_chunks(old, new)
    assert [c.id for c in result] == ["a", "b", "c"]


def test_merge_higher_score_wins() -> None:
    old = [make_book_chunk("a", 10)]
    new = [make_book_chunk("a", 90)]
    assert merge_chunks(old, new)[0].score == 90

    # Higher score in old also wins.
    old2 = [make_book_chunk("a", 90)]
    new2 = [make_book_chunk("a", 10)]
    assert merge_chunks(old2, new2)[0].score == 90


def test_merge_score_beats_none() -> None:
    # Existing None, new scored -> scored kept.
    old = [make_book_chunk("a", None)]
    new = [make_book_chunk("a", 5)]
    assert merge_chunks(old, new)[0].score == 5

    # Existing scored, new None -> scored kept.
    old2 = [make_book_chunk("a", 5)]
    new2 = [make_book_chunk("a", None)]
    assert merge_chunks(old2, new2)[0].score == 5


def test_merge_tie_keeps_existing() -> None:
    old = [make_book_chunk("a", 50, text="OLD")]
    new = [make_book_chunk("a", 50, text="NEW")]
    result = merge_chunks(old, new)
    assert result[0].text == "OLD"


def test_merge_preserves_first_seen_order() -> None:
    old = [make_book_chunk("x", 1)]
    new = [make_book_chunk("y", 1), make_book_chunk("x", 99)]
    result = merge_chunks(old, new)
    assert [c.id for c in result] == ["x", "y"]
    assert result[0].score == 99  # x updated in place, still first


def test_merge_empty_inputs() -> None:
    assert merge_chunks([], []) == []
    chunks = [make_book_chunk("a")]
    assert merge_chunks(chunks, []) == chunks
    assert merge_chunks([], chunks) == chunks


def test_merge_does_not_mutate_inputs() -> None:
    old = [make_book_chunk("a", 10)]
    new = [make_book_chunk("a", 90), make_book_chunk("b", 5)]
    old_snapshot = copy.deepcopy(old)
    new_snapshot = copy.deepcopy(new)
    result = merge_chunks(old, new)
    assert old == old_snapshot
    assert new == new_snapshot
    assert result is not old and result is not new


# --- AgentState shape ---------------------------------------------------------


def test_agent_state_has_expected_keys() -> None:
    expected = {
        "question",
        "current_query",
        "chunks",
        "strategies_tried",
        "answer",
        "cited_chunk_ids",
    }
    assert set(AgentState.__annotations__) == expected
