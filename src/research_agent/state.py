"""Core state types and reducers for the agent.

This module defines the frozen ``Chunk`` contract, the ``merge_chunks`` reducer
used to combine retrieved chunks across graph steps, and the ``AgentState``
TypedDict shape. No LLM or LangGraph graph code lives here; only the pure data
contract and its reducer.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, model_validator

# The ordered set of query-rewrite strategies. Shared by config and state.
Strategy = Literal["step_back", "specific", "fusion"]


class Chunk(BaseModel):
    """A single retrievable unit of text.

    Book chunks require ``book_slug`` and ``pdf_page_start``; web chunks require
    ``url``. ``score`` is the grader relevance (0-100) or ``None`` when ungraded.
    The model is frozen: update a field (e.g. score) via ``model_copy(update=...)``.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    text: str
    source: Literal["book", "web"]
    book_slug: str | None = None
    book_title: str | None = None
    section_path: list[str] = Field(default_factory=list)
    pdf_page_start: int | None = None
    pdf_page_end: int | None = None
    printed_page_start: int | None = None
    printed_page_end: int | None = None
    url: str | None = None
    title: str | None = None
    score: int | None = None

    @model_validator(mode="after")
    def _validate_source_fields(self) -> Chunk:
        if self.source == "book":
            if self.book_slug is None:
                raise ValueError("book chunks require book_slug")
            if self.pdf_page_start is None:
                raise ValueError("book chunks require pdf_page_start")
        elif self.source == "web":
            if self.url is None:
                raise ValueError("web chunks require url")
        if self.score is not None and not (0 <= self.score <= 100):
            raise ValueError("score must be between 0 and 100 when set")
        return self


def _prefer(existing: Chunk, candidate: Chunk) -> Chunk:
    """Return the chunk to keep for a duplicate id.

    Any numeric score beats ``None``; on equal scores keep the existing chunk.
    """
    if existing.score is None:
        return candidate if candidate.score is not None else existing
    if candidate.score is None:
        return existing
    return candidate if candidate.score > existing.score else existing


def merge_chunks(old: list[Chunk], new: list[Chunk]) -> list[Chunk]:
    """Merge two chunk lists by ``id``.

    Dedupes by id, keeping the higher-scored chunk on duplicates (any score
    beats ``None``; a tie keeps the existing one). Preserves first-seen order
    and never mutates its inputs.
    """
    merged: dict[str, Chunk] = {}
    order: list[str] = []
    for chunk in (*old, *new):
        existing = merged.get(chunk.id)
        if existing is None:
            merged[chunk.id] = chunk
            order.append(chunk.id)
        else:
            merged[chunk.id] = _prefer(existing, chunk)
    return [merged[chunk_id] for chunk_id in order]


class AgentState(TypedDict):
    """Single-turn Q&A state.

    ``chunks`` uses :func:`merge_chunks` as its reducer; ``strategies_tried``
    accumulates via list concatenation. There is no ``retry_count`` field: the
    number of retries equals ``len(strategies_tried)``.
    """

    question: str
    current_query: str
    chunks: Annotated[list[Chunk], merge_chunks]
    strategies_tried: Annotated[list[Literal["step_back", "specific", "fusion"]], operator.add]
    answer: str | None
    cited_chunk_ids: list[str]
