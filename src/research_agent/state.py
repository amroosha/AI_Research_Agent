"""Persistent state contracts for all agent stages."""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, model_validator

Strategy = Literal["step_back", "specific", "fusion"]


class Chunk(BaseModel):
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
        if self.source == "book" and (self.book_slug is None or self.pdf_page_start is None):
            raise ValueError("book chunks require book_slug and pdf_page_start")
        if self.source == "web" and self.url is None:
            raise ValueError("web chunks require url")
        if self.score is not None and not 0 <= self.score <= 100:
            raise ValueError("score must be between 0 and 100 when set")
        return self


def _prefer(existing: Chunk, candidate: Chunk) -> Chunk:
    if existing.score is None:
        return candidate if candidate.score is not None else existing
    if candidate.score is None:
        return existing
    return candidate if candidate.score > existing.score else existing


def merge_chunks(old: list[Chunk], new: list[Chunk]) -> list[Chunk]:
    merged: dict[str, Chunk] = {}
    order: list[str] = []
    for chunk in (*old, *new):
        if chunk.id not in merged:
            order.append(chunk.id)
            merged[chunk.id] = chunk
        else:
            merged[chunk.id] = _prefer(merged[chunk.id], chunk)
    return [merged[key] for key in order]


class AgentState(TypedDict):
    question: str
    current_query: str
    mode: Literal["fast", "deep"]
    queries: list[str]
    tried_queries: Annotated[list[str], operator.add]
    chunks: Annotated[list[Chunk], merge_chunks]
    strategies_tried: Annotated[list[Strategy], operator.add]
    web_used: bool
    answer: str | None
    cited_chunk_ids: list[str]
    status: str
    warnings: Annotated[list[str], operator.add]
    llm_calls: Annotated[int, operator.add]


# Named alias retained for importers that used the first Stage 3 draft.
Stage3State = AgentState
