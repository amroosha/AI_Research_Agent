"""Small pure helpers used by retrieval, web search, and answer generation."""

from __future__ import annotations

import re
from collections.abc import Iterable

from research_agent.state import Chunk

_CITATION_RE = re.compile(r"\[([^\]]+)\]")


def unique_chunks(chunks: Iterable[Chunk]) -> list[Chunk]:
    """Deduplicate chunks by id while retaining first-seen order."""

    seen: set[str] = set()
    result: list[Chunk] = []
    for chunk in chunks:
        if chunk.id not in seen:
            seen.add(chunk.id)
            result.append(chunk)
    return result


def format_context(chunks: Iterable[Chunk], limit: int | None = None) -> str:
    """Format chunks with stable citation markers for an answer prompt."""

    selected = list(chunks)[:limit] if limit is not None else list(chunks)
    return "\n\n".join(f"[{chunk.id}]\n{chunk.text}" for chunk in selected)


def extract_citations(answer: str) -> list[str]:
    """Extract bracket citations, preserving order and removing duplicates."""

    seen: set[str] = set()
    citations: list[str] = []
    for value in _CITATION_RE.findall(answer):
        value = value.strip()
        if value and value not in seen:
            seen.add(value)
            citations.append(value)
    return citations
