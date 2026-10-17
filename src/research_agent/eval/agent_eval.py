"""Pure evaluation helpers for Stage 3 orchestration."""

from __future__ import annotations

from dataclasses import dataclass

from research_agent.generate import AnswerResult
from research_agent.helpers import extract_citations


@dataclass(frozen=True)
class AnswerEvaluation:
    has_answer: bool
    citations_valid: bool
    citation_count: int


def evaluate_answer(result: AnswerResult, chunk_ids: set[str]) -> AnswerEvaluation:
    """Check answer presence and that every citation points at supplied context."""

    citations = result.cited_chunk_ids or extract_citations(result.answer)
    return AnswerEvaluation(
        has_answer=bool(result.answer.strip()),
        citations_valid=all(citation in chunk_ids for citation in citations),
        citation_count=len(citations),
    )
