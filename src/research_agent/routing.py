"""Pure routing logic: classify scores and decide the next action.

No I/O, no LLM, no LangGraph. Every function takes ``cfg: Settings`` so the
thresholds and rewrite order are injected rather than hardcoded.
"""

from __future__ import annotations

from typing import Literal

from research_agent.config import DEFAULT_SETTINGS, Settings
from research_agent.state import AgentState, Chunk

Route = Literal["generate", "rewrite", "web_search"]
Classification = Literal["relevant", "ambiguous", "irrelevant"]


def classify(score: int, cfg: Settings = DEFAULT_SETTINGS) -> Classification:
    """Classify a grader score into a band.

    Boundaries belong to the higher bucket: ``>= relevant_threshold`` is
    relevant, ``>= ambiguous_threshold`` is ambiguous, otherwise irrelevant.
    """
    if score >= cfg.relevant_threshold:
        return "relevant"
    if score >= cfg.ambiguous_threshold:
        return "ambiguous"
    return "irrelevant"


def _book_chunks(state: AgentState) -> list[Chunk]:
    """Book chunks only; web chunks are ignored by the router."""
    return [c for c in state["chunks"] if c.source == "book"]


def route(state: AgentState, cfg: Settings = DEFAULT_SETTINGS) -> Route:
    """Decide the next action from the current state.

    Only book chunks count. If any book chunk is ungraded (``score is None``),
    raise ``ValueError``. Any relevant book chunk routes to ``generate`` (even
    at max retries). Otherwise, rewrite while retries remain, else ``web_search``.
    """
    book_chunks = _book_chunks(state)
    if any(chunk.score is None for chunk in book_chunks):
        raise ValueError("cannot route with ungraded book chunks (score is None)")

    if any(classify(chunk.score, cfg) == "relevant" for chunk in book_chunks):
        return "generate"
    if len(state["strategies_tried"]) < cfg.max_retries:
        return "rewrite"
    return "web_search"


def usable_chunks(state: AgentState, cfg: Settings = DEFAULT_SETTINGS) -> list[Chunk]:
    """Return chunks usable for generation.

    Book chunks scoring ``>= ambiguous_threshold``, sorted by score descending
    (stable, preserving original order among equal scores), followed by all web
    chunks in their original order.
    """
    graded_books = [
        chunk
        for chunk in _book_chunks(state)
        if chunk.score is not None and chunk.score >= cfg.ambiguous_threshold
    ]
    graded_books.sort(key=lambda chunk: chunk.score, reverse=True)
    web_chunks = [chunk for chunk in state["chunks"] if chunk.source == "web"]
    return [*graded_books, *web_chunks]


def next_strategy(state: AgentState, cfg: Settings = DEFAULT_SETTINGS) -> str:
    """Return the first rewrite strategy not yet tried.

    Raises ``ValueError`` if every strategy in ``cfg.rewrite_order`` has been
    tried.
    """
    tried = set(state["strategies_tried"])
    for strategy in cfg.rewrite_order:
        if strategy not in tried:
            return strategy
    raise ValueError("all rewrite strategies have been tried")
