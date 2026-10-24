"""Presentation helpers for the Stage 4 Streamlit application."""

from __future__ import annotations

from dataclasses import dataclass, field

from research_agent.agent import AgentEvent
from research_agent.state import Chunk, merge_chunks

EVENT_LABELS = {
    "expand": "Framing the question",
    "retrieve": "Searching the library",
    "grade": "Checking evidence",
    "rewrite": "Refining the search",
    "web_search": "Checking the web",
    "generate": "Writing the answer",
}


@dataclass
class ChatTurn:
    """One completed conversation turn kept in Streamlit session state."""

    question: str
    answer: str
    view: RunView


@dataclass
class RunView:
    """UI-safe projection of the graph's incremental state."""

    events: list[str] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    answer: str | None = None
    cited_chunk_ids: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    status: str = "starting"
    web_used: bool = False

    def apply(self, event: AgentEvent) -> None:
        update = event.update
        self.events.append(event.node)
        if update.get("chunks"):
            self.chunks = merge_chunks(self.chunks, update["chunks"])
        if update.get("answer") is not None:
            self.answer = update["answer"]
        if update.get("cited_chunk_ids"):
            self.cited_chunk_ids = list(update["cited_chunk_ids"])
        if update.get("warnings"):
            self.warnings.extend(update["warnings"])
        if update.get("status"):
            self.status = update["status"]
        if update.get("web_used"):
            self.web_used = True


def build_history_context(turns: list[ChatTurn], limit: int = 4) -> str:
    """Create a compact, explicit context block for follow-up questions."""

    if not turns:
        return ""
    recent = turns[-limit:]
    lines = [
        (
            "Conversation context from earlier turns. Use it only to understand "
            "references and direction; answer the current question."
        ),
    ]
    for turn in recent:
        lines.extend(
            (
                f"User: {turn.question}",
                f"Assistant: {turn.answer}",
            )
        )
    return "\n".join(lines)


def event_label(node: str) -> str:
    return EVENT_LABELS.get(node, node.replace("_", " ").title())


def cited_chunks(view: RunView) -> list[tuple[int, Chunk]]:
    """Resolve generated [n] citations against the ordered context window."""

    resolved: list[tuple[int, Chunk]] = []
    for citation in view.cited_chunk_ids:
        try:
            position = int(citation)
        except (TypeError, ValueError):
            continue
        if 1 <= position <= len(view.chunks):
            resolved.append((position, view.chunks[position - 1]))
    return resolved


def source_label(chunk: Chunk) -> str:
    if chunk.source == "web":
        return chunk.title or chunk.url or "Web source"
    title = chunk.book_title or chunk.book_slug or "Library source"
    if chunk.pdf_page_start is None:
        return title
    page_end = chunk.pdf_page_end or chunk.pdf_page_start
    pages = str(chunk.pdf_page_start) if page_end == chunk.pdf_page_start else (
        f"{chunk.pdf_page_start}-{page_end}"
    )
    return f"{title} · pp. {pages}"
