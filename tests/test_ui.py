from __future__ import annotations

from research_agent.agent import AgentEvent
from research_agent.state import Chunk
from research_agent.ui import (
    ChatTurn,
    RunView,
    build_history_context,
    cited_chunks,
    event_label,
    source_label,
)


def _book(chunk_id: str, score: int | None = None) -> Chunk:
    return Chunk(
        id=chunk_id,
        text=f"passage {chunk_id}",
        source="book",
        book_slug="ml-systems",
        book_title="Designing ML Systems",
        pdf_page_start=12,
        pdf_page_end=14,
        score=score,
    )


def test_run_view_accumulates_chunks_and_resolves_numbered_citations() -> None:
    view = RunView()
    view.apply(AgentEvent("grade", {"chunks": [_book("a", 90), _book("b", 80)]}, "grade"))
    view.apply(
        AgentEvent(
            "generate",
            {
                "generation_chunks": [_book("a", 90), _book("b", 80)],
                "answer": "Synthesis [2]",
                "cited_chunk_ids": ["2"],
                "status": "done",
            },
            "generate",
        )
    )

    assert view.answer == "Synthesis [2]"
    assert [chunk.id for _, chunk in cited_chunks(view)] == ["b"]
    assert view.status == "done"


def test_run_view_exposes_only_generation_chunks() -> None:
    view = RunView()
    view.apply(
        AgentEvent(
            "grade",
            {"chunks": [_book("unrelated", 15), _book("relevant", 90)]},
            "grade",
        )
    )
    view.apply(
        AgentEvent(
            "generate",
            {"generation_chunks": [_book("relevant", 90)]},
            "generate",
        )
    )

    assert [chunk.id for chunk in view.chunks] == ["unrelated", "relevant"]
    assert [chunk.id for chunk in view.generation_chunks] == ["relevant"]


def test_run_view_keeps_higher_score_for_duplicate_chunk() -> None:
    view = RunView()
    view.apply(AgentEvent("retrieve", {"chunks": [_book("a")]}, "retrieve"))
    view.apply(AgentEvent("grade", {"chunks": [_book("a", 95)]}, "grade"))

    assert view.chunks[0].score == 95


def test_source_labels_cover_book_and_web() -> None:
    web = Chunk(id="w", text="result", source="web", url="https://example.com", title="A result")
    assert source_label(_book("a")) == "Designing ML Systems · pp. 12-14"
    assert source_label(web) == "A result"
    assert event_label("web_search") == "Checking the web"


def test_history_context_is_explicit_and_bounded() -> None:
    turns = [
        ChatTurn(f"question {index}", f"answer {index}", RunView())
        for index in range(5)
    ]

    context = build_history_context(turns, limit=2)

    assert "question 2" not in context
    assert "question 3" in context
    assert "answer 4" in context
    assert "answer the current question" in context
