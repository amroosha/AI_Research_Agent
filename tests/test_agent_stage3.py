from __future__ import annotations

from types import SimpleNamespace

from research_agent.agent import AgentDeps, stream_agent
from research_agent.grader import grade_chunks
from research_agent.modes import FAST, AgentMode, coerce_mode
from research_agent.state import Chunk, Stage3State
from research_agent.web import TavilySearcher, WebResult


def _chunk(name: str = "c1") -> Chunk:
    return Chunk(
        id=name,
        text="relevant context",
        source="book",
        book_slug="book",
        pdf_page_start=1,
        pdf_page_end=1,
    )


class FakeRetriever:
    def retrieve(self, _query):
        return [_chunk()]


class FakeStructured:
    def __init__(self, value):
        self.value = value

    def invoke(self, _prompt):
        return self.value


class FakeLLM:
    def with_structured_output(self, schema):
        if schema.__name__ == "ChunkLabel":
            return FakeStructured(SimpleNamespace(label="direct"))
        if schema.__name__ == "GradeBatch":
            return FakeStructured(SimpleNamespace(refs=[1], scores=[95]))
        if schema.__name__ == "AnswerResult":
            return FakeStructured(SimpleNamespace(answer="answer [c1]", cited_chunk_ids=["c1"]))
        return FakeStructured(SimpleNamespace(queries=["broader query"]))


def test_modes_are_fast_and_deep() -> None:
    assert coerce_mode("fast") is AgentMode.FAST
    assert coerce_mode("deep") is AgentMode.DEEP


def test_fast_graph_streams_answer() -> None:
    deps = AgentDeps(FakeRetriever(), FakeLLM(), FakeLLM(), grade_chunks, lambda q, n: [], FAST)
    updates = list(stream_agent("question", deps))
    assert updates
    assert any("answer" in value for update in updates for value in update.values())


def test_stage3_state_extends_legacy_state() -> None:
    assert {"mode", "status", "warnings", "llm_calls"} <= set(Stage3State.__annotations__)


def test_tavily_adapter_maps_results() -> None:
    fake = SimpleNamespace(
        search=lambda **kwargs: {"results": [{"title": "T", "url": "https://t", "content": "S"}]}
    )
    results = TavilySearcher(client=fake).search("q")
    assert results == [WebResult("T", "https://t", "S")]
