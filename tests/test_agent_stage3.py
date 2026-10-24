from __future__ import annotations

from types import SimpleNamespace

from research_agent.agent import AgentDeps, generate, stream_agent
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


class EmptyRetriever:
    def retrieve(self, _query):
        return []


class IrrelevantRetriever:
    def retrieve(self, _query):
        return [_chunk().model_copy(update={"score": 10})]


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


class CaptureAnswerLLM(FakeLLM):
    def __init__(self):
        self.answer_prompt = ""

    def invoke(self, prompt):
        self.answer_prompt = prompt
        return SimpleNamespace(content="answer [1]")


def test_modes_are_fast_and_deep() -> None:
    assert coerce_mode("fast") is AgentMode.FAST
    assert coerce_mode("deep") is AgentMode.DEEP


def test_fast_graph_streams_answer() -> None:
    deps = AgentDeps(FakeRetriever(), FakeLLM(), FakeLLM(), grade_chunks, lambda q, n: [], FAST)
    updates = list(stream_agent("question", deps))
    assert updates
    assert any("answer" in value for update in updates for value in update.values())


def test_fast_graph_uses_web_for_irrelevant_local_results() -> None:
    web_calls = []

    def keep_irrelevant(_question, chunks):
        return chunks

    def search(query, limit):
        web_calls.append((query, limit))
        return [WebResult("Web answer", "https://example.com", "fresh context")]

    deps = AgentDeps(
        IrrelevantRetriever(),
        FakeLLM(),
        FakeLLM(),
        keep_irrelevant,
        search,
        FAST,
    )

    updates = list(stream_agent("best anime of all time", deps))

    assert any(event.node == "web_search" for event in updates)
    assert web_calls == [("best anime of all time", FAST.web_max_results)]


def test_fast_graph_uses_web_even_with_relevant_local_results() -> None:
    web_calls = []

    def search(query, limit):
        web_calls.append((query, limit))
        return [WebResult("Web answer", "https://example.com", "fresh context")]

    deps = AgentDeps(
        FakeRetriever(),
        FakeLLM(),
        FakeLLM(),
        grade_chunks,
        search,
        FAST,
    )

    updates = list(stream_agent("top rated anime", deps))

    assert any(event.node == "web_search" for event in updates)
    assert web_calls == [("top rated anime", FAST.web_max_results)]


def test_generation_omits_unrelated_local_chunks() -> None:
    llm = CaptureAnswerLLM()
    deps = AgentDeps(EmptyRetriever(), llm, llm, grade_chunks, None, FAST)
    state = {
        "question": "top rated anime",
        "current_query": "top rated anime",
        "mode": "fast",
        "queries": ["top rated anime"],
        "tried_queries": [],
        "chunks": [
            _chunk().model_copy(update={"score": 15}),
            Chunk(
                id="web-0",
                text="Anime ranking result",
                source="web",
                url="https://example.com",
                title="Anime rankings",
            ),
        ],
        "strategies_tried": [],
        "web_used": True,
        "answer": None,
        "cited_chunk_ids": [],
        "status": "generating",
        "warnings": [],
        "llm_calls": 0,
    }

    generate(deps, state)

    assert "relevant context" not in llm.answer_prompt
    assert "Anime ranking result" in llm.answer_prompt


def test_fast_graph_uses_web_when_local_index_is_empty() -> None:
    deps = AgentDeps(
        EmptyRetriever(),
        FakeLLM(),
        FakeLLM(),
        grade_chunks,
        lambda query, limit: [WebResult("T", "https://example.com", "S")],
        FAST,
    )

    updates = list(stream_agent("how to get a girlfriend", deps))

    assert any(event.node == "web_search" for event in updates)


def test_web_search_uses_current_question_without_history_wrapper() -> None:
    queries = []

    def search(query, _limit):
        queries.append(query)
        return []

    deps = AgentDeps(
        EmptyRetriever(),
        FakeLLM(),
        FakeLLM(),
        grade_chunks,
        search,
        FAST,
    )

    list(stream_agent("Earlier context\n\nCurrent question: best anime", deps))

    assert queries == ["best anime"]


def test_stage3_state_extends_legacy_state() -> None:
    assert {"mode", "status", "warnings", "llm_calls"} <= set(Stage3State.__annotations__)


def test_tavily_adapter_maps_results() -> None:
    fake = SimpleNamespace(
        search=lambda **kwargs: {"results": [{"title": "T", "url": "https://t", "content": "S"}]}
    )
    results = TavilySearcher(client=fake).search("q")
    assert results == [WebResult("T", "https://t", "S")]
