"""LangGraph Stage 3 agent with the exact dependency and event contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langgraph.graph import END, START, StateGraph

from research_agent.generate import generate_answer
from research_agent.modes import ModeProfile, run_metadata
from research_agent.rewrite import expand_queries
from research_agent.state import AgentState, Chunk


@dataclass(frozen=True)
class AgentDeps:
    retriever: Any
    llm: Any
    llm_short: Any
    grade_fn: Any
    web_search_fn: Any | None
    profile: ModeProfile


@dataclass(frozen=True)
class AgentEvent:
    node: str
    update: dict
    text: str

    def values(self):
        """Mapping-compatible view retained for simple stream consumers."""

        return (self.update,)


def expand(deps: AgentDeps, state: AgentState) -> dict:
    if not deps.profile.pre_expand:
        return {
            "queries": [state["question"]],
            "current_query": state["question"],
            "llm_calls": 0,
            "warnings": [],
        }
    result = expand_queries(deps.llm_short, state["question"], deps.profile.n_variants)
    return {
        "queries": result["queries"],
        "current_query": result["queries"][0],
        "llm_calls": result["llm_calls"],
        "warnings": ["rewrite fallback"] if result["fallback"] else [],
    }


def retrieve(deps: AgentDeps, state: AgentState) -> dict:
    queries = state["queries"] or [state["current_query"]]
    chunks = [chunk for query in queries for chunk in deps.retriever.retrieve(query)]
    return {"chunks": chunks, "tried_queries": queries, "status": "grading"}


def grade(deps: AgentDeps, state: AgentState) -> dict:
    try:
        graded = deps.grade_fn(state["question"], state["chunks"])
    except TypeError:
        # Stage 2's public helper still accepts its injected LLM as arg 1.
        graded = deps.grade_fn(deps.llm, state["question"], state["chunks"])
    return {"chunks": graded, "llm_calls": 1}


def rewrite(deps: AgentDeps, state: AgentState) -> dict:
    index = len(state["strategies_tried"])
    if index >= deps.profile.routing.max_retries:
        return {"status": "web_search"}
    strategy = deps.profile.routing.rewrite_order[index]
    result = expand_queries(deps.llm_short, state["question"], 1, strategy)
    return {
        "queries": result["queries"],
        "current_query": result["queries"][0],
        "strategies_tried": [strategy],
        "status": "retrieving",
        "llm_calls": result["llm_calls"],
        "warnings": ["rewrite fallback"] if result["fallback"] else [],
    }


def web_search(deps: AgentDeps, state: AgentState) -> dict:
    if deps.web_search_fn is None:
        return {"web_used": False, "status": "generating", "warnings": ["web disabled"]}
    query = state["current_query"]
    if "Current question:" in query:
        query = query.rsplit("Current question:", 1)[-1].strip()
    results = deps.web_search_fn(query, deps.profile.web_max_results)
    chunks = [
        Chunk(
            id=f"web-{i}",
            text=item.snippet or item.title,
            source="web",
            url=item.url,
            title=item.title,
        )
        for i, item in enumerate(results[: deps.profile.web_slots])
    ]
    return {"chunks": chunks, "web_used": True, "status": "generating"}


def generate(deps: AgentDeps, state: AgentState) -> dict:
    relevant = [
        chunk
        for chunk in state["chunks"]
        if chunk.source == "book"
        and chunk.score is not None
        and chunk.score >= deps.profile.routing.relevant_threshold
    ]
    web = [chunk for chunk in state["chunks"] if chunk.source == "web"]
    selected = (relevant + web)[: deps.profile.generate_k]
    context_parts = []
    for index, chunk in enumerate(selected, 1):
        if chunk.source == "web":
            source = f"Web result: {chunk.title or 'Untitled'} ({chunk.url})"
        else:
            source = "Local document"
        context_parts.append(f"[{index}] {source}\n{chunk.text}")
    context = "\n\n".join(context_parts)
    result = generate_answer(
        deps.llm,
        state["question"],
        context,
        state["web_used"],
        deps.profile.routing.relevant_threshold,
    )
    return {
        "generation_chunks": selected,
        "answer": result["answer"],
        "cited_chunk_ids": [str(i) for i in result["citations"]],
        "status": result["status"],
        "warnings": result["warnings"],
        "llm_calls": result["llm_calls"],
    }


def _route(deps: AgentDeps, state: AgentState) -> str:
    # Web research is an additional evidence source, not only a fallback for
    # questions that happen to match no local documents.
    if deps.web_search_fn is not None:
        return "web_search"
    if any(
        (chunk.score or 0) >= deps.profile.routing.relevant_threshold for chunk in state["chunks"]
    ):
        return "generate"
    if deps.profile.name == "fast":
        return "web_search" if deps.web_search_fn is not None else "generate"
    return (
        "rewrite"
        if len(state["strategies_tried"]) < deps.profile.routing.max_retries
        else "web_search"
    )


def build_graph(deps: AgentDeps):
    graph = StateGraph(AgentState)
    for name, fn in (
        ("expand", expand),
        ("retrieve", retrieve),
        ("grade", grade),
        ("rewrite", rewrite),
        ("web_search", web_search),
        ("generate", generate),
    ):
        graph.add_node(name, lambda state, fn=fn: fn(deps, state))
    graph.add_edge(START, "expand")
    graph.add_edge("expand", "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade",
        lambda state: _route(deps, state),
        {"generate": "generate", "rewrite": "rewrite", "web_search": "web_search"},
    )
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("web_search", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


def stream_agent(question: str, deps: AgentDeps, extra_metadata: dict | None = None):
    if not question.strip():
        raise ValueError("question must not be empty")
    initial: AgentState = {
        "question": question.strip(),
        "current_query": question.strip(),
        "mode": deps.profile.name,
        "queries": [question.strip()],
        "tried_queries": [],
        "chunks": [],
        "generation_chunks": [],
        "strategies_tried": [],
        "web_used": False,
        "answer": None,
        "cited_chunk_ids": [],
        "status": "starting",
        "warnings": [],
        "llm_calls": 0,
    }
    metadata = run_metadata(deps.profile, deps.retriever)
    metadata.update(extra_metadata or {})
    config = {
        "metadata": metadata,
        "run_name": "research-agent",
        "tags": [deps.profile.name],
        "recursion_limit": 50,
    }
    for event in build_graph(deps).stream(initial, config=config):
        node, update = next(iter(event.items()))
        yield AgentEvent(node, update, node)


def run_agent(question: str, deps: AgentDeps) -> AgentState:
    final: AgentState = {}
    for event in stream_agent(question, deps):
        final.update(event.update)
    return final


def describe_event(event: AgentEvent | dict) -> str:
    return event.text if isinstance(event, AgentEvent) else ", ".join(event)
