"""Streamlit chat workspace for the local research agent."""

from __future__ import annotations

import streamlit as st

from research_agent.agent import AgentDeps, stream_agent
from research_agent.config import CHUNKS_DIR, DEFAULT_INGEST_SETTINGS, INDEX_DIR
from research_agent.env import load_env
from research_agent.grader import grade_chunks
from research_agent.ingest.embedder import BgeEmbedder
from research_agent.llm import get_llm
from research_agent.modes import get_profile
from research_agent.retrieval import HybridRetriever
from research_agent.ui import (
    ChatTurn,
    RunView,
    build_history_context,
    cited_chunks,
    event_label,
    source_label,
)
from research_agent.web import make_tavily_search

st.set_page_config(
    page_title="Research desk",
    page_icon=":material/auto_awesome:",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def load_embedder(model_name: str) -> BgeEmbedder:
    """Load the embedding model once and share it across Streamlit reruns."""

    return BgeEmbedder(model_name)


@st.cache_resource(show_spinner=False)
def build_dependencies(mode: str, web_enabled: bool) -> AgentDeps:
    """Build expensive retrieval and model resources once per mode."""

    load_env()
    profile = get_profile(mode)
    llm = get_llm()
    return AgentDeps(
        retriever=HybridRetriever(
            INDEX_DIR,
            CHUNKS_DIR,
            load_embedder(DEFAULT_INGEST_SETTINGS.embed_model),
            profile.retrieval,
        ),
        llm=llm,
        llm_short=get_llm(num_predict=128),
        grade_fn=grade_chunks,
        web_search_fn=make_tavily_search() if web_enabled else None,
        profile=profile,
    )


def initialize_state() -> None:
    st.session_state.setdefault("turns", [])
    st.session_state.setdefault("mode", "fast")
    st.session_state.setdefault("web_enabled", True)


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## Research desk")
        st.caption("A local-first chat for cited, inspectable answers.")
        st.divider()
        st.session_state.mode = st.segmented_control(
            "Research mode",
            options=("fast", "deep"),
            format_func=lambda value: "Fast" if value == "fast" else "Deep",
            default=st.session_state.mode,
            help="Deep mode can rewrite the query and use more retrieval passes.",
        )
        st.session_state.web_enabled = st.toggle(
            "Use Tavily web research",
            value=st.session_state.web_enabled,
            help="Search Tavily for every question and combine results with local evidence.",
        )
        st.divider()
        st.markdown("**How it works**")
        st.caption("1. Search the indexed library\n\n2. Score the evidence\n\n3. Synthesize with citations")
        if st.session_state.turns:
            st.divider()
            if st.button("Clear conversation", icon=":material/delete_sweep:", width="stretch"):
                st.session_state.turns = []
                st.rerun()


def render_sources(view: RunView) -> None:
    with st.expander(
        f"Evidence used · {len(view.chunks)} passages",
        expanded=False,
        icon=":material/library_books:",
    ):
        cited = {chunk.id for _, chunk in cited_chunks(view)}
        if not view.chunks:
            st.caption("No passages were returned.")
            return
        for index, chunk in enumerate(view.chunks[:10], 1):
            marker = "Cited" if chunk.id in cited else "Context"
            st.markdown(f"**[{index}] {marker} · {chunk.source.title()}**")
            if chunk.url:
                st.markdown(f"[{source_label(chunk)}]({chunk.url})")
            else:
                st.markdown(source_label(chunk))
            st.caption(chunk.text)


def render_turn(turn: ChatTurn) -> None:
    with st.chat_message("user"):
        st.markdown(turn.question)
    with st.chat_message("assistant", avatar=":material/auto_awesome:"):
        st.markdown(turn.answer or "The agent did not produce an answer.")
        if turn.view.warnings:
            st.caption(" · ".join(dict.fromkeys(turn.view.warnings)))
        render_sources(turn.view)


def run_research(question: str, history: list[ChatTurn]) -> ChatTurn | None:
    history_context = build_history_context(history)
    agent_question = f"{history_context}\n\nCurrent question: {question}" if history_context else question
    with st.status("Preparing research models", expanded=True) as startup_status:
        startup_status.write("Loading the embedding model and retrieval index...")
        try:
            deps = build_dependencies(st.session_state.mode, st.session_state.web_enabled)
        except (OSError, RuntimeError, ValueError) as exc:
            startup_status.update(label="Could not prepare research models", state="error")
            st.error(f"Could not start the research agent: {exc}", icon=":material/error:")
            st.info("Check that the index exists, Ollama is running, and OLLAMA_MODEL is set.")
            return None
        startup_status.update(label="Research models ready", state="complete", expanded=False)

    view = RunView()
    with st.chat_message("assistant", avatar=":material/auto_awesome:"):
        with st.status("Researching", expanded=True) as status:
            try:
                for event in stream_agent(agent_question, deps):
                    view.apply(event)
                    status.write(event_label(event.node))
                status.update(label="Research complete", state="complete", expanded=False)
            except (OSError, RuntimeError, ValueError) as exc:
                status.update(label="Research failed", state="error")
                st.error(f"The investigation failed: {exc}", icon=":material/error:")
                return None
        st.markdown(view.answer or "The agent did not produce an answer.")
        if view.warnings:
            st.caption(" · ".join(dict.fromkeys(view.warnings)))
        render_sources(view)
    return ChatTurn(question=question, answer=view.answer or "", view=view)


def main() -> None:
    initialize_state()
    render_sidebar()
    selected = None

    st.title("Ask the research desk")
    st.caption("Follow the thread. Earlier questions and answers guide follow-up research.")

    if not st.session_state.turns:
        st.info(
            "Start with a focused question. Ask a follow-up when you want to narrow, "
            "compare, or challenge the previous answer.",
            icon=":material/chat:",
        )
        suggestions = {
            "Explain a concept": "What makes a machine-learning system reliable after deployment?",
            "Compare approaches": "Compare batch and online feature computation.",
            "Challenge an answer": "What are the main limitations of retrieval-augmented generation?",
        }
        selected = st.pills("Try a prompt", list(suggestions), label_visibility="collapsed")
    for turn in st.session_state.turns:
        render_turn(turn)

    prompt = st.chat_input(
        "Ask a question or follow up on the thread",
        key="chat_input",
        submit_mode="disable",
    )
    if selected:
        prompt = suggestions[selected]
    if prompt:
        completed_turn = run_research(prompt, st.session_state.turns)
        if completed_turn:
            st.session_state.turns.append(completed_turn)
            st.rerun()


if __name__ == "__main__":
    main()
