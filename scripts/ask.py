"""Ask the Stage 3 LangGraph agent."""

from __future__ import annotations

import argparse

from research_agent.agent import AgentDeps, describe_event, stream_agent
from research_agent.config import CHUNKS_DIR, DEFAULT_INGEST_SETTINGS, INDEX_DIR
from research_agent.env import load_env
from research_agent.grader import grade_chunks
from research_agent.ingest.embedder import BgeEmbedder
from research_agent.llm import get_llm
from research_agent.modes import get_profile
from research_agent.retrieval import HybridRetriever
from research_agent.web import make_tavily_search


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--mode", choices=("fast", "deep"), default="fast")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--no-web", action="store_true")
    args = parser.parse_args()
    load_env()
    llm = get_llm()
    deps = AgentDeps(
        HybridRetriever(
            INDEX_DIR,
            CHUNKS_DIR,
            BgeEmbedder(DEFAULT_INGEST_SETTINGS.embed_model),
        ),
        llm,
        get_llm(num_predict=128),
        grade_chunks,
        None if args.no_web else make_tavily_search(),
        get_profile(args.mode),
    )
    for event in stream_agent(args.question, deps):
        if args.stream:
            print(describe_event(event))
        for update in event.values():
            if update.get("answer"):
                print(update["answer"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
