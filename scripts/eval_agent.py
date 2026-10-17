"""Evaluate the Stage 3 graph against a YAML set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from research_agent.agent import AgentDeps, run_agent
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
    parser.add_argument("--set", dest="set_path", type=Path, required=True)
    parser.add_argument("--mode", choices=("fast", "deep"), default="fast")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
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
        make_tavily_search() if args.mode == "deep" else lambda *_: [],
        get_profile(args.mode),
    )
    loaded = yaml.safe_load(args.set_path.read_text(encoding="utf-8")) or []
    items = loaded.get("queries", []) if isinstance(loaded, dict) else loaded
    if args.limit:
        items = items[: args.limit]
    report = [{"id": item.get("id"), **run_agent(item["query"], deps)} for item in items]
    # State contains Pydantic objects; emit only stable evaluation fields.
    output = {
        "mode": args.mode,
        "count": len(report),
        "statuses": [row["status"] for row in report],
    }
    text = json.dumps(output, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
