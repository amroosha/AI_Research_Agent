"""Run one hybrid query against the built index and show the fused results.

Usage:
    python scripts/try_retrieve.py "what is data leakage" [--final-k 6]

Prints, for each result: rank, chunk id, book, pdf pages, section path,
bm25_rank, vector_rank, rrf_score, and the first 120 characters of text.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from research_agent.config import CHUNKS_DIR, DEFAULT_INGEST_SETTINGS, DEFAULT_RETRIEVAL, INDEX_DIR
from research_agent.ingest.embedder import BgeEmbedder
from research_agent.retrieval import HybridRetriever


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hybrid retrieval smoke query.")
    parser.add_argument("query", help="the question to retrieve for")
    parser.add_argument("--final-k", type=int, default=None, help="results to show")
    parser.add_argument("--index", type=Path, default=INDEX_DIR)
    parser.add_argument("--chunks", type=Path, default=CHUNKS_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.query.strip():
        print("error: query is empty", file=sys.stderr)
        return 2

    started = time.perf_counter()
    embedder = BgeEmbedder(DEFAULT_INGEST_SETTINGS.embed_model)
    retriever = HybridRetriever(args.index, args.chunks, embedder, DEFAULT_RETRIEVAL)
    ready = time.perf_counter()

    items = retriever.retrieve_debug(args.query, final_k=args.final_k)
    done = time.perf_counter()

    print(f"corpus_version={retriever.corpus_version[:12]}  results={len(items)}")
    print(f"init={ready - started:.2f}s  query={done - ready:.2f}s")
    for rank, item in enumerate(items, start=1):
        chunk = item.chunk
        section = " > ".join(chunk.section_path)
        print(
            f"\n[{rank}] {chunk.id}\n"
            f"    book={chunk.book_slug}  pdf_pages={chunk.pdf_page_start}-{chunk.pdf_page_end}\n"
            f"    section={section}\n"
            f"    bm25_rank={item.bm25_rank}  vector_rank={item.vector_rank}  "
            f"rrf_score={item.rrf_score:.6f}\n"
            f"    {chunk.text[:120]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
