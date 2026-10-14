"""Verify the built index against the chunk files. Exit code != 0 on any failure.

Usage:
    python scripts/verify_index.py --chunks data/chunks --index data/index
        [--smoke tests/smoke_queries.yaml]

Uses the real BGE embedder (needed for self-retrieval + smoke queries). Smoke
query ranks are report-only and never affect the exit code.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from research_agent.config import DEFAULT_INGEST_SETTINGS
from research_agent.ingest.embedder import BgeEmbedder
from research_agent.ingest.indexing import verify_index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify the built index.")
    parser.add_argument("--chunks", type=Path, default=Path("data/chunks"))
    parser.add_argument("--index", type=Path, default=Path("data/index"))
    parser.add_argument("--smoke", type=Path, default=Path("tests/smoke_queries.yaml"))
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    settings = DEFAULT_INGEST_SETTINGS
    embedder = BgeEmbedder(settings.embed_model)

    report = verify_index(
        chunks_dir=args.chunks,
        index_dir=args.index,
        embedder=embedder,
        smoke_path=args.smoke,
        show_progress=True,
    )

    print("== checks ==")
    for check in report.checks:
        status = "PASS" if check.passed else "FAIL"
        suffix = f" ({check.detail})" if check.detail else ""
        print(f"[{status}] {check.name}{suffix}")

    if report.smoke:
        print("== smoke queries (report only) ==")
        for row in report.smoke:
            chroma = row["chroma_rank"] if row["chroma_rank"] is not None else "miss"
            bm25 = row["bm25_rank"] if row["bm25_rank"] is not None else "miss"
            print(f"  chroma={chroma!s:>4}  bm25={bm25!s:>4}  | {row['query']}")

    if report.ok:
        print("VERIFY OK")
        return 0
    print(f"VERIFY FAILED: {len(report.failures)} check(s) failed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
