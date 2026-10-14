"""Build the Chroma + BM25 indexes from chunk JSONL into ``data/index``.

Usage:
    python scripts/build_index.py --chunks data/chunks --index data/index [--rebuild]

Uses the real BGE embedder (downloads the model on first use). ``--rebuild``
wipes ``data/index`` first. Refuses when the corpus changed unless ``--rebuild``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from research_agent.config import DEFAULT_INGEST_SETTINGS
from research_agent.ingest.embedder import BgeEmbedder
from research_agent.ingest.indexing import IndexStateError, build_index

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Chroma + BM25 indexes from chunks.")
    parser.add_argument("--chunks", type=Path, default=Path("data/chunks"))
    parser.add_argument("--index", type=Path, default=Path("data/index"))
    parser.add_argument("--embed-batch", type=int, default=32, help="Embedding batch size.")
    parser.add_argument("--rebuild", action="store_true", help="Wipe data/index and rebuild.")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    settings = DEFAULT_INGEST_SETTINGS
    embedder = BgeEmbedder(settings.embed_model, batch_size=args.embed_batch)
    try:
        manifest = build_index(
            chunks_dir=args.chunks,
            index_dir=args.index,
            embedder=embedder,
            settings=settings,
            embed_batch=args.embed_batch,
            rebuild=args.rebuild,
        )
    except IndexStateError as error:
        logger.error("%s", error)
        return 1

    if manifest.get("status") == "up_to_date":
        print("up to date")
        return 0
    print(
        f"built index: {manifest['total_chunks']} chunks, "
        f"dim={manifest['embedding_dim']}, corpus_version={manifest['corpus_version'][:12]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
