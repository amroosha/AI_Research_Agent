"""Chunk parsed pages in ``data/pages`` into ``data/chunks``.

Usage:
    python scripts/chunk_books.py --pages data/pages --output data/chunks [--only SLUG]

Always rebuilds. Uses the bge tokenizer (no special tokens) for token counts,
so the first run downloads the tokenizer files.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import transformers
from transformers import AutoTokenizer

from research_agent.config import DEFAULT_INGEST_SETTINGS
from research_agent.ingest.chunking import chunk_directory


def build_tokenizer(model_name: str):
    """Load the bge tokenizer once (downloads on first use)."""
    return AutoTokenizer.from_pretrained(model_name)


def make_count_tokens(tokenizer):
    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    return count_tokens


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chunk parsed book pages.")
    parser.add_argument("--pages", type=Path, default=Path("data/pages"))
    parser.add_argument("--output", type=Path, default=Path("data/chunks"))
    parser.add_argument("--only", default=None, help="Only chunk the book with this slug.")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    settings = DEFAULT_INGEST_SETTINGS
    # Chunking tokenizes over-target candidate chunks (current + next block) only
    # to decide where to split; those transient measurements can exceed the
    # model's 512-token limit even though every final chunk stays <= max_tokens
    # (450). Silence that specific transformers warning so the run output is clean.
    transformers.logging.set_verbosity_error()
    tokenizer = build_tokenizer(settings.embed_model)
    manifest = chunk_directory(
        pages_dir=args.pages,
        output_dir=args.output,
        settings=settings,
        count_tokens=make_count_tokens(tokenizer),
        only=args.only,
    )
    for slug, entry in sorted(manifest.items()):
        stats = entry["token_stats"]
        print(
            f"{slug}: {entry['chunk_count']} chunks, "
            f"tokens min/p50/p95/max = {stats['min']}/{stats['p50']}/{stats['p95']}/{stats['max']}, "
            f"above_max={entry['above_max']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
