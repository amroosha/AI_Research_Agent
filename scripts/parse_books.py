"""Parse PDFs in ``data/raw`` into cleaned page JSONL in ``data/pages``.

Usage:
    python scripts/parse_books.py --input data/raw --output data/pages
        [--only SLUG] [--force]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from research_agent.ingest.pdf_parse import parse_directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parse O'Reilly PDFs into page JSONL.")
    parser.add_argument("--input", type=Path, default=Path("data/raw"), help="Directory of PDFs.")
    parser.add_argument(
        "--output", type=Path, default=Path("data/pages"), help="Directory for page JSONL output."
    )
    parser.add_argument("--only", default=None, help="Only parse the book with this slug.")
    parser.add_argument(
        "--force", action="store_true", help="Reparse even when the PDF sha256 is unchanged."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    manifest = parse_directory(
        input_dir=args.input,
        output_dir=args.output,
        only=args.only,
        force=args.force,
    )
    for slug, entry in sorted(manifest.items()):
        print(
            f"{slug}: kept {entry['kept_pages']}/{entry['total_pages']} pages "
            f"(low_text={entry['low_text']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
