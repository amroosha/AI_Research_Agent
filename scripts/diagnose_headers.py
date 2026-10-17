"""Diagnose repeated first and last lines in parsed page JSONL files.

Usage:
    python scripts/diagnose_headers.py --pages data/pages
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose repeated page headers and footers.")
    parser.add_argument("--pages", type=Path, default=Path("data/pages"))
    return parser


def load_pages(pages_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Load kept page records grouped by book slug."""
    books: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(pages_dir.glob("*.jsonl")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise TypeError(f"{path}:{line_number}: expected a JSON object")
            slug = str(record["book_slug"])
            books.setdefault(slug, []).append(record)
    return books


def nonempty_lines(text: str) -> list[str]:
    """Return stripped, non-empty lines from page text."""
    return [line.strip() for line in text.splitlines() if line.strip()]


def normalized_line(line: str) -> str:
    """Replace every digit with ``#`` for variant grouping."""
    return re.sub(r"\d", "#", line)


def print_frequencies(label: str, counts: Counter[str], total: int) -> None:
    print(label)
    for line, count in counts.most_common(15):
        print(f"  {count:5d}  {count / total * 100:6.2f}%  {line}")


def print_samples(pages: list[dict[str, Any]], rng: random.Random) -> None:
    print("sample pages")
    for page in rng.sample(pages, min(5, len(pages))):
        lines = nonempty_lines(str(page.get("text", "")))
        print(f"  pdf_page={page.get('pdf_page')}")
        print(f"    first 2: {lines[:2]}")
        print(f"    last 2:  {lines[-2:] if lines else []}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    books = load_pages(args.pages)
    rng = random.Random(0)
    for slug, pages in books.items():
        first_lines: Counter[str] = Counter()
        last_lines: Counter[str] = Counter()
        for page in pages:
            lines = nonempty_lines(str(page.get("text", "")))
            if lines:
                first_lines[normalized_line(lines[0])] += 1
                last_lines[normalized_line(lines[-1])] += 1
        print(f"== {slug} ({len(pages)} pages) ==")
        print_frequencies("first lines", first_lines, len(pages))
        print_frequencies("last lines", last_lines, len(pages))
        print_samples(pages, rng)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
