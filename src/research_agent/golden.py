"""Golden-set loading and validation (Stage 2, T3).

The golden set is the frozen evaluation corpus: one YAML file of queries with
the book slug and the inclusive PDF page range(s) where the answer lives. This
module loads that schema -- both the single-range form ``[28, 32]`` and the
multi-source form ``[[28, 32], [104, 109]]`` -- and validates it against the
built index. Validation problems are errors unless marked as warnings (ranges
too loose to be a useful target).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from research_agent.ingest.indexing import load_chunks
from research_agent.page_ranges import normalize_ranges

# A range wider than this many pages is too loose to be a useful target.
MAX_USEFUL_RANGE_WIDTH = 10


@dataclass(frozen=True)
class GoldenItem:
    """One golden query.

    ``book`` is ``None`` for negatives (``expected_book: none``); ``ranges``
    holds inclusive ``(start, end)`` PDF page pairs and is empty for negatives.
    """

    id: str
    query: str
    book: str | None
    ranges: tuple[tuple[int, int], ...]
    is_negative: bool
    source_file: str


@dataclass(frozen=True)
class GoldenValidation:
    """Result of :func:`validate_golden`: problems, split by severity."""

    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def normalize_query(query: str) -> str:
    """Case- and whitespace-normalized query, for duplicate/leakage checks."""
    return " ".join(query.lower().split())


def load_golden(path: Path | str) -> list[GoldenItem]:
    """Load a golden YAML file into ``GoldenItem`` records.

    Permissive on structure: missing fields become empty strings / empty page
    lists, which :func:`validate_golden` then reports as errors. Malformed page
    ranges raise a ``ValueError`` immediately (a schema violation, not a
    validation finding).
    """
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    items: list[GoldenItem] = []
    for index, entry in enumerate(data.get("queries") or []):
        item_id = f"{path.stem}:{index:03d}"
        if not isinstance(entry, dict):
            raise TypeError(f"{item_id}: entry is not a mapping: {entry!r}")
        query = str(entry.get("query") or "").strip()
        raw_book = entry.get("expected_book")
        book_text = "" if raw_book is None else str(raw_book).strip()
        is_negative = book_text.lower() == "none"
        items.append(
            GoldenItem(
                id=item_id,
                query=query,
                book=None if is_negative else book_text,
                ranges=normalize_ranges(entry.get("expected_pdf_pages"), item_id),
                is_negative=is_negative,
                source_file=str(path),
            )
        )
    return items


def validate_golden(
    items: Sequence[GoldenItem],
    index_dir: Path | str,
    chunks_dir: Path | str,
) -> GoldenValidation:
    """Validate golden items against the built index.

    Errors: empty query; empty ``expected_book``; a negative with pages; a
    positive without ranges; ranges with ``start < 1`` or ``start > end``; a
    positive book slug absent from the index manifest; a range end beyond the
    book's largest ``pdf_page_end``; duplicate (normalized) queries inside one
    file. Warning: any range wider than 10 pages.
    """
    errors: list[str] = []
    warnings: list[str] = []

    index_dir = Path(index_dir)
    manifest_path = index_dir / "index_manifest.json"
    if not manifest_path.exists():
        return GoldenValidation(errors=(f"index manifest not found: {manifest_path}",), warnings=())
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    known_books = set(manifest.get("per_book_counts", {}))

    max_page_end: dict[str, int] = {}
    for record in load_chunks(Path(chunks_dir)):
        end = record.get("pdf_page_end")
        if end is not None:
            slug = record["book_slug"]
            max_page_end[slug] = max(max_page_end.get(slug, 0), int(end))

    seen_queries: dict[str, str] = {}
    for item in items:
        if not item.query:
            errors.append(f"{item.id}: query is empty")
        else:
            normalized = normalize_query(item.query)
            if normalized in seen_queries:
                errors.append(f"{item.id}: duplicate query (same as {seen_queries[normalized]!r})")
            else:
                seen_queries[normalized] = item.id

        if item.is_negative:
            if item.ranges:
                errors.append(f"{item.id}: negative query has expected_pdf_pages")
            continue

        if not item.book:
            errors.append(f"{item.id}: empty expected_book")
            continue
        if not item.ranges:
            errors.append(f"{item.id}: positive query has no expected_pdf_pages")
            continue
        if item.book not in known_books:
            errors.append(f"{item.id}: book {item.book!r} not in index manifest")
        for start, end in item.ranges:
            if not (1 <= start <= end):
                errors.append(f"{item.id}: invalid range [{start}, {end}]")
                continue
            if end - start + 1 > MAX_USEFUL_RANGE_WIDTH:
                warnings.append(f"{item.id}: range [{start}, {end}] wider than 10 pages")
            limit = max_page_end.get(item.book, 0)
            if end > limit:
                errors.append(
                    f"{item.id}: range end {end} beyond {item.book}'s max pdf_page_end {limit}"
                )
    return GoldenValidation(errors=tuple(errors), warnings=tuple(warnings))


def find_leakage(dev_items: Sequence[GoldenItem], test_items: Sequence[GoldenItem]) -> list[str]:
    """Queries appearing (normalized) in both the dev and the test file."""
    dev_queries = {normalize_query(item.query): item.id for item in dev_items if item.query}
    leaked: list[str] = []
    for item in test_items:
        normalized = normalize_query(item.query)
        if normalized and normalized in dev_queries:
            leaked.append(f"{item.id} duplicates {dev_queries[normalized]}")
    return leaked
