"""Top-level chunking entry points (Stage 1, T4).

``chunk_book`` turns parsed pages into frozen chunk records; ``chunk_directory``
drives a whole ``data/pages`` directory into ``data/chunks`` with validation and
a manifest. Heavy lifting lives in :mod:`.helpers`.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from research_agent.config import DEFAULT_INGEST_SETTINGS, IngestSettings

from .classes import Block, Piece
from .helpers import (
    _pack_group,
    chunk_id,
    count_tokens_whitespace,
    group_blocks,
    make_index_text,
    merge_blocks,
    page_to_blocks,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Book-level chunking
# --------------------------------------------------------------------------- #
def _page_range(pieces: list[Piece]) -> tuple[int | None, int | None, int | None, int | None]:
    pdf_starts = [p.pdf_page_start for p in pieces]
    pdf_ends = [p.pdf_page_end for p in pieces]
    printed_starts = [p.printed_page_start for p in pieces if p.printed_page_start is not None]
    printed_ends = [p.printed_page_end for p in pieces if p.printed_page_end is not None]
    return (
        min(pdf_starts),
        max(pdf_ends),
        min(printed_starts) if printed_starts else None,
        max(printed_ends) if printed_ends else None,
    )


def chunk_book(
    pages: list[dict[str, Any]],
    book_slug: str,
    settings: IngestSettings,
    count_tokens: Callable[[str], int],
) -> list[dict[str, Any]]:
    """Turn parsed page records into frozen chunk records for one book.

    Blocks are merged across page breaks, grouped by ``section_path[:2]``, packed
    into chunks, then assigned ids and metadata. Output is deterministic.
    """
    blocks: list[Block] = []
    for page in sorted(pages, key=lambda p: int(p["pdf_page"])):
        blocks.extend(page_to_blocks(page))
    blocks = merge_blocks(blocks)

    book_title = pages[0]["book_title"] if pages else book_slug
    records: list[dict[str, Any]] = []

    def make_header(path: list[str]) -> Callable[[str], int]:
        def header(text: str) -> int:
            return count_tokens(make_index_text(book_title, path, text)) - count_tokens(text)

        return header

    for group in group_blocks(blocks):
        section_path = group[0].section_path
        header = make_header(section_path)
        for chunk in _pack_group(group, settings, count_tokens, header):
            text = chunk.text
            index_text = make_index_text(book_title, section_path, text)
            pdf_start, pdf_end, pr_start, pr_end = _page_range(chunk.pieces)
            records.append(
                {
                    "book_slug": book_slug,
                    "book_title": book_title,
                    "section_path": section_path,
                    "chapter": section_path[0] if section_path else "",
                    "pdf_page_start": pdf_start,
                    "pdf_page_end": pdf_end,
                    "printed_page_start": pr_start,
                    "printed_page_end": pr_end,
                    "content_type": chunk.content_type,
                    "text": text,
                    "index_text": index_text,
                    "token_count": count_tokens(index_text),
                }
            )

    # Assign ids and chunk_index in order (id depends on text, not index).
    for idx, record in enumerate(records):
        record["chunk_index"] = idx
        record["id"] = chunk_id(book_slug, idx, record["text"])
    return records


# --------------------------------------------------------------------------- #
# Validation + manifest + directory driver
# --------------------------------------------------------------------------- #
def validate_chunks(records: list[dict[str, Any]], max_tokens: int) -> None:
    """Raise ``ValueError`` on any frozen-contract violation in ``records``.

    Checks: duplicate ids, ``token_count > max_tokens``, inverted page range,
    non-consecutive ``chunk_index``, empty text, and a chunk whose ``chapter``
    disagrees with its ``section_path[0]`` (the observable symptom of a chunk
    spanning two ``section_path[:2]`` groups).
    """
    seen_ids: set[str] = set()
    for expected_index, record in enumerate(records):
        cid = record["id"]
        if cid in seen_ids:
            raise ValueError(f"duplicate chunk id: {cid}")
        seen_ids.add(cid)

        if record["token_count"] > max_tokens:
            raise ValueError(f"chunk {cid} exceeds max_tokens: {record['token_count']}")
        if record["pdf_page_start"] > record["pdf_page_end"]:
            raise ValueError(f"chunk {cid} has inverted page range")
        if record["chunk_index"] != expected_index:
            raise ValueError(
                f"non-consecutive chunk_index: expected {expected_index}, "
                f"got {record['chunk_index']}"
            )
        if not record["text"].strip():
            raise ValueError(f"chunk {cid} has empty text")

        section_path = record.get("section_path", [])
        expected_chapter = section_path[0] if section_path else ""
        if record.get("chapter", "") != expected_chapter:
            raise ValueError(
                f"chunk {cid} chapter {record.get('chapter')!r} does not match "
                f"section_path[0] {expected_chapter!r}"
            )


def build_manifest_entry(records: list[dict[str, Any]], max_tokens: int) -> dict[str, Any]:
    """Per-book chunk manifest: counts and token stats."""
    counts = [r["token_count"] for r in records]
    content_counts = {"prose": 0, "code": 0, "mixed": 0}
    for record in records:
        content_counts[record["content_type"]] = content_counts.get(record["content_type"], 0) + 1

    def percentile(values: list[int], pct: float) -> int:
        if not values:
            return 0
        ordered = sorted(values)
        k = max(0, min(len(ordered) - 1, round(pct * (len(ordered) - 1))))
        return ordered[k]

    return {
        "chunk_count": len(records),
        "token_stats": {
            "min": min(counts) if counts else 0,
            "p50": percentile(counts, 0.50),
            "p95": percentile(counts, 0.95),
            "max": max(counts) if counts else 0,
        },
        "above_max": sum(1 for c in counts if c > max_tokens),
        "content_type_counts": content_counts,
    }


def load_pages(pages_dir: Path, slug: str) -> list[dict[str, Any]]:
    """Load one book's page JSONL."""
    path = Path(pages_dir) / f"{slug}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def chunk_directory(
    pages_dir: Path,
    output_dir: Path,
    settings: IngestSettings = DEFAULT_INGEST_SETTINGS,
    count_tokens: Callable[[str], int] = count_tokens_whitespace,
    only: str | None = None,
) -> dict[str, Any]:
    """Chunk every ``{slug}.jsonl`` in ``pages_dir`` into ``output_dir``.

    Always rebuilds. Validates each book's chunks after writing and returns the
    merged manifest mapping.
    """
    pages_dir = Path(pages_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"

    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8") or "{}")

    for pages_file in sorted(pages_dir.glob("*.jsonl")):
        slug = pages_file.stem
        if only and slug != only:
            continue
        pages = load_pages(pages_dir, slug)
        if not pages:
            logger.warning("no pages for %s; skipping", slug)
            continue
        records = chunk_book(pages, slug, settings, count_tokens)
        validate_chunks(records, settings.max_tokens)

        out_file = output_dir / f"{slug}.jsonl"
        with out_file.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        manifest[slug] = build_manifest_entry(records, settings.max_tokens)
        logger.info("chunked %s: %d chunks", slug, len(records))

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
