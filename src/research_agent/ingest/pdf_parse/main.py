"""Top-level PDF parsing entry points (Stage 1, T3).

``parse_book`` turns one PDF into page records + a manifest entry;
``parse_directory`` drives a whole ``data/raw`` directory with idempotency.
Pure/extraction helpers live in :mod:`.helpers`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pymupdf

from research_agent.config import DEFAULT_INGEST_SETTINGS, IngestSettings

from .classes import PageContent
from .helpers import (
    assemble_page_text,
    coalesce_drops,
    compute_repeated,
    compute_structural_drops,
    extract_page,
    load_overrides,
    page_label,
    parse_printed_page,
    section_path_for_page,
    sha256_file,
    slugify,
)

logger = logging.getLogger(__name__)


def parse_book(
    pdf_path: Path,
    output_dir: Path,
    overrides: dict[str, Any],
    settings: IngestSettings = DEFAULT_INGEST_SETTINGS,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Parse one PDF into page records and a manifest entry.

    Writes ``output_dir/{slug}.jsonl`` and returns ``(manifest_entry, records)``.
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = slugify(pdf_path.stem)
    sha = sha256_file(pdf_path)

    doc = pymupdf.open(pdf_path)
    total = doc.page_count
    book_title = (doc.metadata.get("title") or "").strip() or pdf_path.stem
    toc = doc.get_toc(simple=True)
    has_outline = len(toc) >= 5

    pages: list[PageContent] = [
        extract_page(doc[i], settings.header_footer_zone) for i in range(total)
    ]
    repeated = compute_repeated(pages, total, settings.repeat_ratio)
    section_paths = [section_path_for_page(toc, p) for p in range(1, total + 1)]
    printed_pages = [parse_printed_page(page_label(doc, i)) for i in range(total)]

    override = overrides.get(slug)
    structural = compute_structural_drops(total, section_paths, override, has_outline, settings)

    records: list[dict[str, Any]] = []
    reasons = list(structural)
    kept_chars = 0
    for i in range(total):
        if structural[i] is not None:
            continue
        text = assemble_page_text(pages[i], repeated)
        if len(text) < settings.min_page_chars:
            reasons[i] = "empty"
            continue
        kept_chars += len(text)
        records.append(
            {
                "book_slug": slug,
                "book_title": book_title,
                "pdf_page": i + 1,
                "printed_page": printed_pages[i],
                "section_path": section_paths[i],
                "text": text,
            }
        )

    out_jsonl = output_dir / f"{slug}.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    kept_pages = len(records)
    low_text = kept_pages > 0 and (kept_chars / kept_pages) < settings.low_text_avg_chars
    manifest_entry = {
        "book_slug": slug,
        "book_title": book_title,
        "sha256": sha,
        "total_pages": total,
        "kept_pages": kept_pages,
        "dropped_ranges": coalesce_drops(reasons),
        "low_text": low_text,
    }
    doc.close()
    logger.info("parsed %s: %d/%d pages kept (low_text=%s)", slug, kept_pages, total, low_text)
    return manifest_entry, records


def parse_directory(
    input_dir: Path,
    output_dir: Path,
    only: str | None = None,
    force: bool = False,
    settings: IngestSettings = DEFAULT_INGEST_SETTINGS,
    overrides_path: Path | None = None,
) -> dict[str, Any]:
    """Parse every PDF in ``input_dir`` into ``output_dir`` with idempotency.

    A book is skipped when its PDF sha256 is unchanged and its JSONL exists,
    unless ``force`` is set. Returns the merged manifest mapping.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    overrides = load_overrides(overrides_path)
    manifest_path = output_dir / "manifest.json"

    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8") or "{}")

    for pdf in sorted(input_dir.glob("*.pdf")):
        slug = slugify(pdf.stem)
        if only and slug != only:
            continue
        out_jsonl = output_dir / f"{slug}.jsonl"
        sha = sha256_file(pdf)
        if not force and out_jsonl.exists() and manifest.get(slug, {}).get("sha256") == sha:
            logger.info("skip %s (unchanged sha256)", slug)
            continue
        entry, _records = parse_book(pdf, output_dir, overrides, settings)
        manifest[slug] = entry

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
