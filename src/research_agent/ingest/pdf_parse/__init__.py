"""PDF parsing package (Stage 1, T3).

Re-exports the public parsing API so ``research_agent.ingest.pdf_parse`` keeps
working as before the split into ``config``/``classes``/``helpers``/``main``.
"""

from __future__ import annotations

from .classes import Block, Line, PageContent
from .config import OVERRIDES_PATH
from .helpers import (
    assemble_page_text,
    assemble_text,
    coalesce_drops,
    compute_repeated,
    compute_structural_drops,
    extract_page,
    is_mono_font,
    join_hyphenated,
    load_overrides,
    normalize_header_footer,
    page_label,
    parse_printed_page,
    section_path_for_page,
    sha256_file,
    slugify,
)
from .main import parse_book, parse_directory

__all__ = [
    "OVERRIDES_PATH",
    "Block",
    "Line",
    "PageContent",
    "assemble_page_text",
    "assemble_text",
    "coalesce_drops",
    "compute_repeated",
    "compute_structural_drops",
    "extract_page",
    "is_mono_font",
    "join_hyphenated",
    "load_overrides",
    "normalize_header_footer",
    "page_label",
    "parse_book",
    "parse_directory",
    "parse_printed_page",
    "section_path_for_page",
    "sha256_file",
    "slugify",
]
