"""Chunking package (Stage 1, T4).

Re-exports the public chunking API so ``research_agent.ingest.chunking`` keeps
working as before the split into ``config``/``classes``/``helpers``/``main``.
"""

from __future__ import annotations

from .classes import Block, Piece, _Chunk
from .config import CODE_FENCE, PARA_END, SENTENCE_SPLIT
from .helpers import (
    _pack_group,
    chunk_id,
    count_tokens_whitespace,
    group_blocks,
    make_index_text,
    merge_blocks,
    page_to_blocks,
    split_block,
)
from .main import (
    build_manifest_entry,
    chunk_book,
    chunk_directory,
    load_pages,
    validate_chunks,
)

__all__ = [
    "CODE_FENCE",
    "PARA_END",
    "SENTENCE_SPLIT",
    "Block",
    "Piece",
    "_Chunk",
    "_pack_group",
    "build_manifest_entry",
    "chunk_book",
    "chunk_directory",
    "chunk_id",
    "count_tokens_whitespace",
    "group_blocks",
    "load_pages",
    "make_index_text",
    "merge_blocks",
    "page_to_blocks",
    "split_block",
    "validate_chunks",
]
