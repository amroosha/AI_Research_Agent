"""Pure helpers for chunking: text building, block splitting, and packing.

All functions here are deterministic and side-effect free; the injected
``count_tokens`` callable keeps them testable without a real tokenizer.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from typing import Any

from research_agent.config import IngestSettings

from .classes import Block, Piece, _Chunk
from .config import CODE_FENCE, PARA_END, SENTENCE_SPLIT


# --------------------------------------------------------------------------- #
# Text + id helpers
# --------------------------------------------------------------------------- #
def count_tokens_whitespace(text: str) -> int:
    """Cheap fake token counter for tests: split on whitespace."""
    return len(text.split())


def make_index_text(book_title: str, section_path: list[str], text: str) -> str:
    """Build the embedding/BM25 input: a title/section header line + the text."""
    header_parts = [book_title, *section_path[:3]]
    return f"{' > '.join(header_parts)}\n{text}"


def chunk_id(book_slug: str, chunk_index: int, text: str) -> str:
    """Stable chunk id: ``{slug}-{index:05d}-{sha1(text)[:8]}``."""
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return f"{book_slug}-{chunk_index:05d}-{digest}"


# --------------------------------------------------------------------------- #
# Page -> blocks
# --------------------------------------------------------------------------- #
def _split_code_and_prose(page_text: str) -> list[tuple[str, str]]:
    """Split page text into ordered ``(kind, text)`` segments.

    Fenced code blocks (triple-backtick) are atomic ``code`` segments; the rest
    is split on blank lines into ``prose`` segments.
    """
    segments: list[tuple[str, str]] = []
    # Walk fences: alternate between prose and code regions.
    parts = page_text.split(CODE_FENCE)
    for i, part in enumerate(parts):
        in_code = i % 2 == 1
        if in_code:
            body = part.strip("\n")
            if body.strip():
                segments.append(("code", body))
            continue
        for para in re.split(r"\n\s*\n", part):
            if para.strip():
                segments.append(("prose", para.strip()))
    return segments


def page_to_blocks(page: dict[str, Any]) -> list[Block]:
    """Convert one page record into ordered blocks with page metadata."""
    pdf_page = int(page["pdf_page"])
    printed = page.get("printed_page")
    section_path = list(page.get("section_path", []))
    blocks: list[Block] = []
    for kind, text in _split_code_and_prose(page["text"]):
        blocks.append(
            Block(
                kind=kind,
                text=text,
                pdf_page_start=pdf_page,
                pdf_page_end=pdf_page,
                printed_page_start=printed,
                printed_page_end=printed,
                section_path=section_path,
            )
        )
    return blocks


# --------------------------------------------------------------------------- #
# Cross-page merge + grouping
# --------------------------------------------------------------------------- #
def _mergeable(a: Block, b: Block) -> bool:
    """Two prose blocks merge across a page break when ``a`` doesn't end a
    paragraph and ``b`` continues it in the same section starting lowercase.
    """
    if a.kind != "prose" or b.kind != "prose":
        return False
    if a.section_path != b.section_path:
        return False
    if a.text.rstrip().endswith(PARA_END):
        return False
    return bool(b.text.lstrip()[:1].islower())


def merge_blocks(blocks: list[Block]) -> list[Block]:
    """Merge prose blocks across page breaks per the continuation rule."""
    merged: list[Block] = []
    for block in blocks:
        if merged and _mergeable(merged[-1], block):
            prev = merged[-1]
            prev.text = f"{prev.text.rstrip()} {block.text.lstrip()}"
            prev.pdf_page_end = block.pdf_page_end
            prev.printed_page_end = block.printed_page_end
        else:
            merged.append(block)
    return merged


def group_blocks(blocks: list[Block]) -> list[list[Block]]:
    """Group consecutive blocks by ``section_path[:2]`` (a chunk never spans two)."""
    groups: list[list[Block]] = []
    for block in blocks:
        key = tuple(block.section_path[:2])
        if groups and tuple(groups[-1][-1].section_path[:2]) == key:
            groups[-1].append(block)
        else:
            groups.append([block])
    return groups


# --------------------------------------------------------------------------- #
# Oversize splitting
# --------------------------------------------------------------------------- #
def _hard_split(
    text: str,
    count_tokens: Callable[[str], int],
    max_tokens: int,
    header: Callable[[str], int],
    kind: str,
) -> list[str]:
    """Last-resort split of text that is over the limit even by sentence/line."""

    def stored(candidate: str) -> str:
        return f"{CODE_FENCE}\n{candidate}\n{CODE_FENCE}" if kind == "code" else candidate

    def fits(candidate: str) -> bool:
        return count_tokens(stored(candidate)) + header(candidate) <= max_tokens

    def split_long_word(word: str) -> list[str]:
        pieces: list[str] = []
        remaining = word
        while remaining:
            size = len(remaining)
            while size and not fits(remaining[:size]):
                size //= 2
            if size == 0:
                raise ValueError("cannot split text to fit max_tokens")
            pieces.append(remaining[:size])
            remaining = remaining[size:]
        return pieces

    words = text.split()
    pieces: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join([*current, word])
        if current and not fits(candidate):
            pieces.append(" ".join(current))
            current = []
        if not current and not fits(word):
            pieces.extend(split_long_word(word))
        else:
            current.append(word)
    if current:
        pieces.append(" ".join(current))
    return pieces or [text]


def _piece_from(block: Block, text: str) -> Piece:
    kind = block.kind
    body = f"{CODE_FENCE}\n{text}\n{CODE_FENCE}" if kind == "code" else text
    return Piece(
        kind=kind,
        text=body,
        pdf_page_start=block.pdf_page_start,
        pdf_page_end=block.pdf_page_end,
        printed_page_start=block.printed_page_start,
        printed_page_end=block.printed_page_end,
        section_path=list(block.section_path),
    )


def split_block(
    block: Block, count_tokens: Callable[[str], int], max_tokens: int, header: Callable[[str], int]
) -> list[Piece]:
    """Split a single block whose ``index_text`` exceeds ``max_tokens``.

    ``header`` maps a candidate text to the header length used by ``index_text``
    so the limit is checked on the full indexed text, not just the body. Prose
    is split by sentences, code by lines, and anything still over the limit is
    hard-split by tokens. Code pieces are re-wrapped in fences, and the fence
    overhead is included when measuring against the limit.
    """

    def stored(kind: str, text: str) -> str:
        return f"{CODE_FENCE}\n{text}\n{CODE_FENCE}" if kind == "code" else text

    def over(text: str) -> bool:
        return count_tokens(stored(block.kind, text)) + header(text) > max_tokens

    if block.kind == "code":
        lines = block.text.split("\n")
        pieces_text: list[str] = []
        current: list[str] = []
        for line in lines:
            candidate = "\n".join([*current, line])
            if current and over(candidate):
                pieces_text.append("\n".join(current))
                current = [line]
            else:
                current.append(line)
        if current:
            pieces_text.append("\n".join(current))
    else:
        sentences = [s for s in SENTENCE_SPLIT.split(block.text) if s.strip()]
        pieces_text = []
        current = ""
        for sentence in sentences:
            candidate = f"{current} {sentence}".strip()
            if current and over(candidate):
                pieces_text.append(current)
                current = sentence
            else:
                current = candidate
        if current:
            pieces_text.append(current)

    result: list[Piece] = []
    for text in pieces_text:
        if over(text):
            for hard in _hard_split(text, count_tokens, max_tokens, header, block.kind):
                result.append(_piece_from(block, hard))
        else:
            result.append(_piece_from(block, text))
    return result


# --------------------------------------------------------------------------- #
# Packing a section group into chunks
# --------------------------------------------------------------------------- #
def _overlap_pieces(
    prev: _Chunk, count_tokens: Callable[[str], int], overlap_tokens: int
) -> list[Piece]:
    """Trailing whole sentences of the previous prose chunk, up to overlap_tokens.

    None if no whole sentence fits within the budget.
    """
    if prev.content_type != "prose":
        return []
    sentences = [s for s in SENTENCE_SPLIT.split(prev.text) if s.strip()]
    chosen: list[str] = []
    total = 0
    for sentence in reversed(sentences):
        cost = count_tokens(sentence)
        if total + cost > overlap_tokens:
            break
        chosen.append(sentence)
        total += cost
    chosen.reverse()
    if not chosen:
        return []
    last = prev.pieces[-1]
    return [
        Piece(
            kind="prose",
            text=" ".join(chosen),
            pdf_page_start=last.pdf_page_start,
            pdf_page_end=last.pdf_page_end,
            printed_page_start=last.printed_page_start,
            printed_page_end=last.printed_page_end,
            section_path=list(last.section_path),
            is_overlap=True,
        )
    ]


def _overlap_ok(overlap: list[Piece], piece: Piece) -> bool:
    """Overlap only when both sides are prose (never around code)."""
    return bool(overlap) and piece.kind == "prose"


def _merge_tiny_final(
    chunks: list[_Chunk],
    settings: IngestSettings,
    count_tokens: Callable[[str], int],
    header_len: Callable[[str], int],
) -> list[_Chunk]:
    """A final chunk below ``min_chunk_tokens`` merges into the previous chunk if
    the result stays within ``max_tokens``; otherwise it is kept as-is.
    """
    if len(chunks) < 2:
        return chunks
    last = chunks[-1]
    if count_tokens(last.text) + header_len(last.text) >= settings.min_chunk_tokens:
        return chunks
    prev = chunks[-2]
    final_pieces = list(last.pieces)
    while final_pieces and final_pieces[0].is_overlap:
        final_pieces.pop(0)
    if not final_pieces:
        return chunks
    merged = _Chunk(pieces=[*prev.pieces, *final_pieces])
    if count_tokens(merged.text) + header_len(merged.text) <= settings.max_tokens:
        return [*chunks[:-2], merged]
    return chunks


def _pack_group(
    group: list[Block],
    settings: IngestSettings,
    count_tokens: Callable[[str], int],
    header_len: Callable[[str], int],
) -> list[_Chunk]:
    """Pack one section group's blocks into chunks honoring target/max/overlap.

    Rule 4: add blocks greedily while ``token_count(index_text) <= target``; when
    the next block would exceed the target, close the chunk. An empty chunk always
    takes its first block even above the target (so an oversize-but-under-max
    block stands alone). Overlap is prepended to a new prose chunk.
    """
    # Expand each block into pieces, splitting any single block that is oversize.
    pieces: list[Piece] = []
    for block in group:
        stored = _piece_from(block, block.text).text
        if count_tokens(stored) + header_len(block.text) > settings.max_tokens:
            pieces.extend(split_block(block, count_tokens, settings.max_tokens, header_len))
        else:
            pieces.append(_piece_from(block, block.text))

    def over_target(chunk: _Chunk) -> bool:
        return count_tokens(chunk.text) + header_len(chunk.text) > settings.target_tokens

    def over_max(chunk: _Chunk) -> bool:
        return count_tokens(chunk.text) + header_len(chunk.text) > settings.max_tokens

    def start_new(piece: Piece, prev: _Chunk) -> _Chunk:
        overlap = _overlap_pieces(prev, count_tokens, settings.overlap_tokens)
        if _overlap_ok(overlap, piece):
            candidate = _Chunk(pieces=[*overlap, piece])
            if not over_max(candidate):
                return candidate
        return _Chunk(pieces=[piece])

    chunks: list[_Chunk] = []
    current = _Chunk()
    for piece in pieces:
        if not current.pieces:
            current.pieces.append(piece)  # empty chunk always takes the block
            continue
        candidate = _Chunk(pieces=[*current.pieces, piece])
        if not over_target(candidate):
            current.pieces.append(piece)
            continue
        # Adding this piece exceeds the target: close and start a new chunk.
        chunks.append(current)
        current = start_new(piece, current)
    if current.pieces:
        chunks.append(current)

    return _merge_tiny_final(chunks, settings, count_tokens, header_len)
