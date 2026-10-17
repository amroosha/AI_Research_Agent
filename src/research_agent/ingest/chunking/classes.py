"""Core data classes for chunking (Stage 1, T4)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Block:
    """A unit of page text that stays together: a fenced code listing or a
    prose paragraph. Carries page numbers and the section path it came from.
    """

    kind: str  # "prose" | "code"
    text: str
    pdf_page_start: int
    pdf_page_end: int
    printed_page_start: int | None
    printed_page_end: int | None
    section_path: list[str]


@dataclass
class Piece:
    """A slice of a block placed into a chunk, carrying its own page numbers."""

    kind: str
    text: str
    pdf_page_start: int
    pdf_page_end: int
    printed_page_start: int | None
    printed_page_end: int | None
    section_path: list[str]
    is_overlap: bool = False


@dataclass
class _Chunk:
    """A chunk under construction: an ordered list of pieces."""

    pieces: list[Piece] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(piece.text for piece in self.pieces)

    @property
    def content_type(self) -> str:
        kinds = {piece.kind for piece in self.pieces}
        if kinds == {"code"}:
            return "code"
        if kinds == {"prose"}:
            return "prose"
        return "mixed"
