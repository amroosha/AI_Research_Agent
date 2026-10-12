"""Extraction data classes for PDF parsing (Stage 1, T3)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Line:
    """A single rendered text line with geometry and a code/prose flag."""

    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    size: float
    is_code: bool


@dataclass
class Block:
    """A PyMuPDF text block tagged for header/footer handling."""

    lines: list[Line]
    y0: float
    y1: float
    is_zone: bool
    norm_text: str | None


@dataclass
class PageContent:
    """Raw per-page extraction result (blocks + geometry)."""

    blocks: list[Block]
    page_height: float
