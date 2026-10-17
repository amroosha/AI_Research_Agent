"""Normalization helpers for inclusive PDF page ranges."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def normalize_ranges(raw: Any, item_id: str) -> tuple[tuple[int, int], ...]:
    """Normalize single- and multi-range page formats."""
    if not raw:
        return ()
    if not isinstance(raw, (list, tuple)):
        raise TypeError(f"{item_id}: expected_pdf_pages must be a list, got {raw!r}")
    pairs: list[tuple[int, int]] = []
    if isinstance(raw[0], (list, tuple)):
        elements: Sequence[Any] = raw
    else:
        elements = [raw]
    for element in elements:
        if not isinstance(element, (list, tuple)) or len(element) != 2:
            raise ValueError(f"{item_id}: malformed page range {element!r}")
        pairs.append((int(element[0]), int(element[1])))
    return tuple(pairs)
