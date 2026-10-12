"""Paths and constants for PDF parsing (Stage 1, T3).

Shared settings (zones, thresholds, drop titles) live in the top-level
``research_agent.config.IngestSettings``; this module only locates the
user-editable ``book_overrides.json`` (kept at the ``ingest/`` level).
"""

from __future__ import annotations

from pathlib import Path

# book_overrides.json lives in the ingest/ package directory (one level up).
OVERRIDES_PATH = Path(__file__).resolve().parent.parent / "book_overrides.json"
