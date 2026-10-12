"""Frozen configuration for routing (and, in later stages, ingestion).

``Settings`` holds routing thresholds and rewrite ordering. ``IngestSettings``
is added in T3. Config is pure data: no I/O, no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Routing configuration.

    ``rewrite_order`` values must be valid strategy names. Thresholds follow
    the frozen grader bands: relevant >= ``relevant_threshold``, ambiguous
    between the two thresholds, irrelevant below ``ambiguous_threshold``.
    """

    relevant_threshold: int = 75
    ambiguous_threshold: int = 50
    max_retries: int = 3
    rewrite_order: tuple[str, ...] = ("step_back", "specific", "fusion")


DEFAULT_SETTINGS = Settings()


@dataclass(frozen=True)
class IngestSettings:
    """PDF parsing, cleaning, and chunking configuration (Stage 1 ingest).

    Thresholds follow the frozen plan: header/footer zones are the top/bottom
    ``header_footer_zone`` fraction of page height; a zone block is dropped when
    its digit-normalized text repeats on ``repeat_ratio`` of pages.
    """

    header_footer_zone: float = 0.06
    repeat_ratio: float = 0.30
    min_page_chars: int = 50
    low_text_avg_chars: int = 500
    default_skip_front: int = 20
    default_skip_back: int = 50
    drop_section_titles: tuple[str, ...] = (
        "preface",
        "foreword",
        "copyright",
        "contents",
        "table of contents",
        "index",
        "colophon",
        "about the author",
        "about the authors",
        "acknowledgments",
        "acknowledgements",
    )
    # Chunking (used in T4)
    target_tokens: int = 350
    max_tokens: int = 450
    overlap_tokens: int = 50
    min_chunk_tokens: int = 80
    embed_model: str = "BAAI/bge-base-en-v1.5"


DEFAULT_INGEST_SETTINGS = IngestSettings()
