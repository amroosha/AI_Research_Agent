"""Pure helpers for PDF parsing: text normalization, extraction, and assembly.

Side-effect free apart from PyMuPDF page reads in :func:`extract_page`; the
I/O orchestration lives in :mod:`.main`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from statistics import median
from typing import Any

from research_agent.config import IngestSettings

from .classes import Block, Line, PageContent
from .config import OVERRIDES_PATH

logger = logging.getLogger(__name__)

CODE_FENCE = "```"


# --------------------------------------------------------------------------- #
# Small pure helpers
# --------------------------------------------------------------------------- #
def slugify(name: str) -> str:
    """Lowercase a filename stem and collapse non-alphanumerics into ``-``."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def normalize_header_footer(text: str) -> str:
    """Normalize a candidate header/footer for cross-page comparison.

    Digits become ``#`` (so changing page numbers match) and whitespace is
    collapsed to single spaces.
    """
    without_digits = re.sub(r"\d", "#", text)
    return re.sub(r"\s+", " ", without_digits).strip()


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_printed_page(label: str | None) -> int | None:
    """Return the integer printed page if the label is all digits, else None."""
    if label is None:
        return None
    label = label.strip()
    return int(label) if label.isdigit() else None


def is_mono_font(font_name: str) -> bool:
    """True when a font name looks monospaced (mono/courier/code)."""
    lowered = font_name.lower()
    return "mono" in lowered or "courier" in lowered or "code" in lowered


def join_hyphenated(left: str, right: str) -> str:
    """Join two soft-wrapped prose lines, healing a mid-word line break.

    A trailing hyphen at a line break is treated as a break artifact and
    removed to reform the word: ``"archi-"`` + ``"tecture"`` -> ``"architecture"``
    and ``"well-writ-"`` + ``"ten"`` -> ``"well-written"``. The break only heals
    when the line ends with a hyphen and the next line starts lowercase (a
    genuine mid-word split); otherwise a space is inserted.
    """
    left = left.rstrip()
    right = right.lstrip()
    if left.endswith("-") and right[:1].islower():
        return left[:-1] + right
    return f"{left} {right}".strip()


def section_path_for_page(toc: list[list[Any]], page: int) -> list[str]:
    """Full level-1..N title path of the deepest TOC entry starting at/before ``page``.

    ``toc`` entries are ``[level, title, start_page]`` (1-based) in document
    order, as returned by ``Document.get_toc(simple=True)``.
    """
    stack: list[tuple[int, str]] = []
    for entry in toc:
        level, title, start = int(entry[0]), str(entry[1]), int(entry[2])
        if start < 1:
            continue
        if start > page:
            break
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
    return [title for _level, title in stack]


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def _line_from_dict(line: dict[str, Any]) -> Line:
    spans = line.get("spans", [])
    text = "".join(span.get("text", "") for span in spans)
    visible = [span for span in spans if span.get("text", "").strip()]
    # A line is code only if it has visible text and every visible span is mono.
    is_code = bool(visible) and all(is_mono_font(span.get("font", "")) for span in visible)
    size = max((span.get("size", 0.0) for span in spans), default=0.0)
    x0, y0, x1, y1 = line["bbox"]
    return Line(text=text, x0=x0, y0=y0, x1=x1, y1=y1, size=size, is_code=is_code)


def extract_page(page: Any, zone: float) -> PageContent:
    """Extract text blocks (reading order) and tag header/footer zone blocks."""
    page_height = max(page.rect.height, 1.0)
    zone_top = zone * page_height
    zone_bottom = (1.0 - zone) * page_height

    data = page.get_text("dict", sort=True)
    blocks: list[Block] = []
    for raw in data.get("blocks", []):
        if raw.get("type") != 0:  # skip images
            continue
        lines = [_line_from_dict(line) for line in raw.get("lines", [])]
        lines = [line for line in lines if line.text.strip()]
        if not lines:
            continue
        by0, by1 = raw["bbox"][1], raw["bbox"][3]
        is_zone = by0 < zone_top or by1 > zone_bottom
        norm = normalize_header_footer(" ".join(line.text for line in lines)) if is_zone else None
        blocks.append(Block(lines=lines, y0=by0, y1=by1, is_zone=is_zone, norm_text=norm))
    return PageContent(blocks=blocks, page_height=page_height)


def compute_repeated(pages: list[PageContent], total_pages: int, repeat_ratio: float) -> set[str]:
    """Return normalized header/footer texts appearing on >= repeat_ratio of pages."""
    if total_pages <= 0:
        return set()
    page_counts: dict[str, int] = {}
    for content in pages:
        seen = {b.norm_text for b in content.blocks if b.is_zone and b.norm_text}
        for norm in seen:
            page_counts[norm] = page_counts.get(norm, 0) + 1
    threshold = repeat_ratio * total_pages
    return {norm for norm, count in page_counts.items() if count >= threshold}


# --------------------------------------------------------------------------- #
# Text assembly
# --------------------------------------------------------------------------- #
def _is_para_break(prev: Line, cur: Line) -> bool:
    """Heuristic paragraph break: vertical gap exceeds half the previous line height."""
    line_height = max(prev.y1 - prev.y0, 1.0)
    gap = cur.y0 - prev.y1
    return gap > 0.5 * line_height


def _render_prose(lines: list[Line]) -> str:
    out = ""
    for index, line in enumerate(lines):
        text = line.text.strip()
        out = text if index == 0 else join_hyphenated(out, text)
    return out.strip()


def _render_code(lines: list[Line]) -> str:
    min_x = min(line.x0 for line in lines)
    sizes = [line.size for line in lines if line.size > 0]
    size = median(sizes) if sizes else 10.0
    char_width = 0.6 * size if size > 0 else 6.0
    rendered: list[str] = []
    for line in lines:
        indent = max(round((line.x0 - min_x) / char_width), 0) if char_width else 0
        rendered.append((" " * indent) + line.text.strip())
    return f"{CODE_FENCE}\n" + "\n".join(rendered) + f"\n{CODE_FENCE}"


def assemble_text(lines: list[Line]) -> str:
    """Assemble cleaned page text from ordered lines.

    Consecutive prose lines are merged into paragraphs (blank line between
    paragraphs); consecutive code lines are fenced as one block. Paragraphs and
    code blocks are separated by a blank line.
    """
    segments: list[list[Any]] = []  # each: [kind, list[Line]]
    for line in lines:
        kind = "code" if line.is_code else "prose"
        if not segments:
            segments.append([kind, [line]])
            continue
        last_kind, last_lines = segments[-1]
        starts_new = kind != last_kind or (kind == "prose" and _is_para_break(last_lines[-1], line))
        if starts_new:
            segments.append([kind, [line]])
        else:
            last_lines.append(line)

    parts: list[str] = []
    for kind, seg_lines in segments:
        part = _render_code(seg_lines) if kind == "code" else _render_prose(seg_lines)
        if part:
            parts.append(part)
    return "\n\n".join(parts)


def assemble_page_text(content: PageContent, repeated: set[str]) -> str:
    """Drop repeated header/footer blocks, then assemble the surviving lines."""
    kept_lines: list[Line] = []
    for block in content.blocks:
        if block.is_zone and block.norm_text in repeated:
            continue
        kept_lines.extend(block.lines)
    return assemble_text(kept_lines)


# --------------------------------------------------------------------------- #
# Structural page selection
# --------------------------------------------------------------------------- #
def compute_structural_drops(
    total: int,
    section_paths: list[list[str]],
    override: dict[str, Any] | None,
    has_outline: bool,
    settings: IngestSettings,
) -> list[str | None]:
    """Return a per-page drop reason (or None to keep) from outline/override/fallback.

    Precedence: book override wins over outline and fallback; else outline-based
    title dropping when the outline has >= 5 entries; else front/back fallback.
    """
    reasons: list[str | None] = [None] * total

    skip_ranges = (override or {}).get("skip_ranges")
    if skip_ranges:
        for start, end in skip_ranges:
            for page in range(int(start), int(end) + 1):
                if 1 <= page <= total:
                    reasons[page - 1] = "override"
        return reasons

    if has_outline:
        top_titles = [path[0].strip().lower() if path else None for path in section_paths]
        first_kept = next(
            (
                i
                for i, title in enumerate(top_titles)
                if title and title not in settings.drop_section_titles
            ),
            None,
        )
        for i, title in enumerate(top_titles):
            if title is None:
                reasons[i] = "before_first_section"
            elif title in settings.drop_section_titles:
                reasons[i] = f"section:{title}"
            elif first_kept is not None and i < first_kept:
                reasons[i] = "before_first_section"
        return reasons

    logger.warning(
        "outline missing or <5 entries; using fallback skip front=%d back=%d",
        settings.default_skip_front,
        settings.default_skip_back,
    )
    for i in range(total):
        if i < settings.default_skip_front or i >= total - settings.default_skip_back:
            reasons[i] = "fallback"
    return reasons


def coalesce_drops(reasons: list[str | None]) -> list[dict[str, Any]]:
    """Coalesce consecutive equal non-None reasons into ``{start, end, reason}`` ranges."""
    ranges: list[dict[str, Any]] = []
    i = 0
    n = len(reasons)
    while i < n:
        reason = reasons[i]
        if reason is None:
            i += 1
            continue
        j = i
        while j + 1 < n and reasons[j + 1] == reason:
            j += 1
        ranges.append({"start": i + 1, "end": j + 1, "reason": reason})
        i = j + 1
    return ranges


# --------------------------------------------------------------------------- #
# Overrides
# --------------------------------------------------------------------------- #
def load_overrides(path: Path | None = None) -> dict[str, Any]:
    """Load ``book_overrides.json`` (missing/empty file -> ``{}``)."""
    target = path or OVERRIDES_PATH
    if not target.exists():
        return {}
    text = target.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    return json.loads(text)


def page_label(doc: Any, index: int) -> str | None:
    """Return a page's label string, or None when the document defines no labels."""
    if not doc.get_page_labels():
        return None
    return doc[index].get_label()
