"""Tests for PDF parsing and cleaning (T3).

Builds synthetic PDFs with PyMuPDF and asserts outline-based dropping, the
no-outline fallback, override precedence, header/footer removal, hyphen joining,
code fencing, section paths, manifest contents, idempotency, and --force.
"""

from __future__ import annotations

import json
from pathlib import Path

import pymupdf

from research_agent.config import IngestSettings
from research_agent.ingest import pdf_parse


# --------------------------------------------------------------------------- #
# Pure-helper tests
# --------------------------------------------------------------------------- #
def test_slugify() -> None:
    assert pdf_parse.slugify("My Book (2nd Ed)!") == "my-book-2nd-ed"
    assert pdf_parse.slugify("AI_Research") == "ai-research"


def test_normalize_header_footer_digits() -> None:
    assert pdf_parse.normalize_header_footer("Chapter 3    Page 42") == "Chapter # Page ##"


def test_parse_printed_page() -> None:
    assert pdf_parse.parse_printed_page("42") == 42
    assert pdf_parse.parse_printed_page("xlii") is None
    assert pdf_parse.parse_printed_page(None) is None


def test_is_mono_font() -> None:
    assert pdf_parse.is_mono_font("Courier")
    assert pdf_parse.is_mono_font("DejaVuSansMono")
    assert pdf_parse.is_mono_font("CodeText")
    assert not pdf_parse.is_mono_font("Helvetica")


def test_join_hyphenated() -> None:
    # Mid-word break: trailing hyphen removed, word reformed.
    assert pdf_parse.join_hyphenated("archi-", "tecture") == "architecture"
    assert pdf_parse.join_hyphenated("well-writ-", "ten paragraph") == "well-written paragraph"
    # No trailing hyphen -> simple space join.
    assert pdf_parse.join_hyphenated("foo bar", "baz") == "foo bar baz"
    # Trailing hyphen but next line uppercase -> not a mid-word break, keep as is.
    assert pdf_parse.join_hyphenated("end-", "Start") == "end- Start"


def test_section_path_for_page() -> None:
    toc = [
        [1, "Chapter 1", 3],
        [2, "Intro", 3],
        [2, "Details", 5],
        [1, "Chapter 2", 7],
    ]
    assert pdf_parse.section_path_for_page(toc, 1) == []
    assert pdf_parse.section_path_for_page(toc, 3) == ["Chapter 1", "Intro"]
    assert pdf_parse.section_path_for_page(toc, 6) == ["Chapter 1", "Details"]
    assert pdf_parse.section_path_for_page(toc, 8) == ["Chapter 2"]


def test_section_path_skips_toc_entries_before_page_one() -> None:
    toc = [[1, "Invalid", -1], [1, "Chapter 1", 2]]
    assert pdf_parse.section_path_for_page(toc, 1) == []
    assert pdf_parse.section_path_for_page(toc, 2) == ["Chapter 1"]


def test_coalesce_drops() -> None:
    reasons = ["a", "a", None, "b", "b", "b", None]
    assert pdf_parse.coalesce_drops(reasons) == [
        {"start": 1, "end": 2, "reason": "a"},
        {"start": 4, "end": 6, "reason": "b"},
    ]


# --------------------------------------------------------------------------- #
# Fixture builders
# --------------------------------------------------------------------------- #
def _add_body_page(doc: pymupdf.Document, header: str, footer: str) -> None:
    """A content page with a repeated header/footer and two prose paragraphs."""
    page = doc.new_page(width=400, height=700)
    page.insert_text((50, 25), header, fontsize=9)
    page.insert_text((50, 120), "The transformer archi-", fontsize=11)
    page.insert_text((50, 135), "tecture relies on attention to weigh tokens.", fontsize=11)
    page.insert_text((50, 220), "A second paragraph adds more detail here.", fontsize=11)
    page.insert_text((50, 690), footer, fontsize=9)


def _preface_page(doc: pymupdf.Document) -> None:
    page = doc.new_page(width=400, height=700)
    page.insert_text((50, 120), "Preface text about the making of this book.", fontsize=11)


def _index_page(doc: pymupdf.Document) -> None:
    page = doc.new_page(width=400, height=700)
    page.insert_text(
        (50, 120),
        "Index entries: attention mechanisms, backpropagation, convolution, and more.",
        fontsize=11,
    )


def _build_outlined_book(path: Path, num_body: int = 3) -> None:
    """Book: preface(2) + 2 chapters of body pages + index(2), with an outline.

    The outline has >= 5 entries so outline-based dropping is used.
    """
    doc = pymupdf.open()
    toc: list[list[int | str]] = [[1, "Preface", 1]]
    _preface_page(doc)
    _preface_page(doc)
    toc.append([1, "Chapter 1", 3])
    toc.append([2, "Attention", 3])
    for i in range(num_body):
        _add_body_page(doc, f"Chapter 1   Page {i + 1}", f"BOOK NAME   {i + 1}")
    toc.append([1, "Chapter 2", 3 + num_body])
    toc.append([2, "Training", 3 + num_body])
    for i in range(num_body):
        _add_body_page(doc, f"Chapter 2   Page {i + 1}", f"BOOK NAME   {i + 1}")
    toc.append([1, "Index", 3 + 2 * num_body])
    _index_page(doc)
    _index_page(doc)
    doc.set_toc(toc)
    doc.save(path)
    doc.close()


def _build_no_outline_book(path: Path, total: int = 90) -> None:
    doc = pymupdf.open()
    for i in range(total):
        page = doc.new_page(width=400, height=700)
        page.insert_text(
            (50, 120),
            f"Body content for page {i + 1} of the chapter with plenty of words here.",
            fontsize=11,
        )
    doc.save(path)
    doc.close()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# --------------------------------------------------------------------------- #
# Integration tests
# --------------------------------------------------------------------------- #
def test_outline_based_dropping(tmp_path: Path) -> None:
    pdf = tmp_path / "My Great Book.pdf"
    out = tmp_path / "pages"
    _build_outlined_book(pdf)
    entry, records = pdf_parse.parse_book(pdf, out, overrides={})

    assert entry["book_slug"] == "my-great-book"
    assert entry["total_pages"] == 10  # 2 preface + 2*3 body + 2 index
    assert entry["kept_pages"] == 6
    reasons = {r["reason"] for r in entry["dropped_ranges"]}
    assert "section:preface" in reasons
    assert "section:index" in reasons
    kept_pages = {r["pdf_page"] for r in records}
    assert kept_pages == {3, 4, 5, 6, 7, 8}
    assert all(r["section_path"][:1] == ["Chapter 1"] for r in records if r["pdf_page"] <= 5)
    assert all(r["section_path"][:1] == ["Chapter 2"] for r in records if r["pdf_page"] >= 6)


def test_header_footer_and_hyphen(tmp_path: Path) -> None:
    pdf = tmp_path / "My Great Book.pdf"
    out = tmp_path / "pages"
    _build_outlined_book(pdf)
    _entry, records = pdf_parse.parse_book(pdf, out, overrides={})
    text = next(r for r in records if r["pdf_page"] == 3)["text"]
    assert "BOOK NAME" not in text  # repeated footer removed
    assert "Chapter 1   Page 1" not in text  # repeated header removed
    assert "architecture" in text  # hyphen healed across the soft wrap
    assert "archi-\n" not in text  # the break was removed
    assert "\n\n" in text  # paragraph break preserved


def test_fallback_when_no_outline(tmp_path: Path) -> None:
    pdf = tmp_path / "No Outline.pdf"
    out = tmp_path / "pages"
    _build_no_outline_book(pdf, total=90)
    settings = IngestSettings(default_skip_front=20, default_skip_back=50)
    entry, _records = pdf_parse.parse_book(pdf, out, overrides={}, settings=settings)
    assert entry["kept_pages"] == 20  # 90 - 20 front - 50 back
    assert {r["reason"] for r in entry["dropped_ranges"]} == {"fallback"}


def test_override_precedence(tmp_path: Path) -> None:
    pdf = tmp_path / "My Great Book.pdf"
    out = tmp_path / "pages"
    _build_outlined_book(pdf)
    overrides = {"my-great-book": {"skip_ranges": [[1, 5]]}}
    entry, records = pdf_parse.parse_book(pdf, out, overrides=overrides)
    kept = {r["pdf_page"] for r in records}
    # Override wins over the outline: only 1-5 dropped, index pages 9-10 kept.
    assert kept == {6, 7, 8, 9, 10}
    assert {r["reason"] for r in entry["dropped_ranges"]} == {"override"}


def test_manifest_and_idempotency(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    pdf = raw / "My Great Book.pdf"
    _build_outlined_book(pdf)
    out = tmp_path / "pages"

    manifest = pdf_parse.parse_directory(raw, out)
    assert "my-great-book" in manifest
    jsonl = out / "my-great-book.jsonl"
    assert jsonl.exists()
    first_mtime = jsonl.stat().st_mtime_ns

    # Second run without --force: unchanged sha256 -> skip (file untouched).
    manifest2 = pdf_parse.parse_directory(raw, out)
    assert jsonl.stat().st_mtime_ns == first_mtime
    assert manifest2["my-great-book"]["kept_pages"] == manifest["my-great-book"]["kept_pages"]

    # --force reparses and rewrites the manifest.
    manifest3 = pdf_parse.parse_directory(raw, out, force=True)
    assert manifest3["my-great-book"]["sha256"] == manifest["my-great-book"]["sha256"]


def test_code_fencing(tmp_path: Path) -> None:
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=700)
    page.insert_text((50, 120), "Here is a listing that demonstrates the concept:", fontsize=11)
    page.insert_text((50, 200), "def foo():", fontname="cour", fontsize=10)
    page.insert_text((60, 214), "return 1", fontname="cour", fontsize=10)
    doc.set_toc(
        [
            [1, "Chapter 1", 1],
            [2, "Code", 1],
            [2, "More", 1],
            [2, "Even More", 1],
            [2, "Last", 1],
        ]
    )
    tmp = tmp_path / "code.pdf"
    doc.save(tmp)
    doc.close()

    _entry, records = pdf_parse.parse_book(tmp, tmp_path / "pages", overrides={})
    text = records[0]["text"]
    assert "```" in text
    assert "def foo():" in text
    assert "return 1" in text


def test_empty_page_dropped(tmp_path: Path) -> None:
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=700)
    page.insert_text((50, 120), "tiny", fontsize=11)  # under min_page_chars -> dropped
    doc.set_toc(
        [
            [1, "Chapter 1", 1],
            [2, "Short", 1],
            [2, "More", 1],
            [2, "Even More", 1],
            [2, "Last", 1],
        ]
    )
    tmp = tmp_path / "tiny.pdf"
    doc.save(tmp)
    doc.close()
    entry, records = pdf_parse.parse_book(tmp, tmp_path / "pages", overrides={})
    assert records == []
    assert entry["kept_pages"] == 0
    assert entry["low_text"] is False  # no kept pages -> not flagged


def test_records_written_as_jsonl(tmp_path: Path) -> None:
    pdf = tmp_path / "My Great Book.pdf"
    out = tmp_path / "pages"
    _build_outlined_book(pdf)
    _entry, _records = pdf_parse.parse_book(pdf, out, overrides={})
    written = _read_jsonl(out / "my-great-book.jsonl")
    assert len(written) == 6
    assert set(written[0]) == {
        "book_slug",
        "book_title",
        "pdf_page",
        "printed_page",
        "section_path",
        "text",
    }
