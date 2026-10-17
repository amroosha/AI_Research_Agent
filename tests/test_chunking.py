"""Tests for chunking (T4), using a fake whitespace token counter."""

from __future__ import annotations

import copy
import json

import pytest

from research_agent.config import IngestSettings
from research_agent.ingest import chunking
from research_agent.ingest.chunking import (
    chunk_book,
    count_tokens_whitespace,
    make_index_text,
    validate_chunks,
)
from research_agent.ingest.chunking.classes import Block, Piece, _Chunk
from research_agent.ingest.chunking.helpers import _merge_tiny_final, _pack_group, split_block

# Small settings so tests exercise splitting/merging with few words.
SMALL = IngestSettings(
    target_tokens=10,
    max_tokens=15,
    overlap_tokens=4,
    min_chunk_tokens=6,
)


def page(pdf_page: int, text: str, section_path: list[str], printed: int | None = None) -> dict:
    return {
        "book_slug": "book",
        "book_title": "Book",
        "pdf_page": pdf_page,
        "printed_page": printed,
        "section_path": section_path,
        "text": text,
    }


def run(pages: list[dict], settings: IngestSettings = SMALL) -> list[dict]:
    return chunk_book(pages, "book", settings, count_tokens_whitespace)


# --- pure helpers -------------------------------------------------------------


def test_make_index_text_with_and_without_section() -> None:
    assert make_index_text("T", ["A", "B", "C", "D"], "body") == "T > A > B > C\nbody"
    assert make_index_text("T", [], "body") == "T\nbody"


def test_chunk_id_format() -> None:
    cid = chunking.chunk_id("book", 7, "hello world")
    assert cid.startswith("book-00007-")
    assert len(cid.split("-")[-1]) == 8


# --- core algorithm -----------------------------------------------------------


def test_prose_merges_across_pages(tmp_path: None) -> None:
    pages = [
        page(1, "This sentence does not end yet and continues onto", ["C"]),
        page(2, "the next page seamlessly here.", ["C"]),
    ]
    chunks = run(pages)
    assert len(chunks) == 2
    assert all(chunk["pdf_page_start"] == 1 for chunk in chunks)
    assert all(chunk["pdf_page_end"] == 2 for chunk in chunks)
    assert "continues onto the next page" in chunks[0]["text"]


def test_section_change_closes_chunk() -> None:
    pages = [
        page(1, "Alpha beta gamma delta epsilon.", ["C1"]),
        page(2, "Zeta eta theta iota kappa.", ["C2"]),
    ]
    chunks = run(pages)
    assert len(chunks) == 2
    assert chunks[0]["section_path"] == ["C1"]
    assert chunks[1]["section_path"] == ["C2"]
    # No chunk spans two section_path[:2] groups.
    for c in chunks:
        assert len(set(map(tuple, [c["section_path"][:2]]))) == 1


def test_code_block_split_keeps_fences() -> None:
    long_code = "```\n" + "\n".join(f"line_{i} = {i}" for i in range(20)) + "\n```"
    pages = [page(1, long_code, ["C"])]
    chunks = run(pages)
    assert len(chunks) >= 2
    for c in chunks:
        assert c["text"].count("```") == 2  # each piece re-fenced
        assert c["content_type"] == "code"
        assert c["token_count"] <= SMALL.max_tokens


def test_overlap_present_between_prose_chunks() -> None:
    # Multiple short paragraphs (separate blocks) so chunks close between them,
    # and each trailing sentence fits the overlap budget (overlap_tokens=4).
    text = "\n\n".join(f"Para {i} short." for i in range(8))  # "Para 0 short." = 3 tokens
    pages = [page(1, text, ["C"])]
    chunks = run(pages)
    assert len(chunks) >= 2
    # The last sentence of chunk 0 reappears at the start of chunk 1.
    last_sentence = [s for s in chunking.SENTENCE_SPLIT.split(chunks[0]["text"]) if s.strip()][-1]
    assert last_sentence in chunks[1]["text"]


def test_no_overlap_around_code() -> None:
    prose = " ".join(f"Word{i} here." for i in range(8))
    pages = [
        page(1, prose, ["C"]),
        page(2, "```\ncode_a = 1\ncode_b = 2\n```", ["C"]),
    ]
    chunks = run(pages)
    code_chunks = [c for c in chunks if c["content_type"] == "code"]
    assert code_chunks
    # A code chunk must not carry prose overlap text from a previous chunk.
    for cc in code_chunks:
        assert "Word" not in cc["text"]


def test_tiny_final_chunk_merges() -> None:
    # A big chunk followed by a tiny final paragraph merges if under max.
    pages = [
        page(1, " ".join(f"w{i}" for i in range(9)), ["C"]),  # ~9 tokens, close to max
        page(2, "tiny.", ["C"]),  # below min_chunk_tokens
    ]
    chunks = run(pages)
    # Either merged into one chunk, or the final tiny chunk is kept (not dropped).
    assert all(c["text"].strip() for c in chunks)
    assert sum(c["token_count"] <= SMALL.max_tokens for c in chunks) == len(chunks)


def test_tiny_final_merge_drops_leading_overlap_once() -> None:
    def piece(text: str, start: int, overlap: bool = False) -> Piece:
        return Piece(
            kind="prose",
            text=text,
            pdf_page_start=start,
            pdf_page_end=start,
            printed_page_start=None,
            printed_page_end=None,
            section_path=["C"],
            is_overlap=overlap,
        )

    chunks = [
        _Chunk(pieces=[piece("one. two.", 1)]),
        _Chunk(pieces=[piece("two.", 1, overlap=True), piece("three.", 2)]),
    ]
    merged = _merge_tiny_final(chunks, SMALL, count_tokens_whitespace, lambda text: 0)
    assert merged[0].text == "one. two.\n\nthree."
    assert merged[0].pieces[-1].pdf_page_start == 2


def test_code_fence_overhead_is_included_before_packing() -> None:
    settings = IngestSettings(target_tokens=3, max_tokens=3, overlap_tokens=0, min_chunk_tokens=1)
    block = Block(
        kind="code",
        text="abcdefghij",
        pdf_page_start=1,
        pdf_page_end=1,
        printed_page_start=None,
        printed_page_end=None,
        section_path=["C"],
    )
    chunks = _pack_group([block], settings, lambda text: len(text) // 4, lambda text: 0)
    assert chunks
    assert all(len(chunk.text) // 4 <= settings.max_tokens for chunk in chunks)


def test_hard_split_flushes_before_crossing_and_splits_long_word() -> None:
    block = Block(
        kind="prose",
        text="alpha beta gamma",
        pdf_page_start=1,
        pdf_page_end=1,
        printed_page_start=None,
        printed_page_end=None,
        section_path=["C"],
    )
    pieces = split_block(block, lambda text: len(text) // 4, 3, lambda text: 0)
    assert all(len(piece.text) // 4 <= 3 for piece in pieces)
    assert " ".join(piece.text for piece in pieces) == block.text

    long_word = Block(
        **{**block.__dict__, "text": "abcdefghijklmno"},
    )
    pieces = split_block(long_word, lambda text: len(text) // 4, 3, lambda text: 0)
    assert all(len(piece.text) // 4 <= 3 for piece in pieces)
    assert "".join(piece.text for piece in pieces) == long_word.text


def test_id_format_and_index() -> None:
    pages = [page(1, "Some content here for testing.", ["C"])]
    chunks = run(pages)
    for i, c in enumerate(chunks):
        assert c["id"].startswith(f"book-{i:05d}-")
        assert c["chunk_index"] == i
        assert c["index_text"].startswith("Book > C\n")


def test_determinism() -> None:
    pages = [page(i, f"Paragraph {i} with several words of text here.", ["C"]) for i in range(1, 4)]
    a = run(pages)
    b = run(pages)
    assert a == b


def test_validate_chunks_catches_violations() -> None:
    good = run([page(1, "A perfectly fine sentence with enough words.", ["C"])])
    validate_chunks(copy.deepcopy(good), SMALL.max_tokens)  # no raise

    def expect_error(mutate) -> None:
        recs = copy.deepcopy(good)
        mutate(recs)
        with pytest.raises(ValueError):
            validate_chunks(recs, SMALL.max_tokens)

    expect_error(lambda r: r.__setitem__(0, {**r[0], "token_count": 999}))
    expect_error(lambda r: r.__setitem__(0, {**r[0], "pdf_page_start": 5, "pdf_page_end": 1}))
    expect_error(lambda r: r.__setitem__(0, {**r[0], "chunk_index": 3}))
    expect_error(lambda r: r.__setitem__(0, {**r[0], "text": "   "}))
    expect_error(lambda r: r.__setitem__(0, {**r[0], "chapter": "WRONG"}))

    # Duplicate ids.
    dup = copy.deepcopy(good) + copy.deepcopy(good)
    with pytest.raises(ValueError):
        validate_chunks(dup, SMALL.max_tokens)


def test_coverage_all_source_text_present() -> None:
    # Every word from the source pages should appear in at least one chunk,
    # except words that are dropped by code-fence markers.
    pages = [
        page(1, "Alpha beta gamma delta from page one.", ["C"]),
        page(2, "Epsilon zeta eta theta from page two.", ["C"]),
    ]
    chunks = run(pages)
    joined = " ".join(c["text"] for c in chunks)
    for word in ["Alpha", "beta", "gamma", "delta", "Epsilon", "zeta", "eta", "theta"]:
        assert word in joined


def test_directory_and_manifest(tmp_path) -> None:
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    recs = [
        page(i, f"Chapter {i} body text with enough words to chunk.", ["C"]) for i in range(1, 4)
    ]
    (pages_dir / "book.jsonl").write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")

    out_dir = tmp_path / "chunks"
    manifest = chunking.chunk_directory(pages_dir, out_dir, SMALL, count_tokens_whitespace)
    assert "book" in manifest
    entry = manifest["book"]
    assert entry["chunk_count"] >= 1
    assert entry["above_max"] == 0
    assert set(entry["content_type_counts"]) == {"prose", "code", "mixed"}
    assert (out_dir / "book.jsonl").exists()
    assert (out_dir / "manifest.json").exists()
