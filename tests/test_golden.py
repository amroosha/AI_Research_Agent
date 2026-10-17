"""Tests for golden-set loading and validation (Stage 2, T3).

Uses a synthetic index built with the deterministic ``HashEmbedder`` so no
network or model download is required.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from research_agent.config import DEFAULT_INGEST_SETTINGS
from research_agent.golden import (
    find_leakage,
    load_golden,
    normalize_query,
    validate_golden,
)
from research_agent.ingest.indexing import build_index
from tests.helpers import HashEmbedder, make_chunks, write_chunks

BOOK = "synthetic-book"


@pytest.fixture
def index_env(tmp_path: Path) -> tuple[Path, Path]:
    """A built synthetic index: chunks 1..30, one page each."""
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    write_chunks(make_chunks(30), chunks_dir / f"{BOOK}.jsonl")
    index_dir = tmp_path / "index"
    build_index(
        chunks_dir=chunks_dir,
        index_dir=index_dir,
        embedder=HashEmbedder(dim=64),
        settings=DEFAULT_INGEST_SETTINGS,
        show_progress=False,
    )
    return chunks_dir, index_dir


def _write_golden(path: Path, entries: list[dict]) -> Path:
    path.write_text(yaml.safe_dump({"queries": entries}, sort_keys=False), encoding="utf-8")
    return path


def _entry(query: str = "what is gradient descent", **kwargs) -> dict:
    entry: dict = {"query": query}
    entry.update(kwargs)
    return entry


# --- load_golden --------------------------------------------------------------


def test_load_single_range_format(tmp_path: Path) -> None:
    path = _write_golden(
        tmp_path / "set.yaml",
        [_entry(expected_book=BOOK, expected_pdf_pages=[2, 5])],
    )
    (item,) = load_golden(path)
    assert item.id == "set:000"
    assert item.query == "what is gradient descent"
    assert item.book == BOOK
    assert item.ranges == ((2, 5),)
    assert item.is_negative is False
    assert item.source_file == str(path)


def test_load_multi_range_format(tmp_path: Path) -> None:
    path = _write_golden(
        tmp_path / "set.yaml",
        [_entry(expected_book=BOOK, expected_pdf_pages=[[2, 5], [10, 12]])],
    )
    (item,) = load_golden(path)
    assert item.ranges == ((2, 5), (10, 12))


def test_load_negative(tmp_path: Path) -> None:
    path = _write_golden(
        tmp_path / "set.yaml",
        [_entry(query="capital of Australia", expected_book="none", expected_pdf_pages=[])],
    )
    (item,) = load_golden(path)
    assert item.is_negative is True
    assert item.book is None
    assert item.ranges == ()


def test_load_malformed_range_raises(tmp_path: Path) -> None:
    path = _write_golden(
        tmp_path / "set.yaml",
        [_entry(expected_book=BOOK, expected_pdf_pages=[1, 2, 3])],
    )
    with pytest.raises(ValueError, match="malformed page range"):
        load_golden(path)


# --- validate_golden: errors --------------------------------------------------


def _validate(items, index_env) -> tuple[str, ...]:
    chunks_dir, index_dir = index_env
    return validate_golden(items, index_dir, chunks_dir).errors


def test_empty_query_is_error(tmp_path: Path, index_env) -> None:
    path = _write_golden(
        tmp_path / "set.yaml",
        [_entry(query="   ", expected_book=BOOK, expected_pdf_pages=[2, 5])],
    )
    errors = _validate(load_golden(path), index_env)
    assert any("query is empty" in error for error in errors)


def test_empty_expected_book_is_error(tmp_path: Path, index_env) -> None:
    path = _write_golden(
        tmp_path / "set.yaml",
        [_entry(expected_book="", expected_pdf_pages=[2, 5])],
    )
    errors = _validate(load_golden(path), index_env)
    assert any("empty expected_book" in error for error in errors)


def test_negative_with_pages_is_error(tmp_path: Path, index_env) -> None:
    path = _write_golden(
        tmp_path / "set.yaml",
        [_entry(expected_book="none", expected_pdf_pages=[1, 2])],
    )
    errors = _validate(load_golden(path), index_env)
    assert any("negative query has expected_pdf_pages" in error for error in errors)


def test_positive_without_ranges_is_error(tmp_path: Path, index_env) -> None:
    path = _write_golden(tmp_path / "set.yaml", [_entry(expected_book=BOOK, expected_pdf_pages=[])])
    errors = _validate(load_golden(path), index_env)
    assert any("no expected_pdf_pages" in error for error in errors)


@pytest.mark.parametrize(
    ("lo", "hi"), [(0, 5), (5, 4), (-1, 3)], ids=["start-zero", "inverted", "negative-start"]
)
def test_invalid_range_is_error(tmp_path: Path, index_env, lo: int, hi: int) -> None:
    path = _write_golden(
        tmp_path / "set.yaml", [_entry(expected_book=BOOK, expected_pdf_pages=[lo, hi])]
    )
    errors = _validate(load_golden(path), index_env)
    assert any("invalid range" in error for error in errors)


def test_unknown_book_slug_is_error(tmp_path: Path, index_env) -> None:
    path = _write_golden(
        tmp_path / "set.yaml",
        [_entry(expected_book="some-other-book", expected_pdf_pages=[2, 5])],
    )
    errors = _validate(load_golden(path), index_env)
    assert any("not in index manifest" in error for error in errors)


def test_range_end_beyond_book_is_error(tmp_path: Path, index_env) -> None:
    # Synthetic chunks cover pdf pages 1..30.
    path = _write_golden(
        tmp_path / "set.yaml", [_entry(expected_book=BOOK, expected_pdf_pages=[28, 35])]
    )
    errors = _validate(load_golden(path), index_env)
    assert any("beyond" in error and "max pdf_page_end" in error for error in errors)


def test_duplicate_queries_are_error(tmp_path: Path, index_env) -> None:
    path = _write_golden(
        tmp_path / "set.yaml",
        [
            _entry(
                query="What Is Gradient Descent",
                expected_book=BOOK,
                expected_pdf_pages=[2, 5],
            ),
            _entry(
                query="  what   is gradient descent ",
                expected_book=BOOK,
                expected_pdf_pages=[6, 9],
            ),
        ],
    )
    errors = _validate(load_golden(path), index_env)
    assert any("duplicate query" in error for error in errors)


# --- validate_golden: warning + valid file ------------------------------------


def test_wide_range_is_warning_not_error(tmp_path: Path, index_env) -> None:
    chunks_dir, index_dir = index_env
    path = _write_golden(
        tmp_path / "set.yaml", [_entry(expected_book=BOOK, expected_pdf_pages=[10, 25])]
    )
    result = validate_golden(load_golden(path), index_dir, chunks_dir)
    assert result.ok
    assert any("wider than 10 pages" in warning for warning in result.warnings)


def test_valid_file_passes(tmp_path: Path, index_env) -> None:
    chunks_dir, index_dir = index_env
    path = _write_golden(
        tmp_path / "set.yaml",
        [
            _entry(query="positive single", expected_book=BOOK, expected_pdf_pages=[2, 5]),
            _entry(
                query="positive multi", expected_book=BOOK, expected_pdf_pages=[[6, 8], [10, 12]]
            ),
            _entry(query="negative one", expected_book="none", expected_pdf_pages=[]),
        ],
    )
    result = validate_golden(load_golden(path), index_dir, chunks_dir)
    assert result.errors == ()
    assert result.warnings == ()


# --- leakage ------------------------------------------------------------------


def test_find_leakage_detects_shared_query(tmp_path: Path) -> None:
    dev = _write_golden(
        tmp_path / "dev.yaml",
        [_entry(query="What is ETL", expected_book=BOOK, expected_pdf_pages=[2, 5])],
    )
    test = _write_golden(
        tmp_path / "test_set.yaml",
        [
            _entry(query="what  is etl", expected_book=BOOK, expected_pdf_pages=[2, 5]),
            _entry(query="distinct query", expected_book=BOOK, expected_pdf_pages=[6, 8]),
        ],
    )
    leaked = find_leakage(load_golden(dev), load_golden(test))
    assert len(leaked) == 1
    assert "test_set:000" in leaked[0] and "dev:000" in leaked[0]


def test_find_leakage_empty_when_disjoint(tmp_path: Path) -> None:
    dev = _write_golden(tmp_path / "dev.yaml", [_entry(query="alpha")])
    test = _write_golden(tmp_path / "test_set.yaml", [_entry(query="beta")])
    assert find_leakage(load_golden(dev), load_golden(test)) == []


def test_normalize_query_case_and_whitespace() -> None:
    assert normalize_query("  What   Is ETL? ") == "what is etl?"
    assert normalize_query("what is etl?") == normalize_query("WHAT IS ETL?")
