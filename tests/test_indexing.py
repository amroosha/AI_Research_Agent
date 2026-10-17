"""Tests for indexing and verification (T5).

Uses the deterministic ``HashEmbedder`` and temp dirs so no network or model
download is required. The one ``slow`` test exercises the real ``BgeEmbedder``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import yaml

from research_agent.config import DEFAULT_INGEST_SETTINGS
from research_agent.ingest import indexing
from research_agent.ingest.indexing import (
    COLLECTION_NAME,
    IndexStateError,
    build_bm25,
    build_index,
    chunk_metadata,
    compute_corpus_version,
    load_bm25,
    tokenize,
    verify_index,
)
from tests.helpers import HashEmbedder, make_chunk, make_chunks, write_chunks

SETTINGS = DEFAULT_INGEST_SETTINGS


@pytest.fixture
def embedder() -> HashEmbedder:
    return HashEmbedder(dim=64)


@pytest.fixture
def chunks_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "chunks"
    directory.mkdir()
    write_chunks(make_chunks(30), directory / "synthetic-book.jsonl")
    return directory


def _build(chunks_dir: Path, index_dir: Path, embedder: HashEmbedder, **kwargs):
    return build_index(
        chunks_dir=chunks_dir,
        index_dir=index_dir,
        embedder=embedder,
        settings=SETTINGS,
        show_progress=False,
        **kwargs,
    )


def _chunk_ids(chunks_dir: Path) -> set[str]:
    ids: set[str] = set()
    for path in chunks_dir.glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                ids.add(json.loads(line)["id"])
    return ids


# --- counts + id parity ------------------------------------------------------


def test_counts_equal_across_stores(
    chunks_dir: Path, tmp_path: Path, embedder: HashEmbedder
) -> None:
    import chromadb

    index_dir = tmp_path / "index"
    manifest = _build(chunks_dir, index_dir, embedder)
    client = chromadb.PersistentClient(path=str(index_dir / "chroma"))
    collection = client.get_collection(COLLECTION_NAME)
    bm25 = load_bm25(index_dir / "bm25.pkl")

    assert manifest["total_chunks"] == 30
    assert collection.count() == 30
    assert len(bm25["ids"]) == 30


def test_id_sets_identical(chunks_dir: Path, tmp_path: Path, embedder: HashEmbedder) -> None:
    import chromadb

    index_dir = tmp_path / "index"
    _build(chunks_dir, index_dir, embedder)
    client = chromadb.PersistentClient(path=str(index_dir / "chroma"))
    collection = client.get_collection(COLLECTION_NAME)
    bm25 = load_bm25(index_dir / "bm25.pkl")

    chunk_ids = _chunk_ids(chunks_dir)
    assert chunk_ids == set(collection.get(include=[])["ids"]) == set(bm25["ids"])


# --- idempotency / rebuild / refuse ------------------------------------------


def test_rerun_is_noop(chunks_dir: Path, tmp_path: Path, embedder: HashEmbedder) -> None:
    index_dir = tmp_path / "index"
    _build(chunks_dir, index_dir, embedder)
    result = _build(chunks_dir, index_dir, embedder)
    assert result.get("status") == "up_to_date"


def test_changed_corpus_without_rebuild_refuses(
    chunks_dir: Path, tmp_path: Path, embedder: HashEmbedder
) -> None:
    index_dir = tmp_path / "index"
    _build(chunks_dir, index_dir, embedder)
    # Append one more chunk: new id -> new corpus_version.
    write_chunks(make_chunks(31), chunks_dir / "synthetic-book.jsonl")
    with pytest.raises(IndexStateError):
        _build(chunks_dir, index_dir, embedder)


def test_rebuild_works(chunks_dir: Path, tmp_path: Path, embedder: HashEmbedder) -> None:
    index_dir = tmp_path / "index"
    _build(chunks_dir, index_dir, embedder)
    write_chunks(make_chunks(31), chunks_dir / "synthetic-book.jsonl")
    result = _build(chunks_dir, index_dir, embedder, rebuild=True)
    assert result["total_chunks"] == 31
    assert result.get("status") != "up_to_date"


def test_resume_after_partial_insert(
    chunks_dir: Path, tmp_path: Path, embedder: HashEmbedder
) -> None:
    import chromadb

    index_dir = tmp_path / "index"
    index_dir.mkdir()
    records = make_chunks(30)
    half = records[:15]

    # Pre-insert half the ids (no manifest, no bm25) to simulate interruption.
    client = chromadb.PersistentClient(path=str(index_dir / "chroma"))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    collection.add(
        ids=[r["id"] for r in half],
        embeddings=embedder.embed([r["index_text"] for r in half]),
        documents=[r["text"] for r in half],
        metadatas=[chunk_metadata(r) for r in half],
    )

    manifest = _build(chunks_dir, index_dir, embedder)
    assert manifest["total_chunks"] == 30
    collection = client.get_collection(COLLECTION_NAME)
    ids = collection.get(include=[])["ids"]
    assert len(ids) == 30
    assert len(set(ids)) == 30  # no duplicates
    assert (index_dir / "index_manifest.json").exists()


def test_embedding_model_mismatch_raises(
    chunks_dir: Path, tmp_path: Path, embedder: HashEmbedder
) -> None:
    import chromadb

    index_dir = tmp_path / "index"
    index_dir.mkdir()
    client = chromadb.PersistentClient(path=str(index_dir / "chroma"))
    client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", "embedding_model": "some-other-model"},
    )
    with pytest.raises(IndexStateError):
        _build(chunks_dir, index_dir, embedder)


# --- cosine self-distance ----------------------------------------------------


def test_cosine_distance_of_identical_vectors(
    chunks_dir: Path, tmp_path: Path, embedder: HashEmbedder
) -> None:
    index_dir = tmp_path / "index"
    _build(chunks_dir, index_dir, embedder)
    report = verify_index(chunks_dir, index_dir, embedder, show_progress=False)
    cosine = next(c for c in report.checks if c.name == "cosine_self_distance")
    assert cosine.passed, cosine.detail


def test_verify_smoke_handles_multi_range_and_negative(
    chunks_dir: Path, tmp_path: Path, embedder: HashEmbedder, monkeypatch
) -> None:
    index_dir = tmp_path / "index"
    _build(chunks_dir, index_dir, embedder)
    smoke_path = tmp_path / "smoke.yaml"
    smoke_path.write_text(
        yaml.safe_dump(
            {
                "queries": [
                    {
                        "query": "positive",
                        "expected_book": "synthetic-book",
                        "expected_pdf_pages": [1, 1],
                    },
                    {
                        "query": "multi",
                        "expected_book": "synthetic-book",
                        "expected_pdf_pages": [[1, 1], [2, 2]],
                    },
                    {
                        "query": "negative",
                        "expected_book": "none",
                        "expected_pdf_pages": [],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        indexing,
        "_chroma_ranked",
        lambda collection, query, embedder: [
            ("first", {"book_slug": "synthetic-book", "pdf_page_start": 1, "pdf_page_end": 1}),
            ("second", {"book_slug": "synthetic-book", "pdf_page_start": 2, "pdf_page_end": 2}),
        ],
    )
    monkeypatch.setattr(
        indexing,
        "_bm25_ranked",
        lambda bundle, records, query: [
            ("first", {"book_slug": "synthetic-book", "pdf_page_start": 1, "pdf_page_end": 1}),
            ("second", {"book_slug": "synthetic-book", "pdf_page_start": 2, "pdf_page_end": 2}),
        ],
    )
    report = verify_index(chunks_dir, index_dir, embedder, smoke_path=smoke_path)
    assert [(row["chroma_rank"], row["bm25_rank"]) for row in report.smoke] == [
        (1, 1),
        (1, 1),
        (None, None),
    ]
    assert report.smoke[-1]["negative"] is True


# --- tokenizer ---------------------------------------------------------------


def test_tokenizer_keeps_snake_case_lowercases_drops_punct() -> None:
    assert tokenize("Hello, snake_case_var WORLD!") == ["hello", "snake_case_var", "world"]
    assert tokenize("CamelCase HTTP2 x_y_z") == ["camelcase", "http2", "x_y_z"]
    assert tokenize("...") == []


# --- BM25 --------------------------------------------------------------------


def test_bm25_returns_chunk_with_unique_term_first() -> None:
    records = make_chunks(10)
    target = make_chunk(
        99,
        text="This chunk uniquely mentions xylophone quantization routines.",
    )
    records.append(target)
    ids, bm25 = build_bm25(records)
    top = bm25.get_top_n(tokenize("xylophone"), ids, n=3)
    assert top[0] == target["id"]


# --- corpus version ----------------------------------------------------------


def test_corpus_version_order_independent() -> None:
    records = make_chunks(30)
    assert compute_corpus_version(records) == compute_corpus_version(list(reversed(records)))


def test_corpus_version_changes_when_text_changes() -> None:
    records = make_chunks(30)
    changed = make_chunks(30)
    changed[0] = {
        **changed[0],
        "text": "totally different text here",
        "id": "synthetic-book-00000-deadbeef",
    }
    assert compute_corpus_version(records) != compute_corpus_version(changed)


# --- metadata ----------------------------------------------------------------


def test_chroma_metadata_has_no_none_or_list(
    chunks_dir: Path, tmp_path: Path, embedder: HashEmbedder
) -> None:
    import chromadb

    index_dir = tmp_path / "index"
    _build(chunks_dir, index_dir, embedder)
    client = chromadb.PersistentClient(path=str(index_dir / "chroma"))
    collection = client.get_collection(COLLECTION_NAME)
    metadatas = collection.get(include=["metadatas"])["metadatas"]
    assert metadatas
    for meta in metadatas:
        for key, value in meta.items():
            assert value is not None, f"{key} is None"
            assert not isinstance(value, (list, dict)), f"{key} is a container"


def test_metadata_omits_null_printed_pages() -> None:
    record = make_chunk(0, printed_page_start=None, printed_page_end=None)
    meta = chunk_metadata(record)
    assert "printed_page_start" not in meta
    assert "printed_page_end" not in meta
    record2 = make_chunk(1, printed_page_start=10, printed_page_end=11)
    meta2 = chunk_metadata(record2)
    assert meta2["printed_page_start"] == 10
    assert meta2["printed_page_end"] == 11


# --- manifest ordering -------------------------------------------------------


def test_manifest_written_last_absent_if_interrupted(
    chunks_dir: Path, tmp_path: Path, embedder: HashEmbedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    import chromadb

    from research_agent.ingest import indexing

    index_dir = tmp_path / "index"

    def _boom(_chunks):
        raise RuntimeError("interrupted before BM25")

    monkeypatch.setattr(indexing, "build_bm25", _boom)
    with pytest.raises(RuntimeError):
        _build(chunks_dir, index_dir, embedder)

    # Collection was populated but the completion marker is absent.
    assert not (index_dir / "index_manifest.json").exists()
    client = chromadb.PersistentClient(path=str(index_dir / "chroma"))
    assert client.get_collection(COLLECTION_NAME).count() == 30


# --- end-to-end verify -------------------------------------------------------


def test_verify_report_ok(chunks_dir: Path, tmp_path: Path, embedder: HashEmbedder) -> None:
    index_dir = tmp_path / "index"
    _build(chunks_dir, index_dir, embedder)
    report = verify_index(chunks_dir, index_dir, embedder, show_progress=False)
    assert report.ok, [f.detail for f in report.failures]


def test_verify_detects_missing_manifest(
    chunks_dir: Path, tmp_path: Path, embedder: HashEmbedder
) -> None:
    index_dir = tmp_path / "index"
    report = verify_index(chunks_dir, index_dir, embedder)
    assert not report.ok
    assert report.checks[0].name == "manifest_exists"


# --- slow: real embedder -----------------------------------------------------


@pytest.mark.slow
def test_bge_embedder_real() -> None:
    from research_agent.ingest.embedder import BgeEmbedder

    embedder = BgeEmbedder("BAAI/bge-base-en-v1.5")
    assert embedder.dim == 768

    anchor = embedder.embed(["training a machine learning model"])[0]
    related = embedder.embed(["a model learns from training data"])[0]
    unrelated = embedder.embed(["the cathedral clock chimed at noon"])[0]

    norm = math.sqrt(sum(value * value for value in anchor))
    assert abs(norm - 1.0) < 1e-3  # normalized embeddings

    sim_related = sum(a * b for a, b in zip(anchor, related, strict=True))
    sim_unrelated = sum(a * b for a, b in zip(anchor, unrelated, strict=True))
    assert sim_related > sim_unrelated
