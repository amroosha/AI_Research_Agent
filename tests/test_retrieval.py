"""Tests for the hybrid retriever (Stage 2, T2).

Uses a synthetic index built with the deterministic ``HashEmbedder`` so no
network or model download is required.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_agent.config import DEFAULT_INGEST_SETTINGS, DEFAULT_RETRIEVAL, RetrievalSettings
from research_agent.ingest.indexing import build_index
from research_agent.retrieval import (
    HybridRetriever,
    RetrievalInitError,
    rrf,
)
from tests.helpers import VOCAB, HashEmbedder, make_chunks, write_chunks

BOOK = "synthetic-book"
SETTINGS = DEFAULT_INGEST_SETTINGS


@pytest.fixture
def embedder() -> HashEmbedder:
    return HashEmbedder(dim=64)


@pytest.fixture
def index_env(tmp_path: Path, embedder: HashEmbedder) -> tuple[Path, Path]:
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    write_chunks(make_chunks(30), chunks_dir / f"{BOOK}.jsonl")
    index_dir = tmp_path / "index"
    build_index(
        chunks_dir=chunks_dir,
        index_dir=index_dir,
        embedder=embedder,
        settings=SETTINGS,
        show_progress=False,
    )
    return chunks_dir, index_dir


def _retriever(index_env, embedder: HashEmbedder, cfg=None) -> HybridRetriever:
    chunks_dir, index_dir = index_env
    return HybridRetriever(index_dir, chunks_dir, embedder, cfg or DEFAULT_RETRIEVAL)


# --- rrf ----------------------------------------------------------------------


def test_rrf_single_list_keeps_order() -> None:
    assert rrf([["a", "b", "c"]], k=60) == [
        ("a", pytest.approx(1 / 61)),
        ("b", pytest.approx(1 / 62)),
        ("c", pytest.approx(1 / 63)),
    ]


def test_rrf_consensus_beats_single_top() -> None:
    # "b" is 2nd in both lists: 2/(60+2) > 1/(60+1) for "a".
    fused = rrf([["a", "b"], ["c", "b"]], k=60)
    assert fused[0][0] == "b"
    assert fused[1][0] == "a"  # best rank 1, id "a" < "c"
    assert fused[2][0] == "c"


def test_rrf_custom_k() -> None:
    fused = rrf([["a", "b"]], k=0)
    assert fused[0] == ("a", pytest.approx(1.0))
    assert fused[1][1] == pytest.approx(1 / 2)


def test_rrf_tie_breaking_deterministic() -> None:
    # Equal scores across distinct single lists: best rank asc, then id asc.
    fused = rrf([["a"], ["b"], ["c"]], k=60)
    assert [chunk_id for chunk_id, _ in fused] == ["a", "b", "c"]


def test_rrf_duplicate_ids_count_once() -> None:
    fused = rrf([["a", "a", "a"]], k=60)
    assert fused == [("a", pytest.approx(1 / 61))]


def test_rrf_empty_input() -> None:
    assert rrf([]) == []


# --- init checks --------------------------------------------------------------


def test_init_embedding_model_mismatch(index_env, embedder: HashEmbedder) -> None:
    class WrongName:
        name = "some-other-model"
        dim = 64

        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[0.0] * 64 for _ in texts]

    with pytest.raises(RetrievalInitError, match="embedding model mismatch"):
        _retriever(index_env, WrongName())


def test_init_corpus_version_mismatch(index_env, embedder: HashEmbedder) -> None:
    chunks_dir, index_dir = index_env
    # Modify a chunk file: text change -> new id -> new corpus version.
    records = make_chunks(30)
    records[0] = {**records[0], "id": records[0]["id"] + "-x"}
    write_chunks(records, chunks_dir / f"{BOOK}.jsonl")
    with pytest.raises(RetrievalInitError, match="corpus version mismatch"):
        HybridRetriever(index_dir, chunks_dir, embedder, DEFAULT_RETRIEVAL)


def test_init_tokenizer_mismatch(index_env, embedder: HashEmbedder) -> None:
    chunks_dir, index_dir = index_env
    manifest_path = index_dir / "index_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tokenizer"] = "other_v0"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RetrievalInitError, match="tokenizer"):
        HybridRetriever(index_dir, chunks_dir, embedder, DEFAULT_RETRIEVAL)


def test_init_index_id_missing_from_chunks(index_env, embedder: HashEmbedder) -> None:
    chunks_dir, index_dir = index_env
    # Remove a chunk from the file but keep the manifest's corpus_version in
    # sync, so the init check under test is the missing-id one.
    from research_agent.ingest.indexing import compute_corpus_version

    records = make_chunks(30)
    write_chunks(records[1:], chunks_dir / f"{BOOK}.jsonl")
    manifest_path = index_dir / "index_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["corpus_version"] = compute_corpus_version(records[1:])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RetrievalInitError, match="missing from chunk files"):
        HybridRetriever(index_dir, chunks_dir, embedder, DEFAULT_RETRIEVAL)


def test_corpus_version_exposed(index_env, embedder: HashEmbedder) -> None:
    retriever = _retriever(index_env, embedder)
    manifest = json.loads((index_env[1] / "index_manifest.json").read_text(encoding="utf-8"))
    assert retriever.corpus_version == manifest["corpus_version"]


# --- search_bm25 --------------------------------------------------------------


def test_bm25_unique_term_returns_chunk_first(index_env, embedder: HashEmbedder) -> None:
    retriever = _retriever(index_env, embedder)
    # Each chunk contains a unique token ref{index:04d}; ref0007 is chunk 7.
    expected_id = list(retriever._records_by_id)[7]
    top = retriever.search_bm25("ref0007", 5)
    assert top[0] == expected_id


def test_bm25_drops_zero_scores(index_env, embedder: HashEmbedder) -> None:
    retriever = _retriever(index_env, embedder)
    results = retriever.search_bm25("ref0007", 30)
    scores = {
        chunk_id: score
        for chunk_id, score in zip(
            retriever._bm25_bundle["ids"],
            retriever._bm25_bundle["bm25"].get_scores(["ref0007"]),
            strict=True,
        )
    }
    assert all(scores[chunk_id] > 0 for chunk_id in results)


# --- search_vector ------------------------------------------------------------


def test_vector_own_text_returns_chunk_first(index_env, embedder: HashEmbedder) -> None:
    retriever = _retriever(index_env, embedder)
    records = list(retriever._records_by_id.values())
    record = records[9]
    top = retriever.search_vector(record["text"], 3)
    assert top[0] == record["id"]


# --- retrieve -----------------------------------------------------------------


def test_retrieve_returns_chunk_objects(index_env, embedder: HashEmbedder) -> None:
    retriever = _retriever(index_env, embedder)
    chunks = retriever.retrieve("what is gradient descent")
    assert len(chunks) == DEFAULT_RETRIEVAL.final_k
    for chunk in chunks:
        assert chunk.source == "book"
        assert chunk.score is None
        assert chunk.book_slug == BOOK
        assert chunk.pdf_page_start is not None
        assert chunk.section_path == ["Chapter 1"]


def test_retrieve_respects_final_k(index_env, embedder: HashEmbedder) -> None:
    retriever = _retriever(index_env, embedder)
    assert len(retriever.retrieve("gradient descent", final_k=3)) == 3


def test_retrieve_multi_query_fusion_hand_computed(index_env, embedder: HashEmbedder) -> None:
    retriever = _retriever(index_env, embedder)
    cfg = RetrievalSettings(retrieve_k_each=5, final_k=5, rrf_k=60)
    chunks_dir, index_dir = index_env
    retriever = HybridRetriever(index_dir, chunks_dir, embedder, cfg)
    q1, q2 = "ref0003", "ref0007"
    bm25_1 = retriever.search_bm25(q1, 5)
    vec_1 = retriever.search_vector(q1, 5)
    bm25_2 = retriever.search_bm25(q2, 5)
    vec_2 = retriever.search_vector(q2, 5)

    def hand_rank(rankings):
        scores: dict[str, float] = {}
        for ranking in rankings:
            for position, chunk_id in enumerate(ranking, start=1):
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (60 + position)
        return sorted(scores, key=lambda cid: (-scores[cid], cid))

    expected = hand_rank([bm25_1, vec_1, bm25_2, vec_2])[:5]
    items = retriever.retrieve_debug([q1, q2], final_k=5)
    assert [item.chunk.id for item in items] == expected


def test_retrieve_debug_ranks_match_individual_searches(index_env, embedder: HashEmbedder) -> None:
    retriever = _retriever(index_env, embedder)
    query = "gradient descent neural"
    items = retriever.retrieve_debug(query, final_k=6)
    bm25_ids = retriever.search_bm25(query, DEFAULT_RETRIEVAL.retrieve_k_each)
    vector_ids = retriever.search_vector(query, DEFAULT_RETRIEVAL.retrieve_k_each)
    for item in items:
        if item.bm25_rank is not None:
            assert bm25_ids[item.bm25_rank - 1] == item.chunk.id
        if item.vector_rank is not None:
            assert vector_ids[item.vector_rank - 1] == item.chunk.id


def test_query_instruction_prefixes_query_only(index_env) -> None:
    class RecordingEmbedder:
        name = "hash-embedder-v1"
        dim = 64

        def __init__(self) -> None:
            self.seen: list[str] = []

        def embed(self, texts: list[str]) -> list[list[float]]:
            self.seen.extend(texts)
            return HashEmbedder(dim=64).embed(texts)

    inner = RecordingEmbedder()
    cfg = RetrievalSettings(query_instruction="Q: ")
    chunks_dir, index_dir = index_env
    retriever = HybridRetriever(index_dir, chunks_dir, inner, cfg)
    # Embed documents once during init? No: init embeds nothing. Query embeds only.
    before = len(inner.seen)
    retriever.search_vector("hello", 3)
    assert inner.seen[before:] == ["Q: hello"]


# --- scored variants (T4 additive) --------------------------------------------


def test_scored_variants_return_same_ids(index_env, embedder: HashEmbedder) -> None:
    """search_*_scored must not change the ids or their order (T4 contract)."""
    retriever = _retriever(index_env, embedder)
    query = "gradient descent neural"
    bm25_ids = retriever.search_bm25(query, 10)
    vector_ids = retriever.search_vector(query, 10)
    assert [cid for cid, _ in retriever.search_bm25_scored(query, 10)] == bm25_ids
    assert [cid for cid, _ in retriever.search_vector_scored(query, 10)] == vector_ids


def test_vector_scored_is_cosine_similarity(index_env, embedder: HashEmbedder) -> None:
    retriever = _retriever(index_env, embedder)
    records = list(retriever.chunks_by_id.values())
    own = records[9]
    scored = retriever.search_vector_scored(own.text, 3)
    assert scored[0][0] == own.id
    # Stored vectors are of index_text (title+section prefixed), so self-query
    # similarity is high but strictly < 1; what matters is the ordering.
    assert scored[0][1] > 0.7
    assert scored[0][1] >= scored[1][1]


def test_bm25_scored_scores_positive(index_env, embedder: HashEmbedder) -> None:
    retriever = _retriever(index_env, embedder)
    scored = retriever.search_bm25_scored("ref0007", 5)
    assert scored
    assert all(score > 0 for _cid, score in scored)
    assert scored == sorted(scored, key=lambda pair: -pair[1])


# --- ValueError cases ---------------------------------------------------------


def test_empty_query_raises(index_env, embedder: HashEmbedder) -> None:
    retriever = _retriever(index_env, embedder)
    with pytest.raises(ValueError, match="non-empty"):
        retriever.retrieve("   ")
    with pytest.raises(ValueError, match="non-empty"):
        retriever.retrieve([])


def test_final_k_larger_than_retrieve_k_raises(index_env, embedder: HashEmbedder) -> None:
    retriever = _retriever(index_env, embedder)
    with pytest.raises(ValueError, match="cannot exceed"):
        retriever.retrieve("gradient", final_k=DEFAULT_RETRIEVAL.retrieve_k_each + 1)


# --- vocabulary sanity --------------------------------------------------------


def test_vocabulary_is_in_use() -> None:
    # Guard against helpers drift: the retriever tests rely on VOCAB words.
    assert "gradient" in VOCAB
