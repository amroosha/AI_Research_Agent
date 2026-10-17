"""Hybrid BM25 + vector retrieval (Stage 2, T2).

``HybridRetriever`` wraps the two Stage 1 indexes (a persistent Chroma
collection and a rank_bm25 pickle) behind one API and fuses their rankings
with reciprocal rank fusion (RRF) by chunk ID. It reuses ``tokenize`` from
``ingest.indexing`` and an ``Embedder`` implementation (``BgeEmbedder`` in
production, ``HashEmbedder`` in tests) instead of duplicating them.
"""

from __future__ import annotations

import json
import pickle
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import chromadb

from research_agent.config import RetrievalSettings
from research_agent.ingest.embedder import Embedder
from research_agent.ingest.indexing import (
    COLLECTION_NAME,
    TOKENIZER_ID,
    compute_corpus_version,
    load_chunks,
    tokenize,
)
from research_agent.state import Chunk

MANIFEST_NAME = "index_manifest.json"


class RetrievalInitError(RuntimeError):
    """Raised when the on-disk index does not match the given embedder/config."""


def rrf(rankings: Sequence[Sequence[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal rank fusion of ``rankings`` (each 1-based rank order).

    Score of an id = sum over rankings of ``1 / (k + rank)``. An id repeated
    inside one ranking counts only at its first position. Sorted by score
    descending, then best (minimum) rank ascending, then id ascending, so the
    output is fully deterministic.
    """
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for ranking in rankings:
        seen: set[str] = set()
        for position, chunk_id in enumerate(ranking, start=1):
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + position)
            current_best = best_rank.get(chunk_id)
            if current_best is None or position < current_best:
                best_rank[chunk_id] = position
    ordered = sorted(
        scores, key=lambda chunk_id: (-scores[chunk_id], best_rank[chunk_id], chunk_id)
    )
    return [(chunk_id, scores[chunk_id]) for chunk_id in ordered]


@dataclass(frozen=True)
class RetrievedItem:
    """One fused retrieval result: the chunk plus provenance ranks."""

    chunk: Chunk
    rrf_score: float
    bm25_rank: int | None
    vector_rank: int | None


def _load_bm25(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _load_manifest(index_dir: Path) -> dict:
    manifest_path = Path(index_dir) / MANIFEST_NAME
    if not manifest_path.exists():
        raise RetrievalInitError(f"index manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


class HybridRetriever:
    """BM25 + vector retrieval fused with RRF over shared chunk IDs.

    On init the on-disk index is checked against the caller: the manifest
    tokenizer must be ``regex_v1``, the Chroma collection's embedding model
    must equal ``embedder.name``, and the corpus version recomputed from the
    chunk JSONL ids must equal the manifest's. Any mismatch raises
    :class:`RetrievalInitError` rather than silently mixing incompatible data.
    """

    def __init__(
        self,
        index_dir: Path | str,
        chunks_dir: Path | str,
        embedder: Embedder,
        cfg: RetrievalSettings | None = None,
    ) -> None:
        self.cfg = cfg if cfg is not None else RetrievalSettings()
        self.index_dir = Path(index_dir)
        manifest = _load_manifest(self.index_dir)

        tokenizer = manifest.get("tokenizer")
        if tokenizer != TOKENIZER_ID:
            raise RetrievalInitError(
                f"manifest tokenizer {tokenizer!r} != required {TOKENIZER_ID!r}"
            )

        chunks = load_chunks(Path(chunks_dir))
        self.corpus_version: str = manifest.get("corpus_version", "")
        recomputed = compute_corpus_version(chunks)
        if recomputed != self.corpus_version:
            raise RetrievalInitError(
                "corpus version mismatch: chunks on disk were rebuilt or modified "
                f"(manifest={self.corpus_version[:12]} recomputed={recomputed[:12]}); "
                "re-run scripts/build_index.py --rebuild"
            )

        client = chromadb.PersistentClient(path=str(self.index_dir / "chroma"))
        try:
            self._collection = client.get_collection(COLLECTION_NAME)
        except Exception as exc:  # pragma: no cover - only when absent
            raise RetrievalInitError(f"collection {COLLECTION_NAME!r} not found") from exc
        collection_model = (self._collection.metadata or {}).get("embedding_model")
        if collection_model != embedder.name:
            raise RetrievalInitError(
                f"embedding model mismatch: collection={collection_model!r} embedder={embedder.name!r}"
            )

        self._embedder = embedder
        self._bm25_bundle = _load_bm25(self.index_dir / "bm25.pkl")

        records_by_id: dict[str, dict] = {record["id"]: record for record in chunks}
        missing = [
            chunk_id
            for chunk_id in list(self._bm25_bundle["ids"])
            + list(self._collection.get(include=[])["ids"])
            if chunk_id not in records_by_id
        ]
        if missing:
            raise RetrievalInitError(
                f"{len(missing)} index id(s) missing from chunk files, e.g. {missing[0]!r}"
            )

        self._records_by_id = records_by_id
        self._chunks_by_id = {cid: self._to_chunk(rec) for cid, rec in records_by_id.items()}

    @property
    def chunks_by_id(self) -> dict[str, Chunk]:
        """All corpus chunks as frozen :class:`Chunk` objects, keyed by id."""
        return dict(self._chunks_by_id)

    def get_chunk(self, chunk_id: str) -> Chunk:
        """Return one corpus chunk by id."""
        try:
            return self._chunks_by_id[chunk_id]
        except KeyError as exc:
            raise KeyError(f"unknown chunk id: {chunk_id!r}") from exc

    def get_chunks(self, ids: Sequence[str]) -> list[Chunk]:
        """Return corpus chunks in the order of ``ids``."""
        return [self.get_chunk(chunk_id) for chunk_id in ids]

    def _to_chunk(self, record: dict) -> Chunk:
        return Chunk(
            id=record["id"],
            text=record["text"],
            source="book",
            book_slug=record["book_slug"],
            book_title=record["book_title"],
            section_path=list(record.get("section_path", [])),
            pdf_page_start=record["pdf_page_start"],
            pdf_page_end=record["pdf_page_end"],
            printed_page_start=record.get("printed_page_start"),
            printed_page_end=record.get("printed_page_end"),
            score=None,
        )

    def search_bm25_scored(self, query: str, k: int) -> list[tuple[str, float]]:
        """Top-k ``(chunk_id, bm25_score)``; ties broken by lower corpus index."""
        tokens = tokenize(query)
        scores = self._bm25_bundle["bm25"].get_scores(tokens)
        ids = self._bm25_bundle["ids"]
        positive = [(float(score), index) for index, score in enumerate(scores) if score > 0]
        positive.sort(key=lambda pair: (-pair[0], pair[1]))
        return [(ids[index], score) for score, index in positive[:k]]

    def search_bm25(self, query: str, k: int) -> list[str]:
        """Top-k chunk ids by BM25 score, ties broken by lower corpus index."""
        return [chunk_id for chunk_id, _score in self.search_bm25_scored(query, k)]

    def search_vector_scored(self, query: str, k: int) -> list[tuple[str, float]]:
        """Top-k ``(chunk_id, cosine_similarity)`` with similarity = 1 - distance."""
        vector = self._embedder.embed([self.cfg.query_instruction + query])[0]
        result = self._collection.query(
            query_embeddings=[vector], n_results=k, include=["distances"]
        )
        return [
            (chunk_id, 1.0 - float(distance))
            for chunk_id, distance in zip(result["ids"][0], result["distances"][0], strict=True)
        ]

    def search_vector(self, query: str, k: int) -> list[str]:
        """Top-k chunk ids by cosine similarity (embeddings are normalized)."""
        return [chunk_id for chunk_id, _score in self.search_vector_scored(query, k)]

    def vector_similarities(self, query: str, chunk_ids: Sequence[str]) -> dict[str, float]:
        """Return cosine similarities for the requested chunk ids."""
        for chunk_id in chunk_ids:
            self.get_chunk(chunk_id)
        if not chunk_ids:
            return {}
        query_vector = self._embedder.embed([self.cfg.query_instruction + query])[0]
        result = self._collection.get(ids=list(chunk_ids), include=["embeddings"])
        embeddings_by_id = dict(zip(result["ids"], result["embeddings"], strict=True))
        query_norm = sum(value * value for value in query_vector) ** 0.5
        similarities: dict[str, float] = {}
        for chunk_id in chunk_ids:
            vector = embeddings_by_id[chunk_id]
            vector_norm = sum(value * value for value in vector) ** 0.5
            dot = sum(left * right for left, right in zip(query_vector, vector, strict=True))
            similarities[chunk_id] = dot / (query_norm * vector_norm)
        return similarities

    def retrieve_debug(
        self, queries: str | list[str], final_k: int | None = None
    ) -> list[RetrievedItem]:
        """Fuse both retrievers over every query and return top ``final_k``.

        A list of queries is how Stage 3's RAG-Fusion rewrite reuses this
        function: every query x retriever ranking is fused together.
        """
        if isinstance(queries, str):
            query_list = [queries]
        else:
            query_list = list(queries)
        if not query_list or all(not query.strip() for query in query_list):
            raise ValueError("queries must be a non-empty string or list of strings")
        k = self.cfg.final_k if final_k is None else final_k
        if k > self.cfg.retrieve_k_each:
            raise ValueError(
                f"final_k ({k}) cannot exceed retrieve_k_each ({self.cfg.retrieve_k_each})"
            )

        rankings: list[list[str]] = []
        bm25_rank: dict[str, int] = {}
        vector_rank: dict[str, int] = {}
        for query in query_list:
            if not query.strip():
                continue
            bm25_ids = self.search_bm25(query, self.cfg.retrieve_k_each)
            vector_ids = self.search_vector(query, self.cfg.retrieve_k_each)
            rankings.extend([bm25_ids, vector_ids])
            for rank, chunk_id in enumerate(bm25_ids, start=1):
                if chunk_id not in bm25_rank or rank < bm25_rank[chunk_id]:
                    bm25_rank[chunk_id] = rank
            for rank, chunk_id in enumerate(vector_ids, start=1):
                if chunk_id not in vector_rank or rank < vector_rank[chunk_id]:
                    vector_rank[chunk_id] = rank

        fused = rrf(rankings, self.cfg.rrf_k)
        return [
            RetrievedItem(
                chunk=self._to_chunk(self._records_by_id[chunk_id]),
                rrf_score=score,
                bm25_rank=bm25_rank.get(chunk_id),
                vector_rank=vector_rank.get(chunk_id),
            )
            for chunk_id, score in fused[:k]
        ]

    def retrieve(self, queries: str | list[str], final_k: int | None = None) -> list[Chunk]:
        """The fused chunks, in rank order."""
        return [item.chunk for item in self.retrieve_debug(queries, final_k)]
