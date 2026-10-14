"""Indexing and verification for the Stage 1 data layer (T5).

Builds two indexes that share chunk IDs:

* a persistent Chroma collection (cosine space) over ``index_text`` embeddings,
* a ``rank_bm25`` ``BM25Okapi`` index over ``tokenize(index_text)``.

``build_index`` is idempotent and resumable; ``index_manifest.json`` is written
LAST and acts as the completion marker. ``verify_index`` re-checks the on-disk
state against the chunk files and returns a structured report.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import pickle
import random
import re
import shutil
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import chromadb
import yaml
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from research_agent.config import IngestSettings

from .embedder import Embedder

logger = logging.getLogger(__name__)

COLLECTION_NAME = "books_v1"
TOKENIZER_ID = "regex_v1"
BM25_K1 = 1.5
BM25_B = 0.75

# Lowercase words, keeping intra-word underscores (snake_case) as one token.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)*")


class IndexStateError(RuntimeError):
    """Raised when the on-disk index conflicts with the requested build."""


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def load_chunks(chunks_dir: Path) -> list[dict[str, Any]]:
    """Load every ``{slug}.jsonl`` under ``chunks_dir``, sorted by id order.

    Records are ordered by ``(book_slug, chunk_index)`` so the corpus is stable
    regardless of filesystem iteration order.
    """
    records: list[dict[str, Any]] = []
    for path in sorted(Path(chunks_dir).glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    records.sort(key=lambda record: (record["book_slug"], record["chunk_index"]))
    return records


def tokenize(text: str) -> list[str]:
    """Regex tokenizer: lowercase alphanumeric runs, keeping snake_case.

    No stemming and no stopword removal (tokenizer id ``regex_v1``).
    """
    return _TOKEN_RE.findall(text.lower())


def compute_corpus_version(chunks: Sequence[dict[str, Any]]) -> str:
    """sha256 of the sorted chunk ids joined by newline (order-independent)."""
    joined = "\n".join(sorted(record["id"] for record in chunks))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def chunk_metadata(record: dict[str, Any]) -> dict[str, Any]:
    """Flat scalar Chroma metadata for a chunk (no lists, no ``None``).

    ``section_path`` is joined with ``" > "``; null printed pages are omitted.
    """
    meta: dict[str, Any] = {
        "book_slug": record["book_slug"],
        "book_title": record["book_title"],
        "section_path": " > ".join(record.get("section_path", [])),
        "chapter": record["chapter"],
        "pdf_page_start": record["pdf_page_start"],
        "pdf_page_end": record["pdf_page_end"],
        "chunk_index": record["chunk_index"],
        "content_type": record["content_type"],
        "token_count": record["token_count"],
    }
    for key in ("printed_page_start", "printed_page_end"):
        value = record.get(key)
        if value is not None:
            meta[key] = value
    return meta


def build_bm25(chunks: Sequence[dict[str, Any]]) -> tuple[list[str], BM25Okapi]:
    """Return ``(ids, BM25Okapi)`` built over ``tokenize(index_text)``."""
    ids = [record["id"] for record in chunks]
    corpus = [tokenize(record["index_text"]) for record in chunks]
    return ids, BM25Okapi(corpus, k1=BM25_K1, b=BM25_B)


def collection_metadata(corpus_version: str, embedding_model: str) -> dict[str, Any]:
    """Collection-level metadata: cosine space plus model/corpus provenance."""
    return {
        "hnsw:space": "cosine",
        "embedding_model": embedding_model,
        "corpus_version": corpus_version,
    }


def open_collection(
    client: chromadb.ClientAPI,
    corpus_version: str,
    embedding_model: str,
) -> chromadb.Collection:
    """Get or create the books collection configured for cosine space."""
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata=collection_metadata(corpus_version, embedding_model),
    )


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def _library_versions() -> dict[str, str]:
    """Record versions of the libraries that shape the index for reproducibility."""
    names = ["chromadb", "rank-bm25", "sentence-transformers", "transformers", "numpy"]
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:  # pragma: no cover - defensive
            versions[name] = "unknown"
    return versions


def _batches(items: Sequence[dict[str, Any]], size: int) -> Iterable[Sequence[dict[str, Any]]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _add_chunks(
    collection: chromadb.Collection,
    chunks: Sequence[dict[str, Any]],
    embedder: Embedder,
    embed_batch: int,
    show_progress: bool,
) -> None:
    """Embed and upsert chunks in batches (progress bar over batches)."""
    total = len(chunks)
    iterator = _batches(chunks, embed_batch)
    if show_progress and total:
        iterator = tqdm(iterator, total=(total + embed_batch - 1) // embed_batch, desc="embedding")
    for batch in iterator:
        collection.add(
            ids=[record["id"] for record in batch],
            embeddings=embedder.embed([record["index_text"] for record in batch]),
            documents=[record["text"] for record in batch],
            metadatas=[chunk_metadata(record) for record in batch],
        )


def _per_book_counts(chunks: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in chunks:
        counts[record["book_slug"]] = counts.get(record["book_slug"], 0) + 1
    return dict(sorted(counts.items()))


def _write_manifest(
    manifest_path: Path,
    chunks: Sequence[dict[str, Any]],
    settings: IngestSettings,
    embedder: Embedder,
    corpus_version: str,
) -> dict[str, Any]:
    """Write ``index_manifest.json`` LAST (the completion marker)."""
    manifest: dict[str, Any] = {
        "corpus_version": corpus_version,
        "embedding_model": embedder.name,
        "embedding_dim": embedder.dim,
        "chunk_params": {
            "target": settings.target_tokens,
            "max": settings.max_tokens,
            "overlap": settings.overlap_tokens,
            "min": settings.min_chunk_tokens,
        },
        "tokenizer": TOKENIZER_ID,
        "bm25_params": {"k1": BM25_K1, "b": BM25_B},
        "collection_name": COLLECTION_NAME,
        "per_book_counts": _per_book_counts(chunks),
        "total_chunks": len(chunks),
        "library_versions": _library_versions(),
        "built_at": datetime.now(UTC).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


# --------------------------------------------------------------------------- #
# Verify
# --------------------------------------------------------------------------- #
@dataclass
class CheckResult:
    """One verification check: a name, pass/fail, and a human-readable detail."""

    name: str
    passed: bool
    detail: str = ""


@dataclass
class VerifyReport:
    """Aggregated verification result.

    ``checks`` are hard gates (any failure flips ``ok`` to ``False``). ``smoke``
    rows are report-only and never affect ``ok``.
    """

    checks: list[CheckResult] = field(default_factory=list)
    smoke: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def failures(self) -> list[CheckResult]:
        return [check for check in self.checks if not check.passed]

    def add(self, name: str, passed: bool, detail: str = "") -> None:
        self.checks.append(CheckResult(name=name, passed=passed, detail=detail))


def load_bm25(path: Path) -> dict[str, Any]:
    """Load the pickled BM25 bundle written by ``build_index``."""
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def _rank_first_match(
    ranked: Sequence[tuple[str, dict[str, Any]]],
    expected_book: str,
    page_lo: int,
    page_hi: int,
    limit: int = 10,
) -> int | None:
    """1-based rank of the first result in ``ranked`` whose book matches and whose
    pdf page range overlaps ``[page_lo, page_hi]``; ``None`` if none within ``limit``.
    """
    for position, (_, meta) in enumerate(ranked[:limit], start=1):
        if meta.get("book_slug") != expected_book:
            continue
        start = meta.get("pdf_page_start")
        end = meta.get("pdf_page_end")
        if start is None or end is None:
            continue
        if start <= page_hi and end >= page_lo:
            return position
    return None


def _chroma_ranked(
    collection: chromadb.Collection, query_text: str, embedder: Embedder
) -> list[tuple[str, dict[str, Any]]]:
    """Embed ``query_text`` and return Chroma's ranked ``(id, metadata)`` pairs."""
    vector = embedder.embed([query_text])[0]
    result = collection.query(query_embeddings=[vector], n_results=10, include=["metadatas"])
    ids = result["ids"][0]
    metas = result["metadatas"][0] if result["metadatas"] else [{} for _ in ids]
    return list(zip(ids, metas, strict=True))


def _bm25_ranked(
    bm25_bundle: dict[str, Any],
    records_by_id: dict[str, dict[str, Any]],
    query_text: str,
) -> list[tuple[str, dict[str, Any]]]:
    """Return BM25's ranked ``(id, metadata)`` pairs for ``query_text``."""
    ids: list[str] = bm25_bundle["ids"]
    top = bm25_bundle["bm25"].get_top_n(tokenize(query_text), ids, n=10)
    return [(cid, records_by_id[cid]) for cid in top]


def _wipe_index_dir(index_dir: Path) -> None:
    """Remove ``index_dir`` so a rebuild starts clean.

    On Windows the Chroma persistent client keeps the underlying segment files
    memory-mapped, so a plain ``rmtree`` can fail with a sharing violation even
    after the client object is dropped. We drop any cached client for this path,
    delete the collection, force a GC, and retry briefly before giving up.
    """
    if not index_dir.exists():
        return
    try:
        from chromadb.api.client import SharedSystemClient

        client = chromadb.PersistentClient(path=str(index_dir / "chroma"))
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception as error:  # noqa: BLE001 - collection may not exist yet
            logger.debug("delete_collection during wipe: %s", error)
        del client
        SharedSystemClient.clear_system_cache()
    except Exception as error:  # noqa: BLE001 - best effort; fall through to rmtree
        logger.debug("chroma teardown during wipe: %s", error)
    gc.collect()

    last_error: OSError | None = None
    for _ in range(20):
        try:
            shutil.rmtree(index_dir)
            return
        except PermissionError as error:  # pragma: no cover - Windows timing
            last_error = error
            gc.collect()
            time.sleep(0.1)
    if last_error is not None:  # pragma: no cover - surfaced to the caller
        raise last_error


def build_index(
    chunks_dir: Path,
    index_dir: Path,
    embedder: Embedder,
    settings: IngestSettings,
    embed_batch: int = 32,
    rebuild: bool = False,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Build or refresh the Chroma + BM25 indexes for the chunks in ``chunks_dir``.

    Rules:
      * manifest present and ``corpus_version`` matches -> up to date (no work);
      * manifest present and version differs -> refuse unless ``rebuild``;
      * no manifest but a partial collection exists -> resume (add only missing
        ids); extra collection ids not in the corpus are an error;
      * an existing collection whose ``embedding_model`` differs -> raise.

    ``index_manifest.json`` is written only after both stores are populated, so
    its presence marks a complete build. Returns the manifest dict (or a dict
    with ``status == "up_to_date"``).
    """
    chunks_dir = Path(chunks_dir)
    index_dir = Path(index_dir)
    manifest_path = index_dir / "index_manifest.json"
    chroma_dir = index_dir / "chroma"
    bm25_path = index_dir / "bm25.pkl"

    chunks = load_chunks(chunks_dir)
    corpus_version = compute_corpus_version(chunks)
    current_ids = {record["id"] for record in chunks}

    existing_manifest: dict[str, Any] | None = None
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if rebuild:
        logger.info("rebuild requested: wiping %s", index_dir)
        _wipe_index_dir(index_dir)
        existing_manifest = None
    elif existing_manifest is not None:
        if existing_manifest.get("corpus_version") == corpus_version:
            logger.info("index up to date (corpus_version matches)")
            return {"status": "up_to_date", **existing_manifest}
        raise IndexStateError(
            "corpus_version changed since the index was built; rerun with --rebuild"
        )

    index_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = open_collection(client, corpus_version, embedder.name)

    # Guard: an existing collection must use the same embedding model.
    stored_model = (collection.metadata or {}).get("embedding_model")
    if stored_model is not None and stored_model != embedder.name:
        raise IndexStateError(
            f"collection embedding_model {stored_model!r} != embedder {embedder.name!r}; "
            "use --rebuild"
        )

    existing_ids = set(collection.get(include=[])["ids"]) if collection.count() else set()
    extra_ids = existing_ids - current_ids
    if extra_ids:
        raise IndexStateError(
            f"collection has {len(extra_ids)} id(s) not present in the corpus; use --rebuild"
        )

    missing = [record for record in chunks if record["id"] not in existing_ids]
    if existing_ids and missing:
        logger.info("resuming: adding %d missing chunk(s)", len(missing))
    _add_chunks(collection, missing, embedder, embed_batch, show_progress)

    # BM25 is always rebuilt from the full corpus (cheap, deterministic).
    ids, bm25 = build_bm25(chunks)
    with bm25_path.open("wb") as handle:
        pickle.dump({"ids": ids, "bm25": bm25, "tokenizer": TOKENIZER_ID}, handle)

    return _write_manifest(manifest_path, chunks, settings, embedder, corpus_version)


def verify_index(
    chunks_dir: Path,
    index_dir: Path,
    embedder: Embedder,
    smoke_path: Path | None = None,
    sample_size: int = 20,
    seed: int = 0,
    show_progress: bool = False,
) -> VerifyReport:
    """Re-check the on-disk index against the chunk files.

    Hard checks (any failure makes ``report.ok`` False):
      1. manifest exists and its ``corpus_version`` matches the chunks;
      2. id sets are identical across chunk JSONL, Chroma, and BM25;
      3. collection ``embedding_model`` matches the manifest; stored dim matches;
      4. querying with a stored chunk's own embedding gives distance < 0.01;
      5. self-retrieval: Chroma top-1 and BM25 top-5 over sampled chunks.

    Smoke queries (``smoke_path``) are report-only and never affect ``ok``.
    """
    chunks_dir = Path(chunks_dir)
    index_dir = Path(index_dir)
    report = VerifyReport()

    manifest_path = index_dir / "index_manifest.json"
    chunks = load_chunks(chunks_dir)
    records_by_id = {record["id"]: record for record in chunks}
    chunk_ids = set(records_by_id)
    corpus_version = compute_corpus_version(chunks)

    # 1. Manifest + corpus version.
    if not manifest_path.exists():
        report.add("manifest_exists", False, "index_manifest.json is missing")
        return report
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    version_ok = manifest.get("corpus_version") == corpus_version
    report.add(
        "corpus_version_matches",
        version_ok,
        f"manifest={manifest.get('corpus_version')} recomputed={corpus_version}",
    )

    # Load both stores.
    client = chromadb.PersistentClient(path=str(index_dir / "chroma"))
    collection = client.get_collection(COLLECTION_NAME)
    bm25_bundle = load_bm25(index_dir / "bm25.pkl")

    chroma_ids = set(collection.get(include=[])["ids"])
    bm25_ids = set(bm25_bundle["ids"])

    # 2. ID set parity across all three stores.
    ids_ok = chunk_ids == chroma_ids == bm25_ids
    report.add(
        "id_sets_identical",
        ids_ok,
        f"chunks={len(chunk_ids)} chroma={len(chroma_ids)} bm25={len(bm25_ids)}",
    )

    # 3. Embedding model + dimension.
    model_ok = (collection.metadata or {}).get("embedding_model") == manifest.get("embedding_model")
    report.add(
        "embedding_model_matches",
        model_ok,
        f"collection={(collection.metadata or {}).get('embedding_model')} "
        f"manifest={manifest.get('embedding_model')}",
    )
    stored_dim = len(collection.peek(1)["embeddings"][0]) if collection.count() else 0
    dim_ok = stored_dim == manifest.get("embedding_dim") == embedder.dim
    report.add(
        "embedding_dim_matches",
        dim_ok,
        f"stored={stored_dim} manifest={manifest.get('embedding_dim')} embedder={embedder.dim}",
    )

    if not ids_ok:
        # Downstream checks assume a consistent corpus; stop early.
        return report

    # 4. Cosine self-distance of a stored chunk's own embedding.
    sample_id = chunks[0]["id"]
    own_vec = collection.get(ids=[sample_id], include=["embeddings"])["embeddings"][0]
    self_query = collection.query(query_embeddings=[own_vec], n_results=1, include=["distances"])
    self_distance = self_query["distances"][0][0]
    cosine_ok = self_distance < 0.01
    report.add(
        "cosine_self_distance",
        cosine_ok,
        f"distance={self_distance:.6g}",
    )

    # 5. Self-retrieval over a fixed-seed sample.
    rng = random.Random(seed)
    sampled_ids = rng.sample(sorted(chunk_ids), min(sample_size, len(chunk_ids)))
    chroma_hits = 0
    bm25_hits = 0
    iterator = sampled_ids
    if show_progress:
        iterator = tqdm(sampled_ids, desc="self-retrieval")
    for cid in iterator:
        record = records_by_id[cid]
        vector = embedder.embed([record["index_text"]])[0]
        top = collection.query(query_embeddings=[vector], n_results=1)
        if top["ids"][0] and top["ids"][0][0] == cid:
            chroma_hits += 1
        first_tokens = tokenize(record["text"])[:30]
        top5 = bm25_bundle["bm25"].get_top_n(first_tokens, bm25_bundle["ids"], n=5)
        if cid in top5:
            bm25_hits += 1

    n = len(sampled_ids)
    bm25_rate = bm25_hits / n if n else 0.0
    report.add("chroma_self_top1", chroma_hits == n, f"{chroma_hits}/{n} top-1")
    report.add("bm25_self_top5", bm25_rate >= 0.9, f"{bm25_hits}/{n} in top-5")

    # 6. Smoke queries (report-only).
    if smoke_path is not None and Path(smoke_path).exists():
        data = yaml.safe_load(Path(smoke_path).read_text(encoding="utf-8")) or {}
        for entry in data.get("queries", []) or []:
            query = entry.get("query", "")
            expected_book = entry.get("expected_book", "")
            page_lo, page_hi = entry.get("expected_pdf_pages", [0, 0])
            chroma_rank = _rank_first_match(
                _chroma_ranked(collection, query, embedder), expected_book, page_lo, page_hi
            )
            bm25_rank = _rank_first_match(
                _bm25_ranked(bm25_bundle, records_by_id, query), expected_book, page_lo, page_hi
            )
            report.smoke.append(
                {
                    "query": query,
                    "expected_book": expected_book,
                    "expected_pdf_pages": [page_lo, page_hi],
                    "chroma_rank": chroma_rank,
                    "bm25_rank": bm25_rank,
                }
            )

    return report
