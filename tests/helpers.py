"""Shared test helpers: a deterministic HashEmbedder and synthetic chunk builders.

``HashEmbedder`` is used instead of the real BGE model so tests need no network
or model download. It hashes tokens into fixed buckets and L2-normalizes, which
gives stable, comparable vectors (a related pair sharing tokens scores higher
than an unrelated pair).
"""

from __future__ import annotations

import hashlib
import json
import math

from research_agent.ingest.chunking import count_tokens_whitespace, make_index_text

# A small pool of words so synthetic chunks share vocabulary (related pairs)
# while remaining distinguishable.
VOCAB = [
    "gradient",
    "descent",
    "neural",
    "network",
    "feature",
    "vector",
    "embedding",
    "training",
    "model",
    "dataset",
    "sampling",
    "inference",
    "latent",
    "cluster",
    "entropy",
    "regularize",
]


class HashEmbedder:
    """Deterministic embedder: hash tokens into ``dim`` buckets, L2-normalize.

    Two texts sharing tokens land in overlapping buckets and thus have higher
    cosine similarity than unrelated texts. Fully deterministic across runs.
    """

    name = "hash-embedder-v1"

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def _bucket(self, token: str) -> tuple[int, float]:
        digest = hashlib.sha1(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % self.dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        return index, sign

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dim
            for token in text.lower().split():
                index, sign = self._bucket(token)
                vector[index] += sign
            norm = math.sqrt(sum(value * value for value in vector))
            if norm > 0.0:
                vector = [value / norm for value in vector]
            vectors.append(vector)
        return vectors


def make_chunk(
    index: int,
    book_slug: str = "synthetic-book",
    book_title: str = "Synthetic Book",
    section_path: list[str] | None = None,
    text: str | None = None,
    pdf_page_start: int | None = None,
    pdf_page_end: int | None = None,
    printed_page_start: int | None = None,
    printed_page_end: int | None = None,
    content_type: str = "prose",
) -> dict:
    """Build one synthetic chunk record matching the frozen chunk schema.

    ``text`` is auto-generated from ``VOCAB`` when not supplied, so every chunk
    contains some vocabulary while differing from the others.
    """
    section_path = section_path if section_path is not None else ["Chapter 1"]
    if text is None:
        words = [VOCAB[(index + offset) % len(VOCAB)] for offset in range(12)]
        # A per-chunk unique token guarantees distinct embeddings so that
        # self-retrieval top-1 is unambiguous even with the HashEmbedder.
        text = f"Section {index} discusses ref{index:04d} " + " ".join(words) + "."
    pdf_start = pdf_page_start if pdf_page_start is not None else index + 1
    pdf_stop = pdf_page_end if pdf_page_end is not None else pdf_start
    index_text = make_index_text(book_title, section_path, text)
    return {
        "id": f"{book_slug}-{index:05d}-{hashlib.sha1(text.encode('utf-8')).hexdigest()[:8]}",
        "book_slug": book_slug,
        "book_title": book_title,
        "section_path": section_path,
        "chapter": section_path[0] if section_path else "",
        "pdf_page_start": pdf_start,
        "pdf_page_end": pdf_stop,
        "printed_page_start": printed_page_start,
        "printed_page_end": printed_page_end,
        "content_type": content_type,
        "text": text,
        "index_text": index_text,
        "token_count": count_tokens_whitespace(index_text),
        "chunk_index": index,
    }


def make_chunks(
    count: int = 30,
    book_slug: str = "synthetic-book",
    book_title: str = "Synthetic Book",
    **kwargs,
) -> list[dict]:
    """Build ``count`` synthetic chunk records for one book."""
    return [
        make_chunk(i, book_slug=book_slug, book_title=book_title, **kwargs) for i in range(count)
    ]


def write_chunks(records: list[dict], path) -> None:
    """Write chunk records to a JSONL file (one JSON object per line)."""
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
