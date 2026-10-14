"""Embedder protocol and the production BGE embedder (Stage 1, T5).

``BgeEmbedder`` wraps sentence-transformers on CPU with normalized embeddings.
The heavy ``sentence_transformers`` import is deferred to ``__init__`` so that
importing this module (for the ``Embedder`` protocol) never loads torch. Tests
use :class:`tests.helpers.HashEmbedder` instead of the real model.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# bge models are trained for a 512-token window including special tokens.
BGE_MAX_SEQ_LENGTH = 512


@runtime_checkable
class Embedder(Protocol):
    """Maps a batch of texts to dense float vectors.

    Implementations expose a stable ``name`` (recorded in the index manifest and
    collection metadata) and the embedding ``dim``.
    """

    name: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding per input text, in input order."""
        ...


class BgeEmbedder:
    """sentence-transformers wrapper for a BGE model on CPU.

    Embeddings are L2-normalized (``normalize_embeddings=True``) so that inner
    product equals cosine similarity. Asserts the model window is 512 tokens.
    """

    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, device="cpu")
        assert self._model.max_seq_length == BGE_MAX_SEQ_LENGTH, (
            f"expected max_seq_length {BGE_MAX_SEQ_LENGTH}, got {self._model.max_seq_length}"
        )
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [[float(value) for value in row] for row in vectors.tolist()]
