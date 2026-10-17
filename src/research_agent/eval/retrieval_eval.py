"""Retrieval evaluation harness (Stage 2, T4).

Compares BM25-only, vector-only, and hybrid (RRF) retrieval on a golden set
using hit/recall/MRR metrics, plus a negatives-separability analysis that
checks whether a cheap top-1 similarity gate could pre-screen out-of-corpus
questions. Pure metric functions live here so they can be unit-tested with
hand-computed fixtures; I/O and printing live in ``scripts/eval_retrieval.py``.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from research_agent.golden import GoldenItem
from research_agent.state import Chunk

# The bge query prefix (frozen in the Stage 2 plan). Used only when the harness
# is asked for --query-instruction on/both; never applied to documents.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

RETRIEVE_K = 20
EVAL_KS: tuple[int, ...] = (1, 3, 5, 6, 10, 20)


# ---------------------------------------------------------------------------
# Pure metric functions
# ---------------------------------------------------------------------------
def ranges_intersect(chunk_start: int, chunk_end: int, range_start: int, range_end: int) -> bool:
    """Inclusive interval overlap: touching endpoints intersect, adjacent do not."""
    return chunk_start <= range_end and range_start <= chunk_end


def chunk_hits_range(chunk: Chunk, range_start: int, range_end: int) -> bool:
    """True when the chunk's pages overlap ``[range_start, range_end]`` (inclusive)."""
    if chunk.pdf_page_start is None or chunk.pdf_page_end is None:
        return False
    return ranges_intersect(chunk.pdf_page_start, chunk.pdf_page_end, range_start, range_end)


def chunk_hits_item(chunk: Chunk, item: GoldenItem) -> bool:
    """True when the chunk's book matches and it hits ANY expected range."""
    if item.book is None or chunk.book_slug != item.book:
        return False
    return any(chunk_hits_range(chunk, start, end) for start, end in item.ranges)


def first_hit_rank(chunks: Sequence[Chunk], item: GoldenItem) -> int | None:
    """1-based rank of the first chunk hitting any range, else None."""
    for rank, chunk in enumerate(chunks, start=1):
        if chunk_hits_item(chunk, item):
            return rank
    return None


def ranges_covered(chunks: Sequence[Chunk], item: GoldenItem) -> int:
    """How many of the item's ranges are hit by at least one chunk (top-k set)."""
    covered = 0
    for start, end in item.ranges:
        if any(
            chunk_hits_range(chunk, start, end) for chunk in chunks if chunk.book_slug == item.book
        ):
            covered += 1
    return covered


def hit_at_k(ranks: Sequence[int | None], k: int) -> tuple[int, int, float]:
    """(hits, total, fraction): items whose first hit is within top k."""
    hits = sum(1 for rank in ranks if rank is not None and rank <= k)
    total = len(ranks)
    return hits, total, (hits / total if total else 0.0)


def recall_at_k(coverages: Sequence[tuple[int, int]]) -> float:
    """Mean fraction of ranges covered by the top-k set.

    ``coverages`` are (covered, total_ranges) per item.
    """
    if not coverages:
        return 0.0
    values = [(covered / total) if total else 0.0 for covered, total in coverages]
    return statistics.fmean(values)


def mrr(ranks: Sequence[int | None]) -> float:
    """Mean reciprocal rank of the first hit within the retrieved list (0 if none)."""
    if not ranks:
        return 0.0
    return statistics.fmean(1.0 / rank if rank is not None else 0.0 for rank in ranks)


def summary_stats(values: Sequence[float]) -> dict[str, float | None]:
    """min/median/max of a sequence (None fields when empty)."""
    if not values:
        return {"min": None, "median": None, "max": None}
    return {"min": min(values), "median": statistics.median(values), "max": max(values)}


def separability(
    positive_scores: Sequence[float], negative_scores: Sequence[float]
) -> dict[str, Any]:
    """Can a threshold gate separate negatives from positives by top-1 score?

    ``separable`` is True when max(negatives) < min(positives); ``margin`` is
    min(positives) - max(negatives) (positive => a gap exists).
    """
    pos = summary_stats(positive_scores)
    neg = summary_stats(negative_scores)
    separable: bool | None = None
    margin: float | None = None
    if pos["max"] is not None and neg["max"] is not None:
        margin = float(pos["min"]) - float(neg["max"])
        separable = margin > 0
    return {"positives": pos, "negatives": neg, "separable": separable, "margin": margin}


def percentile(values: Sequence[float], fraction: float) -> float:
    """Linear-interpolation percentile (``fraction`` in [0, 1]) of ``values``."""
    if not values:
        raise ValueError("percentile of empty sequence")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


@dataclass(frozen=True)
class QueryResult:
    """Per-query retrieval outcome for one system."""

    item_id: str
    query: str
    first_hit_rank: int | None
    top5: tuple[str, ...]  # chunk ids
    latency_s: float


def evaluate_system(
    name: str,
    rankings: Mapping[str, Sequence[Chunk]],
    items: Sequence[GoldenItem],
    ks: Sequence[int] = EVAL_KS,
) -> dict[str, Any]:
    """Metrics for one system: hit@k, recall@k, MRR over the positive items.

    ``rankings`` maps item_id -> retrieved chunks (rank order, up to 20).
    Negatives are excluded from hit metrics (they are scored separately by the
    negatives-separability analysis).
    """
    positives = [item for item in items if not item.is_negative]
    ranks: list[int | None] = []
    per_query: list[dict[str, Any]] = []
    for item in positives:
        chunks = rankings.get(item.id, [])
        rank = first_hit_rank(chunks, item)
        ranks.append(rank)
        per_query.append(
            {
                "id": item.id,
                "query": item.query,
                "first_hit_rank": rank,
                "top5": [chunk.id for chunk in chunks[:5]],
            }
        )

    hit_table: dict[str, dict[str, Any]] = {}
    recall_table: dict[str, float] = {}
    for k in ks:
        hits, total, fraction = hit_at_k(ranks, k)
        hit_table[str(k)] = {"hits": hits, "total": total, "fraction": fraction}
        coverage = [
            (ranges_covered(rankings.get(item.id, [])[:k], item), len(item.ranges))
            for item in positives
        ]
        recall_table[str(k)] = recall_at_k(coverage)

    return {
        "system": name,
        "hit_at_k": hit_table,
        "recall_at_k": recall_table,
        "mrr": mrr(ranks),
        "missed_at_6": [
            {"id": row["id"], "query": row["query"]}
            for row in per_query
            if row["first_hit_rank"] is None or row["first_hit_rank"] > 6
        ],
        "per_query": per_query,
    }
