"""Grader calibration metrics (Stage 2, T5).

Proxy labels come from the golden set: a retrieved chunk is ``relevant`` when
it hits an expected range (the T4 hit rule), else ``not_relevant``; for
negatives every chunk is ``not_relevant``. All functions are pure so they can
be unit-tested with hand-computed fixtures; I/O and printing live in
``scripts/eval_grader.py``.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from typing import Any

from research_agent.eval.retrieval_eval import chunk_hits_item
from research_agent.golden import GoldenItem
from research_agent.state import Chunk

# Question-level outcome labels (route() decides "generate" at >= 75).
Outcome = str  # positive: correct | wrong_generate | needless_retry | retrieval_miss
#                                negative: correct | false_generate


def proxy_labels(chunks: Sequence[Chunk], item: GoldenItem) -> list[str]:
    """Classify chunks as relevant, near, or not_relevant by page distance."""

    def label(chunk: Chunk) -> str:
        if chunk_hits_item(chunk, item):
            return "relevant"
        if item.is_negative or item.book is None or chunk.book_slug != item.book:
            return "not_relevant"
        if chunk.pdf_page_start is None or chunk.pdf_page_end is None:
            return "not_relevant"
        for start, end in item.ranges:
            if chunk.pdf_page_end >= start - 1 and chunk.pdf_page_start <= end + 1:
                return "near"
        return "not_relevant"

    return [label(chunk) for chunk in chunks]


def binary_relevant(labels: Sequence[str | bool]) -> list[bool]:
    """Convert categorical or legacy boolean labels to relevant/not relevant."""
    return [label is True or label == "relevant" for label in labels]


def confusion_counts(predicted: Sequence[bool], actual: Sequence[bool]) -> dict[str, int]:
    """Return TP, FP, FN, and TN counts."""
    if len(predicted) != len(actual):
        raise ValueError("predicted and actual must have the same length")
    return {
        "TP": int(sum(p and a for p, a in zip(predicted, actual, strict=True))),
        "FP": int(sum(p and not a for p, a in zip(predicted, actual, strict=True))),
        "FN": int(sum(not p and a for p, a in zip(predicted, actual, strict=True))),
        "TN": int(sum(not p and not a for p, a in zip(predicted, actual, strict=True))),
    }


def base_rate(labels: Sequence[str | bool]) -> float:
    """Share of chunks labeled relevant."""
    return sum(binary_relevant(labels)) / len(labels) if labels else 0.0


def similarity_baseline(
    similarities: Sequence[float], labels: Sequence[str | bool], thresholds: Sequence[float]
) -> dict[float, dict[str, Any]]:
    """Evaluate a cosine-similarity threshold sweep."""
    actual = binary_relevant(labels)
    return {
        threshold: {
            **precision_recall_f1([similarity >= threshold for similarity in similarities], actual),
            "confusion": confusion_counts(
                [similarity >= threshold for similarity in similarities], actual
            ),
        }
        for threshold in thresholds
    }


def top1_baseline(
    rankings: Sequence[Sequence[Chunk]], labels: Sequence[Sequence[str | bool]]
) -> dict[str, Any]:
    """Treat only each query's rank-1 chunk as relevant."""
    predicted = [index == 0 for chunks in rankings for index, _ in enumerate(chunks)]
    actual = [
        binary_relevant(item_labels)[index]
        for item_labels, chunks in zip(labels, rankings, strict=True)
        for index, _ in enumerate(chunks)
    ]
    return {
        **precision_recall_f1(predicted, actual),
        "confusion": confusion_counts(predicted, actual),
    }


def _legacy_proxy_labels(chunks: Sequence[Chunk], item: GoldenItem) -> list[bool]:
    """Legacy boolean proxy retained for compatibility with v1 callers."""
    if item.is_negative:
        return [False] * len(chunks)
    return [chunk_hits_item(chunk, item) for chunk in chunks]


def precision_recall_f1(predicted: Sequence[bool], actual: Sequence[bool]) -> dict[str, float]:
    """Precision/recall/F1 for a boolean classifier (empty -> all zeros)."""
    if len(predicted) != len(actual):
        raise ValueError("predicted and actual must have the same length")
    true_pos = sum(1 for p, a in zip(predicted, actual, strict=True) if p and a)
    false_pos = sum(1 for p, a in zip(predicted, actual, strict=True) if p and not a)
    false_neg = sum(1 for p, a in zip(predicted, actual, strict=True) if not p and a)
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) else 0.0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def score_distribution(scores: Sequence[int]) -> dict[str, float | None]:
    """median and quartiles (q1, q3) of a score sample."""
    if not scores:
        return {"median": None, "q1": None, "q3": None, "n": 0}
    ordered = sorted(scores)

    def quartile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        low = int(position)
        high = min(low + 1, len(ordered) - 1)
        weight = position - low
        return ordered[low] * (1 - weight) + ordered[high] * weight

    return {
        "median": statistics.median(ordered),
        "q1": quartile(0.25),
        "q3": quartile(0.75),
        "n": len(ordered),
    }


def question_outcome(
    item: GoldenItem,
    graded: Sequence[Chunk],
    labels: Sequence[str | bool],
    threshold: int = 75,
) -> Outcome:
    """Classify one question's grading outcome.

    Positive, ``correct``: some chunk >= threshold is truly relevant.
    Positive, ``wrong_generate``: chunks >= threshold exist but none is relevant.
    Positive, ``needless_retry``: no chunk >= threshold although a relevant
    chunk was in the set.
    Positive, ``retrieval_miss``: no chunk >= threshold and no relevant chunk
    was retrieved at all (the retriever, not the grader, failed).
    Negative, ``correct``: no chunk >= threshold. Negative, ``false_generate``:
    any chunk >= threshold (the dangerous failure).
    """
    if item.is_negative:
        return (
            "correct"
            if not any(chunk.score is not None and chunk.score >= threshold for chunk in graded)
            else "false_generate"
        )
    has_generate = any(chunk.score is not None and chunk.score >= threshold for chunk in graded)
    has_relevant = any(binary_relevant(labels))
    if has_generate:
        return (
            "correct"
            if any(
                label in (True, "relevant", "near")
                and chunk.score is not None
                and chunk.score >= threshold
                for chunk, label in zip(graded, labels, strict=True)
            )
            else "wrong_generate"
        )
    return "needless_retry" if has_relevant else "retrieval_miss"


def threshold_sweep(
    items: Sequence[GoldenItem],
    gradeds: Sequence[Sequence[Chunk]],
    labelss: Sequence[Sequence[str | bool]],
    thresholds: Sequence[int] = tuple(range(50, 91, 5)),
) -> dict[int, dict[str, int]]:
    """Question-level outcome counts per candidate relevant-threshold."""
    sweep: dict[int, dict[str, int]] = {}
    for threshold in thresholds:
        counts: dict[str, int] = {}
        for item, graded, labels in zip(items, gradeds, labelss, strict=True):
            outcome = question_outcome(item, graded, labels, threshold)
            counts[outcome] = counts.get(outcome, 0) + 1
        sweep[threshold] = counts
    return sweep


def position_bias(
    scores_a: Sequence[int], scores_b: Sequence[int], threshold: int = 75
) -> dict[str, float]:
    """Compare two gradings of the same chunks (e.g. original vs reversed order).

    Returns the mean absolute score difference and the fraction of chunks whose
    side of ``threshold`` flipped between the two runs.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("score lists must have the same length")
    if not scores_a:
        return {"mean_abs_diff": 0.0, "flip_fraction": 0.0, "n": 0}
    diffs = [abs(a - b) for a, b in zip(scores_a, scores_b, strict=True)]
    flips = sum(
        1 for a, b in zip(scores_a, scores_b, strict=True) if (a >= threshold) != (b >= threshold)
    )
    return {
        "mean_abs_diff": statistics.fmean(diffs),
        "flip_fraction": flips / len(scores_a),
        "n": len(scores_a),
    }


def disagreements(
    questions: Sequence[tuple[GoldenItem, Chunk, bool]], limit: int = 10
) -> list[dict[str, Any]]:
    """Largest grader-vs-proxy disagreements for manual inspection.

    ``questions`` holds (item, graded_chunk, proxy_label) triples; disagreement
    magnitude is ``score`` when the proxy says relevant but the score is low,
    and ``100 - score`` when the proxy says not relevant but the score is high.
    Labels are proxies, so some listed "errors" are label noise.
    """
    rows: list[dict[str, Any]] = []
    for item, chunk, label in questions:
        score = chunk.score if chunk.score is not None else 0
        magnitude = (100 - score) if label else score
        rows.append(
            {
                "query": item.query,
                "chunk_id": chunk.id,
                "book": chunk.book_slug,
                "pdf_pages": [chunk.pdf_page_start, chunk.pdf_page_end],
                "score": score,
                "proxy_label": "relevant" if label else "not_relevant",
                "disagreement": magnitude,
                "text_preview": chunk.text[:200],
            }
        )
    rows.sort(key=lambda row: -row["disagreement"])
    return rows[:limit]
