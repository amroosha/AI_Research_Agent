"""Calibrate the LLM grader against golden-set proxy labels.

Usage:
    python scripts/eval_grader.py --set tests/dev_set.yaml [--top-n 10] [--allow-test]

For every question the top-n chunks are retrieved with the hybrid retriever,
graded by the Ollama model (batches of GraderSettings.grader_batch), and
compared against proxy labels (T4's hit rule). Reports per-chunk
precision/recall/F1 at thresholds 75 and 50, question-level outcomes, a
threshold sweep, position bias (reversed-order regrade), parse failures,
latency, and the 10 largest grader-vs-proxy disagreements.

Writes data/eval/grader_<UTC timestamp>.json. Recommends a threshold but
never edits config.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from research_agent.config import (
    CHUNKS_DIR,
    DATA_DIR,
    DEFAULT_GRADER,
    DEFAULT_INGEST_SETTINGS,
    DEFAULT_LLM,
    DEFAULT_RETRIEVAL,
    INDEX_DIR,
)
from research_agent.eval.grader_eval import (
    disagreements,
    position_bias,
    precision_recall_f1,
    proxy_labels,
    question_outcome,
    score_distribution,
    threshold_sweep,
)
from research_agent.eval.retrieval_eval import percentile
from research_agent.golden import GoldenItem, load_golden
from research_agent.grader import grade_chunks
from research_agent.ingest.embedder import BgeEmbedder
from research_agent.llm import get_llm
from research_agent.retrieval import HybridRetriever
from research_agent.state import Chunk

EVAL_DIR = DATA_DIR / "eval"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Grader calibration harness.")
    parser.add_argument("--set", type=Path, required=True, help="golden YAML file")
    parser.add_argument("--top-n", type=int, default=10, help="chunks retrieved per question")
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="required to run on a file whose name contains 'test'",
    )
    parser.add_argument("--index", type=Path, default=INDEX_DIR)
    parser.add_argument("--chunks", type=Path, default=CHUNKS_DIR)
    return parser


def check_test_protection(set_path: Path, allow_test: bool) -> str:
    """Return the split name; refuse test files without --allow-test."""
    split = "test" if "test" in set_path.name.lower() else "dev"
    if split == "test" and not allow_test:
        print(
            f"refusing to evaluate {set_path}: file name contains 'test'. "
            "Test-set results must not be used for tuning. Re-run with --allow-test "
            "if you really want to score the locked test set.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if split == "test":
        print(
            "WARNING: evaluating a TEST set. These results must not be used for "
            "tuning; tune on the dev set only."
        )
    return split


def _git_commit() -> str | None:
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and commit else None


def chunk_metrics_at_threshold(
    items: list[GoldenItem],
    gradeds: list[list[Chunk]],
    labelss: list[list[bool]],
    threshold: int,
) -> dict[str, Any]:
    """Per-chunk precision/recall/F1 and score distributions at one threshold."""
    predicted: list[bool] = []
    actual: list[bool] = []
    relevant_scores: list[int] = []
    not_relevant_scores: list[int] = []
    for graded, labels in zip(gradeds, labelss, strict=True):
        for chunk, label in zip(graded, labels, strict=True):
            score = chunk.score if chunk.score is not None else 0
            predicted.append(score >= threshold)
            actual.append(label)
            (relevant_scores if label else not_relevant_scores).append(score)
    return {
        "precision_recall_f1": precision_recall_f1(predicted, actual),
        "score_distribution": {
            "relevant": score_distribution(relevant_scores),
            "not_relevant": score_distribution(not_relevant_scores),
        },
    }


def recommend_threshold(sweep: dict[int, dict[str, int]]) -> int:
    """Threshold maximizing correct positives while minimizing dangerous errors.

    wrong_generate and false_generate are weighted far above the other misses
    (an incorrect confident answer is worse than an extra retrieval turn).
    """
    best_threshold: int | None = None
    best_cost: float | None = None
    for threshold, counts in sweep.items():
        cost = (
            10 * counts.get("false_generate", 0)
            + 5 * counts.get("wrong_generate", 0)
            + counts.get("needless_retry", 0)
            + counts.get("retrieval_miss", 0)
        )
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_threshold = threshold
    assert best_threshold is not None  # sweep is never empty
    return best_threshold


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    split = check_test_protection(args.set, args.allow_test)

    items = load_golden(args.set)
    if not items:
        print(f"error: no queries in {args.set}", file=sys.stderr)
        return 2

    embedder = BgeEmbedder(DEFAULT_INGEST_SETTINGS.embed_model)
    retriever = HybridRetriever(args.index, args.chunks, embedder, DEFAULT_RETRIEVAL)
    llm = get_llm()
    model_tag = os.environ.get("OLLAMA_MODEL", "").strip()

    # Retrieve once; grade twice (original + reversed order for position bias).
    retrieved: dict[str, list[Chunk]] = {}
    for item in items:
        retrieved[item.id] = retriever.retrieve(item.query, final_k=args.top_n)

    parse_failures: list[str] = []
    batch_latencies: list[float] = []
    gradeds: list[list[Chunk]] = []
    gradeds_reversed: list[list[Chunk]] = []
    for item in items:
        chunks = retrieved[item.id]
        print(f"grading {item.id}: {len(chunks)} chunks ...")
        graded = grade_chunks(
            llm,
            item.query,
            chunks,
            DEFAULT_GRADER,
            llm_cfg=DEFAULT_LLM,
            parse_failures=parse_failures,
            latencies=batch_latencies,
        )
        graded_rev = grade_chunks(
            llm, item.query, list(reversed(chunks)), DEFAULT_GRADER, llm_cfg=DEFAULT_LLM
        )
        by_id = {chunk.id: chunk for chunk in graded_rev}
        gradeds.append(graded)
        gradeds_reversed.append([by_id[chunk.id] for chunk in graded])

    labelss = [proxy_labels(retrieved[item.id], item) for item in items]
    positives = [item for item in items if not item.is_negative]
    negatives = [item for item in items if item.is_negative]

    # Per-chunk metrics at the two frozen band edges.
    per_chunk = {
        str(threshold): chunk_metrics_at_threshold(items, gradeds, labelss, threshold)
        for threshold in (75, 50)
    }

    # Question-level outcomes at the route() threshold (75).
    outcomes = {
        item.id: question_outcome(item, graded, labels, 75)
        for item, graded, labels in zip(items, gradeds, labelss, strict=True)
    }
    outcome_counts: dict[str, int] = {}
    for outcome in outcomes.values():
        outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1

    sweep = threshold_sweep(items, gradeds, labelss)
    recommended = recommend_threshold(sweep)

    # Position bias: pair original and reversed scores by chunk id.
    bias_values: list[dict[str, float]] = []
    for original, reversed_ in zip(gradeds, gradeds_reversed, strict=True):
        rev_by_id = {chunk.id: chunk.score or 0 for chunk in reversed_}
        bias_values.append(
            position_bias(
                [chunk.score or 0 for chunk in original],
                [rev_by_id[chunk.id] for chunk in original],
            )
        )
    bias = {
        "mean_abs_diff": statistics.fmean(row["mean_abs_diff"] for row in bias_values),
        "flip_fraction": statistics.fmean(row["flip_fraction"] for row in bias_values),
        "n": sum(row["n"] for row in bias_values),
    }

    # Disagreements need (item, graded chunk, label) triples over all chunks.
    triples: list[tuple[GoldenItem, Chunk, bool]] = []
    for item, graded, labels in zip(items, gradeds, labelss, strict=True):
        triples.extend(zip([item] * len(graded), graded, labels, strict=True))
    top_disagreements = disagreements(triples, limit=10)

    n_total_chunks = sum(len(graded) for graded in gradeds)
    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "prompt_version": DEFAULT_GRADER.prompt_version,
        "model": model_tag,
        "corpus_version": retriever.corpus_version,
        "set_path": str(args.set),
        "split": split,
        "top_n": args.top_n,
        "thresholds": {"relevant": 75, "ambiguous": 50},
        "n_positives": len(positives),
        "n_negatives": len(negatives),
        "n_graded_chunks": n_total_chunks,
        "per_chunk": per_chunk,
        "question_outcomes_at_75": outcome_counts,
        "threshold_sweep": {str(k): v for k, v in sweep.items()},
        "recommended_threshold": recommended,
        "position_bias": bias,
        "parse_failures": {
            "count": len(parse_failures),
            "rate": len(parse_failures) / n_total_chunks if n_total_chunks else 0.0,
            "chunk_ids": parse_failures,
        },
        "latency_per_batch_s": {
            "median": statistics.median(batch_latencies) if batch_latencies else None,
            "p95": percentile(batch_latencies, 0.95) if batch_latencies else None,
            "n_batches": len(batch_latencies),
        },
        "disagreements": top_disagreements,
        "per_question": [
            {
                "id": item.id,
                "query": item.query,
                "outcome": outcomes[item.id],
                "scores": [
                    {"id": chunk.id, "score": chunk.score, "relevant": label}
                    for chunk, label in zip(graded, labels, strict=True)
                ],
            }
            for item, graded, labels in zip(items, gradeds, labelss, strict=True)
        ],
        "git_commit": _git_commit(),
    }

    _print_report(report)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = EVAL_DIR / f"grader_{stamp}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out_path}")
    return 0


def _print_report(report: dict[str, Any]) -> None:
    print(f"\nmodel={report['model']}  prompt={report['prompt_version']}  split={report['split']}")
    print(
        f"positives={report['n_positives']}  negatives={report['n_negatives']}  "
        f"graded chunks={report['n_graded_chunks']}"
    )

    print("\n== per-chunk metrics vs proxy labels ==")
    for threshold, block in report["per_chunk"].items():
        prf = block["precision_recall_f1"]
        print(
            f"  threshold {threshold}: precision={prf['precision']:.3f} "
            f"recall={prf['recall']:.3f} f1={prf['f1']:.3f}"
        )
        for label, dist in block["score_distribution"].items():
            if dist["n"]:
                print(
                    f"    {label}: median={dist['median']:.0f} q1={dist['q1']:.0f} "
                    f"q3={dist['q3']:.0f} (n={dist['n']})"
                )

    print("\n== question-level outcomes at threshold 75 ==")
    for outcome, count in sorted(report["question_outcomes_at_75"].items()):
        print(f"  {outcome}: {count}")

    print("\n== threshold sweep (question-level counts) ==")
    header = f"{'thr':>4}  " + "  ".join(
        f"{name}" for name in ("correct", "wrong_gen", "retry", "miss", "false_gen")
    )
    print(header)
    aliases = {
        "correct": "correct",
        "wrong_gen": "wrong_generate",
        "retry": "needless_retry",
        "miss": "retrieval_miss",
        "false_gen": "false_generate",
    }
    for threshold, counts in report["threshold_sweep"].items():
        print(
            f"{threshold:>4}  "
            + "  ".join(f"{counts.get(aliases[name], 0):>7}" for name in aliases)
        )
    print(
        f"\nrecommended relevant threshold: {report['recommended_threshold']} "
        "(recommendation only; config not edited)"
    )

    bias = report["position_bias"]
    print(
        f"\n== position bias ==\n  mean |score diff|={bias['mean_abs_diff']:.2f}  "
        f"threshold flips={bias['flip_fraction'] * 100:.1f}% of {bias['n']} chunks"
    )

    failures = report["parse_failures"]
    latency = report["latency_per_batch_s"]
    print(f"\n== parse failures ==\n  {failures['count']} ({failures['rate'] * 100:.1f}%)")
    if latency["median"] is not None:
        print(
            f"== batch latency ==\n  median={latency['median']:.2f}s "
            f"p95={latency['p95']:.2f}s over {latency['n_batches']} batches"
        )

    print("\n== 10 largest disagreements (grader vs proxy label; inspect by hand) ==")
    for row in report["disagreements"]:
        pages = row["pdf_pages"]
        print(
            f"  [{row['proxy_label']} -> score {row['score']}] {row['query']!r}\n"
            f"    {row['book']} pages {pages[0]}-{pages[1]}  ({row['chunk_id']})\n"
            f"    {row['text_preview'][:120]}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
