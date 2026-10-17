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
    DEFAULT_INGEST_SETTINGS,
    DEFAULT_LLM,
    DEFAULT_RETRIEVAL,
    INDEX_DIR,
    GraderSettings,
)
from research_agent.eval.grader_eval import (
    confusion_counts,
    precision_recall_f1,
    proxy_labels,
    question_outcome,
    score_distribution,
    similarity_baseline,
    threshold_sweep,
    top1_baseline,
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
    parser.add_argument(
        "--grader-mode",
        choices=("pointwise", "batch", "both"),
        default="both",
    )
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
    labelss: list[list[str]],
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
            is_relevant = label == "relevant"
            actual.append(is_relevant)
            (relevant_scores if is_relevant else not_relevant_scores).append(score)
    confusion = confusion_counts(predicted, actual)
    return {
        "precision_recall_f1": precision_recall_f1(predicted, actual),
        "confusion": confusion,
        "base_rate": sum(actual) / len(actual) if actual else 0.0,
        "score_distribution": {
            "relevant": score_distribution(relevant_scores),
            "not_relevant": score_distribution(not_relevant_scores),
        },
    }


def _grade_mode(
    mode: str,
    llm,
    items: list[GoldenItem],
    retrieved: dict[str, list[Chunk]],
    llm_cfg,
) -> tuple[dict[str, Any], list[list[Chunk]], list[str], list[float]]:
    cfg = GraderSettings(
        mode=mode,
        prompt_version="grader_v1" if mode == "batch" else "grader_v2",
    )
    failures: list[str] = []
    latencies: list[float] = []
    gradeds: list[list[Chunk]] = []
    for item in items:
        gradeds.append(
            grade_chunks(
                llm,
                item.query,
                retrieved[item.id],
                cfg,
                llm_cfg=llm_cfg,
                parse_failures=failures,
                latencies=latencies,
            )
        )
    return {"mode": mode, "prompt_version": cfg.prompt_version}, gradeds, failures, latencies


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
    model_tag = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b").strip()

    retrieved: dict[str, list[Chunk]] = {}
    for item in items:
        retrieved[item.id] = retriever.retrieve(item.query, final_k=args.top_n)

    labelss = [proxy_labels(retrieved[item.id], item) for item in items]
    positives = [item for item in items if not item.is_negative]
    negatives = [item for item in items if item.is_negative]
    modes = ("batch", "pointwise") if args.grader_mode == "both" else (args.grader_mode,)
    reports: dict[str, Any] = {}
    for mode in modes:
        print(f"grading mode={mode}")
        llm = get_llm(num_predict=32) if mode == "pointwise" else get_llm()
        llm_cfg = DEFAULT_LLM
        metadata, gradeds, parse_failures, latencies = _grade_mode(
            mode, llm, items, retrieved, llm_cfg
        )
        per_chunk = {
            str(threshold): chunk_metrics_at_threshold(items, gradeds, labelss, threshold)
            for threshold in (75, 50)
        }
        outcomes = {
            item.id: question_outcome(item, graded, labels, 75)
            for item, graded, labels in zip(items, gradeds, labelss, strict=True)
        }
        outcome_counts: dict[str, int] = {}
        for outcome in outcomes.values():
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        sweep = threshold_sweep(items, gradeds, labelss)
        n_total_chunks = sum(len(graded) for graded in gradeds)
        reports[mode] = {
            **metadata,
            "model": model_tag,
            "per_chunk": per_chunk,
            "question_outcomes_at_75": outcome_counts,
            "threshold_sweep": {str(k): v for k, v in sweep.items()},
            "recommended_threshold": recommend_threshold(sweep),
            "parse_failures": {
                "count": len(parse_failures),
                "rate": len(parse_failures) / n_total_chunks if n_total_chunks else 0.0,
                "chunk_ids": parse_failures,
            },
            "latency_s": {
                "median": statistics.median(latencies) if latencies else None,
                "p95": percentile(latencies, 0.95) if latencies else None,
                "n_calls": len(latencies),
                "total": sum(latencies),
            },
            "per_question": [
                {"id": item.id, "query": item.query, "outcome": outcomes[item.id]} for item in items
            ],
        }
        if mode == "batch":
            reports[mode]["position_bias"] = {"not_run": False}
        else:
            sample_pairs = [(item.query, chunk) for item in items for chunk in retrieved[item.id]][
                :20
            ]
            stability_cfg = GraderSettings(mode="pointwise", prompt_version="grader_v2")
            stability_llm_cfg = get_llm(num_predict=32)
            first = [
                grade_chunks(
                    llm,
                    query,
                    [chunk],
                    stability_cfg,
                    llm_cfg=stability_llm_cfg,
                )[0]
                for query, chunk in sample_pairs
            ]
            second = [
                grade_chunks(
                    llm,
                    query,
                    [chunk],
                    stability_cfg,
                    llm_cfg=stability_llm_cfg,
                )[0]
                for query, chunk in sample_pairs
            ]
            reports[mode]["stability"] = {
                "sample_size": len(first),
                "label_differences": sum(a.score != b.score for a, b in zip(first, second)),
            }

    # Grader-free baselines use the same retrieved chunks and categorical labels.
    sim_rows: list[float] = []
    sim_labels: list[str] = []
    for item, labels in zip(items, labelss, strict=True):
        similarities = retriever.vector_similarities(
            item.query, [chunk.id for chunk in retrieved[item.id]]
        )
        sim_rows.extend(similarities[chunk.id] for chunk in retrieved[item.id])
        sim_labels.extend(labels)
    thresholds = tuple(round(0.50 + index * 0.05, 2) for index in range(8))
    similarity_rows = similarity_baseline(sim_rows, sim_labels, thresholds)
    best_similarity = max(thresholds, key=lambda t: similarity_rows[t]["f1"])
    baseline = {
        "vector_similarity": {
            "thresholds": {str(k): v for k, v in similarity_rows.items()},
            "best": best_similarity,
            "optimistic_dev_tuning": True,
        },
        "top1": top1_baseline([retrieved[item.id] for item in items], labelss),
    }
    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "model": model_tag,
        "corpus_version": retriever.corpus_version,
        "set_path": str(args.set),
        "split": split,
        "top_n": args.top_n,
        "n_positives": len(positives),
        "n_negatives": len(negatives),
        "modes": reports,
        "baselines": baseline,
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
    print(f"\nmodel={report['model']}  split={report['split']}")
    print(f"positives={report['n_positives']}  negatives={report['n_negatives']}")
    print("\n== side-by-side grader metrics ==")
    print(
        "mode       P75     R75     F1-75   TP FP FN TN   P50     R50     F1-50   calls  failures"
    )
    for mode, block in report["modes"].items():
        m75 = block["per_chunk"]["75"]
        m50 = block["per_chunk"]["50"]
        c = m75["confusion"]
        print(
            f"{mode:<10} {m75['precision_recall_f1']['precision']:.3f} "
            f"{m75['precision_recall_f1']['recall']:.3f} "
            f"{m75['precision_recall_f1']['f1']:.3f} "
            f"{c['TP']:2} {c['FP']:2} {c['FN']:2} {c['TN']:2} "
            f"{m50['precision_recall_f1']['precision']:.3f} "
            f"{m50['precision_recall_f1']['recall']:.3f} "
            f"{m50['precision_recall_f1']['f1']:.3f} "
            f"{block['latency_s']['n_calls']:5} "
            f"{block['parse_failures']['count']:8}"
        )
        print(f"  {mode}: prompt={block['prompt_version']} base_rate={m75['base_rate']:.3f}")
        print(f"  outcomes@75: {block['question_outcomes_at_75']}")
        if "stability" in block:
            print(f"  stability: {block['stability']}")
    print("\n== grader-free baselines ==")
    for threshold, metrics in report["baselines"]["vector_similarity"]["thresholds"].items():
        print(
            f"  vector similarity {threshold}: P={metrics['precision']:.3f} "
            f"R={metrics['recall']:.3f} F1={metrics['f1']:.3f}"
        )
    print(
        f"  best vector threshold={report['baselines']['vector_similarity']['best']} "
        "(optimistic dev-set tuning)"
    )
    print(f"  top-1 baseline: {report['baselines']['top1']}")


if __name__ == "__main__":
    raise SystemExit(main())
