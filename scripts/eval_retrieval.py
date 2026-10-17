"""Evaluate BM25 / vector / hybrid retrieval on a golden set.

Usage:
    python scripts/eval_retrieval.py --set tests/dev_set.yaml [--query-instruction both]
        [--allow-test]

Systems compared: bm25, vector, hybrid (RRF). With --query-instruction on/both
the vector and hybrid systems are additionally run with the bge query prefix
(separate retrievers; the config default is never changed). Refuses to run on
a file whose name contains "test" unless --allow-test is given.

Writes data/eval/retrieval_<UTC timestamp>.json. All results are report-only:
nothing in the pipeline is tuned or edited by this script.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from research_agent.config import (
    CHUNKS_DIR,
    DATA_DIR,
    DEFAULT_INGEST_SETTINGS,
    DEFAULT_RETRIEVAL,
    INDEX_DIR,
    RetrievalSettings,
)
from research_agent.eval.retrieval_eval import (
    BGE_QUERY_INSTRUCTION,
    EVAL_KS,
    RETRIEVE_K,
    evaluate_system,
    percentile,
    separability,
)
from research_agent.golden import GoldenItem, load_golden
from research_agent.ingest.embedder import BgeEmbedder
from research_agent.retrieval import HybridRetriever
from research_agent.state import Chunk

EVAL_DIR = DATA_DIR / "eval"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retrieval evaluation harness.")
    parser.add_argument("--set", type=Path, required=True, help="golden YAML file")
    parser.add_argument(
        "--query-instruction",
        choices=("off", "on", "both"),
        default="both",
        help="run vector/hybrid with the bge query prefix (default: both)",
    )
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


def _retriever_settings(query_instruction: str) -> RetrievalSettings:
    return RetrievalSettings(
        retrieve_k_each=DEFAULT_RETRIEVAL.retrieve_k_each,
        final_k=DEFAULT_RETRIEVAL.final_k,
        rrf_k=DEFAULT_RETRIEVAL.rrf_k,
        query_instruction=query_instruction,
    )


def run_system(
    name: str,
    retriever: HybridRetriever,
    items: list[GoldenItem],
) -> tuple[dict[str, list[Chunk]], dict[str, Any]]:
    """Retrieve RETRIEVE_K chunks per item for one system.

    Returns (rankings by item_id, latency summary). Latencies are per query.
    """
    rankings: dict[str, list[Chunk]] = {}
    latencies: list[float] = []
    for item in items:
        started = time.perf_counter()
        if name == "bm25":
            ids = retriever.search_bm25(item.query, RETRIEVE_K)
            chunks = retriever.get_chunks(ids)
        elif name.startswith("vector"):
            ids = retriever.search_vector(item.query, RETRIEVE_K)
            chunks = retriever.get_chunks(ids)
        else:  # hybrid
            chunks = retriever.retrieve(item.query, final_k=RETRIEVE_K)
        latencies.append(time.perf_counter() - started)
        rankings[item.id] = chunks
    latency = {
        "median_s": statistics.median(latencies) if latencies else None,
        "p95_s": percentile(latencies, 0.95) if latencies else None,
        "n": len(latencies),
    }
    return rankings, latency


def top1_scores(
    retriever: HybridRetriever, items: list[GoldenItem]
) -> tuple[dict[str, float], dict[str, float]]:
    """Top-1 vector similarity and top-1 BM25 score per item id."""
    vector_scores: dict[str, float] = {}
    bm25_scores: dict[str, float] = {}
    for item in items:
        vector = retriever.search_vector_scored(item.query, 1)
        if vector:
            vector_scores[item.id] = vector[0][1]
        bm25 = retriever.search_bm25_scored(item.query, 1)
        if bm25:
            bm25_scores[item.id] = bm25[0][1]
    return vector_scores, bm25_scores


def _fmt_hit(row: dict[str, Any]) -> str:
    hits, total, fraction = row["hits"], row["total"], row["fraction"]
    return f"{hits}/{total} ({fraction * 100:.0f}%)"


def print_report(report: dict[str, Any]) -> None:
    n_pos = report["n_positives"]
    print(f"\nN positives = {n_pos}; a difference of one question = {100 / n_pos:.1f}%")
    print(f"split={report['split']}  set={report['set_path']}")
    print(f"corpus_version={report['corpus_version'][:12]}  model={report['embedding_model']}")

    print("\n== per-system metrics (positives only) ==")
    header_k = "  ".join(f"hit@{k}".ljust(10) for k in report["eval_ks"])
    print(f"{'system':<16} {header_k} {'MRR':>6}  {'median s':>9} {'p95 s':>8}")
    for name, system in report["systems"].items():
        hits = "  ".join(_fmt_hit(system["hit_at_k"][str(k)]).ljust(10) for k in report["eval_ks"])
        latency = system["latency"]
        print(
            f"{name:<16} {hits} {system['mrr']:>6.3f}  "
            f"{latency['median_s']:>9.4f} {latency['p95_s']:>8.4f}"
        )

    print("\n== recall@k (mean range coverage, positives only) ==")
    print(f"{'system':<16} " + "  ".join(f"r@{k}".ljust(8) for k in report["eval_ks"]))
    for name, system in report["systems"].items():
        recalls = "  ".join(
            f"{system['recall_at_k'][str(k)] * 100:.0f}%".ljust(8) for k in report["eval_ks"]
        )
        print(f"{name:<16} {recalls}")

    for name, system in report["systems"].items():
        missed = system["missed_at_6"]
        print(f"\n== {name}: {len(missed)} query(ies) missed at k=6 ==")
        for row in missed:
            print(f"  {row['id']}: {row['query']}")

    print("\n== top-1 score separability (gate pre-screen analysis) ==")
    for label, analysis in report["separability"].items():
        pos, neg = analysis["positives"], analysis["negatives"]
        print(f"  {label}:")
        print(
            f"    positives: min={pos['min']:.4f} median={pos['median']:.4f} max={pos['max']:.4f}"
            if pos["min"] is not None
            else "    positives: (none)"
        )
        print(
            f"    negatives: min={neg['min']:.4f} median={neg['median']:.4f} max={neg['max']:.4f}"
            if neg["min"] is not None
            else "    negatives: (none)"
        )
        if analysis["separable"] is None:
            print("    separable: n/a")
        else:
            print(f"    separable={analysis['separable']}  margin={analysis['margin']:.4f}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    split = check_test_protection(args.set, args.allow_test)

    items = load_golden(args.set)
    if not items:
        print(f"error: no queries in {args.set}", file=sys.stderr)
        return 2

    embedder = BgeEmbedder(DEFAULT_INGEST_SETTINGS.embed_model)
    base = HybridRetriever(args.index, args.chunks, embedder, _retriever_settings(""))

    systems: dict[str, HybridRetriever] = {"bm25": base, "vector": base, "hybrid": base}
    if args.query_instruction in ("on", "both"):
        instructed = HybridRetriever(
            args.index, args.chunks, embedder, _retriever_settings(BGE_QUERY_INSTRUCTION)
        )
        systems["vector+instr"] = instructed
        systems["hybrid+instr"] = instructed

    report_systems: dict[str, Any] = {}
    for name, retriever in systems.items():
        print(f"running system: {name} ...")
        rankings, latency = run_system(name, retriever, items)
        system_report = evaluate_system(name, rankings, items, ks=EVAL_KS)
        system_report["latency"] = latency
        report_systems[name] = system_report

    # Negatives-separability analysis uses the base (no instruction) retriever.
    vector_scores, bm25_scores = top1_scores(base, items)
    positives = [item for item in items if not item.is_negative]
    negatives = [item for item in items if item.is_negative]
    separability_report = {
        "vector": separability(
            [vector_scores[i.id] for i in positives if i.id in vector_scores],
            [vector_scores[i.id] for i in negatives if i.id in vector_scores],
        ),
        "bm25": separability(
            [bm25_scores[i.id] for i in positives if i.id in bm25_scores],
            [bm25_scores[i.id] for i in negatives if i.id in bm25_scores],
        ),
    }

    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "corpus_version": base.corpus_version,
        "embedding_model": embedder.name,
        "retrieval_settings": {
            "retrieve_k_each": DEFAULT_RETRIEVAL.retrieve_k_each,
            "final_k": DEFAULT_RETRIEVAL.final_k,
            "rrf_k": DEFAULT_RETRIEVAL.rrf_k,
            "query_instruction_mode": args.query_instruction,
        },
        "set_path": str(args.set),
        "split": split,
        "n_positives": len(positives),
        "n_negatives": len(negatives),
        "eval_ks": list(EVAL_KS),
        "systems": report_systems,
        "separability": separability_report,
        "top1_scores": {
            item.id: {
                "query": item.query,
                "vector": vector_scores.get(item.id),
                "bm25": bm25_scores.get(item.id),
                "is_negative": item.is_negative,
            }
            for item in items
        },
        "git_commit": _git_commit(),
    }

    print_report(report)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = EVAL_DIR / f"retrieval_{stamp}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
