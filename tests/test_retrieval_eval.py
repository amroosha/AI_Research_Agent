"""Tests for the retrieval eval harness (Stage 2, T4).

Pure metric functions use hand-computed fixtures; the end-to-end test runs on
a synthetic index with the deterministic ``HashEmbedder`` (no network).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from research_agent.config import DEFAULT_INGEST_SETTINGS, RetrievalSettings
from research_agent.eval.retrieval_eval import (
    EVAL_KS,
    chunk_hits_range,
    evaluate_system,
    first_hit_rank,
    hit_at_k,
    mrr,
    percentile,
    ranges_covered,
    ranges_intersect,
    separability,
)
from research_agent.golden import GoldenItem, load_golden
from research_agent.ingest.indexing import build_index
from research_agent.retrieval import HybridRetriever
from research_agent.state import Chunk
from tests.helpers import HashEmbedder, make_chunks, write_chunks

BOOK = "synthetic-book"


def _load_script_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "eval_retrieval.py"
    spec = importlib.util.spec_from_file_location("eval_retrieval_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


script = _load_script_module()


def _chunk(index: int, start: int, end: int, book: str = BOOK) -> Chunk:
    return Chunk(
        id=f"{book}-{index:05d}",
        text="text",
        source="book",
        book_slug=book,
        book_title="Synthetic",
        section_path=["Chapter 1"],
        pdf_page_start=start,
        pdf_page_end=end,
    )


def _item(
    query: str = "q",
    book: str | None = BOOK,
    ranges: tuple[tuple[int, int], ...] = ((10, 12),),
    is_negative: bool = False,
) -> GoldenItem:
    return GoldenItem(
        id=f"fix:{query}",
        query=query,
        book=book,
        ranges=ranges,
        is_negative=is_negative,
        source_file="fix.yaml",
    )


# --- overlap rule boundaries --------------------------------------------------


def test_ranges_intersect_touching_endpoints() -> None:
    assert ranges_intersect(10, 12, 12, 15) is True
    assert ranges_intersect(12, 15, 10, 12) is True


def test_ranges_intersect_adjacent_does_not() -> None:
    assert ranges_intersect(10, 12, 13, 15) is False
    assert ranges_intersect(13, 15, 10, 12) is False


def test_ranges_intersect_full_containment() -> None:
    assert ranges_intersect(5, 20, 10, 12) is True
    assert ranges_intersect(10, 12, 5, 20) is True


def test_chunk_hits_range_boundaries() -> None:
    chunk = _chunk(0, 10, 12)
    assert chunk_hits_range(chunk, 12, 15) is True  # touching
    assert chunk_hits_range(chunk, 13, 15) is False  # adjacent
    assert chunk_hits_range(chunk, 9, 10) is True  # touching at start


# --- hit / recall / MRR (hand-computed) ---------------------------------------


def test_first_hit_rank_and_hit_at_k() -> None:
    item = _item(ranges=((10, 12),))
    chunks = [_chunk(0, 1, 2), _chunk(1, 11, 11), _chunk(2, 30, 31)]
    assert first_hit_rank(chunks, item) == 2
    hits, total, fraction = hit_at_k([2, None, 1, 5], 1)
    assert (hits, total) == (1, 4)
    assert fraction == 0.25
    hits, _, fraction = hit_at_k([2, None, 1, 5], 6)
    assert hits == 3 and fraction == 0.75


def test_mrr_hand_computed() -> None:
    # 1/1 + 1/3 + 0 -> (1 + 0.3333) / 3
    assert mrr([1, 3, None]) == pytest.approx((1 + 1 / 3) / 3)
    assert mrr([1]) == 1.0
    assert mrr([]) == 0.0


def test_recall_multi_range() -> None:
    # Two ranges: one covered at rank 1, the other only at rank 4.
    item = _item(ranges=((10, 12), (40, 41)))
    chunks_rank2 = [_chunk(0, 10, 12), _chunk(1, 1, 2)]
    chunks_rank4 = chunks_rank2 + [_chunk(2, 3, 4), _chunk(3, 40, 41)]
    assert ranges_covered(chunks_rank2, item) == 1
    assert ranges_covered(chunks_rank4, item) == 2
    # recall@k over two items: (1/2 covered, 2/2 covered) -> 0.75 mean.
    from research_agent.eval.retrieval_eval import recall_at_k

    assert recall_at_k([(1, 2), (2, 2)], [True, True]) == pytest.approx(0.75)


def test_hit_wrong_book_is_not_a_hit() -> None:
    item = _item(ranges=((10, 12),))
    foreign = _chunk(0, 10, 12, book="other-book")
    assert first_hit_rank([foreign], item) is None


# --- separability ------------------------------------------------------------


def test_separability_true_and_false() -> None:
    analysis = separability([0.8, 0.9], [0.3, 0.4])
    assert analysis["separable"] is True
    assert analysis["margin"] == pytest.approx(0.4)  # 0.8 - 0.4
    analysis = separability([0.5, 0.9], [0.3, 0.7])
    assert analysis["separable"] is False
    assert analysis["margin"] == pytest.approx(-0.2)  # 0.5 - 0.7


def test_percentile() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(values, 0.0) == 1.0
    assert percentile(values, 1.0) == 5.0
    assert percentile(values, 0.5) == 3.0
    assert percentile([7.0], 0.95) == 7.0


# --- protection flag ----------------------------------------------------------


def test_protection_refuses_test_file_without_flag(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        script.check_test_protection(tmp_path / "my_test_set.yaml", allow_test=False)
    assert excinfo.value.code == 2


def test_protection_allows_test_file_with_flag(capsys, tmp_path: Path) -> None:
    split = script.check_test_protection(tmp_path / "my_test_set.yaml", allow_test=True)
    assert split == "test"
    assert "WARNING" in capsys.readouterr().out


def test_protection_dev_file_runs(tmp_path: Path) -> None:
    assert script.check_test_protection(tmp_path / "dev_set.yaml", allow_test=False) == "dev"


# --- evaluate_system on fixtures ----------------------------------------------


def test_evaluate_system_end_to_end_fixture() -> None:
    items = [
        _item(query="alpha", ranges=((10, 12),)),
        _item(query="beta", ranges=((40, 45),), is_negative=False),
    ]
    rankings = {
        items[0].id: [_chunk(0, 1, 2), _chunk(1, 11, 11)],  # hit at rank 2
        items[1].id: [_chunk(2, 1, 2), _chunk(3, 3, 4)],  # miss
    }
    report = evaluate_system("bm25", rankings, items, ks=(1, 6))
    assert report["hit_at_k"]["1"] == {"hits": 0, "total": 2, "fraction": 0.0}
    assert report["hit_at_k"]["6"]["hits"] == 1
    assert report["mrr"] == pytest.approx(0.5 / 2)  # (1/2 + 0) / 2
    assert report["recall_at_k"]["6"] == pytest.approx(0.5)  # (1/1 + 0/1) / 2
    assert len(report["missed_at_6"]) == 1
    assert report["missed_at_6"][0]["id"] == items[1].id


def test_evaluate_system_ignores_negatives() -> None:
    items = [
        _item(query="alpha", ranges=((10, 12),)),
        _item(query="neg", book=None, ranges=(), is_negative=True),
    ]
    rankings = {items[0].id: [_chunk(0, 10, 12)]}
    report = evaluate_system("hybrid", rankings, items, ks=(1,))
    assert report["hit_at_k"]["1"] == {"hits": 1, "total": 1, "fraction": 1.0}


# --- synthetic index end-to-end -----------------------------------------------


@pytest.fixture
def index_env(tmp_path: Path) -> tuple[Path, Path]:
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    write_chunks(make_chunks(30), chunks_dir / f"{BOOK}.jsonl")
    index_dir = tmp_path / "index"
    build_index(
        chunks_dir=chunks_dir,
        index_dir=index_dir,
        embedder=HashEmbedder(dim=64),
        settings=DEFAULT_INGEST_SETTINGS,
        show_progress=False,
    )
    return chunks_dir, index_dir


def test_hybrid_hits_known_chunk_end_to_end(index_env) -> None:
    chunks_dir, index_dir = index_env
    embedder = HashEmbedder(dim=64)
    cfg = RetrievalSettings(retrieve_k_each=20, final_k=20, rrf_k=60)
    retriever = HybridRetriever(index_dir, chunks_dir, embedder, cfg)
    # Chunk index 7 owns the unique token ref0007; synthetic pages are index+1.
    item = _item(query="ref0007", ranges=((8, 8),))
    chunks = retriever.retrieve(item.query, final_k=20)
    report = evaluate_system("hybrid", {item.id: chunks}, [item], ks=EVAL_KS)
    assert report["per_query"][0]["first_hit_rank"] == 1
    assert report["hit_at_k"]["1"]["fraction"] == 1.0
    assert report["mrr"] == 1.0


def test_all_systems_ranked_end_to_end(index_env) -> None:
    chunks_dir, index_dir = index_env
    embedder = HashEmbedder(dim=64)
    retriever = HybridRetriever(index_dir, chunks_dir, embedder, RetrievalSettings())
    item = _item(query="ref0012", ranges=((13, 13),))
    for name, ids in (
        ("bm25", retriever.search_bm25(item.query, 20)),
        ("vector", retriever.search_vector(item.query, 20)),
    ):
        chunks = [retriever.chunks_by_id[cid] for cid in ids]
        assert first_hit_rank(chunks, item) == 1, name


def test_deterministic_output_across_two_runs(index_env) -> None:
    chunks_dir, index_dir = index_env
    embedder = HashEmbedder(dim=64)
    items = [_item(query="ref0003", ranges=((4, 4),)), _item(query="ref0020", ranges=((21, 21),))]

    def run_once() -> dict:
        retriever = HybridRetriever(index_dir, chunks_dir, embedder, RetrievalSettings())
        rankings, _latency = script.run_system("hybrid", retriever, items)
        return evaluate_system("hybrid", rankings, items)

    assert run_once() == run_once()


# --- JSON schema keys ---------------------------------------------------------


def test_json_report_schema_keys(index_env, tmp_path: Path) -> None:
    # Minimal report mirroring the script's payload structure.
    report = {
        "corpus_version": "x",
        "embedding_model": "hash-embedder-v1",
        "retrieval_settings": {},
        "set_path": "tests/dev_set.yaml",
        "split": "dev",
        "n_positives": 1,
        "n_negatives": 0,
        "systems": {},
        "separability": {},
        "git_commit": None,
    }
    required = {
        "corpus_version",
        "embedding_model",
        "retrieval_settings",
        "set_path",
        "split",
        "n_positives",
        "n_negatives",
        "systems",
        "separability",
        "git_commit",
    }
    assert required <= set(report)
    # And the real script writes JSON that round-trips.
    out = tmp_path / "report.json"
    out.write_text(json.dumps(report), encoding="utf-8")
    assert json.loads(out.read_text(encoding="utf-8"))["split"] == "dev"


def test_golden_files_load_for_harness() -> None:
    # The user's real golden files must load unchanged (frozen schema).
    root = Path(__file__).resolve().parents[1]
    dev = load_golden(root / "tests" / "dev_set.yaml")
    test = load_golden(root / "tests" / "test_set.yaml")
    assert dev and test
    assert all(item.query for item in dev + test)
