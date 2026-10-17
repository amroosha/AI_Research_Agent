"""Validate golden-set files against the built index.

Usage:
    python scripts/validate_golden.py --dev tests/dev_set.yaml [--test tests/test_set.yaml]

Prints counts (positives, negatives, multi-range), a range-width histogram,
and query length stats. With --test it also errors if any normalized query
appears in both files (leakage). Exit non-zero on any error; warnings are
allowed.
"""

from __future__ import annotations

import argparse
import statistics
from collections import Counter
from pathlib import Path

from research_agent.config import CHUNKS_DIR, INDEX_DIR
from research_agent.golden import find_leakage, load_golden, validate_golden


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate golden-set files.")
    parser.add_argument("--dev", type=Path, required=True, help="dev golden YAML")
    parser.add_argument("--test", type=Path, default=None, help="test golden YAML")
    parser.add_argument("--index", type=Path, default=INDEX_DIR)
    parser.add_argument("--chunks", type=Path, default=CHUNKS_DIR)
    return parser


def _report(name: str, path: Path, items) -> None:
    positives = [item for item in items if not item.is_negative]
    negatives = [item for item in items if item.is_negative]
    multi = [item for item in positives if len(item.ranges) > 1]
    widths = [end - start + 1 for item in positives for start, end in item.ranges]
    lengths = [len(item.query) for item in items]

    print(f"== {name}: {path} ==")
    print(f"  items={len(items)}  positives={len(positives)}  negatives={len(negatives)}")
    print(f"  multi-range items={len(multi)}")
    if widths:
        histogram = Counter(widths)
        print("  range-width pages: count")
        for width in sorted(histogram):
            print(f"    {width:>3}: {histogram[width]}")
    if lengths:
        print(
            f"  query chars: min={min(lengths)} median={statistics.median(lengths):.0f} "
            f"max={max(lengths)}"
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    dev_items = load_golden(args.dev)
    _report("dev", args.dev, dev_items)
    dev_result = validate_golden(dev_items, args.index, args.chunks)

    test_result = None
    test_items = None
    if args.test is not None:
        test_items = load_golden(args.test)
        _report("test", args.test, test_items)
        test_result = validate_golden(test_items, args.index, args.chunks)

    errors: list[str] = list(dev_result.errors)
    warnings: list[str] = list(dev_result.warnings)
    if test_result is not None:
        errors.extend(test_result.errors)
        warnings.extend(test_result.warnings)
        for leaked in find_leakage(dev_items, test_items):
            errors.append(f"cross-file leakage: {leaked}")

    for message in warnings:
        print(f"WARNING: {message}")
    for message in errors:
        print(f"ERROR: {message}")

    if errors:
        print(f"VALIDATION FAILED: {len(errors)} error(s), {len(warnings)} warning(s)")
        return 1
    print(f"GOLDEN OK ({len(warnings)} warning(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
