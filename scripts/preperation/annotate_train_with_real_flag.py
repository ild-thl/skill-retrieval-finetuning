#!/usr/bin/env python3
"""
Annotate a training JSONL file with a `meta.source` tag set to 'real' or 'synthetic'.

Usage (example):
  python annotate_train_with_real_flag.py \
    --input data/eval_split_semantic/train_dataset_remaining.jsonl \
    --combined data/combinedEsco.json \
    --output data/eval_split_semantic/train_dataset_remaining_annotated.jsonl \
    --fuzzy 0.95 --sample 100

This script loads the `query` field from `combinedEsco.json` and compares each record's
`query` in the input JSONL file. If an exact or fuzzy match is found the
record will receive `meta.source = 'real'`, otherwise `meta.source = 'synthetic'`.

The script keeps dependencies minimal and uses Python stdlib only (difflib for fuzzy matching).
"""

from __future__ import annotations

import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable, List, Tuple


def load_combined_queries(path: Path) -> List[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    queries = []
    for item in data:
        q = item.get("query")
        if q:
            queries.append(q)
    return queries


def best_fuzzy_ratio(q: str, candidates: Iterable[str]) -> float:
    best = 0.0
    for c in candidates:
        # short-circuit exact
        if q == c:
            return 1.0
        r = SequenceMatcher(None, q, c).ratio()
        if r > best:
            best = r
            # early exit for near-perfect matches
            if best >= 0.9999:
                break
    return best


def annotate_file(
    input_path: Path,
    combined_path: Path,
    output_path: Path,
    fuzzy_threshold: float = 1.0,
    sample: int | None = None,
    verbose: bool = True,
) -> Tuple[int, int, int]:
    combined_queries = load_combined_queries(combined_path)
    combined_set = set(combined_queries)

    total = 0
    real = 0
    synthetic = 0

    # If sample is set, just process first N lines and print stats (dry-run)
    with input_path.open("r", encoding="utf-8") as inp:
        if output_path:
            out_f = output_path.open("w", encoding="utf-8")
        else:
            out_f = None

        try:
            for line in inp:
                if sample is not None and total >= sample:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    # skip malformed lines
                    continue

                total += 1
                q = rec.get("query") or rec.get("text") or ""

                is_real = False
                # exact q match
                if q in combined_set:
                    is_real = True
                else:
                    # if fuzzy threshold < 1.0, allow approximate matches
                    if fuzzy_threshold < 1.0 and combined_queries:
                        ratio = best_fuzzy_ratio(q, combined_queries)
                        if ratio >= fuzzy_threshold:
                            is_real = True

                src = "real" if is_real else "synthetic"
                # attach into meta
                meta = rec.get("meta")
                if meta is None:
                    meta = {}
                    rec["meta"] = meta
                meta["source"] = src

                if is_real:
                    real += 1
                else:
                    synthetic += 1

                if out_f:
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        finally:
            if out_f:
                out_f.close()

    if verbose:
        print(f"Processed: {total}, real: {real}, synthetic: {synthetic}")

    return total, real, synthetic


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input JSONL training file")
    p.add_argument(
        "--combined", required=True, help="Path to combinedEsco.json (real queries)"
    )
    p.add_argument("--output", required=True, help="Output annotated JSONL path")
    p.add_argument(
        "--fuzzy",
        type=float,
        default=0.95,
        help="Fuzzy match threshold (0..1). 1.0 requires exact match. Default=0.95",
    )
    p.add_argument(
        "--sample",
        type=int,
        default=None,
        help="If set, only process the first N lines (dry-run).",
    )
    p.add_argument("--quiet", action="store_true", help="Reduce output")

    args = p.parse_args()

    input_path = Path(args.input)
    combined_path = Path(args.combined)
    output_path = Path(args.output)

    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")
    if not combined_path.exists():
        raise SystemExit(f"Combined file not found: {combined_path}")

    total, real, synthetic = annotate_file(
        input_path,
        combined_path,
        output_path,
        fuzzy_threshold=args.fuzzy,
        sample=args.sample,
        verbose=not args.quiet,
    )

    print("Done.")


if __name__ == "__main__":
    main()
