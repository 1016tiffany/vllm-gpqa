#!/usr/bin/env python3
# prep_gpqa_tsv.py
"""
Create a VLMEvalKit-compatible MCQ TSV for GPQA.

Usage (defaults):
  python prep_gpqa_tsv.py

Pick a specific split (one of: gpqa_main, gpqa_diamond, gpqa_extended, gpqa_experts):
  python prep_gpqa_tsv.py --config gpqa_diamond

Limit rows for a quick smoke test:
  python prep_gpqa_tsv.py --limit 50

Write to a custom path:
  python prep_gpqa_tsv.py --out /path/to/LMUData/gpqa.tsv

Notes:
- If the dataset is gated, set HUGGINGFACE_TOKEN in your environment or accept terms in the browser.
"""

import os
import re
import sys
import argparse
from typing import Optional, List
import pandas as pd

def letter_from_idx(i: Optional[int]) -> str:
    return {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}.get(i, "")

def idx_from_letter(s: str) -> Optional[int]:
    s = (s or "").strip().upper()
    return {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}.get(s)

def clean_str(x) -> str:
    if x is None:
        return ""
    return str(x).strip()

def detect_options(ex) -> List[str]:
    # Try common field names for choices
    opts = (
        ex.get("options")
        or ex.get("choices")
        or ex.get("answers")  # some mirrors misuse this name
        or []
    )
    opts = list(opts)
    # Normalize to exactly 5 (A..E) for TSV; empty strings if fewer
    return (opts + [""] * 5)[:5]

def detect_question(ex) -> str:
    return clean_str(ex.get("question") or ex.get("problem") or ex.get("prompt") or "")

def parse_gold_label(ex, opts: List[str]) -> str:
    """Return answer letter A-E or '' if unknown."""
    gold = ex.get("answer") or ex.get("label") or ex.get("target") or ex.get("correct_answer")
    # 1) If it's already an index
    if isinstance(gold, int):
        return letter_from_idx(gold)

    # 2) If it's a string, try letter then exact text match
    if isinstance(gold, str):
        g = re.sub(r"\\boxed\{(.+?)\}", r"\1", gold)  # strip LaTeX \boxed{}
        g = re.sub(r'^[\s\.:;()\[\]{}"]+|[\s\.:;()\[\]{}"]+$', '', g).strip()
        # letter A-E?
        idx = idx_from_letter(g)
        if idx is not None:
            return letter_from_idx(idx)
        # match against options (case-insensitive)
        low = [clean_str(x).lower() for x in opts]
        if g.lower() in low:
            return letter_from_idx(low.index(g.lower()))

    # Unknown → blank (VLMEvalKit will skip scoring if key is missing)
    return ""

def main():
    ap = argparse.ArgumentParser(description="Prepare GPQA TSV for VLMEvalKit")
    ap.add_argument("--config", default="gpqa_main",
                    choices=["gpqa_main", "gpqa_diamond", "gpqa_extended", "gpqa_experts"],
                    help="Which GPQA subset to use")
    ap.add_argument("--split", default="train", help="HF split name (usually 'train' for GPQA)")
    ap.add_argument("--out", default=os.path.expanduser("~/LMUData/gpqa.tsv"),
                    help="Output TSV path (default: ~/LMUData/gpqa.tsv)")
    ap.add_argument("--limit", type=int, default=None, help="Optional row limit for a quick test")
    args = ap.parse_args()

    try:
        from datasets import load_dataset  # lazy import to keep envs clean
    except Exception as e:
        print("Please install 'datasets' and 'pandas' first: pip install datasets pandas", file=sys.stderr)
        raise

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    print(f"[GPQA] Loading Idavidrein/gpqa config={args.config} split={args.split} ...")
    ds = load_dataset("Idavidrein/gpqa", args.config, split=args.split)

    rows = []
    total = len(ds)
    take = args.limit if (args.limit is not None and args.limit > 0) else total
    print(f"[GPQA] Converting {take}/{total} rows → {args.out}")

    for i, ex in enumerate(ds):
        if i >= take:
            break
        q = detect_question(ex)
        opts = detect_options(ex)
        A, B, C, D, E = opts
        ans_letter = parse_gold_label(ex, opts)

        rows.append({
            "index": i,
            "question": q,
            "A": A, "B": B, "C": C, "D": D, "E": E,
            "answer": ans_letter,
            "split": "test"  # eval-only for VLMEvalKit usage
        })

    df = pd.DataFrame(rows, columns=["index", "question", "A", "B", "C", "D", "E", "answer", "split"])
    df.to_csv(args.out, sep="\t", index=False, encoding="utf-8")
    print(f"[GPQA] Wrote {len(df)} rows to {args.out}")
    print("[GPQA] Done. Now run your VLMEvalKit command with --data gpqa")

if __name__ == "__main__":
    main()
