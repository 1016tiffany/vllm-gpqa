#!/usr/bin/env python3
import os, re, sys, json, random
import pandas as pd

try:
    from datasets import load_dataset
except Exception:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "datasets"])
    from datasets import load_dataset

OUT_DIR = os.path.expanduser("~/LMUData")
OUT     = os.path.join(OUT_DIR, "gpqa.tsv")
os.makedirs(OUT_DIR, exist_ok=True)

CFG  = os.environ.get("GPQA_CFG", "gpqa_main")   # gpqa_main | gpqa_diamond | gpqa_extended | gpqa_experts
SPLT = os.environ.get("GPQA_SPLIT", "train")     # train | validation | test

def norm(x: str) -> str:
    x = "" if x is None else str(x)
    # strip LaTeX-ish wrappers and excess whitespace
    x = re.sub(r'\\boxed\{(.+?)\}', r'\1', x)
    x = re.sub(r'\s+', ' ', x).strip()
    return x

def letter(i: int) -> str:
    return "ABCDE"[i]

def main():
    print(f"[gpqa] loading Idavidrein/gpqa cfg={CFG} split={SPLT}", flush=True)
    ds = load_dataset("Idavidrein/gpqa", CFG, split=SPLT)

    rows = []
    for idx, ex in enumerate(ds):
        q  = norm(ex.get("Question"))
        ca = norm(ex.get("Correct Answer"))
        ia1 = norm(ex.get("Incorrect Answer 1"))
        ia2 = norm(ex.get("Incorrect Answer 2"))
        ia3 = norm(ex.get("Incorrect Answer 3"))

        # Skip if anything critical is missing
        if not q or not ca or not ia1 or not ia2 or not ia3:
            continue

        # Build options and deterministic shuffle per index
        options = [ca, ia1, ia2, ia3]  # 4-choice
        rng = random.Random(idx)       # reproducible
        perm = list(range(len(options)))
        rng.shuffle(perm)

        shuffled = [options[i] for i in perm]
        correct_pos = shuffled.index(ca)  # 0..3
        ans_letter = letter(correct_pos)  # A..D

        # Pad to 5 columns for VLMEval (E blank)
        A, B, C, D = shuffled
        E = ""

        rows.append(dict(
            index=idx, question=q, A=A, B=B, C=C, D=D, E=E,
            answer=ans_letter, split="test"
        ))

    df = pd.DataFrame(rows, columns=["index","question","A","B","C","D","E","answer","split"])
    df.to_csv(OUT, sep="\t", index=False, encoding="utf-8")
    print(f"[gpqa] wrote {len(df)} rows to {OUT}")
    if len(df):
        print("[gpqa] answer letters:", sorted(df["answer"].unique().tolist()))
        print("[gpqa] preview:\n", df.head(3)[["index","question","A","B","C","D","E","answer"]])

if __name__ == "__main__":
    main()
