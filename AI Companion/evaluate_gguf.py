# evaluate_gguf.py
# ------------------------------------------------------------------
# USAGE
#   pip install llama-cpp-python==0.2.28  # or the latest wheel w/ CUDA
#   python evaluate_gguf.py \
#          --excel   data/eval/…xlsx \
#          --gguf    models/SmolLM3-f16.gguf \
#          --tokenizer_dir models/SmolLM3-tokenizer \
#          --max_examples 100
# ------------------------------------------------------------------

import argparse, re, os, time
import pandas as pd
from tqdm.auto import tqdm

# ------------- NEW: llama-cpp ------------------
from llama_cpp import Llama                       # pure-C++ backend
# ------------------------------------------------

# ---------- OPTIONAL: keep a tokenizer ----------
#    We only need it for the chat template; if your
#    model doesn’t ship a HF tokenizer you can
#    write a tiny custom formatter instead.
from transformers import AutoTokenizer
# ------------------------------------------------


# ------------ helper to split the conversation -------------
def parse_conversation(raw_text: str):
    pat = re.compile(
        r'<<<(USER|ASSISTANT)>>>\s*:\s*'
        r'(.*?)'
        r'(?=(?:<<<(?:USER|ASSISTANT)>>>\s*:|$))',
        re.DOTALL,
    )
    return [(r.lower(), c.strip()) for r, c in pat.findall(raw_text)]


# ------------ ONE turn of generation ------------------------
def generate_one(system, raw_conv, tokenizer, llama, max_new_tokens):
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    for role, content in parse_conversation(raw_conv):
        msgs.append({"role": role, "content": content})

    # build a prompt string (HF chat template is convenient)
    prompt = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )

    # llama_cpp wants a plain string, returns dict like OpenAI
    out = llama(
        prompt,
        max_tokens=max_new_tokens,
        stop=[],
        echo=False,
    )
    return out["choices"][0]["text"].lstrip()   # strip leading space


def main():
    p = argparse.ArgumentParser("Evaluate GGUF model with llama.cpp")
    p.add_argument("--excel", type=str, default="data/eval/Function_Calling_Multi_1_Step_Master_Eval (NMC) (Flattened)(Sheet1).xlsx")
    p.add_argument("--output", type=str, default="predictions.xlsx")
    # path to *.gguf
    p.add_argument("--gguf", type=str, required=True,
                   help="Path to the .gguf model file")
    # tokenizer dir (vocab.json / merges.txt) – can be the same repo you
    # converted from, or any compatible tokenizer.
    p.add_argument("--tokenizer_dir", type=str, required=True)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--max_examples", type=int)
    p.add_argument("--ctx", type=int, default=4096,
                   help="Context window for llama.cpp")
    p.add_argument("--gpu_layers", type=int, default=35,
                   help="How many layers to keep on GPU (-1 = all on GPU, 0 = CPU)")
    args = p.parse_args()

    # -------- load spreadsheet ------------------
    df = pd.read_excel(args.excel, engine="openpyxl")
    if args.max_examples:
        df = df.head(args.max_examples)

    # -------- load tokenizer --------------------
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    tokenizer.pad_token = tokenizer.eos_token   # safety

    # -------- load GGUF with llama-cpp ----------
    llama = Llama(
        model_path=args.gguf,
        n_ctx=args.ctx,
        n_gpu_layers=args.gpu_layers,  # set 0 if you have no CUDA build
        logits_all=False,
        embedding=False,
    )

    # ------------- predict ----------------------
    preds, times = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating"):
        system = row.get("System", "") or ""
        user   = row.get("User",   "") or ""

        try:
            t0 = time.time()
            gen = generate_one(system, user, tokenizer, llama,
                               max_new_tokens=args.max_new_tokens)
            dt = time.time() - t0
        except Exception as e:
            gen, dt = f"[ERROR: {e}]", 0.0

        preds.append(gen)
        times.append(dt)

    # ------------- save & score -----------------
    df["Predicted"] = preds
    df["Time"]      = times
    df.to_excel(args.output, index=False)

    if "Assistant" in df.columns:        # ground-truth column
        acc = (df["Predicted"] == df["Assistant"].astype(str)).mean()
        print(f"Accuracy: {acc:.2%} "
              f"({df['Predicted'].eq(df['Assistant']).sum()}/{len(df)})")

    print(f"Done – wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
