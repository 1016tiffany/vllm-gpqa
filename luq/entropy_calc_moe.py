#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Entropy‑based expert ranking for MoE models.

For each MoE layer we collect the activations **after each expert MLP**,
cluster the tokens routed to that expert with k‑means, measure the entropy
of the cluster distribution, then sort all (layer, expert) pairs by entropy.
"""

import argparse, os, torch, numpy as np
from collections import Counter, defaultdict
from datasets import load_dataset
from tqdm import tqdm
from sklearn.cluster import KMeans
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoProcessor,
)

# -------------------------------------------------------------------------- #
#  Helper: entropy from k‑means clusters
# -------------------------------------------------------------------------- #
def calculate_kmeans_entropy(acts: torch.Tensor, n_clusters: int = 100) -> float:
    """Return entropy of token clusters for one expert."""
    flat = acts.reshape(-1, acts.shape[-1]).cpu().numpy()        # (n_tokens, d)
    if flat.shape[0] < n_clusters:                               # too few pts
        return 0.0

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(flat)

    counts = Counter(labels)
    probs  = np.array([c / len(labels) for c in counts.values()])
    return float(-(probs * np.log2(probs + 1e-12)).sum())


# -------------------------------------------------------------------------- #
#  Main
# -------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",  default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--n_clusters",  type=int, default=20)
    parser.add_argument("--output_dir",  default="./outputs")
    parser.add_argument("--sorted_file", default="sorted_experts.txt")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    sorted_path = os.path.join(args.output_dir, args.sorted_file)

    # ------------------------------------------------------------------ #
    #  Load model / processor
    # ------------------------------------------------------------------ #
    if "switch" in args.model_name.lower():
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=torch.float16,
            device_map=args.device, trust_remote_code=True
        )
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_name)
    breakpoint()

    # ------------------------------------------------------------------ #
    #  Register hooks – one per expert (supports Phi‑MoE, Mixtral, Switch…)
    # ------------------------------------------------------------------ #
    from collections import defaultdict

    expert_acts = defaultdict(list)      # key -> list[tensor]
    hooks       = []

    def make_hook(layer_idx: int, expert_idx: int):
        key = f"layer_{layer_idx}_expert_{expert_idx}"
        def _hook(_, __, output):
            # output shape: (n_tokens_for_this_expert, hidden_dim)
            if output.numel():                      # ignore empty calls
                expert_acts[key].append(output.detach().cpu())
        return _hook

    for l_idx, layer in enumerate(model.model.layers):

        # ---------- 1. Phi‑MoE style -----------------------------------
        experts = None
        if hasattr(layer, "block_sparse_moe") \
        and hasattr(layer.block_sparse_moe, "experts"):
            experts = layer.block_sparse_moe.experts                # ✅ Phi‑MoE

        # ---------- 2. Mixtral / DeepSpeed / Tutel style --------------
        elif hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
            experts = layer.mlp.experts                             

        # ---------- 3. Switch‑Transformer style -----------------------
        elif hasattr(layer, "experts"):
            experts = layer.experts

        # ---------- 4. No experts?  Skip this layer -------------------
        if experts is None:
            continue

        # Register one forward hook per expert
        for e_idx, ex in enumerate(experts):
            hooks.append(ex.register_forward_hook(make_hook(l_idx, e_idx)))

    print(f"‣ Registered {len(hooks)} expert hooks.")


    # ------------------------------------------------------------------ #
    #  Minimal dataset stub: text for language‑only, image+text otherwise
    # ------------------------------------------------------------------ #
    if processor.__class__.__name__.lower().startswith("qwen") and hasattr(processor, "image_processor"):
        ds = load_dataset("lmms-lab/textvqa", split="train[:200]")
    else:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test[:200]")

    # ------------------------------------------------------------------ #
    #  Forward passes
    # ------------------------------------------------------------------ #
    with torch.inference_mode():
        for ex in tqdm(ds.select(range(min(args.num_samples, len(ds))))):
            image = ex.get("image", None)
            prompt = ex.get("question") or ex.get("text")
            if prompt is None or not prompt.strip():
                continue
            inputs = (
                processor(images=image, text=prompt, return_tensors="pt", padding=True)
                if image is not None else
                processor(prompt, return_tensors="pt", padding=True)
            ).to(model.device)

            _ = model(**inputs, use_cache=False)

    # ------------------------------------------------------------------ #
    #  Remove hooks
    # ------------------------------------------------------------------ #
    for h in hooks:
        h.remove()

    # ------------------------------------------------------------------ #
    #  Entropy per expert
    # ------------------------------------------------------------------ #
    entropies = {}
    for key, chunks in tqdm(expert_acts.items(), desc="Entropy per expert"):
        acts = torch.cat(chunks, dim=0)      # (n_tokens_expert, d)
        entropies[key] = calculate_kmeans_entropy(acts, n_clusters=args.n_clusters)

    # ------------------------------------------------------------------ #
    #  Sort & persist
    # ------------------------------------------------------------------ #
    ordered = sorted(entropies.items(), key=lambda kv: kv[1])           # ascending
    with open(sorted_path, "w") as f:
        f.write("\n".join(f"{k}\t{v:.3f}" for k, v in ordered))

    print("\nExperts by ascending entropy:")
    for k, v in ordered:
        print(f"{k:>24s} : {v:.3f}")
    print(f"\nSaved to {sorted_path}")


if __name__ == "__main__":
    main()
