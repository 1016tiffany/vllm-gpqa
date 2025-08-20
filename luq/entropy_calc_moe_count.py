#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Entropy-based expert ranking for MoE models.

For each MoE layer we collect the activations **after each expert MLP**,
cluster the tokens routed to that expert with k-means, measure the entropy
of the cluster distribution, then sort all (layer, expert) pairs by entropy.
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
#  Helper: entropy from k-means clusters
# -------------------------------------------------------------------------- #
def calculate_kmeans_entropy(acts: torch.Tensor, n_clusters: int = 100) -> float:
    """Return entropy of token clusters for one expert."""
    if acts is None or acts.numel() == 0:
        return 0.0

    flat = acts.reshape(-1, acts.shape[-1]).cpu().numpy()  # (n_tokens, d)
    if flat.shape[0] < n_clusters:                         # too few points
        return 0.0

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(flat)

    counts = Counter(labels)
    probs  = np.array([c / len(labels) for c in counts.values()], dtype=np.float64)
    return float(-(probs * np.log2(probs + 1e-12)).sum())


def _parse_key(key: str):
    # key format: "layer_{L}_expert_{E}"
    try:
        left, right = key.split("_expert_")
        L = int(left.replace("layer_", ""))
        E = int(right)
        return L, E
    except Exception:
        return -1, -1


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
    parser.add_argument("--counts_file", default="expert_activation_counts.tsv",
                        help="Tab-separated file with per-expert activation counts and entropy.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    sorted_path = os.path.join(args.output_dir, args.sorted_file)
    counts_path = os.path.join(args.output_dir, args.counts_file)

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

    # ------------------------------------------------------------------ #
    #  Register hooks – one per expert (supports Phi-MoE, Mixtral, Switch…)
    # ------------------------------------------------------------------ #
    expert_acts = defaultdict(list)   # key -> list[tensor]
    hooks       = []

    def make_hook(layer_idx: int, expert_idx: int):
        key = f"layer_{layer_idx}_expert_{expert_idx}"
        def _hook(_, __, output):
            # Some modules return a tuple; normalize to a Tensor.
            out = output[0] if isinstance(output, (tuple, list)) and len(output) > 0 else output
            if torch.is_tensor(out) and out.numel():       # ignore empty calls
                # Expected shape: (n_tokens_for_this_expert, hidden_dim)
                expert_acts[key].append(out.detach().cpu())
        return _hook

    # Try to find experts in each layer (handles common MoE layouts)
    for l_idx, layer in enumerate(getattr(getattr(model, "model", model), "layers", [])):
        experts = None

        # ---------- 1. Phi-MoE style -----------------------------------
        if hasattr(layer, "block_sparse_moe") and hasattr(layer.block_sparse_moe, "experts"):
            experts = layer.block_sparse_moe.experts

        # ---------- 2. Mixtral / DeepSpeed / Tutel style ---------------
        elif hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
            experts = layer.mlp.experts

        # ---------- 3. Switch-Transformer style ------------------------
        elif hasattr(layer, "experts"):
            experts = layer.experts

        if experts is None:
            continue

        # Register one forward hook per expert
        for e_idx, ex in enumerate(experts):
            hooks.append(ex.register_forward_hook(make_hook(l_idx, e_idx)))

    print(f"‣ Registered {len(hooks)} expert hooks.")

    # ------------------------------------------------------------------ #
    #  Minimal dataset stub: text for language-only, image+text otherwise
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
            if prompt is None or not str(prompt).strip():
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
    #  Entropy + counts per expert
    # ------------------------------------------------------------------ #
    entropies = {}
    stats = {}  # key -> dict(n_tokens, used_tokens, used_for_entropy, entropy)

    for key, chunks in tqdm(expert_acts.items(), desc="Entropy per expert"):
        acts = torch.cat(chunks, dim=0) if len(chunks) else torch.empty(0, 0)
        n_tokens = int(acts.shape[0]) if acts.numel() else 0

        entropy = calculate_kmeans_entropy(acts, n_clusters=args.n_clusters)
        entropies[key] = entropy

        used_for_entropy = int(n_tokens >= args.n_clusters)
        used_tokens = n_tokens if used_for_entropy else 0

        stats[key] = {
            "n_tokens": n_tokens,
            "used_tokens": used_tokens,
            "used_for_entropy": used_for_entropy,
            "entropy": entropy,
        }

    # ------------------------------------------------------------------ #
    #  Sort & persist (original sorted file unchanged)
    # ------------------------------------------------------------------ #
    ordered = sorted(entropies.items(), key=lambda kv: kv[1])  # ascending
    with open(sorted_path, "w") as f:
        f.write("\n".join(f"{k}\t{v:.3f}" for k, v in ordered))

    # Also write counts/usage details
    # Columns: key, layer, expert, total_activations, used_for_entropy, used_count, entropy
    with open(counts_path, "w") as f:
        f.write("key\tlayer\texpert\ttotal_activations\tused_for_entropy\tused_count\tentropy\n")
        # Sort by (layer, expert) for readability
        for key in sorted(stats.keys(), key=lambda k: _parse_key(k)):
            L, E = _parse_key(key)
            row = stats[key]
            f.write(
                f"{key}\t{L}\t{E}\t{row['n_tokens']}\t{row['used_for_entropy']}"
                f"\t{row['used_tokens']}\t{row['entropy']:.3f}\n"
            )

    print("\nExperts by ascending entropy:")
    for k, v in ordered:
        print(f"{k:>24s} : {v:.3f}")
    print(f"\nSaved entropy ranking to {sorted_path}")
    print(f"Saved activation counts to {counts_path}")


if __name__ == "__main__":
    main()
