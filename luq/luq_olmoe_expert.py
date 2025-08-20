#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mix HQQ-quantized OLMoE experts by entropy ranking:
- Keep first K experts (by entropy ordering file) at 4-bit
- Replace the rest with 2-bit experts
- Save a new "mixed" checkpoint

Works with allenai/OLMoE-1B-7B-0924 quantized via Transformers+HQQ.

Usage:
  python mix_olmoe_hqq_by_entropy.py \
    --model4 outputs/olmoe-hqq-4bit \
    --model2 outputs/olmoe-hqq-2bit \
    --order  outputs/sorted_experts_olmoe.txt \
    --keep   512 \
    --out    outputs/olmoe-hqq-mix-4b512-2b512
"""

import os, re, copy, argparse, json
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

EXPERT_PATH_RE = re.compile(r"\.experts\.(\d+)$")
LAYER_IN_PATH_RE = re.compile(r"\.layers\.(\d+)\.")

def parse_ordering_file(path, keep):
    """Return (keep4_set, ordered_pairs) where pairs are (layer, expert)."""
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Format: layer_7_expert_25<TAB>score
            # We only need the left token.
            left = line.split()[0]
            m = re.match(r"layer_(\d+)_expert_(\d+)", left)
            if not m:
                raise ValueError(f"Bad line in ordering file: {line}")
            L = int(m.group(1)); E = int(m.group(2))
            pairs.append((L, E))
    if len(pairs) < keep:
        raise ValueError(f"Ordering file has only {len(pairs)} entries; need at least keep={keep}.")
    keep4 = set(pairs[:keep])
    return keep4, pairs

def collect_expert_map(model):
    """
    Discover expert container modules via .named_modules():
    map[(layer_idx, expert_idx)] -> module_path like 'model.layers.7.moe.experts.25'
    We only select entries that end exactly at '.experts.<idx>'.
    """
    mapping = {}
    for name, mod in model.named_modules():
        # must end with ".experts.<digit>"
        mE = EXPERT_PATH_RE.search(name)
        if not mE:
            continue
        expert_idx = int(mE.group(1))
        mL = LAYER_IN_PATH_RE.search(name)
        if not mL:
            # Not in a standard layers.<L>. path; ignore
            continue
        layer_idx = int(mL.group(1))
        # ensure the name is exactly ending in experts.<idx>
        # (and not deeper like experts.<idx>.something)
        if name.endswith(f".experts.{expert_idx}"):
            mapping[(layer_idx, expert_idx)] = name
    return mapping

def resolve_parent_and_key(root: nn.Module, path: str):
    """
    Given a dotted path like 'model.layers.7.moe.experts.25',
    return (parent_module, child_key) so that parent[child_key] is the expert.
    child_key is either a string attr name or a numeric string for ModuleList.
    """
    parts = path.split(".")
    # walk to parent
    parent = root
    for i, part in enumerate(parts[:-1]):
        nxt = parts[i+1] if i+1 < len(parts) else None
        if part.isdigit():
            # indexing a ModuleList by number
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    child = parts[-1]
    return parent, child

def get_module_by_path(root: nn.Module, path: str) -> nn.Module:
    m = root
    for part in path.split("."):
        if part.isdigit():
            m = m[int(part)]
        else:
            m = getattr(m, part)
    return m

def set_child_module(parent: nn.Module, key: str, new_mod: nn.Module):
    """
    Replace child module inside parent for both Module and ModuleList cases.
    """
    if isinstance(parent, nn.ModuleList) and key.isdigit():
        parent[int(key)] = new_mod
    else:
        setattr(parent, key, new_mod)

def find_first_hqq_bits(expert: nn.Module):
    """
    Best-effort: find an HQQ leaf inside the expert and read its 'meta.nbits'.
    Returns int or None.
    """
    for m in expert.modules():
        meta = getattr(m, "meta", None)
        if isinstance(meta, dict) and "nbits" in meta:
            nbits = meta["nbits"]
            # could be tensor/int/str depending on HQQ version; normalize:
            if hasattr(nbits, "item"):
                try:
                    nbits = int(nbits.item())
                except Exception:
                    pass
            if isinstance(nbits, (int,)):
                return nbits
    return None

def count_bits_distribution(model, expert_paths):
    counts = {}
    for path in expert_paths:
        exp = get_module_by_path(model, path)
        bits = find_first_hqq_bits(exp)
        counts[bits] = counts.get(bits, 0) + 1
    return counts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model4", required=True, help="Path to 4-bit HQQ model dir")
    ap.add_argument("--model2", required=True, help="Path to 2-bit HQQ model dir")
    ap.add_argument("--order", required=True, help="Entropy ordering file (layer_X_expert_Y ...)")
    ap.add_argument("--keep", type=int, default=512, help="How many experts to keep as 4-bit")
    ap.add_argument("--out", required=True, help="Output dir for mixed model")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16","float16"], help="Compute dtype during load")
    ap.add_argument("--dry_run", action="store_true", help="Only print what would be replaced")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Parse ordering
    keep4, order_pairs = parse_ordering_file(args.order, keep=args.keep)

    # Use a concrete dtype to avoid the 'auto' dtype bug on load.
    load_kwargs = dict(
        trust_remote_code=True, 
        device_map={"": "cuda:0"}, 
        low_cpu_mem_usage=False,
        attn_implementation="eager",
    )
    if args.dtype == "bfloat16":
        load_kwargs["torch_dtype"] = torch.bfloat16
    else:
        load_kwargs["torch_dtype"] = torch.float16

    print(f"Loading 4-bit model from: {args.model4}")
    model4 = AutoModelForCausalLM.from_pretrained(args.model4, **load_kwargs)
    print(f"Loading 2-bit model from: {args.model2}")
    model2 = AutoModelForCausalLM.from_pretrained(args.model2, **load_kwargs)

    # Build expert maps
    map4 = collect_expert_map(model4)
    map2 = collect_expert_map(model2)

    if not map4 or not map2:
        raise RuntimeError("Could not discover experts in one or both models (maps empty).")

    if set(map4.keys()) != set(map2.keys()):
        missing_4 = set(map2.keys()) - set(map4.keys())
        missing_2 = set(map4.keys()) - set(map2.keys())
        raise RuntimeError(
            f"Mismatch in (layer,expert) keys between models.\n"
            f"Missing in 4-bit: {sorted(list(missing_4))[:10]} ...\n"
            f"Missing in 2-bit: {sorted(list(missing_2))[:10]} ..."
        )

    num_experts = len(map4)
    print(f"Discovered {num_experts} experts total.")
    # Optional sanity: many OLMoE configs use 64 experts x 16 layers = 1024
    # but we don't hard-require exactly 1024.

    # Verify ordering file only references known experts
    unknown = [pair for pair in order_pairs if pair not in map4]
    if unknown:
        raise RuntimeError(f"Ordering file contains unknown experts (first 5): {unknown[:5]}")

    # Report current bit distribution (before swap)
    paths_all = [map4[k] for k in sorted(map4.keys())]
    dist_before = count_bits_distribution(model4, paths_all)
    print(f"4-bit model before mixing — expert bits distribution: {dist_before}")

    # Build replacement list: those NOT in keep4 become 2-bit
    to_replace = [pair for pair in map4.keys() if pair not in keep4]
    print(f"Will replace {len(to_replace)} experts with 2-bit modules "
          f"(keeping {len(keep4)} as 4-bit).")

    if args.dry_run:
        preview = sorted(to_replace)[:20]
        print("Dry-run preview of first replacements:", preview)
        return

    # Perform replacements
    for (L, E) in to_replace:
        path4 = map4[(L, E)]
        path2 = map2[(L, E)]
        parent4, child4 = resolve_parent_and_key(model4, path4)
        expert2 = get_module_by_path(model2, path2)

        # Deepcopy to avoid parameter sharing between models
        expert2_copy = copy.deepcopy(expert2)

        # Replace in 4-bit model
        set_child_module(parent4, child4, expert2_copy)

    # Verify after-swap bits distribution
    dist_after = count_bits_distribution(model4, paths_all)
    print(f"Mixed model after swap — expert bits distribution: {dist_after}")

    # Extra assertion: expect both 4 and 2 in the distribution
    if (2 not in dist_after) or (4 not in dist_after):
        print("WARNING: Could not detect both 2-bit and 4-bit in the mixed model's experts."
              " (This may happen if HQQ meta.nbits schema changed.)")

    # Save mixed model + tokenizer
    print(f"Saving mixed model to: {args.out}")
    model4.save_pretrained(args.out, safe_serialization=True)
    # Copy tokenizer (either dir is fine; they should be identical)
    tok = AutoTokenizer.from_pretrained(args.model4, trust_remote_code=True)
    tok.save_pretrained(args.out)

    # Save a tiny manifest for reproducibility
    manifest = {
        "source_4bit": os.path.abspath(args.model4),
        "source_2bit": os.path.abspath(args.model2),
        "order_file": os.path.abspath(args.order),
        "keep_4bit": args.keep,
        "dtype_load": args.dtype,
        "dist_before": dist_before,
        "dist_after": dist_after,
        "note": "Mixed experts: first K by entropy kept 4-bit; others replaced with 2-bit."
    }
    with open(os.path.join(args.out, "mix_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("Done.")
    
if __name__ == "__main__":
    main()
