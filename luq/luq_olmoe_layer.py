#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mix HQQ-quantized OLMoE *layers* by entropy ranking.

- Take a fully 4-bit HQQ model dir and a fully 2-bit HQQ model dir
- Read a layer entropy ordering file (comma-separated layer indices)
- Replace the first N layers in that ordering with the 2-bit versions
- Save the mixed checkpoint

Works with allenai/OLMoE-1B-7B-0924 quantized via Transformers+HQQ.

Example:
  python luq_olmoe_layer.py \
    --model4 outputs/olmoe-hqq-4bit \
    --model2 outputs/olmoe-hqq-2bit \
    --order_layers outputs/sorted_layers_olmoe.txt \
    --num2 8 \
    --out outputs/olmoe-luq-layer \
    --dtype bfloat16
"""

import os, re, copy, argparse, json
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# Utilities reused from your expert-mixer -------------------------------

def get_module_by_path(root: nn.Module, path: str) -> nn.Module:
    m = root
    for part in path.split("."):
        if part.isdigit():
            m = m[int(part)]
        else:
            m = getattr(m, part)
    return m

def resolve_parent_and_key(root: nn.Module, path: str):
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]

def set_child_module(parent: nn.Module, key: str, new_mod: nn.Module):
    if isinstance(parent, nn.ModuleList) and key.isdigit():
        parent[int(key)] = new_mod
    else:
        setattr(parent, key, new_mod)

def find_first_hqq_bits(mod: nn.Module):
    """Best-effort: find an HQQ leaf and read meta.nbits."""
    for m in mod.modules():
        meta = getattr(m, "meta", None)
        if isinstance(meta, dict) and "nbits" in meta:
            nbits = meta["nbits"]
            if hasattr(nbits, "item"):
                try: nbits = int(nbits.item())
                except Exception: pass
            if isinstance(nbits, int):
                return nbits
    return None

# New helpers for *layer*-level mixing ---------------------------------

def parse_layers_ordering_file(path: str):
    """
    Expect one line like: 0,2,1,3,4,13,9,10,5,11,6,14,12,15,8,7
    Returns a list of ints in rank order (best/lowest-entropy first).
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    # allow accidental whitespace/newlines
    line = content.splitlines()[0].strip()
    # split on comma *or* whitespace
    tokens = [t for t in re.split(r"[,\s]+", line) if t != ""]
    order = [int(t) for t in tokens]
    return order

def autodetect_layers_list_path(model: nn.Module) -> str:
    """
    Try common paths for the transformer block list.
    Returns a dotted path like 'model.layers' (ModuleList).
    """
    candidates = ["model.layers", "transformer.layers", "layers"]
    for cand in candidates:
        try:
            mod = get_module_by_path(model, cand)
            if isinstance(mod, nn.ModuleList) and len(mod) > 0:
                return cand
        except Exception:
            pass

    # Fallback: scan named_modules for something ending with ".layers"
    best_name, best_len = None, -1
    for name, mod in model.named_modules():
        if name.endswith(".layers") and isinstance(mod, nn.ModuleList):
            if len(mod) > best_len:
                best_name, best_len = name, len(mod)
    if best_name is not None:
        return best_name

    raise RuntimeError("Could not locate a ModuleList of layers (e.g., 'model.layers'). "
                       "Pass --layers_path to override.")

def build_layer_paths(model: nn.Module, layers_path: str):
    """
    Return (paths, n_layers) where paths[i] = f'{layers_path}.{i}' exists.
    """
    layers_ml = get_module_by_path(model, layers_path)
    if not isinstance(layers_ml, nn.ModuleList):
        raise RuntimeError(f"'{layers_path}' is not an nn.ModuleList.")
    n = len(layers_ml)
    paths = [f"{layers_path}.{i}" for i in range(n)]
    return paths, n

def count_bits_distribution_by_layer(model: nn.Module, layer_paths):
    counts = {}
    for p in layer_paths:
        layer = get_module_by_path(model, p)
        bits = find_first_hqq_bits(layer)
        counts[bits] = counts.get(bits, 0) + 1
    return counts

# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model4", required=True, help="Path to 4-bit HQQ model dir")
    ap.add_argument("--model2", required=True, help="Path to 2-bit HQQ model dir")
    ap.add_argument("--order_layers", required=True, help="Layer entropy ordering file (CSV)")
    ap.add_argument("--num2", type=int, default=8, help="How many layers (from the ordering head) to swap to 2-bit")
    ap.add_argument("--out", required=True, help="Output dir for mixed model")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16","float16"], help="Compute dtype during load")
    ap.add_argument("--layers_path", default=None, help="Override path to the ModuleList of layers (e.g. 'model.layers')")
    ap.add_argument("--dry_run", action="store_true", help="Only print what would be replaced")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Parse ordering
    ordering = parse_layers_ordering_file(args.order_layers)
    if args.num2 > len(ordering):
        raise ValueError(f"--num2={args.num2} exceeds ordering size {len(ordering)}")
    two_bit_layers = ordering[:args.num2]

    # Concrete dtype + GPU map (avoid 'auto' dtype bug)
    load_kwargs = dict(
        trust_remote_code=True,
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=False,
        attn_implementation="eager",
    )
    load_kwargs["torch_dtype"] = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    print(f"Loading 4-bit model from: {args.model4}")
    model4 = AutoModelForCausalLM.from_pretrained(args.model4, **load_kwargs)
    print(f"Loading 2-bit model from: {args.model2}")
    model2 = AutoModelForCausalLM.from_pretrained(args.model2, **load_kwargs)

    # Locate layers list
    layers_path4 = args.layers_path or autodetect_layers_list_path(model4)
    layers_path2 = args.layers_path or autodetect_layers_list_path(model2)
    if layers_path4 != layers_path2:
        print(f"NOTE: detected different layer paths: 4-bit='{layers_path4}', 2-bit='{layers_path2}'")

    layer_paths_4, n4 = build_layer_paths(model4, layers_path4)
    layer_paths_2, n2 = build_layer_paths(model2, layers_path2)
    if n4 != n2:
        raise RuntimeError(f"Mismatch in number of layers: 4-bit has {n4}, 2-bit has {n2}")
    n_layers = n4

    # Sanity: ensure two_bit_layers indices are valid
    bad = [L for L in two_bit_layers if L < 0 or L >= n_layers]
    if bad:
        raise ValueError(f"Ordering contains invalid layer indices (0..{n_layers-1} expected): {bad[:5]}")

    # Bits distribution before
    dist_before = count_bits_distribution_by_layer(model4, layer_paths_4)
    print(f"4-bit model before mixing — layer bits distribution: {dist_before}")

    print(f"Will replace {len(two_bit_layers)} layers with 2-bit modules: {sorted(two_bit_layers)}")
    if args.dry_run:
        return

    # Perform replacements
    for L in two_bit_layers:
        path4 = layer_paths_4[L]
        path2 = layer_paths_2[L]
        parent4, child4 = resolve_parent_and_key(model4, path4)
        layer2 = get_module_by_path(model2, path2)
        set_child_module(parent4, child4, copy.deepcopy(layer2))

    # Verify after
    dist_after = count_bits_distribution_by_layer(model4, layer_paths_4)
    print(f"Mixed model after swap — layer bits distribution: {dist_after}")
    if (2 not in dist_after) or (4 not in dist_after):
        print("WARNING: Could not detect both 2-bit and 4-bit across layers. "
              "HQQ meta.nbits schema may differ; this is informational only.")

    # Save
    print(f"Saving mixed model to: {args.out}")
    model4.save_pretrained(args.out, safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(args.model4, trust_remote_code=True)
    tok.save_pretrained(args.out)

    manifest = {
        "source_4bit": os.path.abspath(args.model4),
        "source_2bit": os.path.abspath(args.model2),
        "order_layers_file": os.path.abspath(args.order_layers),
        "num2_layers": args.num2,
        "two_bit_layers": sorted(two_bit_layers),
        "dtype_load": args.dtype,
        "n_layers": n_layers,
        "layers_path_used_4bit": layers_path4,
        "layers_path_used_2bit": layers_path2,
        "dist_before": dist_before,
        "dist_after": dist_after,
        "note": "Mixed layers: first N by entropy replaced with 2-bit; remainder left 4-bit.",
    }
    with open(os.path.join(args.out, "mix_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("Done.")

if __name__ == "__main__":
    main()
