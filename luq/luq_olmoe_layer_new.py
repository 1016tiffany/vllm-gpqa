#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mix HQQ-quantized OLMoE *layers* by entropy ranking.

Swap modes:
- experts_only (default):        only swap MoE experts inside chosen layers to 2-bit
- full_layer:                    swap the entire layer module to 2-bit (legacy behavior)
- experts_attn:                  swap experts + the layer's attention block
- experts_router:                swap experts + the layer's MoE router (a.k.a. gate)
- experts_attn_router:           swap experts + attention + router

Examples:
  # Experts only (baseline behavior)
  python luq_olmoe_layer.py \
    --model4 outputs/olmoe-hqq-4bit \
    --model2 outputs/olmoe-hqq-2bit \
    --order_layers outputs/sorted_layers_olmoe.txt \
    --num2 8 \
    --swap experts_only \
    --out outputs/olmoe-luq-layer-8_expertsOnly \
    --dtype bfloat16

  # Experts + Attention
  python luq_olmoe_layer.py ... --swap experts_attn ...

  # Experts + Router
  python luq_olmoe_layer.py ... --swap experts_router ...

  # Experts + Attention + Router
  python luq_olmoe_layer.py ... --swap experts_attn_router ...
"""

import os, re, copy, argparse, json
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# ------------------------------------------------------------
# Regex helpers
# ------------------------------------------------------------
EXPERT_PATH_RE = re.compile(r"\.experts\.(\d+)$")
LAYER_IN_PATH_RE = re.compile(r"\.layers\.(\d+)\.")

# ------------------------------------------------------------
# Generic tree utilities
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# HQQ inspection (best-effort)
# ------------------------------------------------------------
def find_first_hqq_bits(mod: nn.Module):
    """Best-effort: find an HQQ leaf and read meta.nbits."""
    for m in mod.modules():
        meta = getattr(m, "meta", None)
        if isinstance(meta, dict) and "nbits" in meta:
            nbits = meta["nbits"]
            if hasattr(nbits, "item"):
                try:
                    nbits = int(nbits.item())
                except Exception:
                    pass
            if isinstance(nbits, int):
                return nbits
    return None

def count_bits_distribution_by_layer(model: nn.Module, layer_paths):
    counts = {}
    for p in layer_paths:
        layer = get_module_by_path(model, p)
        bits = find_first_hqq_bits(layer)
        counts[bits] = counts.get(bits, 0) + 1
    return counts

def count_bits_distribution_experts(model: nn.Module, expert_paths):
    counts = {}
    for p in expert_paths:
        exp = get_module_by_path(model, p)
        bits = find_first_hqq_bits(exp)
        counts[bits] = counts.get(bits, 0) + 1
    return counts

# ------------------------------------------------------------
# Discovery: experts, attention, router
# ------------------------------------------------------------
def collect_expert_map(model: nn.Module):
    """
    Returns dict[(layer_idx, expert_idx)] -> path like 'model.layers.7.moe.experts.25'
    Only entries that end exactly at '.experts.<idx>'.
    """
    mapping = {}
    for name, _ in model.named_modules():
        mE = EXPERT_PATH_RE.search(name)
        if not mE:
            continue
        expert_idx = int(mE.group(1))
        mL = LAYER_IN_PATH_RE.search(name)
        if not mL:
            continue
        layer_idx = int(mL.group(1))
        if name.endswith(f".experts.{expert_idx}"):
            mapping[(layer_idx, expert_idx)] = name
    return mapping

def autodetect_layers_list_path(model: nn.Module) -> str:
    candidates = ["model.layers", "transformer.layers", "layers"]
    for cand in candidates:
        try:
            mod = get_module_by_path(model, cand)
            if isinstance(mod, nn.ModuleList) and len(mod) > 0:
                return cand
        except Exception:
            pass
    # Fallback: scan
    best_name, best_len = None, -1
    for name, mod in model.named_modules():
        if name.endswith(".layers") and isinstance(mod, nn.ModuleList):
            if len(mod) > best_len:
                best_name, best_len = name, len(mod)
    if best_name is not None:
        return best_name
    raise RuntimeError("Could not locate a ModuleList of layers (e.g., 'model.layers'). Pass --layers_path.")

def build_layer_paths(model: nn.Module, layers_path: str):
    layers_ml = get_module_by_path(model, layers_path)
    if not isinstance(layers_ml, nn.ModuleList):
        raise RuntimeError(f"'{layers_path}' is not an nn.ModuleList.")
    n = len(layers_ml)
    paths = [f"{layers_path}.{i}" for i in range(n)]
    return paths, n

def _find_direct_child_path(parent_mod: nn.Module, parent_path: str, name_contains=("attn","attention")):
    """Return the path to a direct child whose name matches."""
    for child_name, child in parent_mod.named_children():
        cn = child_name.lower()
        if any(tok in cn for tok in name_contains):
            return f"{parent_path}.{child_name}"
    return None

def _find_shallowest_named_module_under(model: nn.Module, base_path: str, want_suffixes):
    """
    Scan named_modules() for modules whose path starts with base_path + '.'
    and whose *last* component matches one of want_suffixes. Return the shortest.
    """
    prefix = base_path + "."
    best = None
    best_depth = 1e9
    for name, mod in model.named_modules():
        if not name.startswith(prefix):
            continue
        last = name.split(".")[-1].lower()
        if last in want_suffixes:
            depth = name.count(".")
            if depth < best_depth:
                best, best_depth = name, depth
    return best

def collect_attention_map(model: nn.Module, layer_paths):
    """
    Returns dict[layer_idx] -> attention_root_path
    Heuristics:
      1) direct child named 'self_attn'/'attn'/'attention'
      2) otherwise shallowest descendant ending with those names
      3) otherwise any direct child whose class name contains 'Attention'
    """
    attn_names = {"self_attn","attn","attention"}
    result = {}
    for L, lp in enumerate(layer_paths):
        layer = get_module_by_path(model, lp)
        # 1) direct child
        p = _find_direct_child_path(layer, lp, name_contains=tuple(attn_names))
        if p is None:
            # 2) shallowest descendant
            p = _find_shallowest_named_module_under(model, lp, attn_names)
        if p is None:
            # 3) class-name fallback on direct children
            for child_name, child in layer.named_children():
                if "attention" in child.__class__.__name__.lower():
                    p = f"{lp}.{child_name}"
                    break
        if p:
            result[L] = p
    return result

def collect_router_map(model: nn.Module, layer_paths):
    """
    Returns dict[layer_idx] -> router_path
    Heuristics (common in MoE):
      - prefer '...layers.N.moe.router' or '...layers.N.moe.gate'
      - fallback to direct child 'router'/'gate'
      - fallback to shallowest descendant named 'router'/'gate'
    """
    prefer_under_moe = ("moe.router","moe.gate")
    router_names = {"router","gate","router_layer","gating","gate_proj"}  # broad
    result = {}
    for L, lp in enumerate(layer_paths):
        # Prefer under 'moe'
        for cand in prefer_under_moe:
            try:
                p = f"{lp}.{cand}"
                _ = get_module_by_path(model, p)
                result[L] = p
                break
            except Exception:
                pass
        if L in result:
            continue
        # Direct child
        p = _find_direct_child_path(get_module_by_path(model, lp), lp, name_contains=tuple(router_names))
        if p is None:
            # Shallowest descendant
            p = _find_shallowest_named_module_under(model, lp, router_names)
        if p:
            result[L] = p
    return result

# ------------------------------------------------------------
# I/O helpers
# ------------------------------------------------------------
def parse_layers_ordering_file(path: str):
    """One line like: 0,2,1,3,4,... -> list[int]"""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    line = content.splitlines()[0].strip()
    tokens = [t for t in re.split(r"[,\s]+", line) if t != ""]
    order = [int(t) for t in tokens]
    return order

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model4", required=True, help="Path to 4-bit HQQ model dir")
    ap.add_argument("--model2", required=True, help="Path to 2-bit HQQ model dir")
    ap.add_argument("--order_layers", required=True, help="Layer entropy ordering file (CSV)")
    ap.add_argument("--num2", type=int, default=8, help="How many layers (from the ordering head) to swap to 2-bit")
    ap.add_argument("--out", required=True, help="Output dir for mixed model")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16","float16"], help="Compute dtype during load")
    ap.add_argument("--layers_path", default=None, help="Override path to ModuleList of layers (e.g. 'model.layers')")
    ap.add_argument("--swap",
                    default="experts_only",
                    choices=["experts_only","full_layer","experts_attn","experts_router","experts_attn_router"],
                    help="What to swap to 2-bit for the selected layers")
    ap.add_argument("--dry_run", action="store_true", help="Only print what would be replaced")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Parse ordering
    ordering = parse_layers_ordering_file(args.order_layers)
    if args.num2 > len(ordering):
        raise ValueError(f"--num2={args.num2} exceeds ordering size {len(ordering)}")
    two_bit_layers = ordering[:args.num2]

    # Concrete dtype + GPU map
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

    bad = [L for L in two_bit_layers if L < 0 or L >= n_layers]
    if bad:
        raise ValueError(f"Ordering contains invalid layer indices (0..{n_layers-1} expected): {bad[:5]}")

    # Info before
    dist_layers_before = count_bits_distribution_by_layer(model4, layer_paths_4)
    print(f"4-bit model before mixing — layer FIRST-leaf bits distribution: {dist_layers_before}")

    # Build expert maps
    map4 = collect_expert_map(model4)
    map2 = collect_expert_map(model2)
    if not map4 or not map2:
        raise RuntimeError("Could not discover experts in one or both models (maps empty).")
    if set(map4.keys()) != set(map2.keys()):
        missing_4 = set(map2.keys()) - set(map4.keys())
        missing_2 = set(map4.keys()) - set(map2.keys())
        raise RuntimeError(
            "Mismatch in (layer,expert) keys between models.\n"
            f"Missing in 4-bit: {sorted(list(missing_4))[:10]} ...\n"
            f"Missing in 2-bit: {sorted(list(missing_2))[:10]} ..."
        )
    # Count expert bits pre-swap
    paths_all_experts = [map4[k] for k in sorted(map4.keys())]
    dist_experts_before = count_bits_distribution_experts(model4, paths_all_experts)
    print(f"4-bit model before mixing — EXPERT bits distribution: {dist_experts_before}")

    # Discover attention + router roots (per layer)
    attn4 = collect_attention_map(model4, layer_paths_4)
    attn2 = collect_attention_map(model2, layer_paths_2)

    router4 = collect_router_map(model4, layer_paths_4)
    router2 = collect_router_map(model2, layer_paths_2)

    want_attn = args.swap in ("experts_attn", "experts_attn_router")
    want_router = args.swap in ("experts_router", "experts_attn_router")

    print(f"Selected layers to act on ({args.swap}): {sorted(two_bit_layers)}")
    if args.dry_run:
        preview = {"experts": [], "attn": [], "router": []}
        # sample first ~40 experts to be swapped
        for L in sorted(two_bit_layers):
            count_in_L = 0
            for (l,e), p in sorted(map4.items()):
                if l != L:
                    continue
                preview["experts"].append((L, e))
                count_in_L += 1
                if len(preview["experts"]) >= 40:
                    break
            if len(preview["experts"]) >= 40:
                break
        for L in sorted(two_bit_layers):
            if want_attn and (L in attn4) and (L in attn2):
                preview["attn"].append((L, attn4[L].split(".")[-1]))
            if want_router and (L in router4) and (L in router2):
                preview["router"].append((L, router4[L].split(".")[-1]))
        print("Dry-run preview:")
        print("  experts (first ~40):", preview["experts"])
        if want_attn:  print("  attn roots:", preview["attn"])
        if want_router: print("  router roots:", preview["router"])
        return

    # --------------------------------------------------------
    # Perform replacements
    # --------------------------------------------------------
    replaced_experts = 0
    replaced_attn = 0
    replaced_router = 0

    if args.swap == "full_layer":
        for L in two_bit_layers:
            path4 = layer_paths_4[L]
            path2 = layer_paths_2[L]
            parent4, child4 = resolve_parent_and_key(model4, path4)
            layer2 = get_module_by_path(model2, path2)
            set_child_module(parent4, child4, copy.deepcopy(layer2))
        print(f"Replaced {len(two_bit_layers)} full layer modules.")
    else:
        # Always swap experts in selected layers
        for L in two_bit_layers:
            for (l_idx, e_idx), path4 in map4.items():
                if l_idx != L:
                    continue
                path2 = map2[(l_idx, e_idx)]
                parent4, child4 = resolve_parent_and_key(model4, path4)
                expert2 = get_module_by_path(model2, path2)
                set_child_module(parent4, child4, copy.deepcopy(expert2))
                replaced_experts += 1

        # Optionally swap attention
        if want_attn:
            for L in two_bit_layers:
                if (L in attn4) and (L in attn2):
                    p4 = attn4[L]; p2 = attn2[L]
                    parent4, child4 = resolve_parent_and_key(model4, p4)
                    attn2_mod = get_module_by_path(model2, p2)
                    set_child_module(parent4, child4, copy.deepcopy(attn2_mod))
                    replaced_attn += 1
                else:
                    print(f"WARNING: Could not find matching attention for layer {L}; skipping.")

        # Optionally swap router
        if want_router:
            for L in two_bit_layers:
                if (L in router4) and (L in router2):
                    p4 = router4[L]; p2 = router2[L]
                    parent4, child4 = resolve_parent_and_key(model4, p4)
                    router2_mod = get_module_by_path(model2, p2)
                    set_child_module(parent4, child4, copy.deepcopy(router2_mod))
                    replaced_router += 1
                else:
                    print(f"WARNING: Could not find matching router for layer {L}; skipping.")

        print(f"Replaced {replaced_experts} experts across {len(two_bit_layers)} layers.")
        if want_attn:  print(f"Replaced {replaced_attn} attention modules.")
        if want_router: print(f"Replaced {replaced_router} routers.")

    # Verify after
    dist_layers_after = count_bits_distribution_by_layer(model4, layer_paths_4)
    print(f"Mixed model after swap — layer FIRST-leaf bits distribution: {dist_layers_after}")

    dist_experts_after = count_bits_distribution_experts(model4, [map4[k] for k in sorted(map4.keys())])
    print(f"Mixed model after swap — EXPERT bits distribution: {dist_experts_after}")

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
        "swap_mode": args.swap,
        "layer_firstleaf_bits_before": dist_layers_before,
        "layer_firstleaf_bits_after": dist_layers_after,
        "experts_bits_before": dist_experts_before,
        "experts_bits_after": dist_experts_after,
        "notes": "experts_* modes replace experts and optionally attention/router; full_layer replaces entire blocks.",
        "counts": {
            "replaced_experts": replaced_experts,
            "replaced_attention": replaced_attn,
            "replaced_router": replaced_router,
        },
    }
    with open(os.path.join(args.out, "mix_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("Done.")

if __name__ == "__main__":
    main()
