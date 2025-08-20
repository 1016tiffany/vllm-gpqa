#!/usr/bin/env python
"""
mix_precision_moe.py  ─  v2
Keep the first `top_k` experts (lowest-entropy) in 2-bit precision and
all remaining experts in 4-bit precision.

Usage example
-------------
python mix_precision_moe.py \
       --model_2bit_dir  outputs/phi_mini_q2 \
       --model_4bit_dir  outputs/phi_mini_q4 \
       --entropy_file    outputs/sorted_experts_phi_mini_moe_v2.txt \
       --top_k           256 \
       --output_dir      outputs/phi_mini_mixed_2of4
"""
import argparse, json, os, re, shutil, torch
from typing import List, Tuple

_EXPERT_RE = re.compile(r"layer_(\d+)_expert_(\d+)")


# ──────────────────────────── helper functions ──────────────────────────────
def load_state(path: str):
    return torch.load(path, map_location="cpu")


def parse_entropy_file(path: str, top_k: int) -> List[Tuple[int, int]]:
    """Return list of the first `top_k` (layer, expert) tuples."""
    experts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            token = line.split()[0]                         # e.g. layer_10_expert_5
            m = _EXPERT_RE.match(token)
            if not m:
                raise ValueError(f"Malformed expert id: {token}")
            experts.append(tuple(map(int, m.groups())))
            if len(experts) == top_k:
                break
    if len(experts) < top_k:
        raise ValueError(f"Entropy list has only {len(experts)} rows (<{top_k}).")
    return experts


def dot2under(name: str) -> str:
    return name.replace(".", "_")


def belongs_to(expert_key: str, expert_set: set) -> bool:
    """Does an underscore key correspond to one of the (layer, expert) pairs?"""
    m = re.search(r"layers_(\d+)_.*experts_(\d+)_", expert_key)
    return m and (int(m.group(1)), int(m.group(2))) in expert_set


# ─────────────────────────────────── main ────────────────────────────────────
def main():
    ap = argparse.ArgumentParser("Mix 2-bit + 4-bit MoE checkpoints (2-bit first)")
    ap.add_argument("--model_2bit_dir", required=True)
    ap.add_argument("--model_4bit_dir", required=True)
    ap.add_argument("--entropy_file", required=True)
    ap.add_argument("--top_k", type=int, default=256,
                    help="Number of *lowest-entropy* experts to stay 2-bit")
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # load checkpoints & configs
    sd2 = load_state(os.path.join(args.model_2bit_dir, "quantised.pt"))
    sd4 = load_state(os.path.join(args.model_4bit_dir, "quantised.pt"))
    cfg2 = json.load(open(os.path.join(args.model_2bit_dir, "quant_config.json")))
    cfg4 = json.load(open(os.path.join(args.model_4bit_dir, "quant_config.json")))

    # figure out which experts stay 2-bit
    low_entropy_experts = set(parse_entropy_file(args.entropy_file, args.top_k))
    print(f"▶ {len(low_entropy_experts)} experts → 2-bit, "
          f"{512-len(low_entropy_experts)} experts → 4-bit")

    # start from 4-bit state-dict, then overwrite selected low-entropy experts with 2-bit
    mixed_sd = sd4.copy()
    replaced = 0
    for k, tensor2 in sd2.items():
        if belongs_to(k, low_entropy_experts):
            mixed_sd[k] = tensor2
            replaced += 1
    print(f"   Replaced {replaced} tensors with 2-bit versions.")

    # merge metadata: start from 4-bit config, overwrite selected experts with 2-bit meta
    mixed_cfg = cfg4
    for module, meta2 in cfg2["layers"].items():
        if belongs_to(dot2under(module), low_entropy_experts):
            mixed_cfg["layers"][module] = meta2
    mixed_cfg["method"]  = "mixed_2bit_4bit"
    mixed_cfg["comment"] = (f"First {args.top_k} entropy-ranked experts in 2-bit, "
                            "remainder in 4-bit.")

    # save
    torch.save(mixed_sd, os.path.join(args.output_dir, "quantised.pt"))
    with open(os.path.join(args.output_dir, "quant_config.json"), "w") as f:
        json.dump(mixed_cfg, f, indent=2)

    # tokenizer & model config (identical between 2-bit/4-bit) – copy from 2-bit dir
    for fn in ("tokenizer.json", "tokenizer_config.json",
               "special_tokens_map.json", "config.json"):
        src = os.path.join(args.model_2bit_dir, fn)
        if os.path.exists(src):
            shutil.copy(src, args.output_dir)

    print(f"✅ Mixed checkpoint written to: {args.output_dir}")


if __name__ == "__main__":
    main()
