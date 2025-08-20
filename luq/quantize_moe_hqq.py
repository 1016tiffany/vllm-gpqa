#!/usr/bin/env python
# Quantize microsoft/Phi-mini-MoE-instruct to 4-bit and 2-bit with HQQ.

import argparse, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, HqqConfig

# MODEL_ID = "microsoft/Phi-mini-MoE-instruct"
MODEL_ID = "allenai/OLMoE-1B-7B-0924"
def quantize_and_save(bits: int, out_dir: str, group_size: int = 64):
    os.makedirs(out_dir, exist_ok=True)
    qcfg = HqqConfig(
        nbits=bits, group_size=group_size,
        quant_zero=True, quant_scale=True, axis=0,
        compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map={"": "cuda:0"},
        low_cpu_mem_usage=False, attn_implementation="eager",
        quantization_config=qcfg,
    )
    # Sanity check: make sure HQQ saved a real torch.dtype, not a string
    bad = []
    for name, mod in model.named_modules():
        meta = getattr(mod, "meta", None)
        if isinstance(meta, dict) and "compute_dtype" in meta:
            if not isinstance(meta["compute_dtype"], torch.dtype):
                bad.append((name, meta["compute_dtype"], type(meta["compute_dtype"])))
    if bad:
        raise RuntimeError(f"HQQ compute_dtype not torch.dtype: {bad[:3]} ...")

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    # ✅ VERIFY: find one HQQ module and print its compute_dtype *type and value*
    for name, mod in model.named_modules():
        cd = getattr(mod, "compute_dtype", None)
        meta = getattr(mod, "meta", None)
        if meta and isinstance(meta, dict) and "compute_dtype" in meta:
            val = meta["compute_dtype"]
            print(f"[VERIFY] {name}.meta.compute_dtype =", val, type(val))
            break

    model.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)


def main():
    ap = argparse.ArgumentParser(description="HQQ-quantize Phi-mini-MoE-instruct to 4-bit and 2-bit.")
    ap.add_argument("--group-size", type=int, default=64, help="HQQ group size (typical: 64 or 128).")
    ap.add_argument("--out-root", type=str, default="outputs", help="Root directory to save outputs.")
    ap.add_argument("--bits", type=str, default="2,4", help="Comma-separated list of bit widths to produce, e.g. '2,4' or '4'.")
    args = ap.parse_args()

    bits_list = [int(x.strip()) for x in args.bits.split(",") if x.strip()]
    for b in bits_list:
        out_dir = os.path.join(args.out_root, f"olmoe-hqq-{b}bit")
        quantize_and_save(b, out_dir, group_size=args.group_size)

if __name__ == "__main__":
    main()
