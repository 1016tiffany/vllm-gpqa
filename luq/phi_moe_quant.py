#!/usr/bin/env python
"""
True N-bit (2- or 4-bit) groupwise weight quantiser for HF Causal-LMs.
Packed storage, vectorised – runs in seconds instead of minutes.
"""
import argparse, json, os, time, torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ───────────────────────────────── quantisation core ─────────────────────────
def _pack_bits(q: torch.Tensor, nbits: int) -> torch.Tensor:
    """Vectorised bit-packing: q is uint8 flat tensor of values < 2**nbits."""
    vals_per_byte = 8 // nbits                   # 4 for 2-bit, 2 for 4-bit
    # pad length so it's divisible
    pad = (-len(q)) % vals_per_byte
    if pad:
        q = torch.cat([q, torch.zeros(pad, dtype=q.dtype)])
    q = q.view(-1, vals_per_byte)                # (n_bytes, vals_per_byte)
    shifts = (torch.arange(vals_per_byte) * nbits).to(q.device)
    packed = (q.to(torch.int16) << shifts).sum(dim=1).to(torch.uint8)
    return packed

def quantise_linear(weight: torch.Tensor, nbits: int = 2, group_size: int = 8):
    """Return packed weights, scale, zero, original_shape."""
    out_f, in_f = weight.shape
    pad_o = (group_size - out_f % group_size) % group_size
    pad_i = (group_size -  in_f % group_size) % group_size
    if pad_o or pad_i:
        weight = torch.nn.functional.pad(weight, (0, pad_i, 0, pad_o))
    g_out, g_in = weight.shape[0] // group_size, weight.shape[1] // group_size

    w = weight.view(g_out, group_size, g_in, group_size)\
              .permute(0, 2, 1, 3).contiguous()          # (Go,Gi,gs,gs)
    flat = w.view(-1, group_size * group_size)           # (n_grp, gs²)

    w_min, w_max = flat.min(1, keepdim=True).values, flat.max(1, keepdim=True).values
    levels = (1 << nbits) - 1                            # 3 for 2-bit, 15 for 4-bit
    scale = (w_max - w_min) / levels
    scale[scale == 0] = 1.0
    zero = w_min
    q = torch.round((flat - zero) / scale).clamp_(0, levels).to(torch.uint8)

    packed = _pack_bits(q.flatten(), nbits)
    return packed, scale.squeeze(1).half(), zero.squeeze(1).half(), (out_f, in_f)

@torch.no_grad()
def quantise_model(model, nbits: int, group_size: int):
    q_state, q_cfg = {}, {}
    for name, mod in tqdm(model.named_modules(), desc=f"Quantising to {nbits}-bit"):
        if not isinstance(mod, torch.nn.Linear):
            continue
        p, s, z, shape = quantise_linear(mod.weight.cpu().float(), nbits, group_size)
        key = name.replace('.', '_')
        q_state[f"{key}.packed"] = p
        q_state[f"{key}.scale"]  = s
        q_state[f"{key}.zero"]   = z
        if mod.bias is not None:
            q_state[f"{key}.bias"] = mod.bias.cpu()
        q_cfg[name] = dict(original_shape=shape,
                           group_size=group_size,
                           bits=nbits,
                           has_bias=mod.bias is not None)
    return q_state, q_cfg

# ────────────────────────────────────── CLI ──────────────────────────────────
def main():
    ap = argparse.ArgumentParser("Groupwise N-bit packer")
    ap.add_argument("--model_input", required=True,
                    help="HF model name or path (MoE or dense).")
    ap.add_argument("--bits", type=int, default=2, choices=[2, 4],
                    help="2 or 4 bit quantisation.")
    ap.add_argument("--group_size", type=int, default=8,
                    help="Side length of square groups (≥ 8 is typical).")
    ap.add_argument("--output_dir", default="./outputs/quantised_nbit")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("▶ Loading model/tokenizer …")
    tok = AutoTokenizer.from_pretrained(args.model_input, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.add_special_tokens({"pad_token": "[PAD]"})

    model = AutoModelForCausalLM.from_pretrained(
        args.model_input,
        torch_dtype=torch.float16,
        device_map="cpu",                 # stay on CPU; quantiser is CPU-bound
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    print(f"▶ Quantising to {args.bits}-bit …")
    t0 = time.time()
    q_state, q_cfg = quantise_model(model, args.bits, args.group_size)
    print(f"   Done in {time.time()-t0:.1f}s.")

    print("▶ Saving …")
    torch.save(q_state, os.path.join(args.output_dir, "quantised.pt"))
    with open(os.path.join(args.output_dir, "quant_config.json"), "w") as f:
        json.dump({"method": "uniform_affine",
                   "bits": args.bits,
                   "group_size": args.group_size,
                   "layers": q_cfg}, f, indent=2)
    tok.save_pretrained(args.output_dir)
    # also save model’s config so HF can reload it
    model.config.save_pretrained(args.output_dir)
    print(f"✅ Quantised model written to {args.output_dir}")

if __name__ == "__main__":
    main()
