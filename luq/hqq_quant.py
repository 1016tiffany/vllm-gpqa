#!/usr/bin/env python
import argparse, os, torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HqqConfig            # ⬅️  built-in HQQ wrapper
)

def main():
    parser = argparse.ArgumentParser(
        description="Quantize any HF model to 2-bit with HQQ")
    parser.add_argument("--model_input", required=True,
                        help="HF model name or path")
    parser.add_argument("--bits", type=int, default=2, choices=[1,2,3,4],
                        help="Number of bits (default: 2)")
    parser.add_argument("--group_size", type=int, default=64,
                        help="HQQ group size (default: 64)")
    parser.add_argument("--save_model_name", default="hqq_quantized")
    parser.add_argument("--output_path",   default="./outputs/")
    args = parser.parse_args()

    print(f"▶ Quantizing {args.model_input} → {args.bits}-bit, gs={args.group_size}")

    # 1) Build the HQQ quantisation config
    quant_cfg = HqqConfig(
        nbits       = args.bits,
        group_size  = args.group_size,
        quant_zero  = True,
        quant_scale = True,
        axis        = 0,          # axis=0 gives better quality; axis=1 enables fused-kernel speed
        # device      = "cuda:0"    # Use specific device instead of "cuda"
    )

    # 2) Load **and** quantise in one call
    model = AutoModelForCausalLM.from_pretrained(
        args.model_input,
        torch_dtype         = torch.float16,
        device_map          = "auto",    # send to specific GPU
        quantization_config = quant_cfg,
        trust_remote_code   = True               # needed for SlimMoE
    )

    # 3) Tokeniser is unchanged
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_input, trust_remote_code=True
    )

    # 4) Save everything
    save_dir = os.path.join(args.output_path, args.save_model_name)
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)       # HQQ weights are saved transparently
    tokenizer.save_pretrained(save_dir)

    print(f"✅ Quantised model written to: {save_dir}")

if __name__ == "__main__":
    main()
