#!/bin/bash

# Run GPTQ quantization for Qwen 2.5 VL model
python gptq_quant.py \
    --model_input "Qwen/Qwen3-0.6B" \
    --bits 4 \
    --group_size 128 \
    --device "cuda:0" \
    --nsamples 128 \
    --save_model_name 'gptq_qwen3_4bit' \
    --output_path "../outputs/"

# Alternative quantization options:
# --bits 3            # For 3-bit quantization
# --bits 2            # For 2-bit quantization
# both 2,3 bit gptq can be used for low bit model layers
