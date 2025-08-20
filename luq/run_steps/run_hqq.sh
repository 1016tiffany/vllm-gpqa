#!/bin/bash

# Run HQQ quantization for Phi-mini-MoE model
python hqq_quant.py \
    --model_input "microsoft/Phi-mini-MoE-instruct" \
    --bits 2 \
    --group_size 64 \
    --save_model_name 'hqq_phi_mini_moe_2bit' \
    --output_path "./outputs/"