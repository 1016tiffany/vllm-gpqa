#!/bin/bash

# Run AWQ quantization for Qwen 2.5 VL model
python awq_quant.py \
    --model_input "Qwen/Qwen2.5-VL-7B-Instruct" \
    --bits 3 \
    --group_size 128 \
    --save_model_name 'awq_qwen2.5_3bit' \
    --output_path "./outputs/"