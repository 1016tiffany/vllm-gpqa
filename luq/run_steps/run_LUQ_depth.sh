#!/bin/bash

python LUQ_model.py \
    --model_input "Qwen/Qwen2.5-VL-7B-Instruct" \
    --low_bit_model_name "1_bit_qwen" \
    --LUQ_depth_based \
    --output_path "./outputs/"

# Alternative call if high bit model not implicit like it is for qwen
# python LUQ_model.py \
#     --model_input "llama3" \ Should be loadable by huggingface, and have the account should have access to it
#     --high_bit_model_name "gptq_llama" \
#     --low_bit_model_name "1_bit_llama" \
#     --output_path "./output_models/" \:

