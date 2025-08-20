#!/bin/bash

python BiLLM_quantize.py \
    --model_input "HuggingFaceTB/SmolLM3-3B" \
    --dataset wikitext2 \
    --low_quant_method braq \
    --blocksize 128 \
    --salient_metric hessian \
    --device "cuda:0" \
    --nsamples 128 \
    --save \
    --save_model_name "1_bit_smollm" \
    --output_path "./outputs/"

# Doesnt work with gemma 3 for now.... only works for qwen / llava 1.5
