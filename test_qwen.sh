#!/usr/bin/env bash
set -euo pipefail

for k in 0 2 4 6 8 10 12 14 16 18 20 22 24 26 28; do
  echo "=== Running with $k one-bit layers ==="
  python3 run.py \
      --data ChartQA_TEST \
      --model Qwen2.5-VL-7B-Instruct \
      --data-limit 500 \
      --n-1bit "$k" \
      --work-dir outputs/qwen_n${k}
done