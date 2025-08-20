#!/usr/bin/env bash
# run_luq_layers_sweep.sh
# Sweep --num2 from 1..15 and store outputs under outputs/olmoe-luq-n_layer/

# Config (edit if needed)
SCRIPT="luq_olmoe_layer.py"
MODEL4="outputs/olmoe-hqq-4bit"
MODEL2="outputs/olmoe-hqq-2bit"
ORDER="outputs/sorted_layers_olmoe.txt"
DTYPE="bfloat16"
OUTROOT="outputs/olmoe-luq-n_layer"

mkdir -p "$OUTROOT"

for N in $(seq 1 15); do
  OUTDIR="$OUTROOT/olmoe-luq-layer-${N}_2bit"
  echo "[$(date '+%F %T')] Running num2=${N} -> ${OUTDIR}"

  # Run and continue even if one configuration fails
  if ! python "$SCRIPT" \
      --model4 "$MODEL4" \
      --model2 "$MODEL2" \
      --order_layers "$ORDER" \
      --num2 "$N" \
      --out "$OUTDIR" \
      --dtype "$DTYPE"; then
    echo "[$(date '+%F %T')] WARNING: num2=${N} failed — continuing..."
  fi
done

echo "[$(date '+%F %T')] Sweep complete. Outputs in: $OUTROOT"
