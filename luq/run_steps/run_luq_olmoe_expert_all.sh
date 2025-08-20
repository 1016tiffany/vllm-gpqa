#!/usr/bin/env bash
# run_luq_olmoe_expert_all.sh
# Sweep the number of 2-bit EXPERTS from 0..(total) in increments of 64.
# Uses luq_olmoe_expert.py (which takes --keep = how many experts stay 4-bit).

set -euo pipefail

# ---- Config (edit if needed) -------------------------------------------------
SCRIPT="luq_olmoe_expert.py"
MODEL4="outputs/olmoe-hqq-4bit"
MODEL2="outputs/olmoe-hqq-2bit"
ORDER="outputs/sorted_experts_olmoe.txt"   # lines like: layer_7_expert_25 <score>
DTYPE="bfloat16"
OUTROOT="outputs/olmoe-luq-n_expert"
STEP=64
# ------------------------------------------------------------------------------

mkdir -p "$OUTROOT"

# Determine how many experts are listed in the ordering file (default 1024)
if [[ -f "$ORDER" ]]; then
  TOTAL_EXPERTS="$(awk 'END{print NR}' "$ORDER")"
else
  echo "WARNING: ORDER file '$ORDER' not found; assuming TOTAL_EXPERTS=1024."
  TOTAL_EXPERTS=1024
fi

echo "[$(date '+%F %T')] Total experts found/listed: ${TOTAL_EXPERTS}"

# Loop N = number of experts to quantize to 2-bit
for N in $(seq 0 $STEP "$TOTAL_EXPERTS"); do
  # keep = how many remain 4-bit
  KEEP=$(( TOTAL_EXPERTS - N ))
  if (( KEEP < 0 )); then KEEP=0; fi

  OUTDIR="$OUTROOT/olmoe-luq-layer-${N}_2bit"   # name uses "layer-<N>_2bit" per request
  echo "[$(date '+%F %T')] Quantizing ${N} experts to 2-bit (keep=${KEEP} 4-bit) -> ${OUTDIR}"

  # Run and continue even if one configuration fails
  if ! python "$SCRIPT" \
        --model4 "$MODEL4" \
        --model2 "$MODEL2" \
        --order  "$ORDER" \
        --keep   "$KEEP" \
        --out    "$OUTDIR" \
        --dtype  "$DTYPE"; then
    echo "[$(date '+%F %T')] WARNING: N=${N} failed — continuing..."
  fi
done

# If TOTAL_EXPERTS is not a multiple of STEP, you might want one last exact pass.
if (( TOTAL_EXPERTS % STEP != 0 )); then
  N="$TOTAL_EXPERTS"
  KEEP=0
  OUTDIR="$OUTROOT/olmoe-luq-layer-${N}_2bit"
  echo "[$(date '+%F %T')] Final exact pass: Quantizing ${N} experts (keep=0) -> ${OUTDIR}"
  if ! python "$SCRIPT" \
        --model4 "$MODEL4" \
        --model2 "$MODEL2" \
        --order  "$ORDER" \
        --keep   "$KEEP" \
        --out    "$OUTDIR" \
        --dtype  "$DTYPE"; then
    echo "[$(date '+%F %T')] WARNING: N=${N} final pass failed — continuing..."
  fi
fi

echo "[$(date '+%F %T')] Sweep complete. Outputs in: $OUTROOT"
