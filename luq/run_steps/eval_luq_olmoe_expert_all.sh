#!/usr/bin/env bash
# run_lm_eval_n_expert_sweep.sh
# Evaluate outputs/olmoe-luq-n_expert/olmoe-luq-layer-*_2bit models and summarize results.

set -euo pipefail

# ---------- Config (edit as needed) ----------
LM_EVAL_BIN="lm_eval"                       # path or just "lm_eval" if on PATH
MODELS_ROOT="outputs/olmoe-luq-n_expert"
RESULTS_ROOT="results/olmoe-luq-n_expert"
TASKS="hellaswag,arc_challenge,winogrande,piqa,boolq"
DEVICE="cuda:0"
BATCH="auto:4"
DTYPE="bfloat16"
# --------------------------------------------

mkdir -p "$RESULTS_ROOT"

# Find candidate model dirs like .../olmoe-luq-layer-64_2bit, sort numerically by the number.
mapfile -t MODEL_DIRS < <(
  find "$MODELS_ROOT" -maxdepth 1 -type d -regex '.*/olmoe-luq-layer-[0-9]+_2bit$' \
  | awk -F'layer-|_2bit' '{print $2, $0}' \
  | sort -n -k1,1 \
  | cut -d' ' -f2-
)

if [ ${#MODEL_DIRS[@]} -eq 0 ]; then
  echo "[$(date '+%F %T')] No matching model dirs under $MODELS_ROOT (pattern: olmoe-luq-layer-*_2bit)."
  exit 0
fi

echo "[$(date '+%F %T')] Found ${#MODEL_DIRS[@]} models to evaluate under $MODELS_ROOT"

for MODEL_DIR in "${MODEL_DIRS[@]}"; do
  # Extract N (the number in olmoe-luq-layer-N_2bit)
  BASENAME="$(basename "$MODEL_DIR")"
  N="${BASENAME#olmoe-luq-layer-}"
  N="${N%_2bit}"

  OUTDIR="${RESULTS_ROOT}/olmoe-luq-layer-${N}_2bit"
  mkdir -p "$OUTDIR"

  # Conditionally set trust_remote_code=True only if custom modeling files are present
  if compgen -G "${MODEL_DIR}/modeling_*.py" > /dev/null; then
    TRUST=",trust_remote_code=True"
  else
    TRUST=""
  fi

  MODEL_ARGS="pretrained=${MODEL_DIR},device_map=${DEVICE},low_cpu_mem_usage=False,attn_implementation=eager,dtype=${DTYPE}${TRUST}"

  echo "[$(date '+%F %T')] Evaluating ${MODEL_DIR} -> ${OUTDIR}"
  # IMPORTANT: Do NOT pass top-level --trust_remote_code (causes datasets error).
  if ! "$LM_EVAL_BIN" \
        --model hf \
        --model_args "$MODEL_ARGS" \
        --tasks "$TASKS" \
        --device "$DEVICE" \
        --batch_size "$BATCH" \
        --num_fewshot 0 \
        --output_path "$OUTDIR" \
        2>&1 | tee "${OUTDIR}/stdout.log"; then
    echo "[$(date '+%F %T')] ERROR: lm_eval failed for N=${N}. See ${OUTDIR}/stdout.log"
    continue
  fi

  if [ -f "${OUTDIR}/results.json" ]; then
    echo "[$(date '+%F %T')] Done N=${N} ✓ -> ${OUTDIR}/results.json"
  else
    echo "[$(date '+%F %T')] WARNING: results.json missing for N=${N}. Check ${OUTDIR}/stdout.log"
  fi
done

# --------- Build a summary CSV across all runs (acc & acc_norm) ---------
export RESULTS_ROOT
python - <<'PY'
import os, json, glob, csv, re, math

root = os.environ['RESULTS_ROOT']

# Keep this in sync with your TASKS above if you change them
TASKS = ["arc_challenge", "boolq", "hellaswag", "piqa", "winogrande"]
# For CSV we’ll include both metrics (if present)
METRICS = ["acc", "acc_norm"]

def get_metric(task_obj, key):
    """Extract a float regardless of lm_eval result nesting."""
    if not isinstance(task_obj, dict):
        return None
    v = task_obj.get(key)
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, dict) and isinstance(v.get("value"), (int, float)):
        return float(v["value"])
    return None

def macro_avg_pref_norm(row):
    """Average across tasks, preferring acc_norm else acc."""
    vals = []
    for t in TASKS:
        val = row.get(f"{t}_acc_norm")
        if not isinstance(val, (int, float)):
            val = row.get(f"{t}_acc")
        if isinstance(val, (int, float)) and not math.isnan(val):
            vals.append(val)
    return sum(vals) / len(vals) if vals else None

rows = []
for d in sorted(glob.glob(os.path.join(root, "olmoe-luq-layer-*_*bit"))):
    m = re.search(r"layer-(\d+)_2bit$", d)
    if not m:
        continue
    n = int(m.group(1))
    jf = os.path.join(d, "results.json")
    if not os.path.isfile(jf):
        continue

    with open(jf, "r") as f:
        data = json.load(f)
    res = data.get("results", {})

    row = {"num2": n}  # keep column name for downstream compatibility
    for task in TASKS:
        tobj = res.get(task, {})
        for metric in METRICS:
            row[f"{task}_{metric}"] = get_metric(tobj, metric)
    row["macro_avg_pref_norm"] = macro_avg_pref_norm(row)
    rows.append(row)

rows.sort(key=lambda r: r["num2"])
if rows:
    out = os.path.join(root, "summary.csv")
    fieldnames = (["num2"] +
                  [f"{t}_{m}" for t in TASKS for m in METRICS] +
                  ["macro_avg_pref_norm"])
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out} with {len(rows)} rows")
else:
    print("No results found; summary not created.")
PY

echo "[$(date '+%F %T')] All done. Per-model outputs in: $RESULTS_ROOT"
