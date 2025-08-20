#!/usr/bin/env python
# aggregate_lm_eval_results.py
#
# Usage:
#   python aggregate_lm_eval_results.py \
#     --results-root results/olmoe-luq-n_layer \
#     --out-wide summary_wide.csv \
#     --out-long summary_long.csv
#
# Notes:
# - Picks the newest JSON per model (by "date" field; fallback to file mtime).
# - Extracts both acc and acc_norm where present.
# - Computes macro averages:
#     * macro_avg_pref_norm: per-task prefers acc_norm else acc, then mean
#     * macro_avg_acc: mean of acc over tasks (skips missing)
#     * macro_avg_acc_norm: mean of acc_norm over tasks (skips missing)

import argparse, os, json, glob, csv, re, math, time
from typing import Dict, Any, Optional, Tuple

TASKS = ["arc_challenge", "boolq", "hellaswag", "piqa", "winogrande"]
METRICS = ["acc", "acc_norm"]

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", required=True, help="Directory that contains per-model result folders")
    ap.add_argument("--out-wide", default="summary_wide.csv", help="Output CSV (wide format)")
    ap.add_argument("--out-long", default="summary_long.csv", help="Output CSV (long/tidy format)")
    return ap.parse_args()

def newest_per_model(json_paths):
    """
    Group JSON files by model (parsed from JSON 'model_name' or path),
    pick the newest by JSON 'date' (float epoch) else by file mtime.
    Returns dict: model_key -> (json_path, json_data)
    """
    groups = {}
    for jp in json_paths:
        try:
            with open(jp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        model_name = data.get("model_name") or data.get("config", {}).get("model_args") or ""
        if not model_name:
            # fallback: derive model from path
            model_name = jp

        # extract a stable model key like "olmoe-luq-layer-6_2bit"
        key = extract_model_key(model_name) or extract_model_key(jp) or model_name

        # determine recency
        date_val = data.get("date")
        if isinstance(date_val, (int, float)):
            recency = float(date_val)
        else:
            try:
                recency = os.path.getmtime(jp)
            except Exception:
                recency = time.time()

        prev = groups.get(key)
        if prev is None or recency > prev[2]:
            groups[key] = (jp, data, recency)
    # strip recency
    return {k: (v[0], v[1]) for k, v in groups.items()}

def extract_model_key(s: str) -> Optional[str]:
    """
    Find 'olmoe-luq-layer-<N>_2bit' inside a string.
    """
    m = re.search(r"(olmoe-luq-layer-\d+_2bit)", s)
    if m:
        return m.group(1)
    return None

def extract_num2(model_key: str) -> Optional[int]:
    m = re.search(r"layer-(\d+)_2bit", model_key)
    return int(m.group(1)) if m else None

def get_metric_value(task_obj: Dict[str, Any], metric_key: str) -> Optional[float]:
    """
    Handle lm_eval’s different result field styles:
      - flat: 'acc': 0.75
      - flat w/ tag: 'acc,none': 0.75
      - nested: 'acc': {'value': 0.75, ...}
    Prefer exact key; else look for any key that starts with '<metric_key>,'
    """
    if not isinstance(task_obj, dict):
        return None
    # 1) direct
    v = task_obj.get(metric_key)
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, dict) and isinstance(v.get("value"), (int, float)):
        return float(v["value"])
    # 2) tagged variants like 'acc,none'
    cand = None
    for k, val in task_obj.items():
        if not isinstance(k, str):
            continue
        if k == metric_key or k.startswith(metric_key + ","):
            # prefer ',none' if multiple
            if cand is None or (isinstance(cand[0], str) and not cand[0].endswith(",none") and k.endswith(",none")):
                cand = (k, val)
    if cand is not None:
        val = cand[1]
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, dict) and isinstance(val.get("value"), (int, float)):
            return float(val["value"])
    return None

def safe_mean(xs):
    vals = [x for x in xs if isinstance(x, (int, float)) and not math.isnan(x)]
    return sum(vals) / len(vals) if vals else None

def main():
    args = parse_args()
    # Find all JSONs recursively
    json_paths = glob.glob(os.path.join(args.results_root, "**", "*.json"), recursive=True)
    if not json_paths:
        print(f"No JSON files found under {args.results_root}")
        return

    per_model = newest_per_model(json_paths)
    if not per_model:
        print("No valid results JSONs found.")
        return

    # Build rows
    wide_rows = []
    long_rows = []
    for model_key, (jp, data) in per_model.items():
        res = data.get("results", {})
        num2 = extract_num2(model_key)
        row = {"model_key": model_key, "num2": num2}

        # Extract metrics per task
        for task in TASKS:
            task_obj = res.get(task, {})
            for metric in METRICS:
                val = get_metric_value(task_obj, metric)
                col = f"{task}_{metric}"
                row[col] = val
                # long/tidy row
                long_rows.append({
                    "model_key": model_key,
                    "num2": num2,
                    "task": task,
                    "metric": metric,
                    "value": val
                })

        # Macro averages
        row["macro_avg_pref_norm"] = safe_mean([
            row.get(f"{t}_acc_norm") if isinstance(row.get(f"{t}_acc_norm"), (int, float))
            else row.get(f"{t}_acc")
            for t in TASKS
        ])
        row["macro_avg_acc"] = safe_mean([row.get(f"{t}_acc") for t in TASKS])
        row["macro_avg_acc_norm"] = safe_mean([row.get(f"{t}_acc_norm") for t in TASKS])

        wide_rows.append(row)

    # Sort by num2 if present, else by model_key
    wide_rows.sort(key=lambda r: (999999 if r["num2"] is None else r["num2"], str(r["model_key"])))
    long_rows.sort(key=lambda r: (999999 if r["num2"] is None else r["num2"], r["task"], r["metric"]))

    # Write wide CSV
    # columns: model_key, num2, per-task metrics, macro averages
    per_task_cols = [f"{t}_{m}" for t in TASKS for m in METRICS]
    fieldnames = ["model_key", "num2"] + per_task_cols + ["macro_avg_pref_norm", "macro_avg_acc", "macro_avg_acc_norm"]

    out_wide = os.path.join(args.results_root, args.out_wide)
    with open(out_wide, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(wide_rows)

    # Write long CSV
    out_long = os.path.join(args.results_root, args.out_long)
    with open(out_long, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model_key", "num2", "task", "metric", "value"])
        w.writeheader()
        w.writerows(long_rows)

    print(f"Wrote:\n  {out_wide}\n  {out_long}\nModels aggregated: {len(wide_rows)}")

if __name__ == "__main__":
    main()
