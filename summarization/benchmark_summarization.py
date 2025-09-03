#!/usr/bin/env python3
"""
Benchmark Phi-4 multimodal‐instruct on abstractive summarization.

Edits lines 80-100 for LUQ Qwen Evaluation
Install packages from requirements.txt 

Usage (defaults shown):
    python benchmark_summarization.py \
        --model Qwen/Qwen2.5-VL-7B-Instruct-AWQ \
        --dataset cnn_dailymail \
        --split test \
        --batch_size 4 \
        --max_new_tokens 128 \
        --out_dir results \
        --device cuda \
        --num_data 100
"""

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    GenerationConfig,
)
import evaluate
from tqdm import tqdm

PROMPT_TMPL = (
    # "Summarize the following article in a concise, factual paragraph.\n\n"
    # "Summarize the following article in 3-4 sentences.\n\n" #CNN/DailyMail
    "Summarize the following text in one sentence.\n\n" #XSum
    "{article}\n\nSummary:"
)

def load_data(name: str, split: str):
    if name == "cnn_dailymail":
        ds = load_dataset("cnn_dailymail", "3.0.0", split=split)
        return ds, "article", "highlights"
    elif name == "xsum":
        ds = load_dataset("xsum", split=split, trust_remote_code=True)
        return ds, "document", "summary"
    else:
        raise ValueError(f"Unknown dataset {name!r}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",        default="microsoft/Phi-4-multimodal-instruct")
    ap.add_argument("--dataset",      choices=["cnn_dailymail","xsum"], default="cnn_dailymail")
    ap.add_argument("--split",        default="test")
    ap.add_argument("--batch_size",   type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--out_dir",      default="results")
    ap.add_argument("--device",       default="cuda")
    ap.add_argument("--num_data",     type=int)
    args = ap.parse_args()

    # prepare output directory
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # load processor, model, and generation config
    print("🔌 Loading model …")
    processor = AutoProcessor.from_pretrained(
        args.model,
        trust_remote_code=True,
        use_fast=True,   # optional but avoids the “slow processor” warning
    )
    
    if "qwen" in args.model.lower():
        from transformers import Qwen2_5_VLForConditionalGeneration
        # Full model
        # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        #     args.model,
        #     device_map="auto",
        #     torch_dtype=torch.float16,
        #     trust_remote_code=True,
        # ).eval()

        # LUQ
        # 4bit: high_quant_model_name = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
        # 16bit: high_quant_model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
        high_quant_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model,  
            device_map="auto", 
            torch_dtype=torch.float16
        ).eval()

        low_quant_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "../VLMEvalKit/1_bit_qwen", 
            torch_dtype=torch.float16,
            device_map="auto",
        ).eval()
        layers_entropy_asc_order = [ 27, 26,18, 24, 23, 20,22, 21, 19, 17,25, 16, 15, 14, 13, 12, 11, 10, 9, 8, 1, 0, 3, 2,6 , 5,7 ,4 ]
        
        n_1bit = 28
        for layer_num in layers_entropy_asc_order[:n_1bit]:
            # Set the new layer to the model
            high_quant_model.model.language_model.layers[layer_num] = low_quant_model.model.language_model.layers[layer_num]

        model = high_quant_model

    else: #--model microsoft/Phi-4-multimodal-instruct
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).eval()
        gen_cfg = GenerationConfig.from_pretrained(
            args.model, trust_remote_code=True
        )
    
    # load & slice dataset
    print("📚 Loading dataset …")
    ds, src_key, tgt_key = load_data(args.dataset, args.split)
    if args.num_data is not None:
        ds = ds.select(range(min(args.num_data, len(ds))))

    # generate
    preds, refs = [], []
    print("📝 Generating summaries …")
    for i in tqdm(range(0, len(ds), args.batch_size)):
        batch = ds[i : i + args.batch_size]
        prompts = [PROMPT_TMPL.format(article=art).strip() for art in batch[src_key]]

        # processor will infer input_mode for TEXT
        inputs = processor(
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096
        ).to(args.device)

        with torch.no_grad():
            if "phi" in args.model.lower():
                out_ids = model.generate(
                    **inputs,
                    generation_config=gen_cfg,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    eos_token_id=processor.tokenizer.eos_token_id,
                )
            else:
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    eos_token_id=processor.tokenizer.eos_token_id,
                )

        dec = processor.batch_decode(out_ids, skip_special_tokens=True)
        for prompt, full in zip(prompts, dec):
            preds.append(full[len(prompt):].strip())
        refs.extend(batch[tgt_key])

    # save raw
    Path(args.out_dir, "preds.txt").write_text("\n".join(preds), encoding="utf-8")
    flat_refs = [r.replace("\n", " ") for r in refs]
    Path(args.out_dir, "refs.txt").write_text("\n".join(flat_refs), encoding="utf-8")

    # compute metrics
    print("📏 Computing metrics …")
    rouge = evaluate.load("rouge")
    bert  = evaluate.load("bertscore")

    rouge_res = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
    bert_res  = bert.compute(predictions=preds, references=refs, lang="en")

    results = {
        "rouge": rouge_res,
        "bertscore": {
            "precision": sum(bert_res["precision"]) / len(bert_res["precision"]),
            "recall":    sum(bert_res["recall"])    / len(bert_res["recall"]),
            "f1":        sum(bert_res["f1"])        / len(bert_res["f1"]),
        },
        "meta": {
            "model":     args.model,
            "dataset":   args.dataset,
            "split":     args.split,
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }

    Path(args.out_dir, "metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
