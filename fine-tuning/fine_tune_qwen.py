#!/usr/bin/env python
# Fine‑tune Qwen2.5‑7B‑Instruct on CNN/DailyMail with LoRA,
# and compare ROUGE + BERTScore against the untuned base model.

import argparse, os, torch, gc
from dataclasses import dataclass
from typing import Dict, List, Optional
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          Trainer, TrainingArguments, BitsAndBytesConfig)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import evaluate

# ────────────────────────── helpers ──────────────────────────
def build_prompt(article: str) -> str:
    return f"Summarize the following news article in 3-4 concise sentences:\n\n{article.strip()}"

def apply_chat_template(tokenizer, user_text: str) -> str:
    msgs = [
        {"role": "system", "content": "You are a helpful assistant that produces concise, factual summaries."},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": ""},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)

@dataclass
class DataCollator:
    tokenizer: AutoTokenizer
    pad_to_multiple_of: Optional[int] = 8
    label_pad_token_id: int = -100
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        labels = [f["labels"] for f in features]
        model_inputs = self.tokenizer.pad(
            [{k: f[k] for k in ("input_ids", "attention_mask")} for f in features],
            padding=True, pad_to_multiple_of=self.pad_to_multiple_of, return_tensors="pt"
        )
        max_len = model_inputs["input_ids"].size(1)
        model_inputs["labels"] = torch.tensor(
            [l + [self.label_pad_token_id]*(max_len-len(l)) for l in labels], dtype=torch.long
        )
        return model_inputs

def preprocess(examples, tokenizer, max_src, max_tgt, eos_id):
    batch = {"input_ids":[], "attention_mask":[], "labels":[]}
    for art, summ in zip(examples["article"], examples["highlights"]):
        prefix = apply_chat_template(tokenizer, build_prompt(art))
        full = prefix + summ.strip() + tokenizer.eos_token
        tok_full = tokenizer(full, truncation=True,
                             max_length=max_src+max_tgt, return_attention_mask=True)
        prefix_len = len(tokenizer(prefix, truncation=True,
                                   max_length=max_src, add_special_tokens=False)["input_ids"])
        labels = tok_full["input_ids"][:]
        labels[:prefix_len] = [-100]*prefix_len
        if tok_full["input_ids"][-1] != eos_id:
            tok_full["input_ids"].append(eos_id)
            tok_full["attention_mask"].append(1)
            labels.append(eos_id)
        batch["input_ids"].append(tok_full["input_ids"])
        batch["attention_mask"].append(tok_full["attention_mask"])
        batch["labels"].append(labels)
    return batch

# ────────────────────────── main ─────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--output_dir", default="./qwen25-cnn_dm-lora")
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size",  type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--max_source_tokens", type=int, default=512)
    p.add_argument("--max_target_tokens", type=int, default=160)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--report_to", default="none")
    args = p.parse_args()
    torch.manual_seed(args.seed)

    # -------- tokenizer & data --------
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, padding_side="right")
    tok.pad_token = tok.eos_token
    raw_train = load_dataset("cnn_dailymail","3.0.0",split="train[:1000]")
    raw_val   = load_dataset("cnn_dailymail","3.0.0",split="validation[:100]")
    eos_id = tok.eos_token_id
    _prep = lambda x: preprocess(x,tok,args.max_source_tokens,args.max_target_tokens,eos_id)
    train_ds = raw_train.map(_prep, batched=True, remove_columns=raw_train.column_names)
    val_ds   = raw_val.map(_prep,   batched=True, remove_columns=raw_val.column_names)
    collator = DataCollator(tok)

    # -------- metrics --------
    rouge = evaluate.load("rouge")
    bert  = evaluate.load("bertscore")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if preds.ndim == 3: preds = preds.argmax(-1)
        labels[labels == -100] = tok.pad_token_id
        dec_preds  = tok.batch_decode(preds,  skip_special_tokens=True)
        dec_labels = tok.batch_decode(labels, skip_special_tokens=True)
        rouge_scores = rouge.compute(predictions=dec_preds, references=dec_labels, use_stemmer=True)
        bert_scores  = bert.compute(predictions=dec_preds, references=dec_labels,
                                    lang="en", rescale_with_baseline=False)
        return {
            **{k: rouge_scores[k] for k in ["rouge1","rouge2","rougeL","rougeLsum"]},
            "bertscore_precision": sum(bert_scores["precision"])/len(bert_scores["precision"]),
            "bertscore_recall":    sum(bert_scores["recall"])/len(bert_scores["recall"]),
            "bertscore_f1":        sum(bert_scores["f1"])/len(bert_scores["f1"]),
        }

    # -------- common training/eval args --------
    common_args = dict(
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        remove_unused_columns=False,
        report_to=None if args.report_to=="none" else args.report_to,
    )

    # # ================== 1) baseline evaluation ==================
    # base_model = AutoModelForCausalLM.from_pretrained(
    #     args.model_name,
    #     torch_dtype=torch.float16,
    #     device_map="auto",
    #     attn_implementation="eager",
    # )
    # base_trainer = Trainer(
    #     model=base_model,
    #     args=TrainingArguments(output_dir="tmp", do_train=False, **common_args),
    #     eval_dataset=val_ds,
    #     data_collator=collator,
    #     tokenizer=tok,
    #     compute_metrics=compute_metrics,
    # )
    # base_metrics = base_trainer.evaluate()
    # print("\n📊  BASE MODEL METRICS:", base_metrics)

    # # free GPU memory
    # del base_trainer, base_model
    # torch.cuda.empty_cache(); gc.collect()

    breakpoint()
    # ================== 2) LoRA fine‑tuning =====================
    ft_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",
    )
    ft_model = prepare_model_for_kbit_training(ft_model)
    ft_model = get_peft_model(ft_model, LoraConfig(r=64,lora_alpha=128,lora_dropout=0.05,
                                                   target_modules=["q_proj","k_proj","v_proj","o_proj",
                                                                   "gate_proj","up_proj","down_proj"],
                                                   task_type="CAUSAL_LM"))
    train_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        save_total_limit=2,
        eval_strategy="epoch",
        logging_steps=20,
        **common_args,
    )
    trainer = Trainer(
        model=ft_model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tok,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    ft_metrics = trainer.evaluate()
    print("\n📊  FINE‑TUNED MODEL METRICS:", ft_metrics)

    # save LoRA adapter + tokenizer
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()