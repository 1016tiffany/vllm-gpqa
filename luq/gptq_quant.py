import sys
import os
import torch
from transformers import AutoTokenizer
import argparse
import time
from pathlib import Path
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from datasets import load_dataset

# from pdb import set_trace as breakpoint # For debugging, can be removed

def main():
    parser = argparse.ArgumentParser(description="Quantize a Hugging Face CausalLM model using AutoGPTQ.")
    parser.add_argument("--model_input", type=str, required=True,
                        help="Hugging Face model name or path (e.g., 'Qwen/Qwen1.5-7B-Chat', 'facebook/opt-125m').")
    parser.add_argument("--bits", type=int, default=4, choices=[2,3, 4],
                        help="Quantization bits (2 or 4). Default is 4.")
    parser.add_argument("--group_size", type=int, default=128,
                        help="Group size for quantization (e.g., 32, 64, 128). Default is 128.")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use for quantization (e.g., 'cuda:0', 'cpu'). GPU is highly recommended. Default is 'cuda:0'.")
    parser.add_argument("--nsamples", type=int, default=128,
                        help="Number of wikitext2 samples to use for calibration. Default is 128.")
    parser.add_argument("--save_model_name", type=str, default="gptq_qwen")
    parser.add_argument("--output_path", type=str, default="./outputs/",
                        help="Path to save the quantized model. Default is './outputs/'.")
    args = parser.parse_args()

    
    print(f"Starting quantization process for model: {args.model_input}")
    print(f"Quantization settings: bits={args.bits}, group_size={args.group_size}, calibration_samples={args.nsamples}")

    save_path_high_bit = os.path.join(args.output_path, args.save_model_name)

    # Load tokenizer
    print(f"Loading tokenizer for '{args.model_input}'...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_input,
        # "Qwen/Qwen3-0.6B",
        # use_fast=False,
        trust_remote_code=True
    )
    

    # Handle pad token for tokenizer if not set
    if tokenizer.pad_token is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            #logger.info(f"tokenizer.pad_token was None. Set to eos_token_id: {tokenizer.eos_token_id}")
        else:
            # Add a new pad token if EOS is also missing, though less common for CausalLMs
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            #logger.info("tokenizer.pad_token and tokenizer.eos_token were None. Added a new pad_token '[PAD]'. You might need to resize model token embeddings if fine-tuning.")


    # Prepare calibration dataset (wikitext2)
    print(f"Loading and preparing 'wikitext-2-raw-v1' dataset for calibration ({args.nsamples} samples)...")

    calib_dataset_full = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    
    raw_calib_data = []
    # Iterate through the dataset and collect non-empty samples up to nsamples
    for record in calib_dataset_full:
        if len(raw_calib_data) >= args.nsamples:
            break
        text_sample = record["text"].strip()
        if text_sample: # Ensure text is not empty
                raw_calib_data.append(text_sample)
    
    
    print(f"Using {len(raw_calib_data)} samples for calibration.")

   

    # Define Quantization Configuration
    print("Defining quantization configuration...")
    quantize_config = BaseQuantizeConfig(
        bits=args.bits,
        group_size=args.group_size,
        desc_act=False,  # desc_act=False is common for GPTQ. Set to True to experiment with "act-order".
        # tokenizer=tokenizer, # Pass tokenizer
        # dataset=raw_calib_data  # Pass raw text data; AutoGPTQ handles tokenization internally for calibration
    )

    # Load model and quantize
    # AutoGPTQForCausalLM.from_pretrained will load the specified model and then quantize it.
    print(f"Loading and quantizing model '{args.model_input}'...")
    
    start_time = time.time()
    model = AutoGPTQForCausalLM.from_pretrained(
        args.model_input,
        quantize_config=quantize_config,
        torch_dtype=torch.float16,  # Load model in float16 for quantization. 'auto' can also be used.
        device_map=args.device,    # Handles device placement e.g. "cuda:0", "auto", "cpu"
        trust_remote_code=True,
        tokenizer=tokenizer,
        dataset=raw_calib_data
    )
    
    quantization_time = time.time() - start_time
    print(f"Model quantization completed in {quantization_time:.2f} seconds.")

    # Save the quantized model
    
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    
    # Sanitize model input name to create a valid directory name
    
    
    print(f"Saving quantized model to '{save_path_high_bit}'...")
    
    # save_quantized will save the model weights, quantize_config.json, and other necessary files.
    # It also attempts to save the tokenizer.
    model.save_quantized(save_path_high_bit, use_safetensors=True)
    # Explicitly saving tokenizer again to ensure it's there, though model.save_quantized usually handles it.
    tokenizer.save_pretrained(str(save_path_high_bit)) # AutoGPTQ save_quantized should handle this.

    print(f"Quantized model (and tokenizer) saved successfully to '{save_path_high_bit}'.")


if __name__ == "__main__":
    main()