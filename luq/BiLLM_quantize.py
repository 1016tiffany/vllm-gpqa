import sys
import os
import torch
from transformers import AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration, AutoModelForCausalLM
import argparse
# Add BiLLM directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
billm_dir = os.path.join(current_dir, 'BiLLM')
sys.path.append(billm_dir)
# Add parent directory to path to import run.py functions
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from BiLLM.run import quant_sequential, get_model
from BiLLM.utils.datautils import get_loaders
from pdb import set_trace as breakpoint



def identity_forward(self, input):
    return input

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_input", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--dataset", type=str, default="wikitext2")  # Changed default to mix for multimodal
    parser.add_argument("--low_quant_method", type=str, default="braq")
    parser.add_argument("--blocksize", type=int, default=128)
    parser.add_argument("--salient_metric", type=str, default="hessian")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--percdamp", type=float, default=0.01)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--save_model_name", type=str, default="1_bit_qwen")
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--output_path", type=str, default="./output_models/")
    args = parser.parse_args()

    # Set attributes needed by quant_sequential
    args.groupsize = args.blocksize
    args.minlayer = -1
    args.maxlayer = 1000
    args.quant_only = ""
    args.invert = False
    args.disable_gptq = False

    # Load model
    print(f"Loading model from {args.model_input}...")
    if "qwen" in args.model_input.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_input, torch_dtype="auto", device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(
        args.model_input, torch_dtype="auto", device_map="auto")
    processor = AutoProcessor.from_pretrained(args.model_input)
    
    #seqlen = 2048

    #breakpoint()
    # Get data loaders
    dataloader = get_loaders(
        args.dataset,
        nsamples=args.nsamples,
        seed=0,
        model=args.model_input,
        seqlen=args.seqlen
    )
    # Store original embed layer and replace with identity
    #embed_layer = model.model.embed_tokens
    #model.model.embed_tokens.forward = identity_forward.__get__(model.model.embed_tokens)
    
    # Quantize model
    print("Starting quantization...")
    quant_sequential(model, dataloader, args.device, args)  # Pass args directly

    # Restore original embed layer
    #model.model.embed_tokens = embed_layer
    
    # SAVING vode moved to quant_sequential
    # if args.save:
    #     save_path = os.path.join(args.output_path, f"qwen_quantized_{args.low_q.ptuant.pt_method}.pt")
    #     os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    #     model.save_pretrained(save_path)
    #     print(f"Quantized model saved to {save_path}")

if __name__ == "__main__":
    main()