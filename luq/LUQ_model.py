from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor,AutoModelForCausalLM
from pdb import set_trace as breakpoint
#from qwen_vl_utils import process_vision_info
import sys
import os
import torch
import argparse


def read_sorted_layers_order(filename):
  """Reads a comma-separated string of integers from a file into a list.
  Args:
    filename: The path to the file to read.
  Returns:
    A list of integers read from the file.
    Returns an empty list if the file is empty or cannot be read.
  """
  sorted_layers = []
  
  with open(filename, "r") as f:
    content = f.read()
    if content:
        sorted_layers = list(map(int, content.split(',')))
  return sorted_layers

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_input", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--dataset", type=str, default="wikitext2")  # Changed default to mix for multimodal
    parser.add_argument("--high_bit_model_name", type=str, default="gptq_qwen")
    parser.add_argument("--low_bit_model_name", type=str, default="1_bit_qwen")

    parser.add_argument("--num_layers_to_replace", type=int, default=10)
    parser.add_argument("--output_path", type=str, default="./outputs/")
    parser.add_arguement("--LUQ_depth_based", action="store_true", help="Use depth-based LUQ instead of entropy ordering") 
    parser.add_argument("--sorted_layers_file", type=str, default="sorted_layers_qwen.txt")
    args = parser.parse_args()
    # default: Load the model on the available device(s)
    
    args.sorted_layers_file = os.path.join(args.output_path, args.sorted_layers_file)
    #Load high-bit model
    if 'qwen' in args.model_input.lower():
        high_quant_model_name = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
        high_quant_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(high_quant_model_name,  device_map="cpu")
    # default processer
        processor = AutoProcessor.from_pretrained(high_quant_model_name)
    
    else:
        full_path_high_bit = os.path.join(args.output_path, args.high_bit_model_name)
        high_quant_model = AutoModelForCausalLM.from_pretrained(full_path_high_bit, device_map="cpu")
        processor = AutoProcessor.from_pretrained(args.high_bit_model_name)

    # Load low-bit model
    if 'qwen' in args.model_input.lower():
        full_path = os.path.join(args.output_path, args.low_bit_model_name)
        low_quant_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(full_path,device_map="cpu") # currently set for BiLLM Qwen
    else:
        full_path = os.path.join(args.output_path, args.low_bit_model_name)
        low_quant_model = AutoModelForCausalLM.from_pretrained(full_path, device_map="cpu")
        
    
    # Read layers in ascending order of entropy
    #layers_entropy_asc_order = [ 27, 26,18, 24, 23, 20,22, 21, 19, 17,25, 16, 15, 14, 13, 12, 11, 10, 9, 8, 1, 0, 3, 2,6 , 5,7 ,4 ]
    layers_entropy_asc_order = read_sorted_layers_order(args.sorted_layers_file)
    for layer_num in layers_entropy_asc_order[:args.num_layers_to_replace]:
        # Set the new layer to the model
        high_quant_model.model.layers[layer_num] = low_quant_model.model.layers[layer_num]

    # Save the mixed model
    #breakpoint()
    mixed_model_name = f"{args.low_bit_model_name}_LUQ_{args.num_layers_to_replace}layers"
    save_path = os.path.join(args.output_path, mixed_model_name)
    high_quant_model = high_quant_model.to("cpu")
    high_quant_model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    #processor.save_pretrained(save_path)
    print(f"Mixed model saved to {save_path}")


if __name__ == "__main__":
    main()


#LLLM part looks like this: 
# (Pdb) model.model
# Qwen2_5_VLModel(
#   (embed_tokens): Embedding(152064, 3584)
#   (layers): ModuleList(
#     (0-27): 28 x Qwen2_5_VLDecoderLayer(
#       (self_attn): Qwen2_5_VLSdpaAttention(
#         (q_proj): WQLinear_GEMM(in_features=3584, out_features=3584, bias=True, w_bit=4, group_size=128)
#         (k_proj): WQLinear_GEMM(in_features=3584, out_features=512, bias=True, w_bit=4, group_size=128)
#         (v_proj): WQLinear_GEMM(in_features=3584, out_features=512, bias=True, w_bit=4, group_size=128)
#         (o_proj): WQLinear_GEMM(in_features=3584, out_features=3584, bias=False, w_bit=4, group_size=128)
#         (rotary_emb): Qwen2_5_VLRotaryEmbedding()
#       )
#       (mlp): Qwen2MLP(
#         (gate_proj): WQLinear_GEMM(in_features=3584, out_features=18944, bias=False, w_bit=4, group_size=128)
#         (up_proj): WQLinear_GEMM(in_features=3584, out_features=18944, bias=False, w_bit=4, group_size=128)
#         (down_proj): WQLinear_GEMM(in_features=18944, out_features=3584, bias=False, w_bit=4, group_size=128)
#         (act_fn): SiLU()
#       )
#       (input_layernorm): Qwen2RMSNorm((3584,), eps=1e-06)
#       (post_attention_layernorm): Qwen2RMSNorm((3584,), eps=1e-06)
#     )
#   )
#   (norm): Qwen2RMSNorm((3584,), eps=1e-06)
#   (rotary_emb): Qwen2_5_VLRotaryEmbedding()