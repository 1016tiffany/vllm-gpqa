# import torch
# import numpy as np
# from collections import Counter
# from sklearn.cluster import KMeans
# from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoModelForCausalLM
# # from qwen_vl_utils import process_vision_info
# from datasets import load_dataset
# import argparse
# from tqdm import tqdm
# import os

# # Hook for capturing activations
# layer_outputs = {}
# def hook_fn(module, input, output, layer_id):
#         # organiz into token wise output here
#         if f"layer_{layer_id}" not in layer_outputs:
#             layer_outputs[f"layer_{layer_id}"] = []
#         if(output[0].shape[0]> 100):
#             layer_outputs[f"layer_{layer_id}"].append(output[0].cpu().clone().detach().squeeze(0))

# def run_single_forward_pass(image, question, processor, model):
#     # If this is a multimodal processor, do image+text, otherwise text-only:
#     if hasattr(processor, "image_processor"):
#         # vision+text model
#         inputs = processor(
#             images=image,
#             text=question,
#             return_tensors="pt",
#             padding=True,
#         ).to(model.device)
#     else:
#         # pure-text model (e.g. Phi-tiny)
#         inputs = processor(
#             question,
#             return_tensors="pt",
#             padding=True,
#         ).to(model.device)
    
    
#     # Run forward pass to get logits
#     with torch.inference_mode():
#         outputs = model(**inputs, use_cache=False)
#         #logits = outputs.logits
    
#     return

# def calculate_kmeans_entropy(activations, n_clusters=100):
#     """Calculate entropy based on token clustering."""
#     # Flatten activations to (n_tokens, hidden_dim)
#     flat_acts = activations.reshape(-1, activations.shape[-1]).cpu().numpy()
   
#     # Run k-means clustering
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#     cluster_labels = kmeans.fit_predict(flat_acts)
   
#     # Calculate entropy from cluster distribution
#     counts = Counter(cluster_labels)
#     total = len(cluster_labels)
#     probs = np.array([count/total for count in counts.values()])
#     entropy = -np.sum(probs * np.log2(probs + 1e-10))
   
#     # Calculate average cluster size
#     avg_cluster_size = np.mean(list(counts.values()))
#     print(f'Average cluster size: {avg_cluster_size}')
   
#     # Calculate average variance (radius) of clusters
#     cluster_variances = []
#     for i in range(n_clusters):
#         cluster_points = flat_acts[cluster_labels == i]
#         if len(cluster_points) > 0:
#             cluster_variances.append(np.var(cluster_points, axis=0).mean())
#     avg_cluster_variance = np.mean(cluster_variances)
#     print(f'Average cluster variance (radius): {avg_cluster_variance}')
   
#     # Calculate average inter-cluster distance
#     cluster_centers = kmeans.cluster_centers_
#     inter_cluster_distances = []
#     for i in range(n_clusters):
#         for j in range(i + 1, n_clusters):
#             inter_cluster_distances.append(np.linalg.norm(cluster_centers[i] - cluster_centers[j]))
#     avg_inter_cluster_distance = np.mean(inter_cluster_distances)
#     print(f'Average inter-cluster distance: {avg_inter_cluster_distance}')
   
#     # Calculate average intra-cluster distance
#     intra_cluster_distances = []
#     for i in range(n_clusters):
#         cluster_points = flat_acts[cluster_labels == i]
#         if len(cluster_points) > 0:
#             distances = np.linalg.norm(cluster_points - cluster_centers[i], axis=1)
#             intra_cluster_distances.append(np.mean(distances))
#     avg_intra_cluster_distance = np.mean(intra_cluster_distances)
#     print(f'Average intra-cluster distance: {avg_intra_cluster_distance}')
    
#     return entropy

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
#     parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
#     parser.add_argument("--num_samples", type=int, default=100)  # Reduced default for testing
#     parser.add_argument("--batch_size", type=int, default=1)
#     parser.add_argument("--n_clusters", type=int, default=10)
#     parser.add_argument("--save_act", action="store_true", help="Save activations to file")
#     parser.add_argument("--output_dir", type=str, default="./outputs/")
#     parser.add_argument("--activation_file", type=str, default="all_layer_activations.pth")
#     parser.add_argument("--sorted_layers_file", type=str, default="sorted_layers_qwen.txt")
#     args = parser.parse_args()
    
#     # Create output directory if it doesn't exist
#     if not os.path.exists(args.output_dir):
#         os.makedirs(args.output_dir)
#     args.activation_file = os.path.join(args.output_dir, args.activation_file)
#     args.sorted_layers_file = os.path.join(args.output_dir, args.sorted_layers_file)
#     # Layers to analyze (all 28 layers)
     
    
#     # Load the model and processor
#     print(f"Loading model {args.model_name}...")
#     print(args.device)
#     if(args.model_name == "Qwen/Qwen2.5-VL-7B-Instruct"):
#         model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#             args.model_name, device_map=args.device, torch_dtype=torch.float16
#         )
#         layers_entropy_calc = list(range(1,28))
#     else:
#         model = AutoModelForCausalLM.from_pretrained(
#             args.model_name, device_map=args.device, torch_dtype=torch.float16, trust_remote_code=True
#         )
#         # skip the projector layer, layer 0
#         num_of_layers = len(model.model.layers)
#         layers_entropy_calc = list(range(0,num_of_layers))
        
    
#     model.to(args.device)
#     processor = AutoProcessor.from_pretrained(args.model_name)
    
#     # Register hooks for each layer
#     hooks = []
#     breakpoint()
#     for layer_num in layers_entropy_calc:
#         # layer = model.model.language_model.layers[layer_num].input_layernorm
#         layer = model.model.layers[layer_num].input_layernorm    
#         temp_hook = layer.register_forward_hook(
#             lambda mod, inp, out, layer_id=layer_num: hook_fn(mod, inp, out, layer_id)
#         )
        
#         hooks.append(temp_hook)
    
#     # Choose dataset based on model type
#     if "Phi-tiny-MoE-instruct" in args.model_name or "Qwen3" in args.model_name:
#         print("Loading WikiText-2 (text-only) dataset...")
#         dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=f"test[:100]")
#     else:
#         print("Loading TextVQA dataset...")
#         dataset = load_dataset("lmms-lab/textvqa", split=f"train[:100]")
    
    
#     # Process images and calculate entropy for each layer
#     #layer_entropies = {f"layer_{layer_num}": [] for layer_num in layers_entropy_calc}
#     num_samples = min(100, len(dataset))
#     print(f"Processing {args.num_samples} samples...")
#     for example in tqdm(dataset.select(range(num_samples)), desc="Processing samples"):
#         if "image" in example and "question" in example:
#             # multimodal example
#             image  = example["image"]
#             prompt = example["question"]
#         elif "text" in example:
#             # text-only example
#             prompt = example["text"]
#             if not prompt.strip():         # <-- NEW guard
#                 continue                   # skip blank line
#             image  = None
#         else:
#             # unexpected schema – skip
#             continue

#         run_single_forward_pass(image, prompt, processor, model)

            
#     for hook in hooks:
#             hook.remove()
    
#     if args.save_act:
#         torch.save(layer_outputs, args.activation_file)
#     # Print layers sorted by entropy
#     # Lists to store layer numbers and their corresponding entropies
#     layer_nums = []
#     entropies = []
    
#     # Calculate entropy for each layer
#     for temp_layer_id in tqdm(layers_entropy_calc, desc="Calculating entropy for layers"):
#         activation_temp = torch.vstack(layer_outputs[f"layer_{temp_layer_id}"])
#         entropy1 = calculate_kmeans_entropy(activation_temp)
#         print(f'Entropy of layer {temp_layer_id} data = {entropy1}')
#         layer_nums.append(temp_layer_id)
#         entropies.append(entropy1)
    
#     # Sort layers by entropy
#     sorted_indices = np.argsort(entropies)
#     sorted_layers = [layer_nums[i] for i in sorted_indices]
#     sorted_entropies = [entropies[i] for i in sorted_indices]
#     # Save sorted layers as txt
    
    
#     # Load sorted layers
#     # with open(sorted_layers_file, "r") as f:
#     #     loaded_layers = [int(x) for x in f.read().split(",")]

#     print("\nLayers sorted by entropy (ascending):")
#     for layer, entropy in zip(sorted_layers, sorted_entropies):
#         print(f"Layer {layer}: {entropy}")
    
#     # sorted_layers.append(0) # add the projector layer back for consistency
#     with open(args.sorted_layers_file, "w") as f:
#         f.write(",".join(map(str, sorted_layers)))
#     print(f"\nSorted layers saved to {args.sorted_layers_file}")
    
# if __name__ == "__main__":
#     main() 

#!/usr/bin/env python
import argparse
import os
import random
from collections import Counter

import numpy as np
import torch
from datasets import load_dataset
from sklearn.cluster import KMeans
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

# ────────────────────────── globals ──────────────────────────
layer_outputs = {}      # { "layer_i": [Tensor[T, H], ...], ... }
_current_mask = None    # attention mask for the most recent batch (used by hooks)


# ────────────────────────── hooks ──────────────────────────
def hook_fn(module, inp, out, layer_id: int):
    """
    Capture per-token hidden states from LayerNorm outputs.
    Expected `out` shape: [B, T, H].
    We (deterministically) take the first example in the batch, and strip padding tokens.
    """
    key = f"layer_{layer_id}"
    hs = out.detach().to("cpu")  # [B, T, H]

    # deterministically keep the first batch element
    if hs.dim() != 3:
        # fallback: if some model returns [T, H] already
        pass
    elif hs.size(0) > 1:
        hs = hs[0]
    else:
        hs = hs.squeeze(0)  # -> [T, H]

    # strip padding positions if we have an attention mask
    global _current_mask
    if _current_mask is not None:
        m = _current_mask[0].to(torch.bool).cpu()  # [T]
        if m.numel() == hs.size(0):
            hs = hs[m]

    layer_outputs.setdefault(key, []).append(hs)  # append [T, H]


# ────────────────────────── forward ──────────────────────────
def run_single_forward_pass(image, question, processor, model):
    """
    Builds inputs for either VL or text-only processors, stores attention_mask globally,
    and runs a single forward pass (no cache, no generation).
    """
    global _current_mask

    if hasattr(processor, "image_processor"):
        # vision+text
        inputs = processor(
            images=image,
            text=question,
            return_tensors="pt",
            padding=True,
        ).to(model.device)
    else:
        # text-only
        inputs = processor(
            question,
            return_tensors="pt",
            padding=True,
        ).to(model.device)

    _current_mask = inputs.get("attention_mask", None)

    with torch.inference_mode():
        _ = model(**inputs, use_cache=False)


# ────────────────────────── metrics ──────────────────────────
def calculate_kmeans_entropy(activations: torch.Tensor, n_clusters: int = 100):
    """
    activations: Tensor of shape [N_tokens, H]
    Returns: Shannon entropy of the token-to-cluster distribution (base-2).
    Also prints a few diagnostic stats in a deterministic way.
    """
    flat_acts = activations.reshape(-1, activations.shape[-1]).to(torch.float32).cpu().numpy()

    # Stable KMeans (random_state + explicit n_init)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(flat_acts)

    # entropy
    counts = Counter(cluster_labels)
    total = len(cluster_labels)
    probs = np.array([count / total for count in counts.values()], dtype=np.float64)
    entropy = -np.sum(probs * np.log2(probs + 1e-10))

    # diagnostics (optional but handy)
    avg_cluster_size = float(np.mean(list(counts.values())))
    print(f"Average cluster size: {avg_cluster_size:.4f}")

    cluster_variances = []
    for i in range(n_clusters):
        pts = flat_acts[cluster_labels == i]
        if len(pts) > 0:
            cluster_variances.append(np.var(pts, axis=0).mean())
    if cluster_variances:
        avg_cluster_variance = float(np.mean(cluster_variances))
        print(f"Average cluster variance (radius): {avg_cluster_variance:.6f}")

    centers = kmeans.cluster_centers_
    inter_d = []
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            inter_d.append(np.linalg.norm(centers[i] - centers[j]))
    if inter_d:
        avg_inter = float(np.mean(inter_d))
        print(f"Average inter-cluster distance: {avg_inter:.6f}")

    intra_d = []
    for i in range(n_clusters):
        pts = flat_acts[cluster_labels == i]
        if len(pts) > 0:
            d = np.linalg.norm(pts - centers[i], axis=1)
            intra_d.append(np.mean(d))
    if intra_d:
        avg_intra = float(np.mean(intra_d))
        print(f"Average intra-cluster distance: {avg_intra:.6f}")

    return float(entropy)


# ────────────────────────── main ──────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--n_clusters", type=int, default=10)
    parser.add_argument("--save_act", action="store_true", help="Save activations to file")
    parser.add_argument("--output_dir", type=str, default="./outputs/")
    parser.add_argument("--activation_file", type=str, default="all_layer_activations.pth")
    parser.add_argument("--sorted_layers_file", type=str, default="sorted_layers_qwen.txt")
    args = parser.parse_args()

    # seeds + deterministic backends
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # outputs
    os.makedirs(args.output_dir, exist_ok=True)
    args.activation_file = os.path.join(args.output_dir, args.activation_file)
    args.sorted_layers_file = os.path.join(args.output_dir, args.sorted_layers_file)

    # load model
    print(f"Loading model {args.model_name} on {args.device} ...")
    if args.model_name == "Qwen/Qwen2.5-VL-7B-Instruct":
        # prefer device_map="auto" for large models (don’t also call .to())
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_name, device_map="auto", torch_dtype=torch.float16
        )
        # Qwen2.5-VL has 28 decoder layers typically; include 0..27
        layers_entropy_calc = list(range(0, 28))
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
        )
        # generic transformer path
        try:
            num_layers = len(model.model.layers)
            layers_entropy_calc = list(range(num_layers))
        except Exception as e:
            raise RuntimeError(f"Could not infer layers from model: {e}")

    model.eval()

    # load processor (AutoProcessor works for both VL and many text models;
    # if it falls back to tokenizer, it still won’t have image_processor attr)
    processor = AutoProcessor.from_pretrained(args.model_name)

    # register hooks on input_layernorm of each decoder layer
    hooks = []
    for i in layers_entropy_calc:
        try:
            layer = model.model.layers[i].input_layernorm
        except Exception as e:
            raise RuntimeError(f"Failed to access input_layernorm for layer {i}: {e}")
        # capture `i` by value
        h = layer.register_forward_hook(lambda m, inp, out, layer_id=i: hook_fn(m, inp, out, layer_id))
        hooks.append(h)

    # dataset selection (fixed conditional)
    # If model name contains either token, we assume text-only; else use TextVQA for VL.
    if ("Phi-tiny-MoE-instruct" in args.model_name) or ("Qwen3" in args.model_name):
        print("Loading WikiText-2 (text-only) dataset...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test[:2000]")
        is_text = True
    else:
        print("Loading TextVQA dataset...")
        dataset = load_dataset("lmms-lab/textvqa", split="train[:2000]")
        is_text = False

    num_samples = min(args.num_samples, len(dataset))
    print(f"Processing {num_samples} samples...")

    # run passes
    sel = dataset.select(range(num_samples))
    for ex in tqdm(sel, desc="Processing samples"):
        if not is_text and ("image" in ex and "question" in ex):
            image = ex["image"]
            prompt = ex["question"]
            run_single_forward_pass(image, prompt, processor, model)
        elif is_text and ("text" in ex):
            prompt = ex["text"]
            if prompt and prompt.strip():
                run_single_forward_pass(None, prompt, processor, model)
        # else: skip unexpected schema

    # remove hooks
    for h in hooks:
        h.remove()

    # optionally save raw activations
    if args.save_act:
        torch.save(layer_outputs, args.activation_file)
        print(f"Saved activations to: {args.activation_file}")

    # compute entropy per layer
    layer_nums = []
    entropies = []
    print("Calculating entropy for layers...")
    for lid in tqdm(layers_entropy_calc, desc="Entropy"):
        key = f"layer_{lid}"
        if key not in layer_outputs or len(layer_outputs[key]) == 0:
            print(f"Warning: no activations captured for {key}; skipping.")
            continue
        # stack all [T, H] blocks → [sum_T, H]
        acts = torch.vstack(layer_outputs[key])
        ent = calculate_kmeans_entropy(acts, n_clusters=args.n_clusters)
        print(f"Entropy of layer {lid} = {ent:.6f}")
        layer_nums.append(lid)
        entropies.append(ent)

    # sort ascending by entropy
    if len(entropies) == 0:
        raise RuntimeError("No entropies computed. Check hooks/dataset paths.")

    sorted_idx = np.argsort(entropies)
    sorted_layers = [layer_nums[i] for i in sorted_idx]
    sorted_entropies = [entropies[i] for i in sorted_idx]

    print("\nLayers sorted by entropy (ascending):")
    for L, E in zip(sorted_layers, sorted_entropies):
        print(f"Layer {L}: {E:.6f}")

    with open(args.sorted_layers_file, "w") as f:
        f.write(",".join(map(str, sorted_layers)))
    print(f"\nSorted layers saved to {args.sorted_layers_file}")


if __name__ == "__main__":
    main()
