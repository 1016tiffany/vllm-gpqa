import torch
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoModelForCausalLM
from qwen_vl_utils import process_vision_info
from datasets import load_dataset
import argparse
from tqdm import tqdm
import os

# Hook for capturing activations
layer_outputs = {}
def hook_fn(module, input, output, layer_id):
        # organiz into token wise output here
        if f"layer_{layer_id}" not in layer_outputs:
            layer_outputs[f"layer_{layer_id}"] = []
        if(output[0].shape[0]> 100):
            layer_outputs[f"layer_{layer_id}"].append(output[0].cpu().clone().detach().squeeze(0))

def run_single_forward_pass(image, question, processor, model):
    # Process the image and text
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,  # Use PIL image object here
                },
                {"type": "text", "text": question},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(  text=[text],  images=image_inputs, padding=True,  return_tensors="pt" ).to(model.device)
    
    
    # Run forward pass to get logits
    with torch.inference_mode():
        outputs = model(**inputs, use_cache=False)
        #logits = outputs.logits
    
    return

def calculate_kmeans_entropy(activations, n_clusters=100):
    """Calculate entropy based on token clustering."""
    # Flatten activations to (n_tokens, hidden_dim)
    flat_acts = activations.reshape(-1, activations.shape[-1]).cpu().numpy()
   
    # Run k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(flat_acts)
   
    # Calculate entropy from cluster distribution
    counts = Counter(cluster_labels)
    total = len(cluster_labels)
    probs = np.array([count/total for count in counts.values()])
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
   
    # Calculate average cluster size
    avg_cluster_size = np.mean(list(counts.values()))
    print(f'Average cluster size: {avg_cluster_size}')
   
    # Calculate average variance (radius) of clusters
    cluster_variances = []
    for i in range(n_clusters):
        cluster_points = flat_acts[cluster_labels == i]
        if len(cluster_points) > 0:
            cluster_variances.append(np.var(cluster_points, axis=0).mean())
    avg_cluster_variance = np.mean(cluster_variances)
    print(f'Average cluster variance (radius): {avg_cluster_variance}')
   
    # Calculate average inter-cluster distance
    cluster_centers = kmeans.cluster_centers_
    inter_cluster_distances = []
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            inter_cluster_distances.append(np.linalg.norm(cluster_centers[i] - cluster_centers[j]))
    avg_inter_cluster_distance = np.mean(inter_cluster_distances)
    print(f'Average inter-cluster distance: {avg_inter_cluster_distance}')
   
    # Calculate average intra-cluster distance
    intra_cluster_distances = []
    for i in range(n_clusters):
        cluster_points = flat_acts[cluster_labels == i]
        if len(cluster_points) > 0:
            distances = np.linalg.norm(cluster_points - cluster_centers[i], axis=1)
            intra_cluster_distances.append(np.mean(distances))
    avg_intra_cluster_distance = np.mean(intra_cluster_distances)
    print(f'Average intra-cluster distance: {avg_intra_cluster_distance}')
    
    return entropy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_samples", type=int, default=100)  # Reduced default for testing
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_clusters", type=int, default=10)
    parser.add_argument("--save_act", action="store_true", help="Save activations to file")
    parser.add_argument("--output_dir", type=str, default="./outputs/")
    parser.add_argument("--activation_file", type=str, default="all_layer_activations.pth")
    parser.add_argument("--sorted_layers_file", type=str, default="sorted_layers_qwen.txt")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.activation_file = os.path.join(args.output_dir, args.activation_file)
    args.sorted_layers_file = os.path.join(args.output_dir, args.sorted_layers_file)
    # Layers to analyze (all 28 layers)
     
    
    # Load the model and processor
    print(f"Loading model {args.model_name}...")
    if(args.model_name == "Qwen/Qwen2.5-VL-7B-Instruct"):
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_name, device_map=args.device, torch_dtype=torch.float16
        )
        layers_entropy_calc = list(range(0,28))
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, device_map=args.device, torch_dtype=torch.float16
        )
        num_of_layers = len(model.model.layers)
        layers_entropy_calc = list(range(0,num_of_layers))
        
    
    model.to(args.device)
    processor = AutoProcessor.from_pretrained(args.model_name)
    
    # Register hooks for each layer
    hooks = []
    for layer_num in layers_entropy_calc:
        layer = model.model.layers[layer_num].input_layernorm
        temp_hook = layer.register_forward_hook(
            lambda mod, inp, out, layer_id=layer_num: hook_fn(mod, inp, out, layer_id)
        )
        
        hooks.append(temp_hook)
    
    # Load TextVQA dataset
    print("Loading TextVQA dataset...")
    
    dataset = load_dataset("textvqa", split="train[:100]")
    
    
    # Process images and calculate entropy for each layer
    #layer_entropies = {f"layer_{layer_num}": [] for layer_num in layers_entropy_calc}
    num_samples = min(100, len(dataset))
    print(f"Processing {args.num_samples} samples...")
    for i in tqdm(range(num_samples)):
        
        example = dataset[i]
        # Process the image
        #if "image" in example:
        image = example["image"]
        prompt = example["question"]
        run_single_forward_pass(image, prompt, processor, model)
            
    for hook in hooks:
            hook.remove()
    
    if args.save_act:
        torch.save(layer_outputs, args.activation_file)
    # Print layers sorted by entropy
    # Lists to store layer numbers and their corresponding entropies
    layer_nums = []
    entropies = []
    
    # Calculate entropy for each layer
    for temp_layer_id in tqdm(layers_entropy_calc, desc="Calculating entropy for layers"):
        activation_temp = torch.vstack(layer_outputs[f"layer_{temp_layer_id}"])
        entropy1 = calculate_kmeans_entropy(activation_temp)
        print(f'Entropy of layer {temp_layer_id} data = {entropy1}')
        layer_nums.append(temp_layer_id)
        entropies.append(entropy1)
    
    # Sort layers by entropy
    sorted_indices = np.argsort(entropies)
    sorted_layers = [layer_nums[i] for i in sorted_indices]
    sorted_entropies = [entropies[i] for i in sorted_indices]
    # Save sorted layers as txt
    
    
    # Load sorted layers
    # with open(sorted_layers_file, "r") as f:
    #     loaded_layers = [int(x) for x in f.read().split(",")]

    print("\nLayers sorted by entropy (ascending):")
    for layer, entropy in zip(sorted_layers, sorted_entropies):
        print(f"Layer {layer}: {entropy}")
    
    # sorted_layers.append(0) # add the projector layer back for consistency
    with open(args.sorted_layers_file, "w") as f:
        f.write(",".join(map(str, sorted_layers)))
    print(f"\nSorted layers saved to {args.sorted_layers_file}")
    
if __name__ == "__main__":
    main() 