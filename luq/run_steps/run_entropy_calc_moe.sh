python3 entropy_calc_moe.py \
    --model_name "allenai/OLMoE-1B-7B-0924" \
    --num_samples 100 \
    --n_clusters 20 \
    --output_dir "./outputs/" \
    --sorted_file "sorted_experts_olmoe.txt"