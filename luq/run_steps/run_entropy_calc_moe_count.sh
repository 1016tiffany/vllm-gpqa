python3 entropy_calc_moe_count.py \
    --model_name "allenai/OLMoE-1B-7B-0924" \
    --num_samples 100 \
    --n_clusters 20 \
    --output_dir "./outputs/" \
    --sorted_file "sorted_experts_olmoe2.txt" \
    --counts_file "expert_activation_counts_olmoe.tsv"