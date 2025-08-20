python luq_phi_mini_moe.py \
        --model_4bit_dir  outputs/phi_mini_moe_2bit \
        --model_2bit_dir  outputs/phi_mini_moe_4bit \
        --entropy_file    outputs/sorted_experts_phi_mini_moe.txt \
        --top_k           256 \
        --output_dir      outputs/phi_mini_moe_luq