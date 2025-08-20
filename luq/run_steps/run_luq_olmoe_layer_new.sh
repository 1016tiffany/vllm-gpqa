# experts only
# python luq_olmoe_layer_new.py \
#   --model4 outputs/olmoe-hqq-4bit \
#   --model2 outputs/olmoe-hqq-2bit \
#   --order_layers outputs/sorted_layers_olmoe.txt \
#   --num2 8 \
#   --swap experts_only \
#   --out outputs/olmoe-luq-layer-8-new \
#   --dtype bfloat16

# Experts + Attention in the lowest-entropy 8 layers
python luq_olmoe_layer_new.py \
  --model4 outputs/olmoe-hqq-4bit \
  --model2 outputs/olmoe-hqq-2bit \
  --order_layers outputs/sorted_layers_olmoe.txt \
  --num2 8 \
  --swap experts_attn \
  --out outputs/olmoe-luq-layer-8_experts+attn

# Experts + Router in the top 8 layers
python luq_olmoe_layer_new.py \
  --model4 outputs/olmoe-hqq-4bit \
  --model2 outputs/olmoe-hqq-2bit \
  --order_layers outputs/sorted_layers_olmoe.txt \
  --num2 8 \
  --swap experts_router \
  --out outputs/olmoe-luq-layer-8_experts+router
