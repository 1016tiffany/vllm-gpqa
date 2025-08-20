python luq_olmoe_layer.py \
    --model4 outputs/olmoe-hqq-4bit \
    --model2 outputs/olmoe-hqq-2bit \
    --order_layers outputs/sorted_layers_olmoe.txt \
    --num2 8 \
    --out outputs/olmoe-luq-layer \
    --dtype bfloat16