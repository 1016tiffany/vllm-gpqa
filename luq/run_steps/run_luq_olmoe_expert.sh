python luq_olmoe_expert.py \
  --model4 outputs/olmoe-hqq-4bit \
  --model2 outputs/olmoe-hqq-2bit \
  --order  outputs/sorted_experts_olmoe.txt \
  --keep   512 \
  --out    outputs/olmoe-luq-expert \
  --dtype  bfloat16
