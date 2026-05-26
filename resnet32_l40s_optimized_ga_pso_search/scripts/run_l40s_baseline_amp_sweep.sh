#!/usr/bin/env bash
set -euo pipefail

batches=(128 256 512 1024)
dtypes=(fp16 bf16)

for batch in "${batches[@]}"; do
  for dtype in "${dtypes[@]}"; do
    python train_resnet32.py \
      --run-name "resnet32_l40s_b${batch}_${dtype}" \
      --epochs 200 \
      --batch-size "$batch" \
      --lr 0.1 \
      --milestones 100,150 \
      --num-workers 8 \
      --amp \
      --amp-dtype "$dtype" \
      --channels-last
  done
done
