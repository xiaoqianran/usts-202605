#!/usr/bin/env bash
set -euo pipefail
python train_block_width_resnet32.py \
  --block-channels 16,16,16,16,16,32,28,24,24,28,64,56,48,48,56 \
  --run-name final_block_example \
  --epochs 80 \
  --milestones 40,60 \
  --batch-size 1024 \
  --num-workers 8 \
  --baseline-ckpt runs/resnet32_baseline/best.pt \
  --amp \
  --amp-dtype bf16 \
  --channels-last
