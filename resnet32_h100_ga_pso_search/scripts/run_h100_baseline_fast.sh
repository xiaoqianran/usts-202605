#!/usr/bin/env bash
set -euo pipefail
python train_resnet32.py \
  --run-name resnet32_baseline_fast \
  --epochs 200 \
  --batch-size 1024 \
  --num-workers 8 \
  --lr 0.1 \
  --milestones 100,150 \
  --amp --amp-dtype bf16 --channels-last
