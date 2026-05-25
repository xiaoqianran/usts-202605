#!/usr/bin/env bash
set -euo pipefail
python train_width_resnet32.py \
  --channels 16,24,48 \
  --run-name final_width_16-24-48 \
  --epochs 80 \
  --milestones 40,60 \
  --amp
