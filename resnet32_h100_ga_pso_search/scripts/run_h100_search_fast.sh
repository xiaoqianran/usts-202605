#!/usr/bin/env bash
set -euo pipefail
python search_channels_ga_pso.py \
  --algorithm both \
  --search-epochs 1 \
  --batch-size 2048 \
  --num-workers 8 \
  --max-train-samples 10000 \
  --max-val-samples 5000 \
  --ga-population 8 \
  --ga-generations 5 \
  --pso-particles 8 \
  --pso-iterations 5 \
  --baseline-ckpt runs/resnet32_baseline/best.pt \
  --amp --amp-dtype bf16 --channels-last
