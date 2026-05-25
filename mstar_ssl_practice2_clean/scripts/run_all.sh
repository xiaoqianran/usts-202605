#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-data/MSTAR}"
EPOCHS="${EPOCHS:-80}"

python train.py \
  --mode supervised \
  --data-root "$DATA_ROOT" \
  --label-ratio 1.0 \
  --epochs "$EPOCHS" \
  --batch-size 64 \
  --out runs/01_supervised_full

python train.py \
  --mode supervised \
  --data-root "$DATA_ROOT" \
  --label-ratio 0.1 \
  --epochs "$EPOCHS" \
  --batch-size 64 \
  --out runs/02_supervised_10percent

python train.py \
  --mode fixmatch \
  --data-root "$DATA_ROOT" \
  --label-ratio 0.1 \
  --epochs "$EPOCHS" \
  --batch-size 32 \
  --mu 4 \
  --threshold 0.95 \
  --lambda-u 1.0 \
  --out runs/03_fixmatch_10percent

python tools/make_presentation_assets.py --runs runs --out presentation_assets

