#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-data/MSTAR}"
EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SEED="${SEED:-42}"

python src/inspect_dataset.py --data-root "$DATA_ROOT"

python train_supervised.py \
  --data-root "$DATA_ROOT" \
  --label-ratio 1.0 \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --seed "$SEED" \
  --out runs/01_supervised_full

python train_supervised.py \
  --data-root "$DATA_ROOT" \
  --label-ratio 0.1 \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --seed "$SEED" \
  --out runs/02_supervised_10percent

python train_fixmatch.py \
  --data-root "$DATA_ROOT" \
  --label-ratio 0.1 \
  --epochs "$EPOCHS" \
  --batch-size 32 \
  --mu 4 \
  --threshold 0.95 \
  --lambda-u 1.0 \
  --seed "$SEED" \
  --out runs/03_fixmatch_10percent
