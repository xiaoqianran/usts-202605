#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-data/MSTAR}"
EPOCHS="${EPOCHS:-80}"
LR="${LR:-0.0015}"
RUNS_DIR="${RUNS_DIR:-runs_batch96}"
ASSETS_DIR="${ASSETS_DIR:-presentation_assets_batch96}"

mkdir -p "$RUNS_DIR"

python train.py \
  --mode supervised \
  --data-root "$DATA_ROOT" \
  --label-ratio 1.0 \
  --epochs "$EPOCHS" \
  --batch-size 96 \
  --lr "$LR" \
  --out "$RUNS_DIR/01_supervised_full" \
  > "$RUNS_DIR/01_supervised_full.parallel.log" 2>&1 &
pid_full=$!

python train.py \
  --mode supervised \
  --data-root "$DATA_ROOT" \
  --label-ratio 0.1 \
  --epochs "$EPOCHS" \
  --batch-size 96 \
  --lr "$LR" \
  --out "$RUNS_DIR/02_supervised_10percent" \
  > "$RUNS_DIR/02_supervised_10percent.parallel.log" 2>&1 &
pid_limited=$!

python train.py \
  --mode fixmatch \
  --data-root "$DATA_ROOT" \
  --label-ratio 0.1 \
  --epochs "$EPOCHS" \
  --batch-size 96 \
  --mu 2 \
  --lr "$LR" \
  --threshold 0.95 \
  --lambda-u 1.0 \
  --out "$RUNS_DIR/03_fixmatch_10percent" \
  > "$RUNS_DIR/03_fixmatch_10percent.parallel.log" 2>&1 &
pid_fixmatch=$!

run_rc=0
wait "$pid_full" || run_rc=1
wait "$pid_limited" || run_rc=1
wait "$pid_fixmatch" || run_rc=1

if [ "$run_rc" -eq 0 ]; then
  python tools/make_presentation_assets.py --runs "$RUNS_DIR" --out "$ASSETS_DIR"
fi

exit "$run_rc"
