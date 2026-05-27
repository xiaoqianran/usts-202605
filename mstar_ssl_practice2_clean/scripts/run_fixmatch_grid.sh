#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-data/MSTAR}"
EPOCHS="${EPOCHS:-80}"
SEED="${SEED:-42}"
LR_POLICY="${LR_POLICY:-scaled}"
RUNS_DIR="${RUNS_DIR:-runs_fixmatch_grid_${LR_POLICY}}"
SUMMARY_OUT="${SUMMARY_OUT:-${RUNS_DIR}/summary}"
PARALLEL_JOBS="${PARALLEL_JOBS:-4}"

batches=(32 64 96 128)
mus=(2 4)

lr_for_batch() {
  local batch="$1"
  case "$LR_POLICY" in
    fixed)
      printf "0.001"
      ;;
    scaled)
      case "$batch" in
        32) printf "0.0005" ;;
        64) printf "0.001" ;;
        96) printf "0.0015" ;;
        128) printf "0.002" ;;
        *) echo "Unsupported batch size: $batch" >&2; exit 2 ;;
      esac
      ;;
    *)
      echo "LR_POLICY must be fixed or scaled, got: $LR_POLICY" >&2
      exit 2
      ;;
  esac
}

mkdir -p "$RUNS_DIR"
run_rc=0
running_jobs=0

wait_for_one() {
  if ! wait -n; then
    run_rc=1
  fi
  running_jobs=$((running_jobs - 1))
}

for batch in "${batches[@]}"; do
  lr="$(lr_for_batch "$batch")"
  for mu in "${mus[@]}"; do
    out_dir="$RUNS_DIR/fixmatch_b${batch}_mu${mu}"
    mkdir -p "$out_dir"
    echo "Starting batch=${batch} mu=${mu} lr=${lr} -> ${out_dir}"
    python train.py \
      --mode fixmatch \
      --data-root "$DATA_ROOT" \
      --label-ratio 0.1 \
      --epochs "$EPOCHS" \
      --batch-size "$batch" \
      --mu "$mu" \
      --lr "$lr" \
      --seed "$SEED" \
      --threshold 0.95 \
      --lambda-u 1.0 \
      --out "$out_dir" \
      > "$out_dir/train.log" 2>&1 &
    running_jobs=$((running_jobs + 1))
    if [ "$running_jobs" -ge "$PARALLEL_JOBS" ]; then
      wait_for_one
    fi
  done
done

while [ "$running_jobs" -gt 0 ]; do
  wait_for_one
done

python tools/summarize_fixmatch_grid.py --runs "$RUNS_DIR" --out "$SUMMARY_OUT"
exit "$run_rc"
