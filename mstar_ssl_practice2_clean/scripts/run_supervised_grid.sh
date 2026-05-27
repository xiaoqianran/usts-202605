#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-data/MSTAR}"
EPOCHS="${EPOCHS:-80}"
SEED="${SEED:-42}"
LR_POLICY="${LR_POLICY:-scaled}"
RUNS_DIR="${RUNS_DIR:-runs_supervised_grid_${LR_POLICY}}"
SUMMARY_OUT="${SUMMARY_OUT:-${RUNS_DIR}/summary}"
PARALLEL_JOBS="${PARALLEL_JOBS:-4}"

batches=(32 64 96 128)
ratios=("1.0:full" "0.1:10percent")

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

echo "=== 监督学习网格实验运行器 ==="
echo "学习率策略=$LR_POLICY  训练轮数=$EPOCHS  随机种子=$SEED  并行任务数=$PARALLEL_JOBS"
echo "Batch 列表: ${batches[*]}   标签比例: 全标签(1.0) + 10%(0.1)"
echo "输出目录=$RUNS_DIR"
echo "汇总输出=$SUMMARY_OUT"
echo

mkdir -p "$RUNS_DIR"
run_rc=0
running_jobs=0

wait_for_one() {
  if ! wait -n; then
    run_rc=1
  fi
  running_jobs=$((running_jobs - 1))
}

total_planned=8
completed_before=0
for batch in "${batches[@]}"; do
  for item in "${ratios[@]}"; do
    ratio="${item%%:*}"
    name="${item##*:}"
    out_dir="$RUNS_DIR/supervised_${name}_b${batch}"
    if [ -f "$out_dir/metrics.json" ]; then
      completed_before=$((completed_before + 1))
    fi
  done
done
echo "本次运行前已完成: $completed_before / $total_planned"
echo

for batch in "${batches[@]}"; do
  lr="$(lr_for_batch "$batch")"
  for item in "${ratios[@]}"; do
    ratio="${item%%:*}"
    name="${item##*:}"
    out_dir="$RUNS_DIR/supervised_${name}_b${batch}"
    mkdir -p "$out_dir"
    if [ -f "$out_dir/metrics.json" ]; then
      echo "[跳过] 已完成: $out_dir"
      continue
    fi
    echo "启动 label_ratio=${ratio} batch=${batch} lr=${lr} -> ${out_dir}"
    python train.py \
      --mode supervised \
      --data-root "$DATA_ROOT" \
      --label-ratio "$ratio" \
      --epochs "$EPOCHS" \
      --batch-size "$batch" \
      --lr "$lr" \
      --seed "$SEED" \
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

echo
echo "=== 所有任务结束 (返回码=$run_rc)。统计完成情况 ==="
completed=0
for batch in "${batches[@]}"; do
  for item in "${ratios[@]}"; do
    ratio="${item%%:*}"
    name="${item##*:}"
    out_dir="$RUNS_DIR/supervised_${name}_b${batch}"
    if [ -f "$out_dir/metrics.json" ]; then
      completed=$((completed + 1))
      acc=$(python3 -c "import json; print(f'{json.load(open(\"$out_dir/metrics.json\"))[\"best_test_acc\"]*100:.2f}%')" 2>/dev/null || echo "?")
      echo "  [完成] $out_dir -> $acc"
    else
      echo "  [未完成] $out_dir (缺少 metrics.json)"
    fi
  done
done
echo "本轮共完成: $completed / $total_planned"

if [ "$completed" -gt 0 ]; then
  echo "正在生成汇总报告..."
  python tools/summarize_supervised_grid.py --runs "$RUNS_DIR" --out "$SUMMARY_OUT" || echo "警告: 汇总脚本报告异常（可能有部分运行未完成）"
else
  echo "没有已完成的运行，跳过生成汇总。"
fi

exit "$run_rc"
