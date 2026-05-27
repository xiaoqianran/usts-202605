#!/usr/bin/env bash
# 网格训练监控脚本（中文版）
# monitor_training.sh - 实时监控 FixMatch / 监督学习网格训练状态
#
# 用法：
#   bash scripts/monitor_training.sh                 # 单次报告所有网格目录状态
#   bash scripts/monitor_training.sh --watch [N]     # 每 N 秒循环监控（默认 30），直到全部完成
#   bash scripts/monitor_training.sh --watch 20 runs_supervised_grid_fixed
#
# 功能：
#   - 显示 GPU 显存、利用率、功耗
#   - 每个 run：状态（已完成 / 运行中）、当前/最优轮次、最新测试精度、日志健康度
#   - 自动扫描 train.log 中的 OOM / CUDA / Killed / 错误
#
# 专为其他 agent 接管监控设计，可直接复制提示词使用。

set -euo pipefail

WATCH_MODE=0
INTERVAL=30
TARGET_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --watch)
      WATCH_MODE=1
      if [[ "$2" =~ ^[0-9]+$ ]]; then
        INTERVAL="$2"; shift
      fi
      ;;
    --once)
      WATCH_MODE=0
      ;;
    *)
      if [ -d "$1" ]; then
        TARGET_DIR="$1"
      else
        echo "未知参数或不是目录: $1" >&2
      fi
      ;;
  esac
  shift
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# 查找所有网格实验目录（FixMatch + 监督，固定/缩放）
if [ -n "$TARGET_DIR" ]; then
  GRID_DIRS=("$TARGET_DIR")
else
  GRID_DIRS=()
  for d in runs_fixmatch_grid_fixed runs_fixmatch_grid_scaled runs_supervised_grid_fixed runs_supervised_grid_scaled; do
    [ -d "$d" ] && GRID_DIRS+=("$d")
  done
fi

# 扫描日志中的致命错误
scan_errors() {
  local logf="$1"
  if [ ! -f "$logf" ]; then
    echo "无日志"
    return
  fi
  if tail -n 50 "$logf" | grep -qiE 'cuda out of memory|oom|killed|segmentation fault|error.*cuda|torch.cuda'; then
    echo "OOM崩溃"
  elif tail -n 20 "$logf" | grep -qiE 'exception|traceback|errno'; then
    echo "错误"
  else
    echo "正常"
  fi
}

# 从 history.csv 获取已训练轮次
get_last_epoch() {
  local hist="$1"
  if [ ! -f "$hist" ]; then
    echo "0"
    return
  fi
  local lines
  lines=$(wc -l < "$hist" 2>/dev/null || echo 1)
  echo $(( lines - 1 ))
}

# 获取最后一行测试精度
get_last_test_acc() {
  local hist="$1"
  if [ ! -f "$hist" ]; then
    echo "n/a"
    return
  fi
  tail -n1 "$hist" 2>/dev/null | awk -F, '{printf "%.2f%%", $6*100}' || echo "n/a"
}

# 打印 GPU 状态
print_gpu() {
  echo "=== GPU 状态 @ $(date '+%H:%M:%S') ==="
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu,power.draw \
      --format=csv,noheader,nounits 2>/dev/null | \
      awk -F, '{printf "  显存: %s MB / %s MB | 利用率=%s%% | 功耗=%s W\n", $2,$3,$4,$5}'
  else
    echo "  （nvidia-smi 不可用）"
  fi
  echo
}

# 打印单个网格目录的状态表格
print_grid_status() {
  local gdir="$1"
  local gname
  gname=$(basename "$gdir")
  echo "=== $gname ==="

  local any_running=0
  local done_cnt=0
  local total_cnt=0

  # 表头
  printf "  %-32s | %-6s | %-6s | %-8s | %-8s\n" "实验目录" "状态" "轮次" "测试精度" "日志状态"
  printf "  %-32s | %-6s | %-6s | %-8s | %-8s\n" "--------------------------------" "------" "------" "--------" "--------"

  for rdir in "$gdir"/*/; do
    [ -d "$rdir" ] || continue
    rname=$(basename "$rdir")
    case "$rname" in
      summary|__pycache__|.*) continue ;;
    esac
    total_cnt=$((total_cnt + 1))

    local status="运行中"
    local acc="?"
    local epochs="?"
    local err="?"

    if [ -f "$rdir/metrics.json" ]; then
      status="已完成"
      acc=$(python3 -c "
import json
try:
  m=json.load(open('$rdir/metrics.json'))
  print(f\"{m.get('best_test_acc',0)*100:.2f}%\")
except:
  print('?')
" 2>/dev/null || echo "?")
      epochs=$(python3 -c "
import json
try:
  m=json.load(open('$rdir/metrics.json'))
  print(m.get('best_epoch', '?'))
except:
  print('?')
" 2>/dev/null || echo "?")
      err="已结束"
      done_cnt=$((done_cnt + 1))
    else
      epochs=$(get_last_epoch "$rdir/history.csv")
      acc=$(get_last_test_acc "$rdir/history.csv")
      err=$(scan_errors "$rdir/train.log")
      any_running=1
    fi

    printf "  %-32s | %-6s | %6s | %8s | %-8s\n" "$rname" "$status" "$epochs" "$acc" "$err"
  done

  echo "  进度: $done_cnt / $total_cnt 已完成"
  echo
  return $any_running
}

one_shot() {
  print_gpu
  local still_running=0
  for gd in "${GRID_DIRS[@]}"; do
    if print_grid_status "$gd"; then
      still_running=1
    fi
  done
  return $still_running
}

if [ "$WATCH_MODE" -eq 0 ]; then
  one_shot
  exit 0
fi

echo "正在持续监控网格实验，每 ${INTERVAL} 秒刷新一次（按 Ctrl-C 停止）..."
echo "监控目录: ${GRID_DIRS[*]}"
echo

while true; do
  clear 2>/dev/null || printf "\033c"
  if ! one_shot; then
    echo ">>> 所有网格实验已全部完成 <<<"
    break
  fi
  echo "下次刷新在 ${INTERVAL} 秒后...（按 Ctrl-C 中止监控）"
  sleep "$INTERVAL"
done

echo "监控结束于 $(date)"
