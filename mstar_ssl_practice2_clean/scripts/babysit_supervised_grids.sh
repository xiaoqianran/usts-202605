#!/usr/bin/env bash
# 监督学习网格实验全自动托管脚本（中文版）
# babysit_supervised_grids.sh
#
# 功能：按顺序自动跑完「固定学习率」+「线性缩放学习率」两套监督网格实验。
#       内部调用 run_supervised_grid.sh（已支持跳过已完成）和 monitor_training.sh。
#
# 典型用法（从项目根目录）：
#   nohup bash scripts/babysit_supervised_grids.sh data/MSTAR > logs/babysit_$(date +%Y%m%d_%H%M).log 2>&1 &
#   tail -f logs/babysit_*.log
#
# 或者直接前台运行，让它一直等到两套全部结束。
# 结束后会自动打印两份汇总 Markdown 表格。

set -euo pipefail

DATA_ROOT="${1:-data/MSTAR}"
EPOCHS="${EPOCHS:-80}"
PARALLEL_JOBS="${PARALLEL_JOBS:-4}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MONITOR="./scripts/monitor_training.sh"
RUNNER="./scripts/run_supervised_grid.sh"

echo "================================================================"
echo "  监督学习 Batch/LR 网格实验 全自动托管"
echo "  数据目录: $DATA_ROOT    训练轮数: $EPOCHS    并行任务数: $PARALLEL_JOBS"
echo "  开始时间: $(date)"
echo "================================================================"

echo
echo ">>> 初始状态检查（开始前）"
$MONITOR --once || true

echo
echo ">>> 第一阶段：固定学习率网格（所有 batch 统一 lr=0.001）"
echo "    将执行的命令："
echo "    LR_POLICY=fixed PARALLEL_JOBS=$PARALLEL_JOBS EPOCHS=$EPOCHS \\"
echo "        $RUNNER $DATA_ROOT"
echo
LR_POLICY=fixed PARALLEL_JOBS="$PARALLEL_JOBS" EPOCHS="$EPOCHS" \
    "$RUNNER" "$DATA_ROOT"

echo
echo ">>> 第一阶段完成。固定学习率网格当前状态："
$MONITOR --once || true

echo
echo ">>> 第二阶段：线性缩放学习率网格（lr 随 batch 线性增大）"
echo "    将执行的命令："
echo "    LR_POLICY=scaled PARALLEL_JOBS=$PARALLEL_JOBS EPOCHS=$EPOCHS \\"
echo "        $RUNNER $DATA_ROOT"
echo
LR_POLICY=scaled PARALLEL_JOBS="$PARALLEL_JOBS" EPOCHS="$EPOCHS" \
    "$RUNNER" "$DATA_ROOT"

echo
echo ">>> 第二阶段完成。最终整体状态："
$MONITOR --once || true

echo
echo "================================================================"
echo "  最终汇总表格（可直接复制进报告/答辩稿）"
echo "================================================================"

echo
echo "--- runs_supervised_grid_fixed/summary/supervised_grid_summary.md (固定学习率) ---"
if [ -f runs_supervised_grid_fixed/summary/supervised_grid_summary.md ]; then
    cat runs_supervised_grid_fixed/summary/supervised_grid_summary.md
else
    echo "（文件不存在，可能有运行异常）"
fi

echo
echo "--- runs_supervised_grid_scaled/summary/supervised_grid_summary.md (线性缩放学习率) ---"
if [ -f runs_supervised_grid_scaled/summary/supervised_grid_summary.md ]; then
    cat runs_supervised_grid_scaled/summary/supervised_grid_summary.md
else
    echo "（文件不存在）"
fi

echo
echo ">>> 托管脚本执行完毕：$(date)"
echo "    两套监督网格（各 8 组，共 16 组）理论上已全部跑完并生成汇总。"
echo "    下一步建议：把四张网格汇总表（FixMatch 2 张 + 监督 2 张）统一整理进 README 和答辩提纲。"
