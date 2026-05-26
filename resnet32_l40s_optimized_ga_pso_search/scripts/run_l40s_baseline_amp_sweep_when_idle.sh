#!/usr/bin/env bash
set -euo pipefail

while pgrep -f "train_block_width_resnet32.py" >/dev/null; do
  date "+%F %T waiting for existing block-width training to finish"
  sleep 300
done

date "+%F %T starting baseline FP16/BF16 sweep"
bash scripts/run_l40s_baseline_amp_sweep.sh
date "+%F %T finished baseline FP16/BF16 sweep"
