#!/usr/bin/env bash
set -euo pipefail
python model_info.py
python train_resnet32.py --epochs 2 --batch-size 128 --lr 0.05 --milestones 1 --run-name resnet32_quick
