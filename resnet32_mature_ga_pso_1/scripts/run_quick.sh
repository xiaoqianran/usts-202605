#!/usr/bin/env bash
set -euo pipefail
python train_resnet32.py --epochs 2 --batch-size 128 --lr 0.05 --milestones 1 --run-name resnet32_quick
python search_block_channels_ga_pso.py --algorithm both --search-epochs 1 --ga-population 4 --ga-generations 2 --pso-particles 4 --pso-iterations 2 --max-train-samples 1000 --max-test-samples 500
