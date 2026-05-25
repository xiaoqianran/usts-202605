#!/usr/bin/env bash
set -euo pipefail
python search_block_channels_ga_pso.py --algorithm both --baseline-checkpoint runs/resnet32_baseline/best.pt --search-epochs 2 --ga-population 8 --ga-generations 5 --pso-particles 8 --pso-iterations 5 --max-train-samples 5000 --max-test-samples 2000 --amp
