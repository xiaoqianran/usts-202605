# USTS AI Training Experiments - 2026.05

Collection of experiments and practices for Artificial Intelligence training course at Suzhou University of Science and Technology (USTS).

## Projects / Practices

- **battery_soh_practice3** - Lithium-ion battery SOH prediction using deep learning (Practice 3)
- **battery_soh_practice3_strict_code** - Strict/structured version of battery SOH experiments with detailed outputs
- **mstar_ssl_experiment** - MSTAR SAR image classification with semi-supervised learning (FixMatch)
- **mstar_ssl_experiment_add** - Extended version with additional reports and figures for Practice 2
- **resnet32_ga_pso_search** and variants - ResNet-32 width/channel optimization using Genetic Algorithm + PSO on H100 GPUs
  - resnet32_ga_pso_search
  - resnet32_ga_pso_search_2
  - resnet32_h100_ga_pso_search
  - resnet32_block_h100_code_only
  - resnet32_mature_ga_pso

## Structure

Each directory is mostly self-contained with:
- `src/` - source code
- `scripts/` or `run_all.*` - entry points
- `requirements.txt`
- `README.md`
- `data/` (where applicable, large datasets not committed)
- `runs/`, `outputs/` (generated results - mostly gitignored)

## Notes

- Large model checkpoints (`.pt`), datasets (MSTAR images, CIFAR tarballs), and experiment run directories are gitignored.
- See individual project READMEs for details.
- Many experiments were run on H100 GPUs.

This repo contains code, reports, and selected results from the course practices in May 2026.
