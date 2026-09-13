# Project 2 experiment log

## Implementation validation

- Goal: verify parent-directory Project 1 imports and finite sampler outputs
  with correct NFE.
- Environment: Python 3.14.0, PyTorch 2.13.0+cu126, one NVIDIA RTX 4060
  Laptop GPU on Windows.
- Commands: `python check_compat.py --ckpt ../runs/exp_cifar10_advanced/ckpt/final.pt`,
  `python -m unittest discover -s tests -v`, and both sampler module self-tests.
- Result: 11 standard-library unit tests passed; compatibility check loaded the
  real nested-EMA Project 1 checkpoint; DDIM and DPM-Solver module smoke tests
  passed. `torchmetrics` is not installed locally, so formal FID was run on
  AutoDL instead.
- Qualitative smoke: MNIST DDIM 50-step, CIFAR-10 DDIM 10-step, and DPM-Solver
  5-step/9-NFE grids were generated and visually inspected as finite, structured
  outputs. These are not final FID measurements.

## Formal FID–NFE experiment

- Hypothesis: DPM-Solver-2 should improve the FID/compute trade-off over
  first-order DDIM at comparable NFE.
- Controls: seed-44 linear EMA checkpoint, generation seed 42, exactly 5,000
  unaugmented real and 5,000 generated images per configuration.
- Command: see `README.md`, “Formal AutoDL experiments”.
- AutoDL: NVIDIA RTX 4090, Python 3.12.3, PyTorch 2.8.0+cu128, CUDA 12.8.
- Git commit: `e8f8af6698570485b9909ca1755f0266a09e4da8`.
- Checkpoint SHA256: `937853559a1377660f7d4cbd1dd3f7c6cf022aed4ab84e9daa918aa6e34541412`.
- Full matrix completed with 5,000 fixed unaugmented real images and 5,000
  generated images per configuration. The best absolute FID is DDPM 19.290 at
  1,000 NFE; the best low-NFE point is DPM-Solver-2 21.074 at 19 NFE. Full
  values are recorded in `runs/benchmark_all.json` and the Pareto plot.

## DDIM inversion experiment

- Hypothesis: moderate-to-high step counts reduce discretization error, but
  model-prediction errors mean the trend need not be monotonic.
- Controlled images: CIFAR-10 test indices 0–63.
- Steps: 10, 20, 50, 100, 250.
- Results are recorded in `runs/inversion_results.json` and
  `runs/inversion_errors.png`. Mean L2 falls monotonically from 18.2820 at 10
  steps to 0.9316 at 250 steps, while PSNR rises from 15.891 to 41.836 dB.

## Post-run checks

- `runs/benchmark_all.json`: 15 configurations, each with 5,000 generated
  samples and matching 5,000 real samples.
- `runs/benchmark_ddim.json`: filtered DDIM-only artifact with all five required
  step counts.
- `runs/trajectory_comparison.json`: `same_initial_noise=true`; DPM-Solver-2
  reports 99 NFE for 50 outer steps.
- AutoDL process and detached monitoring sessions exited cleanly; the read-only
  dashboard may be stopped after the artifacts are downloaded.
