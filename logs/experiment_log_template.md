# Project 1 Experiment Log

> Completed record for the real MNIST baseline run on AutoDL.

## Experiment ID

`exp_20260912_01_mnist_autodl`

## Objective

Can the from-scratch unconditional DDPM baseline train to completion and
produce usable MNIST samples on a GPU instance?

## Environment

| Item | Value |
|---|---|
| Date | 2026-09-12 |
| Hardware | NVIDIA GeForce RTX 5090 |
| Python / PyTorch | Python 3.12.3 / PyTorch 2.8.0+cu128 |
| CUDA | 12.8 |
| Git commit | `99955cac0d37da993a7ede479086baa92fc8be65` |
| Branch | `main` |

## Configuration

- Dataset: MNIST
- Image size: 32
- Batch size: 128
- Epochs: 50
- Diffusion steps `T`: 1000
- Beta schedule: linear, 0.0001 → 0.02
- Learning rate: 2.0e-4
- EMA decay: 0.9999
- Mixed precision: no
- Seed: 42

## Command

```text
python train.py --config configs/mnist.yaml
```

## Results

 - Final logged loss: 0.01151 at step 23,400
 - Training time: 24.8 minutes
 - FID: 32.8913 using EMA weights and 5,000 samples
 - Sample paths: `runs/exp_mnist_baseline/samples_inference_ema/grid.png`

## Conclusion

The MNIST baseline completed successfully. The final checkpoint, loss history,
loss curve, generated sample grid, and FID record are included with the
experiment artifacts.

## Actual Issues Encountered

The GitHub clone initially returned HTTP 503 on AutoDL; retrying with a shallow
clone succeeded. No training-code runtime issue occurred. See `debug_log.md`
for the recorded operational notes.
