# Project 1 Challenge Results

This directory records the completed CIFAR-10 challenge-track measurements.

- Schedules: `linear`, `cosine`
- Seeds: `42`, `43`, `44`
- Training budget: `200` epochs per run
- FID sample count: `5,000`
- Real-data split: CIFAR-10 training split, without random augmentation

`summary.md` contains the per-seed values and mean ± standard deviation. The
small `grid.png` files show qualitative samples for each run. Large training
checkpoints and intermediate sample images remain on AutoDL and are not stored
in this Git repository.
