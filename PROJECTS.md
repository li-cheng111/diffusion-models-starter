# Repository projects

This repository keeps Project 1 at the repository root for compatibility with
its existing history and Git LFS paths. Project 2 is isolated in its own
subdirectory.

| Project | Location | Upstream |
|---|---|---|
| Project 1: DDPM from scratch | repository root | existing repository history |
| Project 2: sampler comparison | [`project2-samplers/`](project2-samplers/) | [Qi-StarterTrain/diffusion-project2-samplers](https://github.com/Qi-StarterTrain/diffusion-project2-samplers), imported from `main` at `774d640f3915c3396e034068beff318fbf421719` |

## Repository rules

- `project2-samplers/` is a normal tracked directory, not a nested Git repository.
- Datasets, checkpoints, Inception weights, and caches are not committed.
- Project 2 commands are run from `project2-samplers/`; its compatibility layer
  imports Project 1 modules from the parent directory.
- Experiment JSON, plots, sample grids, logs, and reports belong under
  `project2-samplers/`, so they cannot collide with Project 1 artifacts.
