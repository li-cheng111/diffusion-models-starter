# Project 2 debug log

## 1. Schedule buffer naming mismatch

- Symptom: Project 2 expected singular buffer names while Project 1 exposes
  `sqrt_alphas_cumprod` and `sqrt_one_minus_alphas_cumprod`.
- Verification: the compatibility check found valid plural equivalents.
- Fix: added a central alias-aware accessor instead of changing Project 1.
- Lesson: sampler code should depend on coefficient meaning, not spelling.

## 2. Nested EMA checkpoint format

- Symptom: the starter attempted to load `checkpoint["ema"]` directly.
- Verification: Project 1 stores `ema = {decay, model}`.
- Fix: added a loader supporting nested and flat EMA state dictionaries.
- Lesson: validate checkpoint structure before long inference runs.

## 3. Unfair FID and inaccurate NFE accounting

- Symptom: the starter used augmented real images, changed RNG state between
  configurations, could overshoot 5,000 real images, and reported two model
  calls for DPM's terminal step.
- Fix: fixed the exact real subset, reset RNG per configuration, enforce equal
  counts, and report `2S-1` for DPM-Solver-2.
- Lesson: Pareto comparisons require controlled data and compute accounting.

## 4. DPM trajectory did not share initial noise

- Symptom: the starter created common noise but DPM's `sample()` replaced it.
- Fix: extended every sampler with `initial_x` and routed all trajectory runs
  through that interface.
- Lesson: pass controlled tensors explicitly when RNG consumption differs.
