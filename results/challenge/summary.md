# Project 1 challenge-track summary

- Real images: 5,000 CIFAR-10 training images (no random augmentation)
- Generated images per FID run: 5,000
- Seeds present in this summary: 42, 43, 44
- Epoch budgets present in this summary: 200
- Reported spread: sample standard deviation across seeds

## Per-seed results

| Schedule | Epochs | Seed | EMA FID | Raw FID | Raw - EMA |
|---|---:|---:|---:|---:|---:|
| cosine | 200 | 42 | 137.9856 | 283.4196 | +145.4340 |
| cosine | 200 | 43 | 129.2708 | 399.1676 | +269.8968 |
| cosine | 200 | 44 | 145.2832 | 410.0121 | +264.7289 |
| linear | 200 | 42 | 19.2879 | 28.7464 | +9.4585 |
| linear | 200 | 43 | 19.6306 | 42.7537 | +23.1231 |
| linear | 200 | 44 | 18.9593 | 34.9399 | +15.9806 |

## Mean ± std across seeds

| Schedule | Epochs | EMA FID | Raw FID |
|---|---:|---:|---:|
| cosine | 200 | 137.5132 ± 8.0166 | 364.1998 ± 70.1675 |
| linear | 200 | 19.2926 ± 0.3357 | 35.4800 ± 7.0193 |

The causal interpretation and failure analysis belong in `challenge_report.md`; this file records measurements only and must not be filled with estimated values.
