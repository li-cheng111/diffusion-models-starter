# Debug Log

## Entry 1 — AutoDL GitHub clone

- Status: Resolved
- Symptom: `git clone` returned HTTP 503 after AutoDL network acceleration was enabled.
- Hypothesis: Temporary GitHub/acceleration-proxy availability issue.
- Verification: The repository cloned successfully after retrying with a shallow clone and a separate destination directory.
- Fix: Retried with `--depth 1`; no code or dataset change was required.
- Lesson: AutoDL's academic acceleration is useful but not guaranteed; keep a retry or alternate download path available.

## Entry 2 — AutoDL SSH authentication

- Status: Resolved
- Symptom: The first public-key login attempt returned `Permission denied (publickey,password)`.
- Hypothesis: The public key was not yet registered for the active AutoDL instance.
- Verification: Password authentication succeeded for the same host and port; the remote instance and repository were reachable.
- Fix: Used the instance login credentials for the authorized session and did not modify the training code.
- Lesson: Keep the AutoDL console key registration and the active instance/port aligned; never store the password in project files.

## Entry 3 — MNIST baseline

- Status: Completed
- Symptom: No runtime failure occurred during training, sampling, or EMA FID evaluation.
- Hypothesis: The configured baseline should fit the RTX 5090 environment.
- Verification: Training reached step 23,400, produced the final checkpoint and loss curve, and FID evaluation completed with 5,000 samples.
- Fix: None required.
- Lesson: The baseline is reproducible in the recorded AutoDL environment; preserve the configuration and seed with the artifacts.

## Entry 4 — CIFAR-10 data preparation

- Status: Resolved
- Symptom: The original CIFAR-10 download endpoint was limited to roughly 10–55 KB/s on AutoDL.
- Hypothesis: The external Toronto endpoint was the bottleneck rather than the GPU or project code.
- Verification: A public Hugging Face mirror completed the same 170,498,071-byte archive, and its MD5 matched `c58f30108f718f92721af3b95e74349a`.
- Fix: Resumed the archive download from the mirror and kept the validated file at `data/cifar-10-python.tar.gz`.
- Lesson: Validate dataset bytes before training and keep data files on the AutoDL data disk.

## Entry 5 — CIFAR-10 advanced experiment

- Status: Completed
- Symptom: No runtime failure occurred during the 200-epoch run or post-training evaluation.
- Verification: Training reached step 78,000 in 98.2 minutes; EMA FID was 19.2879 and raw FID was 28.7464 on 5,000 training images.
- Fix: None required for stability. The FID target of 15 was not reached, so the result is recorded as-is.
- Lesson: EMA materially improved this run's FID, but a stable run and a target metric are separate acceptance criteria.
