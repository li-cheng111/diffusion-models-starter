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
