# Results Artifacts

This directory contains generated experiment outputs.

## Files
- `results.csv`: per-question outputs and judge annotations
- `report.md`: summary report generated from the CSV
- `metrics.png`: comparison plot for accuracy and unsupported claim rate
- `cost_latency.png`: comparison plot for estimated cost and latency

## Important Note
These artifacts are only as meaningful as the mode used to generate them:

- `USE_MOCK_LLM=1` is useful for local smoke testing and deterministic runs
- `USE_MOCK_LLM=0` with a real API-backed model is the meaningful mode for reporting grounded-generation behavior

If you are using this repository for portfolio or recruiting purposes, make sure you distinguish between mock-mode plumbing validation and live-model evaluation.
