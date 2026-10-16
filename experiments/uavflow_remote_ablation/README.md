# Shared configuration and historical benchmark backend

Active training commands are the [three metric-depth jobs](../../docs/METRIC_EXPERIMENTS.md)
under `experiments/metric_jobs/`. The old ten job wrappers have been removed.

This directory remains because the shared model builders, regression tests and
working `scripts/bench_r1_ascend.sh` import its configuration/split/runtime code.
`matrix_v2.py`, `compact_matrix.tsv` and `run_experiment R1` describe the
historical benchmark; do not mistake its full-rank, relative-depth, two-stage
defaults for the current metric/LoRA experiments. No environment benchmark is
silently redefined as one of the new scientific models.
