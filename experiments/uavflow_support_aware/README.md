# Shared correction helpers

Use the [three maintained metric-depth experiments](../../docs/METRIC_EXPERIMENTS.md).
This package retains shared configuration, checkpoint loading and observation-only
inference used by the two correction models. Historical Candidate B/C internal
helpers and their regression tests are not extra handoff jobs. No uncertainty,
terminal transport bank or extra training stage is enabled by the three entries.
