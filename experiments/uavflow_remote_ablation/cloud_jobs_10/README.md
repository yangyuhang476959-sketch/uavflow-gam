# Priority cloud jobs (one experiment per 8-GPU machine)

Submit each numbered script as one independent cloud job. Every job uses all
eight visible GPUs with DDP (`NPROC=8`, batch 3 per GPU, global batch 24), runs
Stage 1 for five epochs, then initializes and runs the matching joint Stop
Stage 2 for five epochs. Interrupted jobs resume from their own output folder.

| Priority | Script | Question |
|---:|---|---|
| 1 | `01_B0.sh` | Qwen/no-pose/relative/future baseline |
| 2 | `02_P1_pose.sh` | Does direct numeric pose help? |
| 3 | `03_L1_t5.sh` | Frozen T5 versus frozen Qwen VLM |
| 4 | `04_D1LOG_metric_log.sh` | Fixed metric depth with linear/log/gradient loss |
| 5 | `05_D2_scale_separated.sh` | Relative shape plus learned scene scale |
| 6 | `06_W3_dynamic3.sh` | Three-times per-pixel dynamic-object weighting |
| 7 | `07_H0_current.sh` | Current geometry reconstruction only |
| 8 | `08_HB_current_future.sh` | Observed Current + predicted Future joint deep pass |
| 9 | `09_F3_short_horizon.sh` | Shorter coupled action/geometry horizon K=3 |
| 10 | `10_M1_multiwindow.sh` | GAM-style causal contexts H={1,2,3,4}, K=5 |

Example submission payload:

```bash
bash experiments/uavflow_remote_ablation/cloud_jobs_10/01_B0.sh
```

The scripts inherit paths and credentials from repository `server.env`. A
cloud scheduler may set `CUDA_VISIBLE_DEVICES`; otherwise the launcher uses
devices `0,1,2,3,4,5,6,7`. Do not combine numbered scripts in one allocation.

## Memory expectation

The full-rank configuration has about 1.033 billion trainable parameters.
DDP replicates parameters and AdamW state on every GPU; eight GPUs improve
throughput but do not shard this memory. Based on the measured 19.6 GiB LoRA
peak at batch 8/GPU and the extra full-rank gradients/Adam states, ordinary
H=1 jobs are expected to peak around 30--45 GiB/GPU at batch 3/GPU. `M1` is
heavier because H can reach four; budget roughly 40--55 GiB/GPU. These are
engineering estimates rather than measurements on the target 64-GiB GPU, but
leave useful headroom. Preserve global batch 24 across all jobs for a fair
comparison.

Deferred until Phase 2: scheduler control, stronger dynamic weights, K=7/10,
and the remaining Current/Future attention architectures. Select these only
after inspecting the ten priority runs.
