# Priority cloud jobs: complete 25-cell matrix

The directory name is retained for compatibility with the first ten submitted
jobs, but it now contains one executable script for every cell in the compact
25-experiment matrix. The original scripts `01`--`10` are unchanged; scripts
`11`--`25` only add the previously deferred cells. `priority.tsv` is the
machine-readable submission order.

Submit each numbered script as one independent cloud job. Every job uses all
eight visible GPUs with DDP (`NPROC=8`, batch 3 per GPU, global batch 24), runs
Stage 1 for five epochs, then initializes and runs the matching joint Stop
Stage 2 for five epochs. Interrupted jobs resume from their own output folder.
Stage 2 uses the CLIP Stop fine-tuning ratio: all existing trainable groups
run at one tenth of their Stage-1 peak learning rates (Predictor `1e-6`,
deep DA3 `5e-6`, Action head `5e-5` at the defaults). Its newly initialized
Stop head has an independent `5e-4` peak learning rate, equal to the Stage-1
Action-head peak learning rate, and receives its own
normalized current-pose input, independent of the main backbone's pose setting.
Both stages retain
their respective schedules; Stage 2 uses 500 warmup steps then cosine decay.
The Stage-2 output is `stage2_stop_actionpose_stoplr5e4/`, so older Stage-2 checkpoints
made with the former learning rates cannot be silently resumed.

| Priority | Script | Question |
|---:|---|---|
| 1 | `01_B0.sh` | Qwen current image + raw instruction, no pose, relative/future baseline |
| 2 | `02_P1_pose.sh` | Does direct numeric pose help? |
| 3 | `03_L1_t5.sh` | Frozen T5 versus frozen Qwen VLM |
| 4 | `04_D1LOG_metric_log.sh` | Fixed metric depth with linear/log/gradient loss |
| 5 | `05_D2_scale_separated.sh` | Relative shape plus learned scene scale |
| 6 | `06_W3_dynamic3.sh` | Three-times per-pixel dynamic-object weighting |
| 7 | `07_H0_current.sh` | Current geometry reconstruction only |
| 8 | `08_HB_current_future.sh` | Observed Current + predicted Future joint deep pass |
| 9 | `09_F3_short_horizon.sh` | Shorter coupled action/geometry horizon K=3 |
| 10 | `10_M1_multiwindow.sh` | GAM-style causal contexts H={1,2,3,4}, K=5 |
| 11 | `11_HE_DUALPRED.sh` | Predict Current and Future, then jointly decode them with Action |
| 12 | `12_HF_BRIDGE.sh` | Read-only Current context feeding Future/Action |
| 13 | `13_CA1_HB.sh` | Per-layer Action cross-attention to Current |
| 14 | `14_HC_DIRECT.sh` | Direct Current deep path without the causal Predictor |
| 15 | `15_D1_metric_no_log.sh` | Metric-depth control without log loss |
| 16 | `16_S1COS.sh` | Stage-1 cosine-schedule control |
| 17 | `17_W5_dynamic5.sh` | Five-times dynamic-pixel weight |
| 18 | `18_W10_dynamic10.sh` | Ten-times dynamic-pixel weight |
| 19 | `19_F7_horizon7.sh` | Coupled horizon K=7 |
| 20 | `20_F10_horizon10.sh` | Coupled horizon K=10 |
| 21 | `21_C1_pose_t5.sh` | Pose × language interaction |
| 22 | `22_C2_scale_both.sh` | Scale separation × Current/Future interaction |
| 23 | `23_C3_dynamic3_both.sh` | Dynamic×3 × Current/Future interaction |
| 24 | `24_C4_horizon3_dynamic3.sh` | K=3 × Dynamic×3 interaction |
| 25 | `25_C5_horizon10_both.sh` | K=10 × Current/Future interaction |

Example submission payload:

```bash
bash experiments/uavflow_remote_ablation/cloud_jobs_10/01_B0.sh
```

The scripts inherit paths and credentials from repository `server.env`. A
cloud scheduler may set `CUDA_VISIBLE_DEVICES`; otherwise the launcher uses
devices `0,1,2,3,4,5,6,7`. Do not combine numbered scripts in one allocation.

The Qwen baseline receives only the current image and raw dataset instruction.
It has neither textual numeric state nor a dedicated pose token. `P1` changes
only the latter by adding the normalized numeric pose token.

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

Submit in numeric order when capacity is limited. Priorities 1--10 retain the
original core screen; 11--14 resolve the remaining geometry-architecture
question; 15--20 are controls/strength/horizon sweeps; 21--25 are targeted
interactions and therefore come last.
