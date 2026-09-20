# UAV-Flow-Sim Depth Sidecars

## What is stored

The official UAV-Flow-Sim dataset supplies RGB, language, and trajectory data.
This project adds depth sidecars keyed by episode and frame index.

Reviewed dynamic-object sidecars contain:

| Field | Type | Meaning |
|---|---|---|
| `hybrid_depth_m` | `float16 [T,H,W]` | Metric depth in metres. |
| `valid_mask` | `bool [T,H,W]` | Pixels eligible for depth supervision. |
| `semantic_mask` | `bool [T,H,W]` | Reviewed person, vehicle, or robotic-dog pixels. |
| `frame_indices` | `int32 [T]` | Index into the official episode. |
| `da3_depth_raw` | `float16 [T,H,W]` | DA3 pseudo-depth before metric calibration. |
| `episode_scale` | `float32` | DA3-to-replay metric calibration scale. |

The current reviewed hybrid set covers 3,119 episodes:

- person: 1,428
- robotic dog: 1,013
- vehicle: 678

All remaining episodes use calibrated PnP-to-UE replay depth.  Therefore the 8.4 GiB reviewed
hybrid directory is not a standalone full dataset; the canonical HF release
must merge it with replay-depth fallbacks.

Here, `replay` never means the obsolete raw-pose export.  Every selected
replay episode first uses the saved camera-pose reference and per-episode
iterative PnP calibration, writes the corrected camera pose back into UE, and
then renders metric depth.  The first 1,040 episodes in the primary working
directory were exported before the camera-pose-reference fix; none of those
obsolete files are selected.  They are replaced either by the corrected
reverse rerun (768 selected episodes) or by reviewed hybrid sidecars (272
episodes).

## Minimal training payload

The trainer does not read replay RGB, clean-background RGB, or saved camera
poses.  For calibrated replay episodes it only opens `depth.npy`.  For hybrid
episodes it only opens:

- `hybrid_depth_m`
- `valid_mask`
- `semantic_mask`
- `frame_indices`

Consequently `da3_depth_raw` and `episode_scale` can be omitted from the
training-only HF artifact.  The selected source payload is:

| Source | Episodes | Existing files | Minimal release |
|---|---:|---:|---:|
| primary calibrated PnP-to-UE replay depth | 6,222 | 16.153 GiB | 16.153 GiB before transport compression |
| corrected reverse PnP-to-UE rerun depth | 768 | 2.163 GiB | 2.163 GiB before transport compression |
| reviewed hybrid | 3,119 | 8.299 GiB | approximately 2.293 GiB after unused fields are removed |
| total | 10,109 | 26.615 GiB | approximately 20.609 GiB unpacked |

The published six-shard ModelScope artifact is 7.27 GiB compressed. Remote
training should download the compressed shards once and extract them to the
approximately 20.6 GiB mmap-friendly layout.

## Instruction corrections

Seven released simulation episodes contain a person/dog instruction mismatch.
The official parquet shards remain immutable.  `data/instruction_overrides.json`
is applied by episode ID after the official instruction is read:

```text
corrected_instruction = overrides.get(episode_id, official_instruction)
```

The correction file must be distributed with both the GitHub code and HF
depth dataset.  Users should not manually edit downloaded UAV-Flow-Sim
parquet files.

## Canonical selection

The release manifest resolves one source per episode using the following
working-directory labels:

```text
hybrid > corrected_replay > replay
```

Both `corrected_replay` and `replay` are calibrated PnP-to-UE renders.  The
difference is only that `corrected_replay` is the replacement reverse rerun
for the obsolete prefix; `replay` contains the already-correct primary run.

The source files remain immutable.  Packaging reads the manifest and writes
new sharded artifacts, so no original data are moved or overwritten.

## Planned HF layout

```text
UAV-Flow-Sim-Depth/
  README.md
  metadata/episodes.parquet
  metadata/manifest_sha256.json
  data/train-00000.tar
  data/train-00001.tar
  ...
```

Shards should stay near 1--2 GiB to support resumable transfer to remote
servers.  The dataset card must cite UAV-Flow and describe replay calibration,
SAM masks, DA3 pseudo-depth, invalid pixels, and known camera-alignment noise.
