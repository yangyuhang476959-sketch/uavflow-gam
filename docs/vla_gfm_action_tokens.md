# Maintained VLA–GFM action-token architectures

The release now exposes only [three metric-depth experiments](METRIC_EXPERIMENTS.md).
Old Query/OFT/dual-view/provenance ablation launchers are not part of the handoff.

- GAM baseline: frozen T5 semantic conditioning, existing GAM future Predictor
  and action head, numeric pose history, blocks13–39 DA3 LoRA, future metric depth.
- Old correction: last-layer five native Qwen action slots (all-linear LoRA),
  a directly decoded VLM proposal, GAM future Predictor and DA3 residual head.
  Action slots read the Current Geometry Bank at output layers19/27/33/39.
- RGB-warp correction: the same five-slot VLM proposal controls a differentiable
  RGB warp using predicted Current depth. A second forward through the same DA3
  adds five action tokens at the deep boundary; Future patches read matching
  Current patches via correspondence CA. No Future Predictor. Full40-block LoRA.

Qwen retains causal attention/RoPE and learned per-slot embeddings. There is no
post-Qwen bidirectional mixer or autoregressive action decoder. The extra learned
step encoding is applied only at the continuous ActionHeadV2 decoder. Slots are
not averaged into one action hidden; each outputs one four-dimensional increment.

New correction models use an MLP-injected numeric Pose Token in first-yaw-frame
centimetres/degrees, shared Dataset/inference conversion, not printed numeric
prompt text. Historical full-first-rotation checkpoints keep their old meaning
and are not exact resumes into the corrected state convention.

Detailed warp/mask/gradient behavior: [dual_stream_rgbwarp.md](dual_stream_rgbwarp.md).
Environment and data instructions remain in [NPU_SETUP.md](../NPU_SETUP.md).
