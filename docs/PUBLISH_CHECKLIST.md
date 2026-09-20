# Publication checklist

Before pushing the staging tree:

1. Confirm the GitHub origin is
   `https://github.com/yangyuhang476959-sketch/uavflow-gam.git`.
2. Review the compact-21 and detailed-64 designs with the experiment owner.
3. Confirm all six derived-depth shards and `metadata/` are visible in
   `acetaffy123/UAV-Flow-Sim-Depth`; record its revision in the release notes.
5. Publish selected checkpoints separately. Do not commit `.pt`, `.pth`, model
   safetensors, official UAV-Flow parquet or extracted depth to GitHub.
6. Run the smoke command in `docs/REMOTE_TRAINING.md` on a clean server.
7. Confirm every scientific run records `project_commit`, `project_dirty`,
   split SHA-256 and depth-manifest SHA-256 in `run_state.txt`.
8. Confirm `git status --short` contains only intended source and docs. The
   local `hf_dataset/`, `results/` and downloaded model directories are ignored.

Suggested final sequence:

```bash
git add .
git status --short
git --no-pager diff --cached --stat
git commit -m "Release UAV-Flow GAM ablations and reproducibility scripts"
git remote set-url origin ssh://git@ssh.github.com:443/yangyuhang476959-sketch/uavflow-gam.git
git push -u origin main
```

Do not run the final `git add`, commit or push until the repository URL and
dataset repository ID have been reviewed.
