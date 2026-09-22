# One-click command list

The handoff operator only needs to choose paths and a scheduler once.

```bash
git clone ssh://git@ssh.github.com:443/yangyuhang476959-sketch/uavflow-gam.git UAVFlow-GAM
cd UAVFlow-GAM
cp server.env.example server.env
```

Defaults are relative to the cloned repository. Edit `server.env` only when the
server uses shared mounts, or to select `SCHEDULER`. Then run everything,
including the 22 production submissions:

```bash
bash scripts/uavflow_handoff.sh all
```

For a safer handoff, prepare and validate first, inspect the smoke result, and
submit separately:

```bash
bash scripts/uavflow_handoff.sh prepare
bash scripts/uavflow_handoff.sh submit
```

Progress and interrupted-job recovery:

```bash
bash scripts/uavflow_handoff.sh status
bash scripts/uavflow_handoff.sh submit  # safe to resubmit
```

Completed stages are skipped. Interrupted stages resume from the latest epoch
checkpoint with optimizer, scaler, scheduler and data cursor restored.
