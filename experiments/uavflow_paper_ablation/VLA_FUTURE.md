# VLA+Future

当前 RGB、指令和 native 5D Pose 输入 Qwen。五个 Action Slots 经公共动作投影进入 DA3，DA3 同时预测 t+5 深度和完整五步动作。只有一次当前 RGB 的 DA3 前向，未来 RGB 不进入模型；深度目标是数据集的真实 t+5 深度。这是 VLA+Current 的未来深度监督消融，预测深度与真实未来相机坐标下的深度按像素监督。

沿用 VLA+Current 的最终动作头与公共初始化，深度固定 100 米尺度，linear/log/gradient 各 1/3。损失为 3 × 最终动作 L1 + 3 × 未来深度损失。没有 Predictor、Proposal 监督或额外 Future Pose token。独立 branch_mode 为 future_only，checkpoint 配置保存分支和目标，训练恢复校验二者。epoch、last、best 保留完整训练状态及优化器。

基准配置来自 `configs/current_only.yaml`，显式变化是分支名称、深度监督目标和输出目录；数据划分、初始化、采样与优化器设置沿用该配置。与 Warp 的共享条件依照 `docs/EXPERIMENT_BASELINE.md` 及快照核对，结构和监督变化按上述说明记录。

查看配置：

```bash
PYTHONPATH=src /root/miniconda3/envs/uav-gam/bin/python -m experiments.uavflow_paper_ablation.future_train --print-config
```

训练入口 `experiments.uavflow_paper_ablation.future_train` 支持 `--output`、`--epochs`、`--set` 和 `--resume`。默认配置按三卡每卡 microbatch 8、累积 1 设置。启动前应核对实际 GPU 空间及运行资源。尚未进行完整模型 GPU 测试。
