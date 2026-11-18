# Current-only：已实现，待配置核对

## 实际计算路径

当前RGB＋指令＋5D Pose → 同一Qwen五个Action Slots → 同一LayerNorm/Linear投影。

投影后的五槽特征分成两条：

1. 原Proposal ActionHeadV2直接给5步Proposal，保留辅助动作监督。
2. 同一特征作为Action Tokens，注入当前RGB的DA3；一次DA3深层传播同时产生当前深度和五槽Action hidden。后者进入另一个独立ActionHeadV2，直接给完整动作，不加Proposal。

没有Future RGB、Warp、Future Predictor、Current Bank CA、correspondence CA、residual gate或warp mask embedding。GT动作和未来帧不进入模型；数值Proposal也不进入当前DA3。复用DA3原有Action Tokens注入位置，不另加新token编码器。

动作Pose仍为[x_cm,y_cm,z_cm,relative_yaw_deg,is_start]，除[1000,1000,1000,180,1]，MLP为5→256→GELU→2048，带bias。坐标系/180度处理沿用当前native训练和推理helper。

## 与Warp的实际差异

| 项目 | Warp基准 | Current-only |
|---|---|---|
| 当前流DA3 | 当前RGB→深度 | 当前RGB＋Action Tokens→深度和最终动作 |
| 后续几何 | 当前深度＋数值Proposal→Warp RGB→Future DA3 | 无 |
| 两个动作头 | Proposal＋Future完整动作 | Proposal＋Current完整动作 |
| GT深度监督 | 当前＋真实t+5深度 | 只用当前深度 |
| 深度loss到Qwen的路径 | Future Warp及Action Tokens路径 | 当前DA3的Action Tokens路径 |
| mask embedding | Warp无效区embedding | 移除，没有Warp孔洞 |
| 推理 | 原RGBWarpPolicy、原checkpoint loader | 复用该loader，按持久化branch_mode自动选择CurrentOnlyCorrection |

底层保留dual_stream_rgbwarp_enabled及parallel_action_decode_mode=geometry_residual两个历史兼容字段，以复用原训练器的checkpoint、优化器、数据游标、日志和五槽头构建。**实际拓扑由rgbwarp_correction.branch_mode=current_only与action_output_mode=direct选择；这些兼容字段不代表Current-only仍跑双流或residual。** 模型测试检查不存在未来计算图。

## 损失，当前默认待核对

```
L = 3 × (L_proposal + L_current_action) / 2 + 3 × L_current_depth
L_current_depth = (L_linear + L_log + L_gradient) / 3
```

保留总深度外权重3，故单个Current深度项系数3；Warp原来每个深度项系数1.5。这是显式监督差异，不描述成所有单项系数相同。深度GT有效mask和归一化100m沿用原实现；未来深度不参与loss，也不伪造future指标。

## 默认配置

实际文件：[configs/current_only.yaml](configs/current_only.yaml)。参考文件是原20轮Warp完整配置的静态副本[configs/warp_reference.yaml](configs/warp_reference.yaml)。

- 20轮，无Stop、额外repeat0，保留原生absorbing。
- 单卡micro8×累积3，有效batch24。三卡可用每卡micro8×累积1。
- Qwen LoRA LR=5e-5×0.2=1e-5；DA3 LoRA=5e-5；Pose/动作投影/两个动作头=5e-5×10=5e-4。
- Qwen rank32/alpha16；DA3全层rank64/alpha128；constant，无warmup；原BF16/checkpointing配置。
- 固定1% split，train309145 / val3095窗口；每轮12881更新，20轮257620更新。
- 每5000更新和每epoch全量验证，best按最终动作all-window L1。
- 原完整optimizer checkpoint/resume机制；epoch、last、best均为完整状态。
- 默认新输出results/paper_vla_current20；无resume时拒绝非空输出目录，防止覆盖旧实验。
- 启动时按旧Warp构建顺序生成参数，并执行legacy_initialization_reference的四组公共参数hash校验；不把Proposal头复制给最终头。CPU小模型验证公共初始化逐张量相同；完整Qwen/DA3启动审计仍待实际运行。

## 采样与恢复

新增Warp24MicrogroupSampler：重建原DistributedSampler(seed42＋epoch)的三卡索引，包括原padding和drop_last。单卡每次更新依次处理原rank0、rank1、rank2各8条；三卡各处理对应8条。已有数据集、动作归一化、absorbing和标注均不修改。

原深度loss按每8条的有效像素数归一化；单卡累积按三组等权平均，与原三卡的局部分组口径一致，不改成一次24条共同分母。保留组合与分组不代表跨设备浮点逐位相同。

当前数据读取仍保留原Ft/F(t+5)窗口封装，避免重新定义正常/absorbing窗口；模型和loss只读取当前RGB和当前GT深度。训练日志沿用每25次更新的global interval统计，Current-only只汇总当前深度，不输出伪造的Future深度或Warp覆盖率；累积期间的有效动作计数、原始动作矩和窗口计数按总和统计。

恢复游标以optimizer update计数，sampler直接跳过已完成的update；支持本配置相同卡数的resume。当前不开放1↔3卡的中途迁移，原训练器会拒绝world_size改变。

## 命令（尚未执行训练）

只查看解析后的实际配置：

```bash
PYTHONPATH=src /root/miniconda3/envs/uav-gam/bin/python -m experiments.uavflow_paper_ablation.current_train --print-config
```

确认后单卡训练命令（GPU编号按当时空闲卡指定）：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /root/miniconda3/envs/uav-gam/bin/python -m experiments.uavflow_paper_ablation.current_train --epochs 20 --output results/paper_vla_current20
```

同一配置完整续训：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /root/miniconda3/envs/uav-gam/bin/python -m experiments.uavflow_paper_ablation.current_train --epochs 20 --output results/paper_vla_current20 --resume results/paper_vla_current20/last.pt
```

三卡候选：

```bash
CUDA_VISIBLE_DEVICES=0,1,3 PYTHONPATH=src /root/miniconda3/envs/uav-gam/bin/torchrun --standalone --nproc_per_node=3 -m experiments.uavflow_paper_ablation.current_train --epochs 20 --output results/paper_vla_current20_three_gpu --set training.gradient_accumulation_steps=1
```

## 验证范围

tests/test_paper_current_only.py使用CPU小模型覆盖单次DA3、禁用Warp路径、公共初始化、两头独立且无残差相加、完整动作/深度反传、未来输入不影响loss、无效GT mask、原评测入口、原checkpoint推理loader及动作一致、optimizer状态恢复、错误拓扑拒载、三卡原分组/单卡累积与游标恢复。没有启动正式训练或申请GPU显存。

仍需完整模型GPU前反向/首次optimizer更新测试后，才能报告真实显存和吞吐。现在不承诺单卡micro8必定放得下。

2026-11-10用户按main.pdf表3确认：新 Current-only 使用 `current_final_only=true`，不调用 Proposal 辅助头、不计算其loss；该头仅保留冻结的兼容state。总loss改为 `3*L_final_action + 3*L_current_depth`。旧checkpoint未包含该标志时保留双头旧行为。公共初始化、Qwen Action Tokens→当前DA3→最终头路径、全层DA3 LoRA及Current深度监督不变。新配置的显存需按该路径复测，不引用旧双头方案实测为新方案结果。
