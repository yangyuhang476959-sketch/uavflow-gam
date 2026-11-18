# 对齐版 UAV-GAM / VLA-GAM（未启动训练）

这两组采用当前正式Warp24的公共条件；复用现有GAMFuturePredictor、DA3深层传播、ActionHeadV2及完整checkpoint训练器。旧metric_jobs、remote_ablation和历史checkpoint不改写。此次是新配置，不追溯声称历史GAM已经对齐。

## 真正执行的架构

| 项目 | UAV-GAM | VLA-GAM | Warp参考 |
|---|---|---|---|
| 前端 | 冻结T5＋可训练文本投影；Pose走独立SA槽 | 当前RGB/指令/Pose→Qwen五槽 | 同VLA-GAM |
| 当前流 | 当前RGB→共享DA3→当前深度 | 相同 | 相同 |
| Predictor条件 | T5文本经原Predictor语言CA读取；Pose经原SA proprio槽；现有五槽query | Qwen五槽经公共动作投影及Predictor原in_proj初始化五槽；Pose经相同SA proprio槽，无重复语言CA | 无Predictor |
| 未来生成 | 原GAMFuturePredictor预测浅层视觉特征及动作hidden | 相同 | Proposal＋当前深度→Warp RGB |
| 前几何动作头 | Predictor输出的五槽hidden→动作头 | Qwen公共投影→Proposal动作头 | 同VLA-GAM |
| 最终动作 | 预测未来特征＋动作hidden→DA3深层→独立完整动作头 | 相同 | Warp RGB→DA3→独立完整动作头 |
| 当前/未来深度 | 同一共享DA3，分别监督 | 相同 | 相同 |
| residual / bank / correspondence CA | 均无 | 均无 | 无residual/CA，保留Warp无效区embedding |

UAV-GAM无Qwen，也不会把T5伪装成五个Qwen Action Slots。保留Predictor原语言cross-attention不是Current Bank/Correspondence CA。VLA-GAM已由Qwen隐藏特征携带语义，故不再启用Predictor语言CA；相应参数冻结，不进入优化器。

GAM模块内保留原Predictor的LayerScale及QK-norm初始化；LayerScale为0是现有GAM Transformer子层的初始化，不是动作residual gate。未来特征直接预测，不加当前特征残差。

## 公共Pose，严格保留层结构与初值

两组均使用native起始yaw坐标系[x_cm,y_cm,z_cm,relative_yaw_deg,is_start]，除[1000,1000,1000,180,1]。

动作2026-11-07用户确认改为SA入口；此前未启动的语言CA Pose版本不再是当前配置。

Pose MLP严格为Linear(5,256,bias=True)→GELU→Linear(256,2048,bias=True)，无额外归一化/Dropout，第一层seed43、第二层seed44。UAV-GAM该MLP输出经独立Linear(2048,1024)宽度适配，进入Predictor原生SA proprio token；T5自身768维输出通过单独Linear(768,2048)适配（独立CPU seed46），不改变公共Pose MLP。VLA-GAM仍插入Qwen Pose Token，同时复用同一次公共MLP计算的embedding，经相同宽度适配进入Predictor SA proprio槽；不新增第二个Pose MLP。

两组Predictor的Pose融合位置现已统一到旧GAM的SA proprio槽，不再把Pose拼进T5的语言CA记忆。宽度适配器独立于公共5→256→2048 Pose MLP，seed47、LR1e-5；不会再次归一化Pose。与旧GAM相比，SA融合位置相同，输入表示和公共Pose MLP则采用本次Warp配方。UAV-GAM因没有Qwen，不具有Qwen内部Pose Token这条附加路径。两组不使用GT终点距离、未来Pose或累计GT进度。

实际保存的legacy_reconstructed_common_init.pt中Pose hash与上述CPU重建完全相同，已核对。训练首次启动还会执行公共初始化审计：VLA核对Qwen/pose/projection/proposal-head四组；UAV无Qwen，只核对其余三组。公共构建顺序在新增Predictor之前保持，最终动作头仍与Proposal头独立初始化，不复制Proposal权重。

Predictor用独立CPU RNG seed45构建，不改变先前公共头参数；两组Predictor初值逐张量相同，再冻结VLA未使用的语言CA路径。构建使用CPU generator，避免重置GPU训练RNG。完整预训练模型首次启动审计尚未执行；CPU小模型已验证公共参数与Warp逐张量一致。

## LR和损失

| 参数组 | UAV-GAM | VLA-GAM | Warp |
|---|---:|---:|---:|
| Qwen LoRA | 无 | 1e-5 | 1e-5 |
| T5本体 | 冻结 | 无 | 无 |
| T5文本适配投影 | 1e-5 | 无 | 无 |
| 动作Pose MLP | 5e-4 | 5e-4 | 5e-4 |
| 公共Qwen动作投影 | 保留冻结兼容状态，无实际输入 | 5e-4 | 5e-4 |
| 两个ActionHeadV2 | 5e-4 | 5e-4 | 5e-4 |
| GAM Future Predictor（含独立Pose宽度适配） | 1e-5 | 1e-5 | 无 |
| 全层DA3 LoRA | 5e-5 | 5e-5 | 5e-5 |

Predictor：d_model1024、depth12、16heads、FFN ratio4、dropout0、5槽、LayerNorm输入投影、原RoPE/QK-norm；两组均相同。CPU自动使用原dense SDPA fallback，CUDA保留原flex路径。

两组都使用与Warp相同的loss外系数及监督分配：

```
L = 3 × (L_pre_geometry_action + L_final_action) / 2
  + 3 × (L_current_depth + L_future_depth) / 2
```

动作：同一normalizer、有效mask、归一化L1。深度：同一GT有效mask、GT米/100、linear/log/gradient各1/3，Future目标为真实t+5深度；未将GT未来图像或动作传入模型。最终是完整动作，不监督GT残差。

**此处刻意统一新增控制组的监督预算，不沿用历史GAM的单动作头/单Future深度/Feature MSE1。两组feature_weight=0，与当前Warp相同。** 这是本次对齐版GAM定义，旧GAM默认值保留原样。

## 数据、预算、日志与保存

- 原固定1% episode split不改；train10008条/309145窗口，val101条/3095窗口。
- 正常259187＋原生absorbing49958；额外首尾repeat/self-pair均0，无Stop。
- 单当前RGB224，无新增强、无F0参考、无多窗口历史、无新增时间设置。
- 默认三卡每卡micro8×累积1＝24；可从头选择单卡micro8×累积3，是否放得下需GPU实测。
- 复用Warp24MicrogroupSampler，每次更新同样24条，保持原三组8条的padding/drop_last、有效像素分母、epoch seed和游标；12881更新/epoch，20轮257620更新。
- AdamW原参数、constant、warmup0、BF16及原checkpointing。
- 每5000更新和每epoch全量3095窗口验证；best按all-window最终动作L1。
- 每25更新global interval统计；GAM只输出真实Current/Future深度及动作指标，不伪造Warp覆盖率。
- 每epoch/last/best均保留完整优化器、游标、scaler等原训练状态。新增Predictor完整保存在geometry_architecture_state中；顶层Identity只是旧trainer接口占位，实际只有一个future_predictor。
- 同一配置、相同卡数下完整resume。禁止把Warp或另一GAM分支checkpoint解释成当前分支；严格拓扑检查会拒绝。不是加载历史模型后改名续训。

## 文件和命令

配置：[uav_gam.yaml](configs/uav_gam.yaml)、[vla_gam.yaml](configs/vla_gam.yaml)。新输出分别results/paper_uav_gam20和results/paper_vla_gam20；非空目录须显式resume。

只核对配置，不启动训练：

```bash
PYTHONPATH=src /root/miniconda3/envs/uav-gam/bin/python -m experiments.uavflow_paper_ablation.gam_train --variant uav_gam --print-config
PYTHONPATH=src /root/miniconda3/envs/uav-gam/bin/python -m experiments.uavflow_paper_ablation.gam_train --variant vla_gam --print-config
```

以下为确认后可用的三卡命令，当前未执行：

```bash
CUDA_VISIBLE_DEVICES=0,1,3 PYTHONPATH=src /root/miniconda3/envs/uav-gam/bin/torchrun --standalone --nproc_per_node=3 -m experiments.uavflow_paper_ablation.gam_train --variant uav_gam --epochs 20 --output results/paper_uav_gam20
```

```bash
CUDA_VISIBLE_DEVICES=0,1,3 PYTHONPATH=src /root/miniconda3/envs/uav-gam/bin/torchrun --standalone --nproc_per_node=3 -m experiments.uavflow_paper_ablation.gam_train --variant vla_gam --epochs 20 --output results/paper_vla_gam20
```

完整续训在对应命令后加`--resume 对应目录/last.pt`。单卡候选改CUDA_VISIBLE_DEVICES为一张卡、直接python运行同模块，并加`--set training.gradient_accumulation_steps=3`；不是改变有效batch。

推理和闭环服务继续复用原checkpoint loader与已核对的执行协议；branch_mode持久化后会自动选择MatchedGAMCorrection。CPU测试已覆盖此loader及同输入动作恢复一致，未启动任何正式闭环评测。

## 当前验证界限

CPU小模型：完整前反向、冻结DPT、所有可训练参数连通、无Warp、最终动作非残差、公共初始化、未来RGB/未来动作不进入策略、相同loss、全量评测入口、global日志、optimizer状态恢复、原推理loader、错分支拒载、T5/Pose编码及缩放均已测试。

没有启动正式训练，没有改动既有训练配置，也未分配GPU显存。完整模型吞吐、GPU编译和OOM边界仍需独立短测，不能仅根据代码对齐就保证三卡micro8必定能跑。

## 2026-11-09 当前修订：原生文本路径、Future-only和最终动作监督

用户确认公共改动边界为动作头、Pose MLP、深度loss；不需要几何前动作输出或额外Current深层／深度分支。新的native_gam_path=true明确与上面的旧适配版区分：T5冻结输出768直接交Predictor原生lang_proj(768,1024)作语言CA，不再有768→2048文本Linear。Pose仍为公共5→256→GELU→2048＋独立Linear(2048,1024)，进SA proprio；动作采用既有1536维ActionHeadV2、5槽逐槽解码，只调用最终头，兼容保留的辅助头权重冻结且不参与前向和优化器。

当前RGB只编码pre-global浅层特征，Predictor生成未来浅层特征和动作hidden，再经过DA3后半段，只有Future深度。真实未来RGB仅在策略前向后、no_grad下编码为浅层feature监督；不进入策略。复用原feature_loss的patch MSE，权重1，CLS/register权重0。总loss=3*final_action_L1 + 1*future_feature_MSE + 3*future_depth；深度保留固定100m尺度与linear/log/gradient各1/3、GT有效mask。仅改变这次用户确认项，数据/划分/归一化/LoRA/LR/初始种子/保存格式不变。两GAM在相同native_language_dim=768下构建Predictor，VLA仍只注入Qwen五槽seed、冻结未用语言CA；共同Predictor初值回归核验。

旧UAV-GAM在687步暂停，输出目录保留。新正式输出results/six_gpu_uavgam_unified_20261109/uav_gam_native_t5，从0初始化，不把旧拓扑的优化器强行迁移。旧checkpoint未带native_gam_path/paper_t5_native_text时继续走原兼容路径。公共构建使用独立action_interface_dim=2048以保留未使用的公共投影和动作头初值；该属性不是一层投影，native T5实际输出与hidden_size均768，forward中没有768→2048运算。

2026-11-10用户最终确认的 VLA-GAM：指令、native5D Pose 均由 Qwen 融合，不保留 T5 语言 CA，也不重复把 Pose embedding 送入 Predictor SA。当前 RGB 的 frozen DA3 shallow features 与 Qwen 五 Action Slots（经现有公共投影、Predictor in_proj 后作为 action_slot_seed）进入 Predictor。采用 `native_gam_path=true`, `native_frozen_shallow=true`, `native_sdpa=true`, `vla_pose_in_vlm_only=true`。禁用 proprio slot 时同步更新每步 token 数，相关模块保留初值、冻结且不前向，以保留公共模块构建顺序和 state 兼容性。无 Current deep decode/loss、无 proposal 辅助 loss、无 bank/warp/residual/Stop。只监督最终 action、future patch MSE 和 future depth，权重3/1/3，与当前修正 UAV-GAM 一致。

准备配置 `results/vla_gam_vlm_pose_only_20261110/config.yaml` 从实际运行的 unified UAV-GAM v3 配置派生，继承数据/采样/batch/优化器/保存协议，Qwen 分支按现有 Pure/Warp 配方。尚未启动训练或新的 GPU probe。此前不带 `vla_pose_in_vlm_only` 的旧 VLA-GAM 代码保留原运行语义，不追溯描述为这个配置。当前正在运行的 UAV-GAM 不改动。

## VLA＋GAM 对照修正版 UAV-GAM

修正版 UAV-GAM 启用了五槽独立的 Predictor embedding，使用独立 seed 48、标准差 0.02 的正态初始化。新 VLA＋GAM 对照用 `--match-gam-slot-embeddings` 继承该设置，Qwen 五槽投影仍作为额外的 slot seed。Pose 和语言只在 Qwen 融合。Future DA3 和最终动作的监督沿用上面的 Future-only 配方。

先导出配置核对数据、有效 batch 和完整 checkpoint 设置；该命令只读配置。

```bash
PYTHONPATH=src /root/miniconda3/envs/uav-gam/bin/python -m experiments.uavflow_paper_ablation.gam_train --variant vla_gam --match-gam-slot-embeddings --print-config
```

正式续训使用首跑保存的完整配置与 checkpoint，保留原来的 slot embedding 设置。
