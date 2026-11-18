# 实验基准：当前 Warp（用户于 2026-10-29 指定）

## 2026-11-11 六卡：GAM续训＋detach/Pose探索＋Current-only

同日用户授权只迁移物理GPU布局：Warp由1/2/3改0/1/2（同NUMA0、Gen4×8），单卡GAM由0改3（NUMA1、实测Gen3×8），Current保持4/5且原进程不重启。`migrate_six_topology`等待Warp2000、GAM90000完整last；90000先完成原定全量验证，再核对config/Adam全部step/epoch-batch游标及world size，固定完整状态、清理对应全部旧rank后恢复。batch、rank顺序、累积、采样、LR、模型和损失不变，启用逐张量优化器恢复检查；不承诺缺少旧RNG的checkpoint逐位不停机等价。状态与逐实验审计在`six_gpu_exploration_20261111/topology_migration`；是否实际完成迁移以`launched.json`及恢复后真实更新为准，不能只凭计划声称已换卡。

用户明确分配GPU0继续Unified UAV-GAM，GPU1～3训练改进non-unified Warp，GPU4～5训练Unified VLA＋Current。入口`experiments.uavflow_paper_ablation.launch_six_exploration`，独立配置、启动审计和状态保存在`results/six_gpu_exploration_20261111`；不覆盖历史Warp/Pure结果，不自动关机。

GAM保留原配置单卡24×1，从完整last的86790步、epoch6 batch9504恢复；473项Adam状态步数均86790，恢复时逐张量检查优化器。旧checkpoint未保存独立RNG，不能声称与不停机逐位一致。新增两条从预训练/原构建流程初始化，各20轮257620更新、无Stop、原absorbing保留、无额外repeat。统一native5D Pose/公共MLP、LoRA与动作头LR、固定1% episode划分、每5000及每epoch全量3095验证、完整epoch/last/best。

改进Warp仅在原non-unified配置上启用`detach_warp_depth`、`detach_warp_proposal`、`future_pose_token_enabled`及seed47的Future Pose适配；三卡micro8×累积1保持原24样本与深度归约分组。Proposal动作loss及Current depth自身监督保留，Future hidden/Pose/共享DA3仍有梯度。它是两个改动的组合探索，不能单独归因detach或Pose；不能再声称数值Warp路径反传修正Proposal。

VLA＋Current复用已验收`current_only.yaml`，Unified指令，两卡micro12×累积1＝24，显式启用`allow_warp24_two_gpu_repartition`保持每更新样本集合/顺序和游标。深度loss原三组8改为两组12，不声称梯度归约或BF16逐位等价。Qwen五槽hidden投影后直接注入Current DA3，最后动作头直接预测；不解码或监督数值Proposal、不做Future/Warp/CA/残差。总loss为`3*final_action+3*current_depth`；最终动作和Current深度梯度可经Action Tokens回到Qwen。公共初值启动时由既有legacy hash审计检查，不将CPU重建描述为历史GPU零步快照。

数据盘启动约42GiB，不能保证三条训练全部完成。沿用5GiB共享暂停、串行完整checkpoint写入，并为两条新实验各预分配1800MiB系统盘应急状态空间；原应急文件不清理。触底时完整保存并暂停，不删除优化器或自行缩减逐轮保存。

## 2026-11-11：detach＋Future Pose，仅做单卡容量探针

用户明确要求先测 microbatch12×累积2（全局24），且只测显存和吞吐、不长训。`results/warp_detach_pose_capacity_20261111` 从当前 non-unified Warp 原配置生成原版/组合版两份临时配置；等 Pure best 与 Warp last 的1/5步全量评测均成功且GPU1释放后，逐版本执行8次真实前向、反向、裁剪及AdamW更新。12×2若OOM再试8×3、4×6、2×12；不写任何正式checkpoint，不自动转1000步训练，不影响GPU0 Unified UAV-GAM。

组合版仅在Warp数值输入处detach Proposal与Current depth，保留两者各自动作/深度监督，以及Future Action hidden、Pose和共享DA3的梯度。Future Pose复用公共`5→256→GELU→2048` MLP输出，原native坐标系/缩放及初始化不改；新增独立seed47的普通随机`Linear(2048,1536,bias=True)`，LR归入head组5e-4，不零初始化。作为第六个非视觉条件prefix参与Future DA3自注意力，但只解码前五个Action Tokens，Pose不作为第六个动作，也不进入DPT空间patch。所有开关默认关闭，旧参数初始化经回归验证保持一致。

12×2保持全局样本数24，但改变原3卡micro8的深度loss归约分组；本次不声称数值、梯度或收敛等价。8×3可保留原三组micro8采样布局；即使分组相同也不承诺浮点逐位一致。该实验不用于评价收敛或闭环SR。

后续新训练/消融默认以 `results/warp_direct_pose5_qwen1e5_20261028/warp` 为基准，不能沿用旧 Pure 的默认配置并声称仅有架构差异。已有训练保留其原始配置，不追溯修改。

配置与固定划分的仓库内快照位于 [experiment_baselines/warp_20261029](experiment_baselines/warp_20261029/)，来源与 SHA256 见 `provenance.json`。路径与硬件可迁移；实验语义以这些配置及本说明为准。

## 2026-11-02新增Pure24 Stop队列（用户最终确认）

Pure24当前纯动作训练至10轮/128810步，完成后从完整epoch10恢复，追加10轮动作+Stop联训；**Stop配方参考实际跑过的Pure PoseA 5+5，不再把未执行的Warp Stop计划当作已验证实验**。新目录 `results/pure24_joint_stop10_20261102`，配置快照 `docs/experiment_baselines/pure24_joint_stop_20261102/`。

后10轮保留当前batch24×累积1、native5D Pose、固定缩放、动作MLP/头/LR/constant和原生窗口；每条额外追加5个终点self-pair，**新增样本动作loss mask=0、Stop硬标签=1**，Stop梯度仍进入共享策略。Stage2共359185窗口，每轮14966步，新增149660步，总计278470步。Stop宽512、5x1536动作hidden、专用Pose5→512→GELU→512；LR5e-4、BCE正权重5、loss权重1。公共Stop初始化逐张量核对与旧Pure PoseA epoch5保存的未训练Stop相同。该协议不代表当前正在跑的Warp32已经训练Stop；未来Warp消融必须显式复用同一采样和mask，不得仅设置native self-pair次数5后让动作也监督。

Stage2的动作/Stop训练、验证和服务均统一native helper，修正±180数值边界；当前动作训练进程不改动。独立Stop-Pose MLP不与动作Pose MLP共享参数。每epoch/last/best/best_stop保存完整状态和优化器。主关机控制器已注册此队列，须等所有已注册训练完成才正常关机，异常检测保留。

## 当前生效的新配对实验（2026-10-29后续明确指令）

用户已明确批准：Pure和Warp都从预训练初始化开始，20轮全程无Stop；每条训练轨迹末帧追加5份全零动作窗口。Pure GPU0 microbatch32×累积1；Warp GPU1/2各microbatch8×累积2，全局batch均32。这是对下面旧Warp基准中batch24、repeat0、10+10阶段的显式覆盖，其他公共条件继续沿用并核对。

新输出 `results/matched_repeat5_pure_warp_20261029/`：`pure.json`、`warp.yaml`、`manifest.json`、`common_init.pt`、`preflight.json`。公共Qwen LoRA、Pose MLP、动作投影及动作头从同一份初始state_dict加载，均有张量hash校验；Warp proposal与future两个动作头同样采用ActionHeadV2(input/hidden=1536，2块，learned positions)，普通完整action输出。

训练359185窗口＝normal259187＋原有absorbing49958＋末帧repeat50040；验证不repeat，仍3095窗口。固定seed42＋epoch，同一窗口索引序列；两边每次更新同样32个样本，每轮丢弃同样17条尾部，11224次更新，20轮224480次更新。Pure训练动作loss外权重3；Warp沿用3×双头动作均值＋3×双流深度均值。两边best按全部验证窗口的最终动作L1选择。

用户要求恢复原保存方式：每个epoch、last和best全部保存完整训练状态（含优化器、游标，以及各trainer原有的scaler/RNG等状态），不能以仅评测权重替代。当前GPU进程不重启；监控将每轮精简中间文件替换为同一步、同epoch、同batch的完整last硬链接，并验证optimizer存在。新启动/续训进程直接保存完整epoch。空间不足时仅归档完整文件并保留软链接，不删除优化器状态。旧轨迹停在64405步/epoch5，不覆盖也不继续10+10。

## 原Warp基准公共条件（下表中的变更项以上述新配对设置为准）

| 项目 | 基准 |
|---|---|
| 全局有效 batch | 24（当前三卡，每卡 microbatch8，累积1） |
| 数据 | train 10008 episodes / 309145 windows；val 101 episodes / 3095 windows |
| 划分 | 固定1% episode stratified_instruction，seed42；使用快照中的 episode IDs |
| 窗口 | train normal259187＋extra absorbing49958；val normal2592＋extra absorbing503 |
| absorbing / repeat | 保留 absorbing；每条末尾最多5个起点各1份；人工首尾repeat、自配对、终点窗口额外重复均0 |
| 动作标签 | yaw-only、5槽、相同合法mask及人工补零监督；沿用现有动作归一化文件 |
| 指令 | instruction_unified及相同instruction_overrides；原始instruction属于另一个显式实验变量 |
| 图像 | 单当前RGB、224×224、原有解码/缩放；无增强 |
| Pose | 起始yaw坐标系的相对[x_cm,y_cm,z_cm,yaw_deg,is_start]；除[1000,1000,1000,180,1] |
| Pose MLP | 5→256→Qwen hidden；GELU；普通随机初始化，独立seed43/44 |
| Qwen | 同一qwen3.5-2b预训练权重；layer23，5个causal Action Slots＋Pose Token |
| Qwen LoRA | all_linear，rank32，alpha16，dropout0，LR1e-5 |
| Pose / 公共动作投影 / 动作头 LR | 5e-4 |
| 公共动作头 | ActionHeadV2，隐藏宽度1536、2个MLPResNet块、learned chunk position；与旧Pure结构相同；运行时从DA3实际embed_dim核对，不能误用Qwen hidden2048 |
| 优化器 | AdamW，betas(0.9,0.999)，eps1e-8，weight_decay0，global grad clip1 |
| LR调度 | constant，无warmup |
| 随机种子 | seed42、model_init_seed42；相同seed不等于公共参数逐位相同，新的严格对照需检查/保存公共初始化 |
| 精度 | BF16；记录实际attention/FLA内核、microbatch和checkpointing设置，不声称它们数值等价 |
| 验证 | 每5000次更新及每个epoch，全量3095个唯一验证窗口 |
| best选择 | 全部验证窗口的最终动作L1（当前Warp H1_refine_action）；旧Pure按normal选best，不可混用 |

## 采样顺序与预算

当前Warp使用DistributedSampler(seed42＋epoch)，三卡按rank交错分片，每卡batch8、DataLoader drop_last=True。每轮实际训练309144条（末尾不足一批的1条不训练），12881次优化更新。

后续公共对照应沿用相同窗口列表及顺序、epoch seed、每次更新的24条样本组合、尾部丢弃规则。仅改成相同有效batch但使用不同随机采样，不应称为顺序已对齐。单卡可合并原三卡采样；物理卡数不是实验变量，但需记录实际microbatch/累积与数值设置。

同时记录epoch、优化更新数和已曝光窗口数；禁止将batch32与batch24的相同步数解释为相同数据曝光。

## Warp方法本身的设置

- 单窗口；不启用多窗口时间causal；Current/Future共享DA3。
- Bilinear RGB Warp，保留无效区mask embedding，删除correspondence CA。
- Proposal与Future使用独立的普通初始化动作头；Future直接预测完整action，无residual相加。
- DA3全层LoRA，rank64、alpha128、LR5e-5。
- `L = 3 × (L_proposal + L_final)/2 + 3 × (L_depth_current + L_depth_future)/2`。
- 深度linear/log/gradient各1/3，固定metric scale100m。

去掉几何的Pure对照中，几何分支及其深度loss是预期移除项；公共Qwen、Pose、公共动作投影/动作头、动作归一化和动作loss外权重应对齐。若只剩一个动作头，明确使用 `3 × L_action`，不能悄悄沿用旧Pure的 `1 × L_action`。新增/移除监督本身须列为方法差异。

## 阶段与可比较范围

当前Warp计划前10轮动作＋深度，后10轮动作＋深度＋Stop联训；第10轮128810步，第20轮257620步。Stop为现有512宽MLP＋专用5D Pose编码器，LR5e-4，硬current标签，正样本权重5，loss权重1；Stop梯度进入策略，不额外新增终点重复样本。

纯动作对照先比较前10轮；要比较后10轮，Pure也需相同Stop联训阶段、标签、采样及选择规则。旧Pure20轮纯动作与Warp10＋10不能声称阶段完全对齐。

## Stop 的 Pose 对齐要求（用户再次明确）

后续 Pure/Warp 的 Stop 训练与推理必须使用同一当前观测Pose：`[x_cm,y_cm,z_cm,relative_yaw_deg,is_start]`，起点与起始yaw坐标系相同，缩放均为 `[1000,1000,1000,180,1]`。不得切换成旧CLIP的sin/cos＋分位数归一化、旧7D里程状态或绝对世界坐标；不得输入GT终点距离或未来轨迹。is_start仅当前观测为任务起始帧时为1。

Stop使用自己的Pose MLP，与动作策略的Pose编码器不共享参数；两边Stop专用MLP的结构、初始化规则、LR及训练预算也应对齐。相同Pose输入不意味着共用训练后的Stop权重。当前Warp训练 `rgbwarp_objectives.py` 和推理 `inference.py` 均使用上述5D固定缩放。

## Pose MLP 架构必须逐项相同（用户硬性要求）

后续Pure/Warp对照，动作分支Pose MLP与Stop专用Pose MLP必须分别完全对齐，不能仅保持5维输入或学习率相同：

- 动作Pose MLP：`Linear(5,256,bias=True) → GELU → Linear(256,2048,bias=True)`，输出作为Qwen Pose Token；无额外LayerNorm、Dropout或其他层。LR `5e-4`。
- Stop专用Pose MLP：`Linear(5,512,bias=True) → GELU → Linear(512,512,bias=True)`；无额外LayerNorm、Dropout或其他层。LR `5e-4`。
- “完全一致”包括层顺序、层数、每层input/hidden/output size、激活、bias、归一化、dropout、输入缩放与插入/融合位置。动作Pose MLP与Stop Pose MLP各自独立，并不要求这两个用途彼此用同一网络。
- 公共Pose模块采用相同初始化规则；严格新对照需保存/加载同一份公共初始state_dict，并校验各张量一致，不能只根据seed相同作保证。
- 启动前列出模块结构并检查参数名称、shape及初始值；任何不符必须明确报告，不得称为已经对齐。禁止为追溯对齐而改写历史checkpoint。

## 已存在的历史差异

旧Pure参考 `results/pose_A_action15epochs_20261027/A`：有效batch32、动作头宽1536（与Warp相同）、action loss外权重1、按normal val选best、20轮action-only。当前Warp：batch24、宽1536、双头action均值外权重3、按all val选best、10轮action/depth＋10轮joint Stop。

旧实验结果保留作探索/配方比较，不追溯标记为严格控制变量的架构消融。以后启动前列出与本基准的差异；用户显式指定的实验变量优先。

2026-10-29启动核对纠正：当前DA3/Warp实际embed_dim及动作头hidden均1536；此前文档误写2048，误把Qwen hidden维度当成DA3维度。旧Pure同为1536，这一项不是历史架构差异。新配对已改用实际1536维，并从同一份初始权重加载。

## 2026-10-30 五卡续训（用户明确批准）

Pure从完整`pure/paused_step_0066000.pt`恢复，保持GPU0 micro32×累积1。Warp从完整`warp/paused_step_0022448.pt`（epoch=2，batch=0，已完成第2轮全量验证）恢复到GPU1～4，每卡micro8×累积1；全局batch仍32，总预算仍224480步，不重新计轮。运行配置为同目录`warp_four_gpu.yaml`，保留原`warp.yaml`作为双卡快照。

`preserve_two_rank_microgroups`显式启用原双卡8条microbatch分组映射：四个rank依次对应原(rank0,micro0)、(rank1,micro0)、(rank0,micro1)、(rank1,micro1)，并沿用各组原seed。每轮11224次更新、尾部丢弃17条均不变。初次迁移仅允许epoch边界；后续四卡checkpoint支持相同映射及batch游标续跑。完整校验配置只允许累积次数和保存格式标志变化，优化器各参数组与state张量加载后逐项检查；不清空Adam。

CPU审计见`results/matched_repeat5_pure_warp_20261029/four_gpu_recovery_audit.json`：两checkpoint优化器计数均与保存步一致；epoch0/2/3/19全部44896次更新、179584组microbatch逐一对齐，并验证中途游标seek。此校验不声称跨卡数BF16逐位一致。回归测试`tests/test_four_gpu_migration.py`。每轮/last/best均继续完整保存，停止请求已归档，不在原暂停点再次停止。

## 2026-11-01：排队的 Pure24、旧Warp配方5轮探针

用户要求对照旧batch24 Warp，不加额外repeat5，先训练5轮并排在当前best/epoch20闭环评测之后。独立输出`results/pure24_legacy_warp_recipe_5epochs_20261101/`；当前repeat5 Pure/Warp和历史文件均不改写。

- 数据使用旧Warp原固定1% episode划分、309145训练窗口（normal259187＋原absorbing49958），额外self-pair/repeat均0；val3095，逐次全量。5轮共64405次更新，每轮12881；每5000步及每轮全量验证。
- Pure单卡GPU0 micro24×累积1，总batch24。旧Warp是三卡各micro8；实际硬件/microbatch布局不同，不宣称BF16逐位等价。已逐一核对5个epoch，单卡每次更新24条样本与旧三卡全局样本及顺序相同，每轮丢弃同一条真实尾部样本。
- Qwen LoRA1e-5，Pose/动作投影/头5e-4，constant、warmup0；公共Pose为5→256→GELU→2048，固定缩放[1000,1000,1000,180,1]；同1536维ActionHeadV2。动作loss外权重3，无Stop前向/监督；Pure去掉几何、第二动作头和深度监督是方法差异。
- 原Warp没有留0步初始化权重。已用当前代码在CPU执行旧Warp模型构建路径（相同预训练、seed42及Pose43/44），在训练/优化器创建前截获proposal路径公共状态，保存`common_init.pt`，并逐项检查可严格加载到Pure。它是旧流程的CPU重建，不是历史GPU初值逐位复刻，历史代码/CPU与GPU随机流差异仍是比较限制。
- 实际重建hash对比：Pose与当前repeat5公共bundle相同；Qwen LoRA、动作投影、动作头及新增token embedding不同。旧流程proposal/future头独立随机初始化，当前repeat5把两头加载成同一个公共初始head。新Pure使用重建的旧proposal头。
- 每轮epoch/last/best均保留完整优化器与原有训练状态。best按全量all-window动作L1。队列待两份273条闭环均完成、GPU0显存释放后启动；新训练异常看门狗在25步稳定启动后启用。其余训练继续。

审计：`initialization_audit.json`、`sampler_audit.json`、`resolved_plan.json`；运行监控`queue_state.json`；独立图`live/loss.png`。不把该历史对照追溯描述为除架构外完全逐位对齐的消融。

### 2026-11-01 后续明确覆盖：Pure24采用当前公共初始化

用户随后明确要求“24按照我们现在的这个commoninit”。上述Pure24探针的初始化改为直接加载`results/matched_repeat5_pure_warp_20261029/common_init.pt`，与当前repeat5实验公共初始化的四组hash和6行冻结token embedding完全相同；CPU实际加载验证已通过。旧流程重建bundle保留为该探针目录的`legacy_reconstructed_common_init.pt`，不参与训练。旧初始化审计/配置已保留备份，新的`initialization_audit.json`记录当前实际来源。此覆盖只改初始化：batch24、5轮64405步、不加额外repeat5、保留原absorbing、LR/损失/全量验证/完整checkpoint策略均不变。队列仍等待当前两份273条闭环评测完成、GPU0释放后启动。

## 2026-11-01 Pure24 延长至10轮

用户要求将当前 Pure24 从总计5轮延长为总计10轮。保持 batch24、累积1、额外末帧repeat0及当前 commoninit，其他设置不变。原5轮进程继续，结束后通过 `--resume last.pt --extend-action-budget` 恢复优化器/RNG/数据游标，续至128810步。配置 `results/pure24_legacy_warp_recipe_5epochs_20261101/pure_10epochs.json`；监督进程 `experiments/uavflow_matched_repeat/extend_pure24.py`。所有epoch/last/best仍保存完整训练状态。

## 2026-11-02 Pose入口审计（未修改运行中训练）

审计文件 `results/pose_entry_audit_20261102/`。新Pure best/epoch20闭环日志共24983次请求，Pose列顺序、XYZ/yaw与commanded反馈、is_start按首次决策1其后0均吻合。真实checkpoint前向hook确认state_encoder实际收到 `[x/1000,y/1000,z/1000,yaw/180,is_start]`，不是No-pose路径。左绕人6m第二帧固定图像/XYZ/yaw，仅将is_start0改1，repeat5 Pure20首步平移由0.0198cm变10.4337cm；Pure24 epoch5则9.7606cm与9.7609cm，提示前者对start标志异常敏感，尚不能证明repeat因果或全部Surround根因。

发现数值边界差异：matched训练Pose使用episode_yaw4d_state，对精确+180度输出-180；验证/服务progress→atan2路径可能输出+180。全训练episode每条抽首帧/次帧/中间/末帧共40032点找到6个Rotate末帧，404个验证抽样点为0。两者几何同角但MLP输入为-1/+1，不能作为数值对齐通过；原data_checks.json的errors按mod360检查，仅证明角度几何一致。该边界不能解释yaw约1度的Surround第二帧停住。未修补现有训练或评测。

## 2026-11-02 成功任务Pose对照与Surround圈数混淆

`results/successful_tasks_pose_audit_20261102/` 保存16条Pure32 epoch20已成功轨迹（Move/Shift/Rotate/AD/Approach/Pass/Land/Retreat各2条）的起点/中段/末段及中段XYZ/yaw置零干预，两模型各160输入。Pure原始48个决策的小动作判定48/48复现闭环；所有选取的同输入重复检查完全一致。Warp192k为同输入离线重放，不能记作其成功闭环。Pure末段start0全16条小动作，强改start1仅6条恢复运动，说明start不是全任务通用停走开关。

另发现5条Surround提前停止的Pose与完整GT圆周终点接近：XYZ差0.40/7.46/2.36/4.53/0.47cm，yaw差0.009/1.002/0.352/1.006/0.275deg。左绕人6m GT完整路径37.80m、累计yaw360.96deg，但当前5维只保留wrapped yaw和起点相对XYZ，不能直接区分绕圈前后的访问次数；早期和终点is_start皆0。该事实构成状态混淆线索，不能证明全部失败由此单独造成，也不能解释Turn/Rotate首帧不转。sin/cos同样周期重复，不能单独提供圈数信息。原始数值见 `surround_pose_aliasing.json`。未更改正在运行的训练配置。

## 2026-11-02 全12条Surround扩展检查

`results/surround_all12_and_val_pose_20261102/`：Pure32 epoch20与Warp192k各140输入，包含12条Pure失败任务的首次/首次小动作画面及固定图像下GT中段/终点Pose干预，另含固定1%验证集全部2条Surround的真实图像/真实Pose（首帧、1步后、1/4、中段、终点）。12个首次小动作画面，start0下Pure12/12小动作，Warp3/12；start1下两者皆0/12。Warp三条为左绕狗3.5m、右绕狗6m、左绕人6m。真实验证中段均约10cm，终点GT0：Pure3.448cm与0.0198cm，Warp两条约0.0202cm。因此start0未阻断Pose/停止监督；早末状态混淆及学习边界不稳仍是线索，非已证明唯一原因。Pure一个反事实首帧start0重复未逐位相同，8次新进程补测相同，原/新进程首步幅值相差约0.066cm且均非小动作，详见repeat_audit.json；主要12个停止画面的重复检查一致。

## 2026-11-02 Rotate全部15条测试首帧

`results/rotate_all15_three_models_cpu_20261102/`：同15条A_epoch5保存的TEST首帧、instruction_unified、Pose[0,0,0,0,1]，CPU BF16 native Torch，同一native yaw-only GT（每条首步±3度）。Warp194k最终头15/15转角绝对值<0.1度，Proposal同样15/15近零；Pure32 epoch20/224480与Pure24 epoch5/64405皆15/15转向方向正确且无近零转角，首步yaw MAE分别0.12656/0.13359度。Pure32首步平移均值0.27061cm、最大0.69803cm；Pure24均值0.49157cm、最大4.39895cm（2条>1cm）。所有45输入同进程重复逐位一致。此为不同训练预算的首帧诊断，不是新闭环SR，也不是同轮数架构消融。

## 2026-11-02 Rotate输入扰动与CPU精度复核

`results/rotate_perturbation_cpu_20261102/`：15条同一TEST首帧、13种输入（亮度±5%、右移1像素、Gaussian像素噪声sigma1/255、XYZ各+1/+10cm、yaw±1/±10度、start0、模拟一步yaw后start0，以及原始），两模型各195输入；图像tensor SHA与Pose跨模型全部一致，未编辑原图。Warp194k最终头与Proposal均195/195首步|yaw|<0.1度，最大分别0.000515/0.000984度；Pure32 epoch20为0/195近零，按变体平均|yaw|2.811～2.923度。原始输入重复均逐位一致。

两checkpoint动作normalizer完整state完全相同。另用精确Qwen→proposal_projection→parallel_action_head路径比较CPU BF16与FP32，二者15/15 Rotate近零，同路径Surround对照约-1.83度。FP32只检查Proposal，未检查最终DA3头；FP32转换保留现有存储参数值，不恢复预训练更高精度权重。以上未做GPU训练同核复核，不能将其表述为已验证GPU闭环失败或已定位训练根因。未变更训练。

## 2026-11-02 旧Warp24 Rotate扰动对照

保留的旧Warp24第5轮/64405步通过相同15条TEST首帧×13扰动复测。原始15条转向方向15/15正确，yaw MAE0.017187度、最大0.058593度，首步XYZ平移均值0.167423cm、最大0.333917cm。195输入无|yaw|<0.1度案例；基线重复逐位一致，跨新Warp/Pure32/旧Warp24的图像tensor SHA与Pose全部匹配。结果 `results/rotate_perturbation_cpu_20261102/summary_with_old24.json`。旧新Warp仍存在batch、额外repeat、初始化、训练预算等差异，不能把结果单独归因于batch。仍为CPU诊断，不是闭环SR。

2026-11-02用户再次覆盖关机安排：Pure24动作10轮后追加Stop联训10轮的队列保留，但不作为关机完成依赖。当前Warp32完成20轮/224480步、完整checkpoint确认后按主控制器原60秒完成确认直接关机，不等待Pure24或其他任务；即使Pure24尚未完成也会被关机中断。`completion_dependencies.json`已设为空列表，原异常关机逻辑保留。此条覆盖本文件前文“须等所有已注册训练完成”的关机说明，不改变训练配置和队列。

## 2026-11-02 当前Warp32完成后的新队列（用户再次覆盖关机安排）

用户明确要求当前四卡Warp32结束后，GPU1/2/3续训旧Warp24、GPU0评测Warp32。确认Warp24从保留的第5轮/64405步完整状态续到第10轮/128810步，全程无Stop；保留旧初始化、原生absorbing、额外repeat0、三卡micro8×累积1=24、相同LR/constant/动作及深度loss。每epoch/last/best完整保存优化器。配置仅改训练终点和完整保存标记，不替换commoninit、不回溯对齐。

队列 `results/after_warp32_warp24_eval_20261102/`，监督器 `experiments.uavflow_matched_repeat.after_warp32`。当前Warp32完整结束后启动；单卡GPU0依次评测Warp32 epoch20与best，各全273条，分类倒序每5条轮换，沿用已核对的commanded-waypoint首步执行、重新拍图推理、无逐步yaw校正、Stop-off、小动作连续10次和100步上限。Warp服务复用现有inference loader，显式native5D Pose和同一动作积分函数；没有用Pure权重入口加载Warp。

该队列已注册主控制器完成依赖：须等Warp24第10轮保存确认及两份全量闭环结束才正常关机，异常关机保留；覆盖上一条“Warp32结束立即关机”。Pure24/Stop旧队列不自动恢复，也不作为本次关机依赖。三卡续训前额外逐张量核对已加载optimizer；当前四卡运行进程未重启或改参数。

2026-11-02衔接恢复修复：此前Warp32正常完成后，队列先看到进程退出、后看到控制器complete标记，误报未完成并触发异常关机；Warp24和评测当时均未启动。队列现改为直接验证最终完整checkpoint，并允许最终游标的两种等价形式（epoch20/batch0或epoch19/batch11224均为224480步）；进程退出后未验证成功时给予180秒验证窗口。新增对应回归测试。已归档失败状态和关机记录，用户授权后重启队列；旧Warp24继续从64405完整状态恢复，GPU0运行Warp32 epoch20/best全273条，主看门狗在训练与评测实际运行验证后重新开启。公共权重和优化器未因修复而改写。

## 2026-11-03 停止repeat5评测、清理其权重，恢复Pure24 Stop

用户明确停止repeat5实验后续评测并授权清理其checkpoint。repeat5 Pure32/Warp32的训练权重及同inode诊断/评测硬链接已清理（审计 `results/pure24_stop20_warp24_20261103/cleanup_done.json`），公共common_init、配置、日志和已完成评测结果保留。旧主控制器和队列已停止，Warp24三卡训练未停止。新的监督器 `resume_stop_and_warp24` 将GPU0用于恢复Pure24已有10+10动作/Stop联训，仍以总计20轮/278470步为终点，不改变repeat或mask协议；从145000步完整保存点恢复，策略Adam592组step145000、Stop14组step16190，epoch11/batch1224，RNG/FLA恢复，未保存的775步需重算。原始日志备份后去掉未保存尾部。

新监控根 `results/pure24_stop20_warp24_20261103/`；等Pure24联训总计20轮与Warp24总计10轮均完整保存后关机，异常关机保留，独立guard开启。Pure24/Stop所有epoch/last/best仍含完整优化器。用户提出“无额外repeat5，动作20轮后接Stop”的方案尚属讨论，未将其静默替换成当前训练配方。

2026-11-03单条CPU闭环诊断：用户明确仅运行Land `2025-05-06_23-21-33`（遮阳伞右侧），早先误扩为273任务的队列在完成0条时取消。实际只完成该1条，固定Pure24 Stop152000完整checkpoint，策略CPU BF16原生Torch（CUDA未初始化）、UE GPU0，原commanded-waypoint首步执行后重新推理，无Stop拦截，仅记录概率，小动作连续10次/上限100步。目录 `results/pure24_stop_cpu_closedloop_20261103/single_eval/`，计划以`single_plan.json`和`current_plan.json`为准。64次推理后小动作停止，下降约1.58cm；对应GT相对下降286.088cm。最后Stop概率0.0275853。该CPU数值路径与训练GPU fused kernels不同，不将其与旧GPU评测混称同核严格对照；两条训练未停止。

2026-11-03用户明确要求同一遮阳伞任务执行五步对照：新增wrapper可选`--execute-steps 5`，默认仍1，不改原客户端文件；两项回归确认默认截取1、实验截取5、逐动作小动作检查/100动作上限/全部已执行Pose历史保留。固定相同Pure24 Stop152000 CPU BF16 checkpoint、同1条任务、Stop-off，单独输出`pure24_stop_cpu_closedloop_20261103/chunk5_eval`。15次推理、75个执行动作后因小动作结束；相对下降261.477cm（单步版1.583cm，GT286.088cm），预测终点距GT0.6403m，z差0.2461m。最后一次推理Stop0.299925（执行最后五步前）。结果不是正式自动Land成功标签，仅说明该单条执行chunk5后下降明显改善；不能与step1协议的旧SR混算。

2026-11-03指定8条完成五步执行后，用户要求配对重规划频率对照：固定同一Pure24 Stop152000 CPU BF16权重、同8条及顺序、Stop-off，小动作连续10次和100个执行动作上限；step5已8/8完成，step1追加运行于 `results/pure24_stop_cpu_selected8_step1_20261103`，完成后由 `pure24_stop_cpu_selected8_paired_20261103/summarize_when_done.py` 自动生成对照表。两者都不是全程开环，而是每1/5步重新观测；不混合SR口径。

Warp24最新83000步对比旧第5轮64405步的真实验证末段重放已完成，共31episode×3位置×2模型，CPU BF16，报告 `results/warp24_latest_tail_compare_20261103/summary.md`。真实终点GT0：Rotate残留|yaw|均值1.709→2.996度，小动作1/3→0/3；Turn0.625→0.561度，小动作5/8→5/8；Shift位移13.635→12.583cm，小动作6/20→7/20。运动段Rotate预测约3.016度/GT3度，故当前问题是终点不归零，不能混同已删除repeat5 Warp32的首帧不转。该检测不是新闭环SR。

2026-11-03用户要求最新Warp也跑同8条的重规划频率对照：最新完整Warp24 checkpoint固定于83000步，目录 `results/warp24_latest_cpu_selected8_paired_20261103`，CPU BF16原生Torch策略，UE GPU0；分别step1/step5各8条，Stop head本就关闭，仍为原小动作连续10次/100个执行动作上限。`queue.py`等待当前Pure24同8条step1完全结束后自动启动两份Warp评测（max_concurrent1）。理由：GPU0训练加一个UE后仅余约3.1GiB，第二个UE实测约2.8GiB，故复用单个UE依次评测，避免训练OOM。该队列不修改、不暂停两条训练；Preview已加入两个待评测来源，完成后自动保存Warp的step1/step5对照表。CPU服务增加可选device，不改变默认GPU路径；四项相关回归通过。当前Pure24与历史Warp24仍存在初始化、训练阶段等差异，不称为只移除几何的严格消融。

2026-11-03用户要求排队Pure24加Stop前第10轮作为行为对照：固定epoch10/128810完整checkpoint，stage_steps=[128810,0]，同8条、CPU BF16、执行1/5步各一组、Stop未训练且不拦截；目录`results/pure24_action10_cpu_selected8_paired_20261103`。独立queue等待Warp最新同8条两组结束后复用单个GPU0 UE，训练不中断；Preview已增加两组待生成来源。结束后自动生成step1/step5内部表及`stop_before_after.md/json/csv`，逐条比较该128810保存点和既有Stop152000结果，注明额外训练曝光等混杂因素，不将终点距离直接当语义SR。Pure服务对具有matched_warp_config的5D动作模型也使用native Pose helper，与Stage1训练和当前Stop推理一致；未改历史checkpoint，其他legacy入口保留。11项相关回归通过。

2026-11-03用户新增5条Land并要求Warp优先：固定当前对照的Warp83k/PureStop152k/Pure24 action10权重，各做step5/step1两组、CPU推理、Stop-off、同初始条件和原停止协议。任务为21-56-36右路灯、21-56-20左路灯、21-55-47右雕像、21-55-29右狗、21-54-48右汽车（均2025-05-12）。根目录`results/land5_cpu_models_paired_20261103`，顺序写入`order.json`：当前Warp原8条两组结束→新Land5 Warp→新Land5 PureStop→新Land5 Pure action10→原Pure action10八条对照。只停止并重接了尚未启动的Pure10等待监督器，不中断当前轨迹或训练。新通用`queue_evaluation`严格核对前驱实际完成模型数、任务数和_SUCCESS后复用单个UE，相关3项回归通过；Preview增加6个明确来源，不覆盖旧模型或人工标注。

2026-11-03用户质疑PureStop左路灯(2025-05-12_21-56-20)step1/step5差异是否来自渲染：实测两首帧像素MAE17.681/255，均值亮度137.343/120.741，初始无人机XYZ相同、actor yaw差0.002deg；相机ID3/1为运行时分配，实际camera外参/FOV未probe记录，不能只凭ID判错视角。同进程CPU BF16重放两张原图均精确复现各自日志五槽输出、重复相同，说明首帧图像差异确实影响预测。固定单步原图时首槽横移-0.0188cm，五槽第五waypoint横移-12.6512cm，故执行频率也有直接影响；两完整轨迹差异不能全归于单一因素。审计`results/pure_stop_left_streetlight_step1_step5_audit_20261103/summary.md`。未修改当前训练、UE渲染或原评测协议。

2026-11-03同一左路灯案例继续审计：固定PureStop152k同进程CPU BF16，对执行5/10/50个动作后的两条轨迹做图像×Pose四组合离线干预，另重放五步后续真实输入，全部重复一致、真实配对均与原日志数值精确一致。执行5动作时首槽左移：单图+单Pose0.043cm、五图+单Pose0.035cm、单图+五Pose14.300cm、五图+五Pose11.997cm。执行50动作时首槽下降：0.021/0.029/13.047/8.711cm（同组合顺序）。说明本例横移和下降对Pose整体依赖明显，仅换五步图不恢复这些动作；不能说只由渲染导致，也未隔离具体Pose维度。真实五步后续输入再喂同一模型，首槽会正常左移/下降，模型无执行模式开关。交换图/Pose不一致，仅离线诊断，未用于实际评测控制，也不是中途step5切step1的闭环。报告`results/pure_stop_left_streetlight_replay_cross_20261103/summary.md`。

## 2026-11-03用户更新关机条件

用户明确要求Warp24第10轮完成即关机，不再等待Pure Stop达到20轮。当前附加监控`shutdown_after_warp24.py`（状态`results/pure24_stop20_warp24_20261103/warp_completion_shutdown.json`）先验证Warp128810完整checkpoint与训练退出，再等Pure下一次正常完整last原子保存，核对优化器/RNG/游标后暂停Pure并关机。Pure保持原配置与保存间隔1000，不重启训练；未完成部分以后可续跑。此指令覆盖此前“两条都完成才关机”的正常结束条件，原异常监控在交接前继续有效。

用户随后要求Pure继续训练：此前Warp完成关机流程已正常保存Pure243000步并执行。现在从该完整保存点恢复同目录、同配置，目标278470/总20轮；新监控`results/pure24_stop_resume243000_20261103`，Warp已验证完成，不再因Warp完成提前关闭Pure；正常关机恢复为Pure完成并完整保存后触发。

2026-11-03用户要求暂停Pure训练，GPU评测Pure Stop244k及Pure动作第10轮的同15条历史失败任务后恢复。训练已在250000步完成全量验证及完整保存后暂停，游标epoch18/batch1462（显示第19轮），stage_steps[128810,121190]。暂停硬链接位于原Pure输出目录`resume_gpu_eval_1793710034237531863.pt`，原训练监控已关闭关机权限。接管流程`results/pure_gpu_failed15_then_resume_20261103/workflow.py`依次GPU闭环评测两组，执行1步重规划、Stop仅记录；结果另存，不混入CPU记录。评测完成后从该完整保存点恢复原配置到278470步；失败时记录错误并恢复训练。新恢复监控在该流程目录的`training_resume`下。

## 2026-11-04 Pure Stop20完成后的best/last执行步长对照

用户要求当前Pure Stop完成20轮后，joint阶段按全量all-window动作L1选出的best.pt与最终last.pt各做执行1步/5步的闭环评测，均全273条、同顺序任务、GPU0串行，Stop不拦截但记录概率，保留原连续10次小动作停止及100动作上限。队列results/pure24_stop20_best_last_eval_20261104，完整权重结束后硬链接固定。新监督器等待Pure完成、四组评测完成及Warp24第15轮完整保存后关机，覆盖上一关机依赖；不改变训练配方。Preview已注册四组。

接管旧监督器时，旧screen会话退出导致Warp torchrun收到SIGHUP，于约137725步中断；Pure未被中断。从137000完整保存点(epoch10/batch8190，world3，929项optimizer)原配方恢复，约725步重算。恢复训练改用独立session，旧launcher也增加start_new_session=True，防止监督器退出传播终端HUP。恢复审计及日志保留在该队列目录及原train.log。

2026-11-04 Pure Stop20四组评测启动修复：UnrealCV launcher.read_port使用int(ss[1][-4:])，10000端口被解析为0，best单步首条未开始即失败。保留失败日志，改为9800～9803四位端口，四组从0条重试并采用原有--adopt；训练和单步/五步执行协议未改。Preview四组入口与实时状态已核对。

2026-11-04用户追加评测Pure动作第10轮128810步，分别执行1步/5步，全273条、Stop关闭，与Stop20 best/last同任务顺序及相同闭环协议。固定完整action10_full.pt硬链接为队列checkpoints/action_epoch10.pt，optimizer592项。GPU0最大并行3份，实测两份约15916MiB，单次显存准入阈值显式7800MiB（通用默认8500未变），第三份优先epoch10单步；原best/last进程与进度保留。队列监督器等待6组完整评测及Warp15结束再关机。Preview新增Pure 第10轮无Stop · 执行1步/5步 · 全273条。

## 2026-11-04 GPU0评测后从头Pure15，Warp结束优先关机（最新覆盖）

用户选择从头训练15轮，使用按旧Warp流程、seed42/43/44在CPU重建的proposal路径公共初始化legacy_reconstructed_common_init.pt，而非current common_init.pt。队列results/pure24_legacyinit15_20261104，GPU0六组全273条评测全部结束并释放显存后启动，单卡batch24×accum1，193215步，constant Qwen LoRA1e-5、动作Pose/projection/head5e-4、动作外权重3；原生absorbing、额外repeat/selfpair0、无Stop、同manifest/split/normalizer。15轮每次更新的24条样本全量顺序及丢弃尾部1条已CPU逐轮验证；实际microbatch24与Warp各rank8有数值差异，不声称逐位训练轨迹相同。初始化四组张量hash与既有旧流程重建记录完全相同，已用真实Pure模型严格加载校验；历史Warp缺少0步快照，不能证明与历史GPU初值逐位一致。

新Pure动作Pose MLP保持5→256→GELU→2048、bias及既有缩放，ActionHeadV2 hidden1536不改。新配置native_validation_pose=True，训练/验证/服务均使用native helper，以与Warp对齐精确180度边界；该flag默认不影响其他配置及已运行实验。每5000步及每轮全量3095验证，best按all-window动作L1，保存完整每epoch/last/stage1_best，best.pt作为stage1_best.pt软链接。输出实际在系统盘/root/uavflow_training/pure24_legacyinit15_20261104/pure，数据盘队列pure为软链接。未删除旧数据、checkpoint及标注。

用户随后明确覆盖关机依赖：三卡Warp达到第15轮193215步、完整checkpoint验证后直接关机，不等GPU0新Pure训练完成。若新Pure正在训练，等待按1000步周期的下一份完整last.pt（含optimizer/RNG/cursor/config），核对后停止其进程并sync关机；若Pure尚未启动则不再启动，保留已完成评测直接关机。新独立session监督器接管旧监督器，既有Warp和闭环工作进程未重启。CPU preflight完成、无CUDA初始化，20项相关回归通过。新曲线入口/livepure15/；原/live24/和/livestop/保留。

2026-11-04用户再次覆盖关机安排：“先别关机”，随后要求先核对最新是否改善再讨论Warp续到20轮。已关闭runtime_control.json.shutdown_enabled，新队列监督器读取此开关，state.shutdown_armed=False；Warp与GPU0评测不被暂停，Pure评测后队列保持。当前Warp训练终点暂仍15轮，未擅自延长到20轮。最新186000步固定快照正在进行同32条验证末段CPU诊断。

2026-11-04用户明确同意Warp继续到20轮。当前三卡进程不重启，先正常完成第15轮193215步，再由独立队列监督器从完整last.pt自动恢复到第20轮257620步；新配置results/warp24_constant_to20_20261104/config.yaml仅training.max_epochs由15改20，其他配置逐字段CPU核对完全一致。当前187000步检查world3、epoch14/batch6666、train_size309145、optimizer929项和scaler齐全，游标满足step=epoch*12881+batch。交接时再验证第15轮完整状态、各卡释放显存并启用UAVFLOW_VERIFY_RESUME_OPTIMIZER。三卡仍micro8、accum1、global24、constant、无Stop、repeat0，逐epoch/last/best完整保存。自动关机保持关闭；GPU0六组评测后Pure15原队列不变。

本次空间预算：数据盘实际剩15.80GiB，预留Warp第15～20轮六份完整epoch、原子保存临时文件及best额外增长约11.65GiB，剩余评测图像按最多100步/条估计0.63GiB，预计仍有约3.5GiB余量（日志等另计）；系统盘剩16.76GiB，新Pure15按18份完整状态保守预算12.47GiB，约余4.3GiB。不删除旧checkpoint、数据或标注。

2026-11-04 GPU0显存仅5266MiB调查：最后一组PureStop20_last_step5于14:20（北京时间）第197条任务读取camera2/location时UE进程SIGSEGV退出；UnrealCV receive线程cannot join current thread，客户端主进程未退出，模型HTTP仍ready，评测停在196/273约5小时。监督器只看进程/state心跳未识别真实停滞，导致GPU0 Pure队列未启动。保留196条完成结果，归档未完成obs及UE/client故障尾日志至job recovery_*目录；只接管监控并杀死卡住客户端，复用policy初次重启遇UE SkeletalMesh MeshObject启动断言，随后自动重试成功。重启后跳过已完成196条，从197条重跑，19:29已记录新推理请求，GPU0约7856MiB。Warp训练未暂停/重启。

本次closed_loop新增可配置progress-timeout（默认1200秒）与max-job-retries（默认0），仅当前队列显式设600秒与2次；以client.log和service/requests.jsonl真实活动判断无进度，不以控制器心跳替代，超时/失败清理该job工作进程后重试，保留已有合法轨迹、固定checkpoint、执行5步/Stop-off/种子和任务顺序。新监控接管后继续等待最后77条结束再启动GPU0 Pure15。自动关机仍关闭，Warp15后续20队列保留。4项闭环快照/步长/执行补丁回归通过。

2026-11-04用户询问GPU0 Pure是否启动，并曾要求取消重训改从动作10轮续训；核对时新Pure已经第1轮1325步，暂时仅SIGSTOP队列监控以防后续调度，训练进程未发送任何信号。用户立即更新为“在训了就继续吧”，故保留原从头Pure15（legacy重建公共初始化，无Stop、repeat0），恢复监控SIGCONT；未切换到10轮权重、未重启/暂停Pure训练，确认1450步仍推进。原Warp20目标、关机关闭均保留。

2026-11-05 checkpoint清理：用户先要求确认清单，清单生成后未删除；随后明确授权“都可以删 保证 现在的 和 我要求保留的那几个评测的就ok”。按results/checkpoint_cleanup_20261105_035342/plan.json的102条候选逐一核对设备/inode/大小后删除，执行及journal记录在同目录。释放数据盘73.7869GiB，剩83.518GiB；系统盘剩13.4565GiB。当前Warp全轨迹checkpoint及其已识别快照、系统盘当前Pure全轨迹checkpoint均保留；旧Pure24保留128810动作epoch10、260000 Stop action-best、278470 Stop last及硬链接别名和构建依赖。逐份验证三份保留评测权重的完整optimizer/RNG与inode未变，当前Warp212000步optimizer929项、Pure44000步optimizer592项完整。未改数据、日志、结果图像、标注；训练进程无暂停/重启。其余旧正式实验及旧CPU诊断权重现已删除，不能再声称可加载这些已删历史模型做新推理。

## 2026-11-05 Warp Stop-only重复补齐（未启动）

用户取消了下个保存点暂停及新训练启动安排，当前Pure15与无Stop Warp20保持运行；随后授权只补齐Warp Stop入口。新代码仅在显式joint_stop_terminal_repeat_count=5时，复用Pure的原生窗口＋每episode五个末帧self-pair：原309145窗口监督不变，新增50040窗口动作和两路深度mask为零，Stop硬标签1且梯度进入共享策略。验证保持原3095窗口，无额外重复。Stop公共初值、512宽结构、5D native Pose缩放与Pure24 joint阶段一致；源码变更未注入当前两条已运行训练。

独立配置results/warp24_joint_stop10_20261105/config.yaml，从原128810步/epoch10完整状态显式激活，只增加Stop优化器组并保留原929项moment；新阶段359185窗口、每轮14966更新，总目标278470。修正新增样本后预算不能按20×新epoch步数计算的问题；新分支后续resume校验采样大小、阶段参数和游标，防止再次激活Stop。完整epoch/last/best保存策略不变。脚本scripts/resume_warp24_joint_stop10_to20.sh已准备，只运行了--check；默认4,5,6三卡，未启动训练或新关机流程。详细规则见experiments/uavflow_warp_direct/JOINT_STOP.md。

2026-11-05用户重新要求“Warp结束自动关机”：已启动独立scripts/watch_warp20_shutdown.py，状态results/warp20_shutdown_20261105/state.json；仅接管当前既有Warp/Pure PID，不重启训练或加入新实验。待三卡Warp退出并核验第20轮257620完整优化器/scaler/游标后，等待Pure下一份完整last（含592项optimizer、RNG、stage_steps、windows_seen及游标），原目录硬链接固定恢复点，再终止Pure进程组、sync并执行平台shutdown。该结束条件不等待Pure15训练完成。runtime_control.json.shutdown_enabled=True，改为false可取消尚未执行的关机；不新增异常即关机规则。

## 2026-11-05 当前Warp20结束后的raw instruction五轮对照及四组闭环

用户明确要求当前训练结束后，三卡从头训练唯一更换原始instruction的Warp对照5轮，其他设置与当前Warp一致；GPU0 Pure在完整保存后暂停，用当前Warp最终best和last分别执行1步/5步闭环，全273条。队列results/warp24_raw5_after20_20261105/watch.py已注册，旧warp20_shutdown看门狗取消，避免提前关机；正常关机依赖改为新raw5训练及四组闭环全部完成。

新config.yaml逐叶比较unified_reference.yaml，只有instruction_map及其SHA256、训练预算20→5、独立输出目录、仅审计用legacy_initialization_reference不同。三卡1/2/3 micro8×accum1、global24，constant Qwen1e-5、Pose/projection/head5e-4、DA3 LoRA5e-5及全部原架构/损失/数据/保存策略保持；无Stop、额外repeat0。原309145训练窗口及3095验证窗口、固定episode IDs和源顺序不变，原生采样/过滤完成后仅替换logs中的文本，原位姿数组对象与Torch RNG均保持；10008训练episode中7946条、101验证episode中76条实际措辞改变，其余原始与统一措辞相同。验证也用对应raw instruction，动作/深度标签和归一化文件不变。每轮12881更新、5轮64405，5000及每epoch全量验证，epoch/last/best全部完整状态。

CPU真实构建已逐张量hash校验Qwen LoRA/公共动作token、Pose MLP、proposal projection和proposal head与legacy_reconstructed_common_init.pt相同；proposal/future头仍各自独立随机初始化，未加载成同一头。新GPU rank0启动会在训练前再次校验这四组初值，不符报错；历史Warp无0步快照，沿用之前已明确的按原构建流程和seed42/43/44重建协议。审计preflight.json、config_diff.json，15项相关CPU回归通过，CPU审计无CUDA初始化。

当前Warp20完成257620后先固定最终best_action.pt和last.pt硬链接，再启动新raw5；等待Pure下一次完整last并检查592项optimizer、RNG、stage_steps、windows_seen与游标，固定原目录恢复点后停止其进程组。四组闭环仅占GPU0串行，复用experiments.uavflow_warp_direct.serve及现行closed_loop_client原执行协议、原273条顺序、Stop-off、连续10次小动作及100动作上限；当前unified模型评测仍用instruction_unified。读取真实活动的600秒超时/最多2次重试，UE端口9700起、policy6100起。Preview添加四个独立来源，未覆盖旧结果或人工标注。新队列runtime_control.json.shutdown_enabled可取消队列最终关机。

2026-11-05 raw/unified措辞对照补充核对：发布raw列也含7条此前已人工修正的person/dog目标错误，直接覆盖会撤销纠错。已让instruction_control.apply_instruction_map对两组保留同一instruction_overrides（7条均在训练集），这些条目使用相同已修正文本；其余使用发布raw措辞。实际训练集改变7939/10008条，验证76/101条；不引入错误标注变量。新增回归明确覆盖raw不能撤销目标纠错，相关16项CPU测试通过。原始raw映射文件及其SHA不变，修正表本身也不修改，正式训练入口应用共享修正。

2026-11-05用户进一步明确：7条人工纠错在non-unified中必须保留发布raw原句，只换错的person/dog名词，不可整句使用unified修正文案。此前整句覆盖实现现已修正为repair_raw_target_word，逐条验证每句恰好只改变一个名词token；unified已有纠错入口与原parquet均不改。详细原句/修正raw/修正unified三列保存于results/warp24_raw5_after20_20261105/seven_raw_noun_repairs.json。新raw对照训练仍取消，关机仍关闭，未重启任何训练；前面整句覆盖时跑过约100更新且未保存checkpoint，不算本修正协议的训练结果。修正后训练7944条指令措辞不同、验证76条，7条目标物修复语义一致；23项相关CPU回归通过。

2026-11-05用户确认“按照这个训”，再次强调先取non-unified再替换名词，随后明确“好的继续吧”。已在全新目录results/warp24_raw5_nounfix_20261105从0启动三卡1/2/3 Warp5，目标64405步；不续接此前约100步的整句覆盖试跑。配置相对当前unified Warp仅新增raw文本映射/校验及初始化审计、5轮预算和新输出目录，其余训练/模型/数据设置逐字段一致。raw映射复制后SHA校验相同；7条每句只替换一个person/dog token的逐条审计保留。训练启动再次核对legacy重建公共初值hash；每epoch/last/best完整训练状态照旧。独立只读监控watch.py记录进度/完成和错误，无自动关机。原四组unified Warp best250k/last257620闭环评测继续使用GPU0；Pure保留完整136000步暂停点，不恢复；自动关机保持关闭。

2026-11-05用户要求新训练live图：新增独立/liveraw5/（results/warp24_raw5_nounfix_20261105/live），后台experiments.uavflow_warp_direct.raw_live每30秒刷新。与原unified Warp按相同步数范围比较：训练proposal/final、全量验证all/normal/absorbing proposal/final、训练和验证Current/Future深度；输出完整起步loss.png及4000步后after4000.png。不混入Pure或此前约100步取消试跑。训练使用原三卡25步区间汇总，全量val3095；首个raw5000步验证完成前明确标注尚无验证点。Preview首页增加独立链接，原live24/livestop和评测来源保持。页面/status/PNG均HTTP200，绘图CPU运行，不重启训练。

2026-11-05用户要求先暂停non-unified训练，四卡各跑一组现有Warp20闭环。已在8000步完整last原子保存后固定paused_four_gpu_eval_*.pt，核验world3/train_size309145、epoch0/batch8000、929项optimizer均为8000步、scaler存在，然后停止原训练及其只读进度监控。状态paused_for_four_gpu_evaluation；不重启或换配方，自动关机仍关闭。

评测沿用results/warp24_raw5_after20_20261105/eval原目录、4份固定规格（best250000/last257620×执行1/5步，各273条），仅接管closed_loop控制器并使用--adopt --slots-json四张卡独立模型/UE部署：GPU0 best-step1，GPU1 best-step5，GPU2 last-step1，GPU3 last-step5。原卡0policy/client/UE未中断，143条既有结果保留并继续增加。保持同任务顺序、源client/pose/执行协议、600秒真实活动超时和最多2次重试，结果与人工标注不覆盖。交接审计results/warp24_raw5_nounfix_20261105/pause_for_eval_state.json，原控制器state备份在eval/state_before_four_gpu_adopt.json。此次未安排自动恢复训练或关机。

2026-11-05用户指定优先Approach两条：2025-05-12_21-43-59左侧树、2025-05-12_21-43-45右侧树。best单步两条已完成，不重复；其他三组在当前轨迹json及2D/3D图完整保存后停止该client/UE，保留原policy及所有已完成轨迹，控制器--adopt接管并用task_ids_priority_approach2.json把这两条放前。保留best5已完成47、last1已完成29、last5已完成41条；模型、Pose、执行步数/停止规则、端口和四卡部署不变，未改训练。新client参数顺序已核对，完成两条后继续其余273任务；Preview固定显示顺序不受执行优先级影响。状态approach2_priority_state.json，原state备份state_before_approach_priority.json，关机保持关闭。


## 2026-11-06 评测剩一组后恢复non-unified Warp（用户授权）

Pure从136000步恢复后，用户要求仅剩一组闭环评测时让该评测独占原卡，其余三卡恢复non-unified Warp。调度results/warp24_raw5_nounfix_20261105/resume_when_one_eval.py已排队：当前预计保留GPU2 last-step1评测，Pure在下一份完整last（592项optimizer、RNG和游标核验并硬链接固定）暂停，GPU0/1/3从raw Warp完整8000步恢复至5轮64405步。仍world3、每rank batch8、累积1、全局24，rank和数据游标不变；只更换物理GPU编号。原config、raw先取发布instruction再只修复7条名词、native Pose及各LR均不改，新增deterministic_accumulation不启用。epoch/last/best保留完整状态，启动校验929项优化器状态，自动关机关闭。Pure暂停后不自动重启，后续由用户安排。

## 2026-11-07：逐轮固定输入回查及non-unified Pure20队列

用户要求回查当前Pure24的Turn平移何时消失、Approach 21:47:12/21:46:51及Pass21:51:14的动作演变，以及Unified Warp的Land21:54:48/21:55:29/21:55:47何时退化；同时检查Warp对应Approach/Pass。独立只读诊断results/checkpoint_history_20261107在空闲GPU2按batch1重放Pure第1～20轮、Warp第5～20轮及250000步best，共37份已保留权重。每份240个固定RGB＋真实已观测native Pose输入，包括15条Turn的首帧/第二帧/中后段、6条指定任务的Pure/Warp保存路径，以及原Warp83k Land路径和旧Stop偏移Turn路径。无新UE、无checkpoint复制、无GT进入策略；重放动作不等价于逐轮闭环SR。Pure best206096重放原Pure观测186输入的原结果子集逐值一致。不能据此补造Warp第5轮以前已删除的权重。

用户授权上述检查完成后启动仅换instruction的Pure对照：results/pure24_raw20_nounfix_20261107，GPU2从头20轮/257620更新，无Stop，无额外repeat，batch24×累积1。逐字段继承results/pure24_legacyinit15_20261104/config20.json，仅新增instruction_map及SHA256；不改变初始化文件、动作权重3、LoRA1e-5、Pose/头5e-4、constant/warmup0、Pose5→256→GELU→2048、native缩放、采样/划分/归一化及完整epoch/last/best保存。文本先取发布原始instruction，7条仅修复person/dog名词，保持原句式；训练7944/10008条、验证76/101条措辞变化。原309145/3095窗口、sample_sha256及公共四组初始化hash均已核验，详细preflight.json。控制器experiments.uavflow_eval_diagnostics.raw_pure_after_history先确认全部诊断报告完整、GPU2空闲及磁盘余量，再启动新Pure；诊断失败则停止队列，绝不静默跳过。其他三卡non-unified Warp训练不变，自动关机关闭。

2026-11-07随后用户明确取消固定画面重放：已停止诊断worker及旧“重放后训练”控制器，保留已完成报告，不再补跑37份固定输入。改为用户确认的第5/7/10/13/15/17/20轮＋best，各执行1步/5步真实闭环；Pure测15条Turn＋指定2条Approach/1条Pass（18条），Warp测这2条Approach/1条Pass＋Land21:54:48/21:55:29/21:55:47（6条）。Best与20轮的原完整273条结果直接引用，不重测、不复制标注；新增24份模型部署、288条轨迹，复用96条，共384条检查。新队列results/epoch_history_closedloop_20261107、模块history_closedloop，GPU2先Pure两份并行，再Warp单份（Warp单模型峰值约10.3GiB，避免双Warp＋UE超显存）。仍原batch_run_original_protocol.py和closed_loop_client、native Pose、Stop-off/连续10次小动作/100动作上限，无GT输入。Preview新增独立来源保留原标签及人工标注。所有选定闭环完整且2D/3D图存在后再启动上述non-unified Pure20；不再以固定重放完成作为启动条件。三卡non-unified Warp继续、关机仍关闭。

该逐轮闭环启动时新分配的五位数UE端口10100在旧客户端读取为100，导致首次启动连接失败且尚未产生动作。已仅将此新队列的UE端口改为四位数9040～9051（Pure）、9080～9091（Warp），接管控制器并保留仍存活的模型服务；未修改原评测代码、相机或执行算法。新增端口范围校验防止再次使用五位数。失败启动日志保留。

2026-11-07晚用户要求卡2 Pure完整保存后暂停，改评Unified Warp第15轮193215步，全273条各1步/5步，优先Approach2025-05-12_21-43-45。用户进一步要求同一模型不重新加载，每任务相邻跑1步、5步。新独立队列results/warp15_full_eval_20261107：等待non-unified Pure5000步及该步全量验证完成后，硬链接固定last、校验全部optimizer step/RNG/游标，然后暂停Pure及原等待控制器。模块paired_execute_client仅在原original_protocol经过现有patch_source后，将任务列表展开为每task的(1,5)并设置动作切片长度；每次仍走原任务reset、初始位姿收敛、目标验证及新建observed history。共用一个Warp服务/一个UE，结果及图像分step1/step5目录，服务/相机/执行策略未另改。完成全部546条后从Pure完整暂停点恢复，保持原non-unified Pure配置与预算。新增/livepure-raw/卡2实时图，与同初始化unified Pure比较train及全量val all/normal/absorbing，首个验证前不伪造验证点；Preview增加独立两份Warp15全273来源，原结果/标注保留。自动关机关闭。

2026-11-07 23时暂停控制器故障修正：第一次5000步验证前last_eval_step=None被int转换触发TypeError，原Pure继续运行。用户指出后已修复nullable检查并增加回归，不回退已训练进度；实际在6000步完整last保存后暂停。核验optimizer592项均step6000、epoch0/batch6000、stage_steps[6000,0]、windows_seen[144000,0]、RNG及配置完全一致，固定paused_warp15_eval_step_0006000.pt。原Pure进程已退出，卡2显存已释放，开始加载同一份Warp193215模型供成对1/5步全量评测；评测后从6000恢复，避免回退5000造成进度损失。根目录pause_audit.json及原failure记录均保留。

2026-11-08用户要求Pure Qwen按checkpoint真实闭环跑Warp此前的Land。独立results/pure_land_epoch_closedloop_20261108：统一指令Pure24第5/7/10/13/15/17轮，21:54:48/21:55:29/21:55:47三任务，每任务分别执行1/5步，使用既有paired_execute_client原协议、同一模型服务与UE，checkpoint间更换模型。Best16与epoch20的12条已有完整结果直接引用，新增36条，不复制权重、不做固定画面重放、不新增训练。GPU2当时Warp15推理约13.9GB、空闲10.7GB，新增单份Pure服务＋UE入场要求空闲≥9000MiB，Pure服务保留原32%显存分配上限。Preview新增12个独立来源，不复制人工Land标注。

为避免Warp评测结束立刻恢复raw Pure训练挤占新增推理，已在保持原Warp服务474139/client474435不重启的情况下接管只读监控，adopt_warp15_resume观测原结果完整后清理原UE/服务，并等待新增Pure Land全部完成，卡2空闲后再从6000完整状态恢复non-unified Pure。原三卡non-unified Warp训练未改，关机仍关闭；接管前状态保留state_before_pure_land_dependency.json。

2026-11-08 Pure Land逐轮评测恢复：epoch5的3任务×1/5步共6条完成；epoch7首次UE启动在SkeletalMesh.cpp MeshObject断言崩溃，客户端UnrealCV socket断开后卡住，原监控等1200秒无活动后退出。用户指出未运行后，保留epoch5全部结果及epoch7失败日志，新增断连立即识别、每checkpoint最多两次重试，以及已完整轨迹自动跳过；同目录恢复epoch7及后续，不重新跑epoch5，不更改模型/相机/原执行协议。失败状态不再留作Preview“运行中”。原Warp15及三卡训练不重启。

2026-11-08用户要求当前non-unified Warp下一完整checkpoint暂停，测全部消融显存。独立队列results/ablation_memory_20261108等待126000步：原样硬链接完整last、校验929项optimizer step、world3/train_size309145/epoch/batch/scaler，停止三卡训练及旧监督进程；GPU2既有评测不改。核对发现该旧Warp实际checkpoint没有独立RNG字段，不能追溯声称已保存CPU/CUDA RNG；仅如实保留原格式全部状态，用户已被明确告知。之后在GPU0/1/3分别运行隔离memory_probe，Pure/VLA+Current/UAV-GAM/VLA-GAM/Warp各micro8/12/24，累积3/2/1、global24，真实预训练模型及原窗口、完整loss/backward/clip/首次及后续Adam更新，共3次；仅scratch内更新，不写正式checkpoint、不改原模型/训练配置。micro8保留原三组采样，其他micro仅容量探测，不用作收敛对比。记录allocated/reserved/nvidia-smi、首次/稳定更新峰值，OOM和代码错误分开。测试结束Warp仍暂停；未授权自动启动消融正式训练。

2026-11-08显存测试已完成16项（15项矩阵＋VLA-GAM micro4补测），没有代码错误；每个成功配置真实完成3次AdamW更新。单卡allocated峰值GiB：Pure8/12/24＝9.15/11.17/17.03；Current-only8/12＝17.95/21.23，24 OOM；UAV-GAM8/12＝16.69/19.39，24 OOM；VLA-GAM8首步后OOM、12/24均OOM，micro4×accum6＝19.62成功；Warp8×accum3＝20.47，12/24 OOM。全部global24，保留既有模型、Pose、loss和初始化。micro8几何组保留原3×8分组，其他micro仅容量测试，不将采样差异用于收敛结论。报告results/ablation_memory_20261108/summary.md、summary.json及各report/config/log。0/1/3显存已释放，正式non-unified Warp仍在完整126000点暂停；卡2Warp15闭环继续。未启动任何新的正式消融训练。

2026-11-08用户要求继续Warp：在GPU0/1/3从paused_memory_probe_0126000.pt原配置恢复non-unified Warp20，无Stop/global24/micro8/accum1不变。恢复前config20.yaml逐字段与checkpoint config一致，epoch9/batch10071、world3/train309145、929项Adam step126000全部通过；实际训练打印MIGRATION_OPTIMIZER_EXACTLY_RESTORED states=929并已推进126025步。独立监督results/warp24_raw5_nounfix_20261105/resume_after_memory.py更新原state/continuation20_state，原epoch/last/best完整保存照旧。GPU2闭环继续，自动关机仍关闭。

## 2026-11-09 六卡续训及统一指令 UAV-GAM 对照（用户授权，未启动）

用户增加两卡并要求现有训练完整保存后重启。准备目录 `results/six_gpu_uavgam_unified_20261109`：Pure non-unified 下一完整71000步、Warp non-unified 下一完整187000步暂停，逐份核验optimizer/epoch/batch；Pure保留RNG，旧Warp保持原checkpoint格式，不追溯新增其没有的RNG状态。待六张卡可用后，GPU0续Pure24；GPU1/2/3续原Warp24（rank顺序不变，仅物理GPU编号变）；GPU4/5启动20轮统一指令UAV-GAM，与已完成的正式Unified Warp比较。用户已确认指令选择；现有两条raw轨迹的指令与配置不变。

UAV-GAM沿用paper_ablation/configs/uav_gam.yaml的T5、Predictor SA Pose、DA3、公共Pose/动作头初始化及LR/损失/constant/1%划分/完整epoch-last-best。双卡每卡micro12、累积1、global24，固定全局24个窗口及每轮12881更新。显式allow_warp24_two_gpu_repartition启用新分片：在原3×8拼接次序上分为两段12，保持每次更新样本和游标，但不声称保留原8样本深度loss归一化分组或逐位数值一致。原micro8单卡累积3/三卡累积1路径不改，现有训练没有启用新分片。新增CPU采样回归核验一/二/三卡每更新样本与断点游标；此前single-GPU micro12完整3次Adam峰值19.39GiB通过，尚未在双卡DDP实测。

一键入口scripts/start_six_gpu_uavgam_unified.sh在screen uavflow-six启动，先验证两条轨迹完整暂停、六卡空闲、配置与优化器一致及数据盘至少110GiB空闲。用户同意扩容，建议追加100GB；不自动清理或将epoch权重改为仅评测格式。不自动关机，不安装依赖，不在卡数不足时启动。每次重启优先读取原输出最新完整last，避免退回最初暂停点。

2026-11-09 六卡开机后，用户明确暂不能扩容，授权清理近期CPU测试冗余并先开训。实际仅删除可重建的两个CPU诊断输入缓存和CPU闭环MP4缓存，共131.4MiB；全部checkpoint、轨迹结果、人工标注、训练数据和预训练权重保留，数据盘仍约31.35GiB空闲。不能将UE pak的117个硬链接误认为117份重复物理文件。六卡启动检查改为至少24GiB可启动；显式环境UAVFLOW_MIN_FREE_GIB=20启用每25更新的空间检查，DDP由rank0广播同一决定。空间低于20GiB时先写完整last和_SPACE_PAUSED.json，再全部rank退出；Pure同样保存完整优化器/RNG/游标并暂停。此保护不改loss/LR/采样/逐轮保存，不冒充完整20轮已有空间保障。原环境变量未设置的实验路径不启用该保护。六卡入口和轻量Preview/实时图同时恢复，不自动关机。

2026-11-09 用户再次指定UAV-GAM/Warp公共初始权重与LR对齐，后续VLA-GAM复用相同定义。实际三份配置的公共LR逐字段核验：DA3全层LoRA5e-5、公共Pose MLP及两动作头5e-4、constant/warmup0/seed42；GAM Predictor（含SA Pose宽度适配）两组均1e-5，VLA的Qwen LoRA与Warp均1e-5。UAV实际启动日志Pose/proposal projection/auxiliary head三组hash已匹配重建初始化参考；CPU回归验证两个公共头以及两GAM Predictor初值。后续VLA-GAM须沿用同一Predictor seed45/SA适配seed47/公共Pose seed43及44、公共构建流程，不加载训练后的UAV-GAM作为新的初始化。记录results/six_gpu_uavgam_unified_20261109/shared_initialization_lr_contract.json；区分真实启动hash已验证项与CPU构建回归项，不追溯声称缺失的旧Warp GPU step0已逐位验证。当前训练未重启或改参数。

2026-11-09用户最终确认GAM边界：无几何前动作、无额外Current支路；只保留动作头适配、公共Pose MLP适配与已确认的深度loss配方。新增native_gam_path显式配置：原生T5 768→Predictor lang_proj1024，Pose仍5→256→GELU→2048＋Linear2048→1024走SA；当前RGB仅浅层输入；最终五槽ActionHeadV2只读出一次。训练损失3*最终动作L1＋1*Future patch feature MSE＋3*Future depth，后者固定100m、linear/log/gradient各1/3。真实未来RGB只在策略输出之后no_grad生成浅层监督，不进动作路径；Current GT depth不读取。旧辅助头和公共投影保留兼容state但冻结、无前向和优化器，不将它们报告为实际辅助预测。原checkpoint缺新flag时维持旧拓扑，不原地改写旧训练配置。

旧UAV输出uav_gam停687步保留，新输出uav_gam_native_t5，从0初始化；config路径uav_gam_native_t5_unified20.yaml。37项CPU测试通过：原Warp/旧GAM兼容、native单未来DPT调用、辅助头不调用、无Current GT、未来target与动作分离、全量评价指标、全局平均loss、初值/优化器恢复与GAM Predictor共用。采用adopt_native_uav监控接管现有Pure5908/Warp5909，进程和优化器均未重启；仅GPU4/5重新加载UAV。主一键脚本的launch_plan已指向新配置，保存原旧配置路径和687步原因记录。低空间完整保存暂停机制维持。

2026-11-09 native文本复核进一步恢复原生GAM的固定长度T5语言位置表（native_fixed_language_positions=true），而非沿用Qwen已位置编码token的variable_language_tokens设置。Native T5仍pad256、保留attention mask；Predictor内部768→1024后添加原生lang_pos，逐层语言CA不变。未来VLA-GAM构建同一Predictor初值，但冻结未使用的lang_pos和语言CA。旧路径不带该flag时维持既有checkpoint拓扑。首次修订的短GPU测试保留在uav_gam_native_t5和uav_gam_native_train.log；最终配置uav_gam_native_t5_v2_unified20.yaml、输出uav_gam_native_t5_v2、日志uav_gam_native_v2_train.log。从0重新开始这条，不加载已变拓扑的optimizer。另修正native训练日志weighted_feat使用双卡25更新平均，与feature/total统计一致；未改变损失反传。37项CPU回归含固定语言位置、公共最终头初值、两GAM Predictor同初值、GT仅目标、全量评测和优化器恢复均通过。Pure/Warp仍原进程。

## 2026-11-09 用户指定磁盘暂停底线降到5GiB

三条现有正式训练只调整磁盘保护，不改变配置、batch、LR、初始化、采样或训练预算。在各自下一个完整千步last处固定同一inode并核验config、全部Adam step、epoch/batch；用原命令和原环境恢复，仅新增5GiB共享磁盘策略。Pure保留RNG与FLA配置，Warp/UAV沿用原checkpoint实际包含的状态，不声称旧Warp补存了不存在的RNG。恢复代码再次逐张量核验优化器。控制器experiments.uavflow_paper_ablation.roll_disk_policy，状态results/six_gpu_uavgam_unified_20261109/six_gpu_state.json；roll_target分别79000/191000/3000，未达到前原进程继续训练，不能说新保护已加载。

完整checkpoint写入串行化，系统盘预分配6.089GiB（三份完整状态各加64MiB），用于触及5GiB或写入临时空间不足时完整保存并全体暂停；不精简optimizer、不取消逐轮/last/best、不自动关机。应急文件保留在/root/uavflow-emergency-sixgpu-20261109，输出last链接到对应完整状态；后续显式恢复时回迁数据盘并补足预留。25项CPU回归通过，包含完整应急状态mmap读取、跨epoch游标等价、共享暂停和原有路径兼容。

当前实测Pure每份0.65965GiB、Warp1.45573GiB、UAV3.78658GiB；现有epoch数量6/14/0，剩余逐轮状态约93.70GiB，另需best/last增长和保存余量，按100～105GiB数据盘空闲预算。现有数据盘24.78GiB，缺约75～80GiB；降低底线只推迟暂停，不能保证三条全部完成。

2026-11-09磁盘策略滚动恢复异常修复：UAV-GAM在3000步恢复时，旧rank0 PID25069因自行成为独立进程组而未随torchrun父组退出，占GPU4约17GiB，新rank0加载OOM，GPU5因此空闲。3000步完整checkpoint未损坏；已核对PID命令和孤儿状态后清理该残留，仅重启UAV两卡，从原last恢复，不改任何训练配置。Pure79000/Warp191000的恢复已分别通过592/929项Adam逐张量校验并继续训练。roll_disk_policy补充停止前枚举全部后代、停止后清理仍存活的独立进程组子进程，并修复子任务failed时主状态不应仍为running。当前采用adopt_native_uav接管Pure/Warp，仅恢复UAV，三条均保留5GiB磁盘策略及系统应急完整存储；本次恢复失败由进程清理遗漏导致，不是模型训练loss或checkpoint损坏。

2026-11-09用户确认后清理系统盘旧Pure重复状态：/root/uavflow_training/pure24_legacyinit15_20261104/pure的epoch01～10、139000步last/暂停别名及resume_after_warp20_eval快照，与数据盘pure_storage20内对应文件逐一SHA256完全一致。执行前再次校验源inode/mtime/size及双方SHA256后，将系统盘重复文件原子替换为指向数据盘完整原件的软链接；旧路径仍可读取相同完整optimizer/checkpoint。独有stage1_best.pt保留，原best.pt链接不改；日志、标注、当前训练和应急预留均未改。实际释放7.91584GiB，系统剩12.175GiB。清单和执行审计results/system_pure_duplicate_audit_20261109/{plan,execution}.json。

2026-11-09按用户要求新增CPU实时绘图experiments.uavflow_paper_ablation.live：UAV-GAM native v2独立动作/全量验证all、normal、absorbing/Future depth/feature MSE，读取自身全局双卡25更新日志；Non-unified Pure/Warp独立对比动作L1（Pure、Warp proposal、Warp final），train不拆窗口、val按三类拆分，提供完整范围及4000步后共同更新范围。实际日志parser复用warp_records，保留resume截断语义和各步最后记录，不混用加权总loss。输出six_gpu_uavgam_unified_20261109/live_uavgam、live_raw_comparison；Preview /liveuavgam/、/live-nonunified/，每30秒刷新，首页入口已添加。只重启Preview、未改或重启训练；六卡一键启动脚本补充CPU图进程以便下次恢复。PNG已视觉核对，HTTP四个新页/图均200，首个UAV全量验证尚未完成时明确pending、不补造点。UAV feature MSE上升如实显示，没有因此更改模型或损失。

## 2026-11-09 UAV-GAM原生梯度路径修正、验收及从0重训

用户发现feature MSE由0.94涨至36009后要求对齐原生GAM、确认后从0重训并删除失败checkpoint；随后要求测试冻结浅层后能否单卡micro24。复核发现native v2复用了Warp全层LoRA可微浅层入口，未来target虽no_grad却来自同一可更新浅层，与src/robot/losses/unified_loss.py的固定Teacher和detached浅层输入不一致。该项属于实现偏差，不能将v2描述为已对齐原生GAM；另在5000步首次全量验证时FlexAttention因batch变化失败，最近完整last仅4000步，不是5000步。

新增显式native_frozen_shallow/native_sdpa，仅新GAM路径启用，旧flag缺失时保持旧图，原有Pure/Warp不修改或重启。DA3 pre-global边界alt_start=13，blocks0～12及patch/prefix参数完全冻结并eval；当前RGB特征no_grad/detach，目标只读同一不可变预训练prefix，DA3深层不参与目标提取。它与独立冻结Teacher在该边界的输出等价；真实GPU样本对照原生encode_shallow_visual_slots最大差0，连续200次更新/断点恢复后同一目标逐值不变。无需复制完整DA3教师网络。后半段blocks13～39 LoRA可训练，rank64/alpha128/LR5e-5不变；共42,467,328个可训练DA3 LoRA参数。仍构建全部LoRA后冻结浅层，以保留旧公共构建顺序和深层初始随机流；不能把配置start0误读为浅层仍在更新。Predictor、Pose及动作头仍更新。

Predictor类与QK norm/RoPE/语言CA/SA proprio继续复用原生GAMFuturePredictor，源文件与本地Git HEAD逐字节相同；native_sdpa选择其原有dense block-causal mask，H1内双向、无跨未来泄露，避免当前Flex编译的batch特化错误。没有改任何attention权重结构。保留用户批准的Pose5→256→GELU→2048与SA宽度适配、1536五槽动作头、Future深度固定100m及linear/log/gradient各1/3。总loss仍3*final_action_L1＋future_patch_MSE＋3*future_depth，无Current解码、无proposal动作监督、无Stop。新增native验证前完整last保存，避免验证异常丢失刚完成的训练更新。

47项最终CPU回归通过；GPU4单卡micro24连续8次完整AdamW更新，allocated峰18.3206GiB/reserved峰19.9395GiB。随后真实完整trainer完成100更新＋全量3095窗口验证，保存473项完整Adam状态/epoch0 batch100；中断并清理全部后代后，原配置逐张量恢复优化器，续到200更新再全量验证3095窗口。验证包含24和末尾23条batch，均成功；目标和浅层参数逐值不变；feature MSE从首次batch0.94760降至200步batch0.81492，首次完整val MSE0.889333。验收记录frozen_gate/state.json、single24_training/acceptance.json，原生源码对齐记录native_frozen_alignment_audit.json。短程验收不声称保证后续所有轮次的优化效果。

正式新配置uav_gam_native_frozen_t5_v3_unified20.yaml、输出uav_gam_native_frozen_t5_v3、日志uav_gam_native_frozen_v3_train.log。GPU4单卡24×累积1，从预训练和公共初值重新开始20轮/257620更新；没有使用v2或验收checkpoint续训。原Warp24每次更新全局24样本ID/顺序、seed42、309145训练窗口/3095验证窗口与1%固定划分保持；单卡24替代分布式micro12/8，深度按有效像素的局部归一化及浮点数差异明确保留，不能声称梯度逐位等价。GPU5空闲，未擅自启动别的消融。Pure PID32556/GPU0、Warp torchrun32279/GPU1～3不变；接管监控adopt_native_uav只新建UAV进程，自动关机仍关闭。launch_plan与一键启动入口均支持当前单卡部署，live源切至v3，不混入失败曲线。

用户授权清理的失败v2 last4000/临时resume3000及本次验收临时完整状态已在验收通过、确认新正式命令启动后删除，日志/配置/评测指标/报告保留；实际释放15.66288GiB，数据盘剩24.55024GiB。旧正式Warp/Pure/已评测Stop/公共初始化均未动。清单及逐文件执行记录failed_checkpoint_cleanup_{plan,execution}.json。5GiB共享磁盘暂停策略和系统预分配应急完整状态保留，UAV角色输出指向v3；每轮/last/best仍含完整优化器和原trainer保存的状态。

2026-11-09用户要求空闲GPU5评测non-unified Warp第15轮或best并在图中圈出第15轮。读取完整权重核实：第15轮193215步/epoch15 batch0，当前best155000步/epoch12 batch428。先评第15轮，未擅自添加第二份best评测。results/warp24_raw_epoch15_eval_20261109，raw_warp15_gpu5复用已验证run_warp15_paired原执行器，只显式传入checkpoint/GPU5/服务6725/UE9175/原始instruction；该函数新增可选参数，旧默认GPU2/unified/端口及旧评测行为不变。全273条每条独立reset后执行1步、5步两份，单模型/UE相邻运行，无Stop，无额外执行校正，原客户端sha256483729701b18ba35349fd141b3ccfdf080bae906cbac3a83a485a3ad00690dc8。测试JSON273条原instruction均非空，与7条训练名词修正episode无交集，因此没有替换或重写测试指令、目标或标注。第一条右侧树21:43:45优先，其余沿用此前全量任务顺序，不限制每类5条。已确认GPU policy step193215 ready、至少37次实际推理、第一帧非白屏且渲染正常；GPU5模型+UE约13802MiB。五项配对/原执行协议回归通过。

Preview新增独立来源Non-unified Warp24第15轮193215·执行1步/5步·全273条，保留全部旧标注；原三条训练未暂停。/liveraw5/和/live-nonunified/完整范围图右上全量验证action子图对non-unified final193215加黑圈，best加紫星；best位置按best_action.json实时更新。共同范围图尚未覆盖193215时不强行延长Pure比较范围。控制器不自动关机，完成后仅释放自身GPU5服务与UE。

2026-11-09用户要求当前GPU5 non-unified Warp15优先Land/Pass/Approach/Shift各最后3条、各类倒序，之后继续其余全273条。按当前Preview正式类别顺序取末3倒序（不是跨类别仅按文件时间猜分类）：Land21:56:36/21:56:20/21:56:01，Pass21:52:47/21:52:30/21:52:05，Approach21:55:08/21:55:00/21:48:00（均2025-05-12），Shift2025-03-30_12-54-06/12-53-14/12-50-21。priority_requested_12.json及task_ids_priority12.json记录12条和完整273顺序。保持每条先1步后5步，完成结果跳过。

priority_raw_warp在原正在执行的5步2025-03-30_12-41-38保存有效轨迹、2D/3D图和termination后交接；保留同一个policy PID49875，仅按原配置重启评测client及同一UE副本。首次交接复制/proc/cmdline时误删内部空参数，导致--priority_categories解析失败且尚未启动新UE/产生动作；已修复为保留中间空参数，并用--priority_categories=恢复；增加回归测试。resume_raw_priority接管，原始instruction、GPU5、6725/9175、256分辨率、原执行/停止/初始位姿规则均不变；四项优先级/配对回归通过。已确认第一个优先Land2025-05-12_21-56-36真实开始推理、首帧渲染正常，policy未重载，Pure/Warp/UAV训练未暂停。旧完成轨迹和标注均保留；无关训练和评测协议未改。

2026-11-09用户认为当前non-unified Warp15效果不好，明确取消该评测并改测best，先Turn/Shift各3条。已停止原priority controller/client/policy及其专属UE，保留全部已有结果和标注，Preview旧标签追加“已停止”。当前best读取并硬链接固定为155000步、epoch12 batch428、929项optimizer，results/warp24_raw_best_turn_shift6_20261109/best_snapshot.pt，不受后续best覆盖影响。按上一条要求继续取每类末3倒序：Turn2025-03-30_12-54-13/12-52-50/12-48-18；Shift12-54-06/12-53-14/12-50-21。仅6任务每条1步/5步共12轨迹，不暗中继续273条或恢复已取消Warp15。复用同一配对执行器，新可选task_ids/job_prefix及total=len(ids)保持旧默认273条行为；五项原协议/配对回归通过。GPU5服务6726、UE9176、原始instruction、native Pose、256相机、Stop-off、小动作及100步限制不变，原client hash不变。policy已确认ready step155000且GPU推理正常，Preview新增Non-unified Warp best155000·Turn/Shift各3条·执行1步/5步。训练三条未中断，自动关机仍关闭。

2026-11-09用户取消当前non-unified Warp best六任务闭环，要求查空间并用空闲GPU5启动non-unified GAM。已停止raw_best_turn_shift控制器/客户端/模型/专属UE，保留已有结果、标注和固定best快照，Preview标签与状态标“已停止”。启动的是当前已通过验收的UAV-GAM native frozen v3的原始指令对照，而非另换VLA-GAM架构。

新配置uav_gam_native_frozen_t5_v3_nonunified20.yaml与统一指令版逐字段比较，只有instruction_map、instruction_map_sha256、training.results_dir三处不同；raw映射复用当前non-unified Warp的raw_instructions.json，sha256=0ec7276977ca233ea4236ea7ddf31859778a8f9c2f1d76006e357df621223c0d。运行时核对train10008条中7944条措辞变化，7条人工名词修复保留；val101条中76条措辞变化。公共Pose/投影/头初始化hash与统一版相同，固定浅层边界13、rank64/alpha128的深层LoRA、LR/优化器/scheduler/采样顺序/309145训练及3095验证窗口/1%划分/无Stop无额外repeat、完整每epoch/last/best保存均继承。GPU5单卡24×累积1从预训练初始化重训20轮/257620步，无resume；输出uav_gam_native_frozen_t5_v3_nonunified，日志raw_gam_train.log，独立控制器raw_gam_launch/state文件raw_gam_state.json。已实际超过50更新，feature MSE约0.9388，无非有限异常。其余三条训练未暂停。

空间启动前数据盘22.71GiB、系统12.17GiB；新增第四份完整应急预留约3.62GiB后，系统剩8.55GiB，数据盘仍22.71GiB。共享5GiB磁盘政策原子注册uav_gam_raw角色，四条完整状态预留共9.7099GiB，仍串行保存，触发后四条完整保存暂停；空间不能保证四条全部20轮完成，不缩减完整checkpoint格式。六卡一键启动入口补充raw_GAM控制器和第四份预留恢复，已运行三条不重启。原始指令/磁盘保存相关14项CPU回归通过；raw_gam_preflight.json与raw_gam_started_audit.json留存实际字段差异、进程、初始化及磁盘证据。/liveuavgam/增加instruction_comparison.png显示统一/非统一UAV-GAM动作、三类全量验证、深度和feature MSE，HTTP已确认200，source/status均对应新目录，不混用旧失败训练。


## 2026-11-12：用户要求立即停止旧GAM并重训五槽独立embedding版

旧Unified UAV-GAM native frozen v3在超过161000步后立即停止，最近完整last为164000（epoch12/batch9428），473项Adam状态step一致，固定paused_for_slot_embedding_0164000.pt；不将新参数插入旧训练续跑。GPU3独立从预训练/原公共初始化重新训练20轮257620更新，配置results/six_gpu_exploration_20261111/gam_slots.yaml，输出uav_gam_slot_unified。相对旧配置只变输出目录、gam_action_slot_position_encoding=True、gam_action_slot_init_seed=48。采用原生Predictor action_slot_embed相加机制，[1,1,5,1024]独立可学习向量，Normal(0,.02)，隔离CPU RNG保持原有公共参数初值，归Predictor LR1e-5。原Action/Pose类型embedding、SA proprio、Pose MLP及适配、冻结浅层、文本位置表、最终动作头、loss、batch24×1、原固定episode划分和完整epoch/last/best状态保存均不变。五槽差异、既有参数初值一致及梯度回归11项通过。Warp/VLA+Current原进程和配置不变；磁盘应急角色切到新输出，自动关机仍关闭。旧日志保留，live-six第一列切到新输出，不拼接旧曲线。


## 2026-11-13：本机三条训练预算统一缩至15轮

用户要求不再训练20轮。修正版Unified GAM、改进Non-unified Warp和Unified VLA+Current目标均改为15轮/193215更新；三个原配置文件仅training.max_epochs从20改15，原配置快照保存在results/six_gpu_exploration_20261111/finish_at15。当前运行中的训练器已将20轮预算读入内存，配置编辑不会热更新，因此未重启模型或优化器，而由finish_at15控制器在每个模型完成193215步全量3095窗口验证、epoch15完整状态及last/best检查后独立结束该进程组。运行中的checkpoint config仍如实记载启动时20轮，最终预算以requested_max_epochs=15及endpoint审计为准；constant LR不受终点预算变化影响。三条训练原PID/数据游标/优化器不动，现有CPU闭环队列不动，不自动关机。停止条件（缺全量验证/缺optimizer/错误游标拒绝停止）3项回归通过。


## 2026-11-13：用户加卡前就近完整保存并暂停

通过已有全局磁盘暂停钩子请求三条训练在最近完整optimizer更新边界保存并退出，未等待1000步也未强杀训练。实际GAM57359(epoch4/batch5835,474项Adam)、Warp140545(epoch10/batch11735,931项Adam)、Current176396(epoch13/batch8943,912项Adam)。先写预分配应急完整状态，验证每项optimizer step和数据游标，再迁至数据盘emergency_recovered_*.pt，迁移前后SHA256一致且last同inode。checkpoint原格式无独立RNG字段，不声称不停机逐位一致。三模型目标仍总15轮，已暂停归档；已完成闭环结果保留。复核续跑入口scripts/resume_active_three_20261113.sh（默认原GPU3单卡GAM、GPU0/1/2 Warp、GPU4/5 Current；各自world/batch不变，额外卡待用户另行分配）。只执行--check-only验收，没有启动续训或关机；无自动关机。保留共用旧checkpoint_write.lock，使训练保存与归档最终删除互斥。

2026-11-14 用户要求改进Non-unified Warp15完成后，以释放的三卡排队Non-unified VLA+Current及该Warp last/best闭环评测。队列 experiments/uavflow_paper_ablation/queue_current_raw_after_warp15.py，状态 results/six_gpu_exploration_20261111/after_raw_warp15_20261114。GPU0/1从预训练及原Unified Current legacy公共初始化从头训15轮，micro12×world2×accum1=24；相对current.yaml仅新增原始指令map/SHA并改变输出目录，其他配置逐字段断言一致，无Stop。GPU2顺序加载最终last、best，每模型每任务配对1步/5步，全273条，原执行器与原客户端hash不变。等待源Warp进程正常完成、last193215/epoch15/完整optimizer并释放GPU，才固定last/best硬链接并启动；不会提前抢卡。新Current逐轮/last/best完整保存；checkpoint归档保持用户暂停。独立队列状态不写入现有八卡控制器内存状态；live新增第7列Non-unified Current，Preview新增四份待评测标签。五项原协议/配对/恢复回归通过。自动关机关闭。

2026-11-14 02:18 北京时间修复上述自动交接条件：源Warp最后一步保存游标为epoch14/batch12881（step193215，931项optimizer全部193215），不是epoch15/batch0。原队列误判而退出，造成约11分钟空闲；未丢训练。现接受两个等价的完整epoch边界游标，拒绝不完整/不一致游标，两项CPU回归通过。复用已生成配置/Preview计划恢复队列，GPU0/1启动Non-unified Current、GPU2加载Warp15 last进行原配对评测，训练与评测各独立进程。旧traceback保留在launcher日志，后有明确RESTART记录。

2026-11-14 用户要求15轮Pure对照：Non-unified Pure验证日志限制step<=193215后，最佳all-window normalized L1对应epoch14/180334（0.03153882069522229），last取epoch15/193215，均完整optimizer592项；旧219000last/206096best超过预算不能用。queue_raw_pure15_after_improved_warp注册四份全273条配对1/5评测。用户要求检查显存并行，GPU2实测Warp+UE13862MiB，剩10222MiB，因此启用Pure同卡并行容量检查：加载前至少9500MiB，Pure服务ready后留至少3500MiB启动其独立UE；原协议4项回归通过。服务6791/6792，UE9251/9252；checkpoint保留原路径无需复制，无修改训练。自动标注watch已运行，每30秒添加阈值类别且保留人工标注。

2026-11-14 用户要求补核对本机真实Final动作头初始化。以当前warp.yaml/current.yaml/gam_slots.yaml在CPU执行训练源码原构建段（含DA3/LoRA/Text构建、model_init_seed复位），重现原公共Proposal/Pose/projection启动校验后，直接逐张量比较parallel_action_correction_head.decoder：Warp/Current/修正版GAM的Final头均为ba23f82b9ccb97c709ecc734b33b6943e0a5449f16a2d3addc8d40bff03c0564，完全相同；三个Proposal兼容头均为5231946a58a04dd09cb487c3894e804cb82b4d3f22004cb42508e57dc1855ae4，与Pure实际common_init head相同。因此Pure=A、Warp Proposal=A、Warp/Current/GAM Final=B，A!=B。旧GAM verify_uav_initialization只检查未使用Proposal兼容头的审计缺口属实，但本次重建未发现Final初值不一致。结果及可复现脚本 results/local_final_head_initialization_audit_20261114/{report.json,replay.py}；未存新checkpoint、未使用CUDA、不改运行中训练。范围：按本机当前实际构建流程和seed重建，启动公共哈希吻合；没有历史Final0步独立快照，不能追溯声称历史GPU逐位测量。

2026-11-14 用户追加Pure/Warp第10轮全量闭环。Non-unified Pure epoch10/128810（results/pure24_raw20_nounfix_20261107/pure/epoch_10.pt）排在Pure前15轮best后；改进Non-unified Warp epoch10/128810复用已保留attention目录权重，排在该Warp15 best后。独立队列 queue_raw_pure10、queue_raw_warp10，GPU2共享但各等前序完整成功标记及加载前9500MiB可用/加载后3500MiB UE余量，服务6793/6794、UE9253/9254。均原始instruction，全273任务配对1/5，无Stop无额外执行校正。Preview新增四标签，既有阈值自动标注watch覆盖新增标签。训练未修改。

2026-11-15 00:00 北京时间用户要求立即切换当前评测重测改进Non-unified Warp15 last的Approach2025-05-06_19-15-42，执行1步，测完切回。独立控制器 interrupt_warp10_rerun15_one 已仅暂停Warp10评测，保留所有有效完成结果并隔离中断的未完成轨迹；Pure并行评测/各训练不动。新权重为固定193215步last，GPU2服务6785/UE9260，原执行器单模式execute_modes=(1,)（默认配对1/5不变，4项原协议测试通过）。完整成功后备份该单条旧result/obs并覆盖原Warp15 last step1结果目录，再在finally自动切回原Warp10checkpoint/6794/9254并resume_existing跳过有效任务。记录 results/warp15_last_approach191542_rerun_20261114/。

2026-11-15 用户要求本机修正版Unified GAM15完成后last/best均评测1/5且一起安排。已启动队列 queue_unified_gam15_best_last，results/unified_gam15_best_last_eval_20261115。等待原GAM GPU3完整15轮/193215步、等价epoch游标和完整optimizer并释放显存后，固定last/best硬链接且不改变原checkpoint，单GPU3安排两个独立GAM模型/UE：last先启动，第一条完成并确认剩余>=10000MiB后启动best；每模型ready后额外检查>=3500MiB给UE，不满足则best等待last结束再评。全273条，Unified指令，原执行器每任务1/5独立reset，服务6801/6802，UE9271/9272。Preview新增四标签，现有autolabel覆盖阈值类。其他训练与GPU2评测不改，自动关机关闭，归档仍暂停。

2026-11-15 Pure5队列修复：此前从Pure10脚本派生时误将Preview目录名best_vs_epoch10改为不存在的best_vs_epoch5，初始化注册阶段退出；旧waiting状态并不代表进程存活。用户询问启动后查明，恢复真实Preview目录、保留checkpoint64405与1/5协议，重启PID623794并确认服务开始loading_pure_once。Pure10已完整成功，因此无需再等前序，GPU2现有Warp评测保留。后续检查队列必须同时看进程和launcher traceback，不能只读状态文件。

2026-11-15 用户要求5轮Pure和改进Warp仅执行5步，全273条。仅停止Pure5原配对controller/client/policy并保留已生成文件，修改其plan/Preview为单模式(5,)，复用worker恢复启动；Pure5新PID624587已进入闭环，原step1入口移除且文件不删。新增改进Non-unified Warp5队列queue_raw_warp5_single5，restore ckpt0064405从原远端归档校验SHA256和大小后保留副本，远端原文件不删且archive_worker仍暂停。由于GPU2两个Warp模型无法同时容纳，Warp5等待Pure5与Warp10评测均成功并释放GPU2再启动，服务6796/UE9256。原配方、模型权重、执行协议不变，仅评测模式5。Preview和自动阈值标注接入。

2026-11-15 用户指出Pure5和Warp5可并行，查GPU2仅Pure5约8154MiB、剩15929MiB；原Warp5等待Pure5全完成条件过严。仅重启Warp5等待controller，不动Pure5/其他评测训练；复用已校验本地64405权重不再下载，依赖仅保留已成功Warp10，加载前留>=12000MiB，ready后留>=3500MiB供UE。Warp5服务6796/UE9256、仅5步273条配置不变，独立队列已进入加载。

2026-11-15 用户明确要求Pure5单任务Approach2025-05-06_19-15-42重测并抢先。此前结果已存在，执行25步small_action；按要求备份至Pure5根目录forced_rerun_191542_20261115（原result/obs及旧视频缓存），不删除。新控制器rerun_pure5_approach191542仅替换Pure5 client/UE，保留已经加载的64405 policy PID624596与GPU2数值路径，优先任务放task_ids首项，只执行5步，全局剩余已完成轨迹原地跳过；该条后继续剩余任务。run_warp15_paired新增可选existing_policy_pid，默认行为保持原状，4项原执行协议测试通过。实际服务ready step64405已核对，client已启动。其他模型/训练不变。

2026-11-15 用户指定重测Unified修正版GAM前15轮best（170000步）Land2025-05-07_12-05-23执行5步。rerun_gam_best_land120523仅接管best评测worker/client，保留已加载GPU3 policy507151与服务6802，last评测不动。优先该任务，原step1结果已完整所以自动跳过，仅清空备份其step5 result/obs/视频缓存至best/forced_rerun_land120523_20261115；原协议两模式继续，完成该条后恢复剩余273任务，完整结果跳过。实际served step170000核对通过，新client762748已进入paired_closedloop。其余训练和评测保持原状。

2026-11-15 用户要求Warp5剩余任务对半并行，评完GPU2/3续训未完成Unified Current。队列split_warp5_then_resume_current等待GAM last/best都完整结束并释放GPU3；交接时重新枚举Warp5完整result+termination，只分当时未完成任务，两卡各一半，保留全部旧完整结果。GPU2保留现有Warp5 policy只重开客户端，GPU3新policy，两个worker的完整结果以硬链接合并回原Preview根目录，所有273条验证完整后释放两卡。随后原Current177000完整checkpoint在GPU2/3恢复到193215/15轮，保持world2、micro12、accum1及原current.yaml，断点游标/912项optimizer均预检，启用optimizer恢复逐张量核对。独立状态results/split_warp5_resume_current_20261115/queue_state.json；实际恢复时写RUN/current_external_resume.json，旧八卡state内原Current暂停项不作为新进程状态。不会改其他训练/模型，归档保持暂停、自动关机关闭。


## 2026-11-16：评测完成后关机，五卡原状态续跑

Non-unified GAM best/last均完成273条五步闭环后，按用户要求同步保存并关机。Unified Warp完整状态为190897步（epoch14/batch10563，931项Adam），Non-unified Current为167929步（epoch13/batch476，912项Adam）；校验与SHA在`results/shutdown_after_gam_eval_20261116/resume_plan.json`。五卡续跑只改变物理GPU映射：Current用2/3（同PHB），Warp用0/1/4；world分别2/3、每卡micro12/8、累积1、global24、配置与游标保持原样，目标15轮193215步。恢复审计在`results/resume_five_gpu_20261116/preflight.json`，启用逐张量优化器恢复检查；不恢复自动归档、不新增训练或自动关机。新实例GPU4实际在NUMA1，因此训练入口改为查询PCI sysfs的NUMA/CPU列表，不再按GPU编号<=4判断；这是CPU亲和性修正，不改模型或损失。原状态无独立RNG字段，不宣称与不停机逐位一致。
