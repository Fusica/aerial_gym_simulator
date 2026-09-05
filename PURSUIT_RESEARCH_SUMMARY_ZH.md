# UAV 追逐可观测性风险研究：核心思路与已完成实验汇总

> 汇总日期：2026-08-27
> 来源：`.planning/` 下现有 8 个 Markdown 文件的完整内容，以及汇总时对当前工作区中关键运行产物的只读核查。
> 用途：在删除 `.planning/` 前保留当前研究主线、已经获得的实验事实、证据边界和后续重新研判创新所需的最小上下文。

## 1. 阅读与证据边界

本文严格区分三类状态：

1. **完成的训练或 rollout 实验**：已经运行并产生过数值结果。
2. **完成的工程验证**：代码、静态检查、小规模 smoke 或数据一致性审计已经完成，但不等于方法有效性得到验证。
3. **尚未完成**：只有方法设计、代码入口或采集过程，不能写成论文实验结论。

当前证据保存情况：

- `runs/PE_20260520_110828/`、其策略池、`upd_1300` checkpoint 和 `manifest.jsonl` 当前仍在，可直接复核。
- E0-C1、E0-C2、E0-C3、中间 visibility runs、schema7 500-episode 数据集和早期 branch pilot 的原目录当前已不在工作区；下文相应数值来自 planning 归档，只能视为**历史实验记录**，不能假装为当前已重新复核的原始数据。
- 当前 100k-anchor D0 branch 数据集仍未完成；汇总时 `branch_index.jsonl` 有 `23,661` 行，不能当作冻结数据集或最终实验结果。
- 之前所有策略训练均基于 Aerial Gym 2.0.0 时代的旧 RK4 电机积分实现。迁移到 2.0.1 后电机响应会改变，因此旧训练结果只适合作为设计依据和历史对照，正式论文数据必须重新训练和验证。

## 2. 当前核心研究问题

### 2.1 一句话主线

`特权全状态追逐 teacher -> LiDAR/range-image 降信息输入 -> 动作条件短期可观测性风险 -> 风险调制 PPO`

研究问题是：

> 能否利用特权全状态 rollout 学习候选动作导致目标在短期内脱离 LiDAR 可观测区域的风险，并在只保留自身状态和 LiDAR 输入的条件下，用该风险调制 PPO，从而恢复高速 UAV 追逐的决策质量和目标保持能力？

### 2.2 可辩护的创新范围

当前可辩护的窄主张是：

> 面向 LiDAR/range-image reduced-information UAV pursuit 的 action-conditioned observability-loss risk modulation of PPO。

创新不在“使用 PPO”“使用 LiDAR”“有限视场追逐”或“一般安全强化学习”，而在以下组合：

1. 在线策略输入删除目标和环境真值，只允许自身状态、LiDAR stream 和由这些信息预测的风险。
2. 风险语义不是碰撞风险或目标轨迹预测，而是**执行某个动作后短期内持续丢失目标可观测性**的概率、严重度、首次发生时间和恢复性。
3. 同一状态下比较多个候选动作的风险，用相对风险修正 PPO 学习信号，而不是简单把风险拼进观测或在执行后做动作过滤。

不能声称：

- 首个 partial-observation pursuit、首个 limited-FOV pursuit 或首个 risk-aware PPO。
- 完整机载视觉追逐、真实 500m LiDAR 部署能力或真实 100Hz 全帧 LiDAR 能力。
- 当前 APF/x500 目标是高保真电机级无人机。它只是 PhysX 刚体加高层位置控制、直接 force/torque 注入的规则逃逸目标。
- 仿真 semantic ID、目标 mask、真值 target hit、真值 bearing/range/closing speed 是在线可用输入。

## 3. 任务、传感器与真实性约束

### 3.1 追逐任务

- 追逐者：PPO 输出归一化 CTBR 动作 `[thrust, p_rate, q_rate, r_rate]`。
- 执行链：`PPO -> thrust_bodyrate_control -> control allocation -> motor model -> Isaac Gym`。
- 逃逸者：以 `apf_escape`/规则轨迹生成目标命令，再由 task 内控制器直接写 force/torque；没有 allocator、ESC、propeller delay 或单电机饱和。
- 当前成功语义：进入 `3m`、满足前向对准且终点目标可检测，即 `3m visible strike`。
- 中途短暂不可见不立即终止；需要保留丢失后恢复轨迹。长失锁通过累计 penalty、最长失锁和 horizon 内恢复率暴露。

### 3.2 LiDAR 主设定

- 默认目标：`target_x500`；`target_quad` 仅作显式对照。
- 默认传感器：M3-like 前向高分辨率 LiDAR，`501 x 2401`、约 `0.05° x 0.05°`、水平 `120°`、垂直 `25°`。
- 主实验只把 100/150/200m 看作**极稀疏返回压力测试**。静态 x500 smoke 在这些距离各只有约 2 个目标返回点。
- OS2-64 和 AT128P-like 已证明对当前小目标远距像素不足，只保留为失败对照。
- Warp LiDAR capture 前必须同步/refit 动态 mesh，否则状态、几何标签和 raycast 图像不在同一帧。

### 3.3 Anti-oracle 规则

风险模型和在线 PPO 可以使用：

- 自身线速度、角速度和姿态；
- 前一动作和当前候选/执行动作；
- LiDAR range image/point features；
- 从上述输入学习得到的风险输出。

只能用于离线标签和 QA、不能进入在线模型输入的字段包括：

- target state、relative target truth、墙体/障碍物真值；
- semantic target ID、target pixel mask/bbox；
- privileged geometry detectable；
- reward、terminal truth 和任何真值派生的 bearing/range/closing speed。

## 4. B4 v0 方法契约

### 4.1 风险模型输入

- `s_red = [body_linvel(3), body_angvel(3), rotation_matrix(9), prev_action(4)]`，共 19 维。
- `z_lidar = 64`，来自 `K=3` 帧 LiDAR stack 的 CNN/CNN+GRU 编码。
- 动作特征为 `u`、`u-prev_action`、`abs(u)`、`u^2`。
- `policy_mean/std` 只用于候选生成、OOD 和诊断，不作为风险头的动作 reference。

### 4.2 风险监督与输出

正式标签以 sensor-space semantic target pixels 生成：

- `target_visible := target_pixel_count >= 3`；
- semantic 只生成离线标签和 QA，不进入在线模型输入；
- privileged geometry detectable 只保留为对齐审计字段。

风险头输出：

- `p_loss_50`、`severity_50`；
- `first_loss_50`；
- `p_loss_150`、`severity_150`；
- `p_recover_150`。

PPO v0 使用的标量风险为：

```text
rho(o,u) = 0.40 * p_loss_50
         + 0.35 * severity_50
         + 0.15 * p_loss_150
         + 0.10 * severity_150
```

### 4.3 风险调制 PPO

同一状态生成多个候选动作并预测风险：

```text
rho_bar(o)  = mean_i rho(o, u_i)
delta_risk  = rho(o, u_exec) - rho_bar(o)
A_tilde     = A_task - lambda * clip(norm(delta_risk), -3, 3)
```

- 第一轮 `lambda` 网格：`{0, 0.02, 0.05, 0.1, 0.2}`，设计起点为 `0.05`。
- 环境实际执行动作仍与 PPO logprob/ratio 对应，避免 credit assignment 不一致。
- actor 第一版不重塑为 `pi(u|o) exp(-beta rho)`；critic 第一版仍只估计 `V_task`。
- 风险模型先离线训练并冻结，不与 PPO 同步在线更新。
- 数据闭环设计为：`D0 offline pool -> R0 frozen risk -> B4 PPO -> D1 aggregation -> R1 offline retrain`。

离线 branch 默认 `M_train=16`，覆盖 policy/local sample 和 roll/pitch/yaw/thrust 单轴正负扰动；在线 PPO risk baseline 默认 `M_ppo=8`。主方法用 mean baseline，minimum baseline 只做消融，防止退化成保守 shield。

## 5. 必需比较链与创新判定条件

| ID | 方法 | 作用 |
|---|---|---|
| B0 | visibility-aware 32D 全状态 PPO | 上界参考、teacher、数据来源，不是本文方法 |
| B1 | reduced-state/LiDAR PPO，无风险 | 降信息退化基线 |
| B2 | reduced-state + LiDAR-honest heuristic risk | 手工风险基线 |
| B3 | reduced-state + learned state risk `p_lost(o)` | 检验学习型 bridge 本身 |
| B4 | reduced-state + action-conditioned risk-modulated PPO | 主方法 |

B4 至少需要对比：

- risk concat only；
- state risk 与 action-conditioned risk；
- shuffled-action risk；
- PPO-Lagrangian using the same risk as cost；
- shield/filter using the same risk；
- privileged critic；
- mean baseline 与 minimum baseline；
- fixed lambda 与 adaptive dual lambda；
- 有无 DAgger-like aggregation。

只有当 B4 在保持追击/到达能力的同时，显著优于 B3、risk concat、PPO-Lagrangian 和 shield/filter，并降低 persistent/over-horizon loss，才能支撑当前创新主张。仅靠减速、不追击或更保守地保持可见不算成功。

## 6. 已完成的训练实验

### 6.1 控制器动作范围实验：E0-C1 与 E0-C2

两组均为 1.2B 步、512 env、全状态 PPO、无课程学习，其他 PPO 超参一致。

| 指标 | E0-C1：激进/宽限制 | E0-C2：默认/窄限制 |
|---|---:|---:|
| 最终成功率 | `0.0000` | `0.0097` |
| 峰值成功率 | `0.001953`，孤立峰值 | `0.023256` |
| 最小距离 | `2.40m` | `1.71m` |
| 1m 到达率 | `0` | `0.033` |
| 3m 到达率 | `0.887` | `0.996` |
| 最终熵 | `-4.28` | `-1.71` |
| pre-tanh OOB | `61.6%` | `48.5%` |

历史结论：更宽的动作范围导致更大的搜索空间、更早的熵塌陷和更严重的 tanh/梯度问题；默认较窄控制器表现更好。控制器限制对 RL 同时承担隐式正则化作用，不能简单按“更激进更强”设计。

证据状态：原 run 目录当前缺失，数值仅由 planning 归档保留；升级到 2.0.1 RK4 后必须重做。

### 6.2 手动与自动课程学习：E0-C3

对比条件：相同 robot、controller、PPO 参数和 `5m -> 3m -> 1m` 阈值，总训练量约 1.2B。

- 手动分三次、每次 `total_timesteps=400M` 并 resume：最终成功率 `77.2%`。
- 单次自动课程、`total_timesteps=1.2B`、全局 entropy 衰减：最终成功率 `100%`。

历史结论：主要差异不是 robot 或 controller，而是每阶段重启会把 entropy 重新拉回 `0.01`；高精度 1m 阶段重新进入高探索噪声，导致成功率骤降。课程学习必须让 entropy/学习率等 schedule 感知全局训练步长。

证据状态：原 run 目录当前缺失，数值仅由 planning 归档保留；新 RK4 下需要重做。

### 6.3 Visibility-aware teacher 训练系列

1. `PE_20260514_221640`：visibility reward 从第一阶段启用，课程仍按固定步数切换；两次切换时成功率均为 0，最终成功率仍为 0。该实验否决了“未学会 capture 就强行加入 visibility 并按时切难度”的做法。
2. `PE_20260518_133223`：三个 visibility 阶段 `0.10/0.20/0.30` 的成功率约 `0.998-0.9995`、终端可见率接近 1，但平均累计不可见步数仍约 `682/621/498`。说明终端可见成功不等于过程持续可见。
3. `PE_20260519_103518`：继续提高到 `0.20/0.40/0.60` 后收益递减，最终仍约 `340+` 累计不可见步数和 `300+` 最长连续不可见步数。根因是旧 penalty 在 `K=10` 后饱和。
4. `PE_20260520_000809`：150-step recovery-aware penalty 将最长连续不可见降低到约 `168`，但累计不可见仍约 `403`，horizon 内恢复率约 `0.77`，且长期卡在低权重 stage。说明 recovery 形状改善了长失锁，但仍需平衡总不可见压力和 curriculum gate。

上述四个 run 当前原目录均缺失，属于 planning 保留的历史实验结果。

### 6.4 当前可复核 B0：`PE_20260520_110828/upd_1300`

- 训练量：`2,396,160,000` global steps。
- 最终课程：`stage_idx=3`，`3m + vis0.40`，未进入最终 `vis0.60`。
- 冻结 checkpoint：`policy_pool/ppo_upd_001300_step_2396160000.pth`。

最后 10 个保存点的 planning 汇总：

| 指标 | 均值 ± SD |
|---|---:|
| 成功率 | `0.996887 ± 0.002338` |
| 3m 到达率 | `0.999265 ± 0.000601` |
| 终端可见率 | `0.998244 ± 0.002047` |
| 平均最终距离 | `2.964 ± 0.027m` |
| 累计不可见步数 | `264.591 ± 6.793` |
| 最长连续不可见 | `120.250 ± 1.685` |
| 超过 H=150 的不可见尾部步数 | `7.083 ± 1.112` |
| H=150 内恢复率 | `0.901056 ± 0.008220` |

结论：这是一个强追踪、终端可见、同时仍包含非平凡丢失/恢复样本的 teacher，适合作为旧动力学下的 B0 和数据来源；它不是“全程可见”的理想上界。

证据状态：当前 checkpoint、manifest 和 TensorBoard 文件仍在，可直接复核。但由于 2.0.1 RK4 修正，不能直接作为升级后最终 B0。

## 7. 已完成的传感器与目标实验

### 7.1 静态 LiDAR 返回点实验

| 组合 | 关键结果 | 结论 |
|---|---|---|
| OS2-64 + target quad | 1m=`408`、5m=`8`、10m=`4`，25m 起为 `0`；ray-aligned 也未改善 25m+ | `64 x 512` 角分辨率不足 |
| AT128P-like + target quad | 25m=`4`，50m 起为 `0`；ray-aligned 50m=`2`，100m+ 为 `0` | 不适合作为主配置 |
| M3-like + target quad | 50m=`8`、100m=`2`、150m=`2`、200m=`0` | 仅极稀疏远距可见 |
| AT128P-like + target x500 | 25m=`6`，50m+ 为 `0` | 更大目标仍不足 |
| M3-like + target x500 | 50m=`12`、100/150/200m 均=`2` | 选为主线，但只能称 sparse-return stress test |

这些 smoke 曾实际运行，但输出主要位于 `/tmp`，当前原始目录已不存在；数值由 planning 归档保留。

### 7.2 x500 目标替换 smoke

同一 seed、同一旧策略的一环境回放：

- `target_quad`：966 steps，最终距离约 `0.999m`，success。
- `target_x500`：964 steps，最终距离约 `0.994m`，success。

该 smoke 说明替换 x500 URDF 并按 `1.656kg` 修正 target controller mass 后没有立即导致动力学发散；它只有一个 seed，不能作为多场景性能或高保真目标动力学证据。

## 8. 已完成的数据、图像与 branch 审计

### 8.1 Warp 动态 mesh 同步缺陷

已确认旧 Warp LiDAR 只在 reset 时 refit mesh，动态 target 移动后，几何状态使用当前 pose，而 raycast/semantic 仍使用旧 mesh。修复是在 sensor capture 前同步/refit Warp mesh。

修复前的 semantic pixel 统计不得用于论文或 reward/curriculum 结论。修复后对同一组 checkpoint 的诊断曾得到：

- `visible | geometry = 1.0`；
- `geometry detectable && target_pixel_count=0` 为 0；
- terminal target pixels 恢复到数千至数万级。

### 8.2 单 episode semantic 审计

冻结 B0 `upd_1300` 曾完成 943 帧完整 episode 审计：

- 从约 69m 追近到 terminal success；
- semantic-visible 和 geometry-detectable 均为 617 帧；
- 两类错配均为 0；
- range、target mask、overlay 和 target pixel artifact 逐帧齐全；
- `1 <= target_pixel_count < 3` 的帧数为 0。

结论：在该 episode 中，`>=3 pixels` 标签阈值没有丢掉真实可见帧，且 sensor/state 同帧对齐成立。原 audit 目录当前缺失，数值由 planning 归档保留。

### 8.3 schema7 500-episode 数据集一致性审计

已完成过的全量审计规模：

- 500 episodes；
- 730,020 个非终止 step；
- 45,834 个 LiDAR chunks；
- 重算 body-frame 相对位置最大误差约 `1.14e-4m`；
- 重算 geometry detectable 与存储值错配 0；
- chunk step 序列错配 0；
- `visible_given_detectable = 0.999076`；
- 30m 内 hard mismatch：`0 / 184,073`；
- visible/detectable 帧投影中心到 semantic centroid 的 p95 约 `1.50px`。

`visible && !detectable` 主要来自目标中心略出 FOV、mesh 边缘仍被 ray 命中；`detectable && !visible` 主要来自远距返回少于 3 像素。这是中心点几何与真实 mesh/raycast 标签的定义差异，不是简单帧错位。

证据状态：原 schema7 数据集当前缺失，结果只由 planning 归档保留。

### 8.4 Same-anchor candidate-action branch pilots

第一版 `upd1300_h150`：

- 16 anchors，每个 8 candidates；
- 14/16 anchors 在 severity 或 first-loss 上有分离；
- binary loss 分离仅 1/16；
- 说明只训练 binary BCE 会浪费大量动作条件信号，应保留 severity、offset 和 ranking。

三 teacher risk-edge pilot：

- updates `220/610/1300`，共 48 anchors，每个 12 candidates；
- 任意标签分离：`36/48 = 75%`；
- binary loss 分离：`14/48 = 29.17%`；
- 分 teacher binary：`upd220=6/16`、`upd610=8/16`、`upd1300=0/16`；
- candidate action 无越界和 exact duplicate，最小 pairwise L2 至少约 `0.08125`；
- 轻量 trace 无 NaN/Inf、无 reset-like 跳变。

结论：弱/中 teacher 与边界 anchor 更容易提供二值翻转；强 teacher 更适合提供 severity/first-loss 差异。正式 D0 不能只采最终 B0。

证据状态：早期 branch pilot 原目录当前缺失，数值由 planning 归档保留。

## 9. 已完成的工程工作，但不是方法效果实验

以下工作可以作为后续重建的工程基础，不能写成 B4 已有效：

- policy pool 中间 checkpoint 与 manifest 保存。
- schema7 behavior/branch 数据契约、K=3 LiDAR stack、多 horizon 标签和 anti-leakage 字段声明。
- CNN+GRU action-conditioned risk model、behavior pretrain/branch finetune 和 pairwise ranking 的小样本 CPU smoke。
- 独立 semantic audit 路径和 formal exporter 分离。
- branch exporter 优化：GPU target pixel/bbox 统计、多个 anchor group 并行、candidate action 批量采样、GPU LiDAR ring buffer、action tensor 复用。
- 上述 exporter 优化只通过 `py_compile` 和 `git diff --check`；当时系统 Python 缺少 Isaac Gym/Torch，没有完成真实 GPU 速度基准。因此不能声称已经获得具体加速倍数。
- 100k anchors 未压缩 LiDAR 估算约 `721.74GB`（约 `672.08GiB`），当时可用空间约 `121.6GiB`；因此保留 `np.savez_compressed` 是合理工程决策。

## 10. 当前尚未完成，不能作为结论

- D0 100k-anchor 数据集尚未采集完成、冻结和完成最终 QA。
- 风险模型尚无完整数据训练后的 AUROC、AUPRC、Brier、ECE、false-safe rate、severity/time/recovery 和 heldout checkpoint/source 泛化结果。
- B1、B2、B3、B4 尚未完成统一训练和对比。
- B4 的 lambda 网格、adaptive dual lambda、DAgger aggregation 和必需消融尚未完成。
- H/FOV、LiDAR 噪声/延迟/角分辨率/反射率、场景和逃逸者泛化尚未完成。
- 没有真实 LiDAR 日志、真实飞行、电机 SysID 或 sim-to-real 实验。
- 当前旧 B0 和所有历史训练都没有在 Aerial Gym 2.0.1 修正后的 RK4 动力学下复现。

## 11. 升级到 Aerial Gym 2.0.1 后的重新研判顺序

> 2026-08-27 状态：官方 2.0.1 代码与修正后的 RK4 已合入当前工作区，
> pursuit teacher 显式锁定 RK4，固定房间/目标已关闭新版 15% 随机减障。
> force/RPM step-response 数值回归、真实 GPU controller smoke、X500 1--200m
> Warp LiDAR static smoke，以及 1 个完整 rollout/1 次 PPO update 均已通过。
> 这些只证明升级后的运行链可用；真实电机 SysID、动态 LiDAR 同帧回归和正式
> B0 重训/独立评估仍未完成，不能据此报告新版成功率。

1. **先验证物理底座**：对新旧 RK4 做电机 step response；用真实或可信电机数据标定 thrust constant、升降时间常数、delay 和饱和，而不是仅相信“数值积分已修正”。
2. **重新验证控制器动作范围**：至少复现默认窄限制与激进宽限制的关键差异，确认旧结论不是旧 RK4 的副产物。
3. **重新训练 B0**：在新动力学下冻结新的 visibility-aware full-state teacher，并重新记录 success、reach、collision、invisible、max-loss、over-horizon 和 recovery。
4. **重新跑 LiDAR 动态同步回归**：确认 2.0.1 合并后 Warp mesh sync、sensor/state/semantic 同帧关系仍成立。
5. **完成并冻结 D0**：以多 checkpoint、多难度、risk-edge/hard-case anchor 保证 binary 与 severity/ranking 覆盖。
6. **先证明风险可学**：完成 state-risk baseline 和 action-conditioned risk model 的校准、false-safe 与 matched-state ranking。
7. **再训练 B1-B4**：保持统一动力学、课程、seed、训练预算和评估协议。
8. **最后判断创新**：若 B4 不能显著优于 B3、risk concat、PPO-Lagrangian 和 shield/filter，或只是通过减速换可见性，应放弃或重写当前核心主张。
9. **sim-to-real 门槛**：加入真实电机/执行器标定、传感器噪声与延迟、真实 LiDAR 稀疏返回和至少一组硬件或真实日志验证后，才讨论 sim-to-real 有效性。

## 12. 当前仍可定位的关键证据

- 旧 B0 run：`runs/PE_20260520_110828/`
- 旧 B0 policy pool manifest：`runs/PE_20260520_110828/policy_pool/manifest.jsonl`
- 旧 B0 checkpoint：`runs/PE_20260520_110828/policy_pool/ppo_upd_001300_step_2396160000.pth`
- 当前未完成 D0 index：`runs/risk_dataset/D0_branch_full_v1_10ckpt_100kanchor_h150_k3_m16/branch_index.jsonl`
- 当前 exporter：`aerial_gym/rl_training/cleanrl/pursuit_risk/export_lidar_rollouts.py`
- 当前风险训练模块：`aerial_gym/rl_training/cleanrl/pursuit_risk/`
- 当前追逐 PPO：`aerial_gym/rl_training/cleanrl/ppo_guidance.py`

删除 `.planning/` 后，应把本文件作为历史设计入口；但本文件不能替代已经丢失的 raw run、TensorBoard、dataset 和 QA artifact。
