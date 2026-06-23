# 任务计划：EAAI 强投稿版可观测性风险追逐

## 目标
围绕当前 `pursuit_guidance_task` 分支形成一条单一、强约束的 EAAI 投稿路线：

`特权全状态追逐 -> LiDAR/range-image 传感器代理降信息追逐 -> 动作条件短期可观测性风险预测 -> 风险调制 PPO`

## 活动计划
- **计划 ID：** `2026-05-09-perceptual-risk-pursuit-eaai`
- **当前阶段：** 第 6.2 阶段——策略池构建与风险数据采集
- **总体状态：** in_progress
- **投稿规则：** 所有方法、实验和写作都必须服务于下面锁定的 EAAI 声明。
- **当前写作模板规则：** 论文草稿先使用 `paper/main.tex` 中的 `IEEEtran` / IEEE Transactions 双栏模板撰写和编译；当前阶段不再寻找、安装或切换 `elsarticle.cls`。EAAI/Elsevier 官方模板只在投稿前最终格式化阶段再处理。
- **当前有效 deep-research：** `deep-research-output/perceptual-risk-modulated-ppo-eaai-refresh/`。

## EAAI 主线声明
通过特权全状态 rollout 学习动作条件短期可观测性风险，并用该风险调制 PPO 的动作分布与优势估计，可以让 UAV 追逐者在 LiDAR/range-image 传感器代理降信息条件下恢复高速、大空间追逐决策质量。

> 2026-06-01 定型：B4 第一版采用 **counterfactual risk-adjusted advantage PPO**。风险模型离线训练并冻结，PPO actor 输出分布和 critic 主 value 第一版不改；动作条件风险只通过 mean-baseline 的 advantage 修正进入 PPO actor loss。actor latent/mean/variance modulation、risk value critic、policy distribution reweighting 只作为后续 ablation，不作为 B4 v0 主方法。

## 不可妥协的贡献
1. **LiDAR/range-image 传感器代理降信息追逐定义**
   - 主策略输入为 `self-state + z_lidar + p_lost`。
   - 主线策略输入中必须移除直接目标状态、真实值衍生 bearing/range/closing-speed、墙体真实值和障碍物真实值。
   - 在线输入只能来自自身状态、LiDAR range-image/point features 和由其学习得到的风险量；不得使用目标 ID、真值目标命中标记或仿真语义真值。
   - 全状态 PPO 只作为上界参考，不是本文方法。

2. **动作条件短期可观测性风险模型**
   - 主标签是 `H = 150`、`K_persist = 10` 下的目标丢失概率。
   - 主预测器必须支持动作条件风险评分：`R_obs(s_red, u)` 或等价的候选动作风险评估。
   - 特权状态只允许用于离线标签和辅助监督，不允许在策略推理时使用。

3. **动作条件风险调制 PPO**
   - PPO 仍直接输出归一化 CTBR 动作 `u_rl = [c, p, q, r]`，并由该动作进入环境执行链路。
   - B4 v0 使用 `R_obs(s_red, z_lidar, u)` 形成 `rho(o,u)`，并用同状态候选动作风险的 mean baseline 修正 PPO advantage。
   - 风险模型不是简单 observation concat，也不是 PPO 后处理；它应作为结构化调制信号进入策略学习。
   - 论文 1 主方法不采用连续动作策略分布重塑 `pi(u|o) exp(-beta rho)`；该路线工程风险高，保留为后续 ablation/讨论。

## 仓库事实
- 当前配置：`aerial_gym/config/task_config/pursuit_guidance_task_config.py`。
- 当前控制路径：`base_quad_root_link_control` + `thrust_bodyrate_control`。
- 当前目标运动 baseline：`apf_escape`。
- 当前 target 不是高保真多旋翼执行器模拟：`apf_escape` 生成期望位置/航向后，由 task 内 Lee-like controller 直接写入 `target_force_tensor` / `target_torque_tensor`，跳过 motor allocation、motor model、ESC/prop delay 和单电机推力限制。
- 当前 32D 观测是全状态几何观测，泄露目标相对状态和环境真实值。
- 当前 root-link robot 默认没有启用 LiDAR；仓库 LiDAR 当前走 Warp sensor 路线，任何 LiDAR 主张前必须实现 LiDAR robot/config/export/smoke-test 输入路径。

## 强制运行环境约束
- 本项目所有仿真、PPO、LiDAR/Warp sensor、Isaac Gym/gymtorch、risk rollout 和测试命令必须使用 conda 环境 `aerialgym_v2`。
- 首选命令形式：`conda run -n aerialgym_v2 python ...`。
- 若直接调用 Python，必须使用 `/home/ubuntu/miniconda3/envs/aerialgym_v2/bin/python`，并在涉及 Isaac Gym/gymtorch C++ extension 或 `ninja` 时显式设置 `PATH=/home/ubuntu/miniconda3/envs/aerialgym_v2/bin:$PATH`。

## 必需 Baseline 阶梯
| ID | 方法 | 作用 |
|----|------|------|
| B0 | 当前冻结的 visibility-aware 32D 全状态 PPO | 上界参考、nominal strong-tracking teacher、LiDAR/risk 数据主来源 |
| B1 | 降信息 PPO，无风险 | 主退化 baseline |
| B2 | 降信息 PPO + LiDAR-honest 启发式风险 | 手工风险 baseline |
| B3 | 降信息 PPO + 学习型状态风险 `p_lost(s_red)` | 只含 bridge 的学习 baseline |
| B4 | 降信息 PPO + 学习型动作条件风险调制 | 本文主方法 |

### 当前冻结 B0
- **B0 run：** `runs/PE_20260520_110828`
- **B0 checkpoint：** `policy_pool/ppo_upd_001300_step_2396160000.pth`
- **B0 语义：** 当前 visibility-loss penalty reward 下训练得到的全状态追逐策略；训练结束时处于 `stage_idx=3`，即 `3m + vis0.40`，未进入最终 `vis0.60`。
- **B0 可用性判断：** 可以作为后续 LiDAR rollout 与风险模型预测的 nominal strong-tracking baseline；它不是“完全不丢视野”的理想策略，但成功率、3m 到达率和终端可见性已足够稳定，同时仍保留非平凡丢视野/恢复样本。

## 必需外部比较框架（Intro / Related Work 基础）
本文的科学核心不是“又一个 RL 追逃算法”，而是在 **有限视场/有限可检测条件下的 UAV 追逐** 中，通过 **约束下动作优化** 保持目标可观测性；强化学习是实现和训练支撑。外部比较必须按以下四个轴组织。

1. **有限视场 / 有限可检测条件下的追逐与重捕获**
   - 必比近邻：`[@peng2025limitedvisual]`、`[@huh2026limitedregion]`、`[@shao2026peqmixer]`、`[@wang2025viper]`、`[@zhou2025visibilityocclusion]`。
   - 传感器兼容追逐：`[@li2025bearingonly]`、`[@zheng2025visioncapture]`、`[@mavcapturingmav2024]`。
   - 写作作用：承认 limited-FOV / limited-detectable pursuit 和 reacquisition 已存在；本文区别在于把目标丢失建模成可校准的动作条件可观测性风险，而不是只做 FOV reward、阶段切换或 MARL 协作。

2. **UAV 追逃 / 在线规划 / CTBR 或真实动力学部署**
   - 强近邻：`[@chen2025open]`、`[@zhang2023gameofdrones]`、`[@roncero2025agilecontrollers]`、`[@yan2024lsrctd3]`、`[@giral2026intercept]`、`[@zhao2025msmar]`。
   - 多机/扩展性对照：`[@zhao2024autonomousuavpe]`、`[@luo2024improvedmadrl]`、`[@tan2026scalablefixedwing]`、`[@xiang2025cihrl]`。
   - 写作作用：这些论文构成真实追逃、在线规划、CTBR/体速率控制和真实/高保真部署压力；本文不声称优于 multi-UAV online planning 或 agile PE controller，而是解决 reduced LiDAR/range-image 下的可观测性风险调制。

3. **约束下动作优化、风险动作选择与安全/可观测性保持**
   - 通用 constrained / safe RL 近邻：CPO / PPO-Lagrangian、Recovery RL、shielded RL、risk-sensitive policy gradient、asymmetric actor-critic。
   - 具体近邻：`[@achiam2017cpo]`、`[@thananjeyan2021recoveryrl]`、`[@alshiekh2018shielding]`、`[@zhang2026safetyshieldedflight]`、`[@li2026safeintent]`、`[@liu2026actionriskgating]`、`[@zhao2025msmar]`。
   - 写作作用：这些决定 B4 必须比较 PPO-Lagrangian、shield/filter、risk concat、privileged critic 和 shuffled-action risk。本文不得声称通用 safe PPO 或通用 action-conditioned risk gating 首创；区别是风险语义从 collision/safety 变为 target observability loss。

4. **强化学习作为支撑工具与工程验证语境**
   - RL/工程 scope：`[@yang2025rlpereview]`、`[@schulman2017ppo]`、`[@ren2025losrate]`、`[@liu2026autopilot]`、`[@gao2026mentalstates]`。
   - 传感器策略背景：`[@park2023lidardrone]`、`[@christian2024vtd3]`、`[@cao2019droneschasing]`、`[@zhang2019coarsetofine]`、`[@grape2019riskaware]`。
   - 写作作用：用这些说明 PPO/MARL/DRL 是工程支撑和 EAAI scope 依据，不把“使用 RL”写成创新；传感器背景用于支撑 no-truth-input、return count、dropout 和 sparse-return stress-test 评估。

## 方法定义
### 可观测性标签
- 虚拟可检测区域位于追逐者机体系，由 `R_det`、`alpha_h`、`alpha_v` 定义。
- 若目标在 `H = 150` 步内离开该区域并持续 `K_persist = 10` 步，则 `y_t = 1`。
- 必须做 `H in {100, 150, 200}` 以及至少一组 FOV/range 扰动敏感性分析。

### LiDAR 与风险模型
- 输入：`s_red`、`z_lidar` 和候选/执行动作 `u`。
  - `s_red = [body_linvel(3), body_angvel(3), rotation_matrix(9), prev_action(4)]`，共 19 维。
  - `z_lidar = 64`，由 `K = 3` 帧 LiDAR range-image/point features stack 经 CNN/CNN+GRU 压缩得到；64 维是第一版默认 latent，后续可做 `32/64/128` ablation。
  - `u = [c,p,q,r] in [-1,1]^4` 是实际候选动作或 PPO 执行动作，不把 `policy_mean` 当作 `u_ref` 输入。
- risk head 实际输入特征：`concat(s_red, z_lidar, u, u-prev_action, abs(u), u^2)`；`policy_mean/std` 只作为 metadata、候选生成和 OOD/诊断字段，不作为风险头核心输入。
- 输出：动作条件风险 `R_obs(s_red, z_lidar, u)`，第一版多头为：
  - PPO v0 使用：`p_loss_50`、`severity_50`、`p_loss_150`、`severity_150`。
  - 训练/诊断辅助：`first_loss_50`、`p_recover_150`；这两个不作为 PPO v0 在线输入。
- 训练期辅助输出：由特权监督得到的 range/bearing。
- 推理期规则：无目标真实值、无墙体真实值、无障碍物真实值、无真实值衍生 bearing/range/closing-speed。
- 在线输入真实性规则：`s_red`、`z_lidar`、risk head 和 PPO 风险调制分支不得使用目标 ID、真值 target-hit、仿真语义真值或人工 mask；若未来使用 target-hit/mask/bbox，必须建模为真实 LiDAR 聚类/跟踪器或 detector 输出，并加入漏检、误检、延迟和噪声。
- 锁定 LiDAR/target 设置：论文 1 主线固定使用 `target_x500 + PursuitForwardM3_120x25_UltraHighResLidarConfig`（M3-like 前向高分辨率配置）。静态 smoke 已验证 `target_x500 + M3-like` 在 50m 有 12 个有效返回点，100m/150m/200m 均保留 2 个目标返回点；该组合作为风险模型和可见性几何的强制实验约束。

### 风险调制 PPO（B4 v0 定型）
PPO 仍是控制策略主体，直接输出归一化 CTBR 命令：

`u_rl = [c, p, q, r] in [-1, 1]^4`

B4 v0 将动作条件可观测性风险作为结构化调制信号融入 PPO advantage：

- 风险标量：
  - `rho(o,u) = 0.40*p_loss_50 + 0.35*severity_50 + 0.15*p_loss_150 + 0.10*severity_150`。
- 同状态候选风险 baseline：
  - `rho_bar(o) = mean_i rho(o,u_i)`，主方法使用 mean，不使用 minimum。
  - minimum baseline 只作为 ablation，因为它会把所有非最低风险动作都扣分，容易变成保守 shield/filter。
- PPO advantage 修正：
  - `delta_risk = rho(o,u_exec) - rho_bar(o)`。
  - `A_tilde = A_task - lambda * clip(norm(delta_risk), -3, 3)`。
  - 第一轮固定 `lambda in {0, 0.02, 0.05, 0.1, 0.2}`，推荐起点 `lambda=0.05`。
- actor/critic 契约：
  - actor 第一版仍输出原 PPO Gaussian/tanh CTBR 分布，不做连续动作分布重塑。
  - critic 第一版仍估计 `V_task`，不强制增加 risk value head；risk value head 只作为 logging/adaptive-lambda ablation。
  - risk encoder/head offline pretrain 后 frozen/stop-gradient；不在同一 PPO rollout 中同步更新。

候选动作集合：
- 离线 branch 训练默认 `M_train = 16`，覆盖 policy sample、roll/pitch/yaw/thrust 单轴正负扰动、局部 Gaussian/必要时 uniform fallback，并使用最小 L2 diversity check；`policy_mean` 可以作为候选动作或诊断锚点，但不能作为风险头 reference 输入。
- PPO 在线风险 baseline 默认 `M_ppo = 8`，包含 `u_exec`、额外 policy/local sample、`roll±`、`pitch±`、`yaw±`；thrust 扰动优先用于离线训练覆盖，在线是否加入作为后续 ablation。
- 每个候选动作只执行第一步，随后由 teacher/current policy 接管 horizon rollout；风险语义是单步动作条件短期风险，不是连续 hold 3/5 step 的动作风险。

执行链路保持为：

`risk-modulated PPO -> thrust_bodyrate_control -> control allocation -> motor model -> Isaac Gym`

实现规则：
- PPO 训练仍按原 CleanRL PPO 路径进行。
- 环境执行动作应与 PPO logprob/ratio 对应的动作保持一致，避免 credit assignment 不一致。
- 风险 encoder/head 初始按 offline pretrain + freeze/stop-gradient 使用；是否允许端到端微调作为后续 ablation 再决定。
- 不做 MPC 内环。

B4 必需内部对照：
- risk concat only
- `R_obs(s_red,u)` vs `p_lost(s_red)` vs shuffled-action risk
- risk-conditioned advantage / cost-advantage
- actor latent/mean/variance modulation
- mean baseline vs minimum baseline
- PPO-Lagrangian using the same risk as cost
- shield/filter using the same risk as post-hoc intervention
- privileged critic baseline

工程可信度门槛：
- 100/150/200m LiDAR target returns 必须写成 sparse-return stress-test regime，不得写成真实长距部署保证。
- 必须报告 target return count、range-image occupancy、dropout、longest invisible streak、noise/latency/angular-resolution/reflectivity sensitivity。

## 阶段门槛
### 第 0 阶段：Baseline 审计
- [x] 确认 32D 全状态观测和 APF 目标路径。
- [x] 确认当前 pursuit 路径不是 depth-driven。
- **状态：** complete

### 第 1 阶段：文献与新颖性锁定
- [x] 完成 deep-research。
- [x] 完成 novelty assessment。
- [x] 将强近邻比较项加入 deep-research guidance。
- [x] 按“动作条件风险调制 PPO”完成 targeted related-work / novelty refresh，重点复核 actor risk gating、risk-sensitive PPO、risk-aware actor-critic、safe RL shield/filter 与 asymmetric actor-critic 的近邻关系。
- **状态：** complete（按当前窄 claim 完成）

### 第 2 阶段：问题定义
- [x] 曾锁定为 privileged-to-depth-only sensor-proxy transition；2026-05-14 已被 LiDAR/range-image 路线取代。
- [x] 排除通用目标预测、通用部分可观测和完整双边 self-play 作为论文 1 表述。
- [x] 将问题定义从“动作修正/动作过滤”同步为“动作条件可观测性风险调制 PPO 的动作分布与优势估计”。
- **状态：** complete

### 第 3 阶段：方法设计
- [x] 锁定 `H = 150`、`K_persist = 10`、`K = 3`、`z_lidar = 64`。
- [x] 将动作条件风险调制 PPO 锁定为主方法方向。
- [x] 比较 actor risk gating、risk value/critic、risk-adjusted advantage、risk regularization 的最小实现，并确定 B4 的第一版公式和网络接口。
- [x] 定型 `s_red`、`z_lidar`、`u` 到 `R_obs` 的工程接口，以及 `R_obs` 进入 PPO actor/critic/advantage 的第一版路径。
- **状态：** complete（B4 v0 工程契约已定型；后续变化进入 ablation，不再阻塞 6.3/6.4 实现）

### 第 4 阶段：实验设计
- [x] 锁定 B0-B4。
- [x] 锁定离线/在线指标和统计规则。
- [x] 重审 B4 相关 ablation：risk concat、state-risk bridge、action-conditioned risk modulation、risk critic/advantage modulation 的因果区分。
- [x] 实现前将 PPO-Lagrangian、shield/filter、privileged critic、shuffled-action risk 和 LiDAR sparse-return sensitivity 写入具体实验注册表。
- **状态：** complete（实验设计已按 refresh 收束；后续进入实现阶段）

### 第 5 阶段：Deep-Research 同步
- [x] 加入近邻比较论文和修订后的 gap 表述。
- [x] 新增 `deep-research-output/perceptual-risk-modulated-ppo-eaai-refresh/` 作为当前有效 refresh。
- [x] 将候选论文整合结论同步到 frontier/survey/deep-dive/code/synthesis/report/cross-validation，并重新得出更严格结论。
- **状态：** complete（当前 refresh 已同步）

### 第 6.1 阶段：控制器验证与课程学习 B0 baseline（前置阶段）
- [x] 控制器参数对比：激进（e1, scale=[1,3,3,1.5]）vs 保守（默认, scale=[1,2,2,0.2]）→ 保守碾压激进（0.97% vs 0% SR）。控制器限制 = 隐式正则化器。
- [x] 确认仓库默认控制器参数为 baseline，不再探索放大限制。
- [x] 课程学习对比：手动（每阶段 `total_timesteps=400M` × 3）vs 自动（`total_timesteps=1.2B` × 1）→ 自动 100% SR，手动 77.2%。
- [x] 根因：`total_timesteps` 影响全局 entropy 衰减速度。手动每阶段重置 entropy=0.01，1m 精度阶段噪声过高。自动 `total_timesteps=1.2B` 使 ent 自然匹配难度。
- [x] B0 在默认控制器 + `--curriculum --no-early-stop` + `total_timesteps=1.2B` 下达到 **100% 成功率**。
- **状态：** complete

### 第 6.2 阶段：策略池构建与风险数据采集（当前阶段）
- [x] 设计多维专家/行为策略池（解决单策略数据偏差问题）。
- [x] 确定策略池来源：不同训练阶段、不同课程阈值、不同 seed，并补充少量规则/启发式/噪声行为来源。
- [x] 在 PPO 训练中新增 `policy_pool/` 中间 checkpoint 保存和 `manifest.jsonl` 元数据同步。
- [x] 实现离线 `risk_geometry.py` label/summary 脚本。
- [x] 只保留 LiDAR/range-image 采集方向。
- [x] 收束 LiDAR/target 主配置：严格 OS2-64 与 AT128P-like 作为历史失败对照，主线固定为 `target_x500 + M3-like` 前向高分辨率 LiDAR。
- [x] 新增高速追逐 LiDAR config，并保持在线模型输入仅来自 LiDAR range/point features：M3-like ultra-high-res 前向 profile。
- [x] 实现 LiDAR static smoke test：target 位于 1/5/10/25/50/100/150/200m 时保存 range image 和 return mask，并支持切换 target quad/x500。
- [ ] 在已锁定 `target_x500 + M3-like` 组合上做不同姿态、横向偏移和多帧累积检查；300/500m 不作为论文 1 主线要求。
- [x] 实现 LiDAR rollout exporter，保存 LiDAR range-image keyframes、状态、动作、policy 分布和 checkpoint 元数据。
- [x] 验证 `risk_geometry.py` 标签生成在策略池 rollout 上的基础正确性：`index.jsonl` 路径有效，risk label 长度与轨迹步数对齐。
- [x] 完成第一轮 `runs/PE_20260513_175323` 的 13 checkpoint x500 + LiDAR rollout smoke，输出 `.npz`、PNG 和 manifest。
- [x] 对 13 个筛选 checkpoint 跑完整 LiDAR rollout export + label summary，并检查 visible/lost/risk 分布。
- [x] 针对第一轮 rollout 暴露出的低可见率问题，增加 visibility-aware teacher reward。
- [x] 复盘 `runs/PE_20260514_221640`：visibility reward 从第一阶段直接加入，且课程仍按固定 400M 步切换，两个阈值切换点 SR 均为 0。
- [x] 将 visibility-aware teacher 训练改为两轮渐进式：先无 visibility reward 完成 5m/3m/1m；每阶段 SR 连续 10 个 update >= 95% 后切换；之后开启 visibility reward 并重新从 5m -> 3m -> 1m 优化，stage 3/4/5 分别使用固定低/中/满 visibility 权重档位。
- [x] 根据硬件 LiDAR 近场约束和风险模型数据需求，将 visibility-aware teacher 课程压缩为 5 阶段：`5m,vis0` -> `3m,vis0` -> `3m,vis0.10` -> `3m,vis0.20` -> `3m,vis0.30`；不再回到 1m 终点。
- [x] 将 pursuit task 默认成功阈值定义为 `3m`，success/success bonus 需要最终 `target_detectable=True`；中途短暂丢视野不终止，继续累计 visibility penalty，恢复可见后当前 loss 计数清零。
- [x] 增加 episode 内 visibility recovery 统计：final detectable、当前连续丢失、累计不可见步数、丢失段数、恢复段数、最长连续丢失，并同步到 PPO 日志和 policy-pool manifest。
- [x] 将 LiDAR rollout exporter 升级为 schema v2，增加 `max_consecutive_invisible_steps`、`num_invisible_segments`、`recovered_invisible_segments`、`recovery_rate_within_persist_steps` 等恢复指标。
- [x] 新增 `tests/test_pursuit_visibility_recovery.py`，覆盖 final-detectable success gate、短暂丢失后恢复、visibility counters，以及 exporter recovery stats 实际落盘。
- [x] 基于 `runs/PE_20260518_133223` 验证：后三个 visibility 阶段 SR 基本保持 `0.998-1.000`，final detectable 接近 1，但 final stage 平均 invisible steps 仍约 `498`、最新约 `451.5`；因此支持将 visibility 权重从 `0.10/0.20/0.30` 提高到 `0.15/0.30/0.45`。
- [ ] 同步当前权重上调后的测试和启动日志：`tests/test_curriculum_controller.py` 期望改为 `0.15/0.30/0.45`，`ppo_guidance.py` 启动文案同步为 `3m+vis0.15,3m+vis0.30,3m+vis0.45`，并重跑相关测试。
- [x] 基于 `runs/PE_20260520_110828` 完成当前 B0 visibility-aware teacher 筛选：冻结 `policy_pool/ppo_upd_001300_step_2396160000.pth` 作为 `B0: current vis-reward tracking baseline`。
- [x] 修复动态 Warp LiDAR target mesh stale 问题：sensor render 前同步 Warp mesh，使 geometry detectable 与 semantic/range image 回到同一帧。
- [x] 在 mesh-sync 修复后的冻结 B0 `upd_1300` 上完成 full-episode semantic audit：943 帧从约 69m 到 terminal success，semantic visible 与 geometry detectable 均为 617 帧，两类错配均为 0。
- [x] 将可视化 semantic 审计拆成独立脚本；正式 dataset exporter 不保存 debug PNG/JPG，只写可训练 LiDAR shard 与 metadata。
- [x] 实现 `ladder_v1` checkpoint preset，覆盖弱策略、初学追近、高成功低 visibility、visibility early stage、强 visibility teacher，并支持手动 checkpoint override。
- [x] 实现 schema7 risk dataset shard schema：`lidar_range_norm`、deployable `ego_obs`、history action、behavior action、policy mean/std、multi-horizon semantic-visible labels、QA metadata。
- [x] 实现第一版 QA gates：semantic availability、semantic/geometry hard mismatch、baseline schema/update coverage、QA window jsonl、leakage manifest 声明和 LiDAR finite stats 记录。
- [x] 完成多 teacher branch pilot，并把候选动作扩展为带 diversity check 的 roll/pitch/yaw/thrust 单轴扰动、policy sample 和 fallback 补样；thrust 扰动进入离线 branch 训练覆盖。
- [x] 按 B4 v0 契约补齐 rollout artifact：branch anchor 保存 `K=3` LiDAR stack，branch metadata 显式保存 `h050/h150` 风险标签和 recovery 辅助标签，manifest 将 `policy_mean/std` 标为诊断元数据而非风险头输入。
- [ ] 实现 anchor sampling：`stride/event/stratified`，优先保存 loss/recovery/boundary/low-pixel/hard-case 窗口。（2026-06-01 已完成 branch pilot 用 `risk_edge` 低像素/边界 anchor 过滤；event/stratified 正式采样仍待补。）
- [ ] 将 LiDAR finite stats 升级为可选 hard gate，并补充 split-by-checkpoint/source 的 coverage 检查。
- [ ] 生成 `D0 offline policy-pool` 数据集，过 QA 后冻结为 risk model 训练输入。
- **状态：** in_progress

### 第 6.3 阶段：离线风险模型训练
- [x] 实现 B4 v0 risk head：`RiskHead(s_red, z_lidar, u) -> {p_loss_50, severity_50, first_loss_50, p_loss_150, severity_150, p_recover_150}`。（2026-06-03：`risk_model.py`，CNN+GRU LiDAR backbone + 直接 `s_red/action` 融合）
- [x] 训练 loss 默认权重：`BCE(p_loss_50)=1.0`、`Huber(severity_50)=1.0`、`BCE(p_loss_150)=0.7`、`Huber(severity_150)=0.7`、`Huber(first_loss_50)=0.3`、`BCE(p_recover_150)=0.3`、pairwise ranking `0.5`。
- [ ] Stage 6.3-A：K-frame CNN state-risk baseline，验证 LiDAR/range labels 可学。
- [x] Stage 6.3-B：CNN encoder + GRU temporal risk model，作为默认主力结构。
- [x] Stage 6.3-C：action-conditioned `R(o,u)`，输入使用 `u, u-prev_action, abs(u), u^2`，不使用 `u-policy_mean` 作为核心 reference。
- [ ] Stage 6.3-D：candidate-action ranking / branch rollout，验证模型能否区分不同候选动作的脱视野风险；ranking pair 仅在 `|y_i-y_j| > epsilon_rank=0.02` 时启用。（2026-06-03：ranking 训练入口和 pilot smoke 已通过；完整 branch 数据训练后的排序性能仍待验证）
- [ ] Stage 6.3-E：Mamba/SSM temporal model 作为 GRU baseline 成立后的 creative stage；LSTM/Transformer 只做 ablation。
- [ ] 离线训练和校准，记录 AUROC、AUPRC、Brier、ECE、false-safe rate、severity/time/recovery 指标。
- [ ] 在 heldout checkpoint/source 上验证泛化；不允许同一 checkpoint episode 同时出现在 train/test。
- **状态：** in_progress（代码入口已实现并通过小样本 smoke；完整数据训练/校准/heldout 泛化尚未完成）

### 第 6.4 阶段：降信息策略实现（B1-B4）
- [ ] 实现 B1-B4 reduced-observation modes，并保留 B0 full-state baseline 作为上限参照。
- [ ] 先接入 frozen risk model，不在同一 PPO rollout 中同步更新 risk model。
- [ ] 实现 B4 v0：mean-baseline risk-adjusted advantage，actor 分布和 critic 主 value 不改，PPO logprob/ratio 对应环境实际执行动作。
- [ ] 运行 fixed lambda grid：`{0,0.02,0.05,0.1,0.2}`，定标 risk penalty scale。
- [ ] 在固定网格确认有效后，实现 adaptive dual lambda，约束指标使用 persistent loss / over-horizon risk。
- [ ] 实现 DAgger-like aggregation：`D0 -> R0 -> risk PPO -> D1 -> R1`，每轮混合 offline policy-pool、current risk-PPO、action noise、hard cases、scripted stress cases。
- [ ] large-scale 训练前运行 smoke test、小 rollout 诊断和可见性 QA。
- **状态：** pending

### 第 6.5 阶段：完整训练与评估
- [ ] B0-B4 全部训练完成。
- [ ] 论文图表和统计分析。
- [ ] 敏感性分析（H ∈ {100, 200}、FOV 扰动）。
- **状态：** pending

## 明确推迟
- 完整 RGB 视觉追逐。
- 完整机载 detector/tracker 声明。
- 双边 self-play 或学习型逃逸者作为中心贡献。
- 除非已经实现并验证，否则 full occlusion-aware sensing 不作为主声明。

## 错误记录
| 错误 | 处理 |
|------|------|
| planning 中保留了较弱中间路线 | 已重写为单一 EAAI 强主线 |
| `risk_geometry` 测试直接导入 `aerial_gym.task...` 会触发 Isaac Gym/gymtorch 初始化 | 将 `risk_geometry.py` 保持为轻量纯 PyTorch 模块，测试用文件路径加载，避免离线单测依赖仿真初始化 |
| 最初分析认为 robot_name 是手动/自动 run 差异来源 | 用户在旧 run 前已手动修改 robot name，两者实际相同。根因是 `total_timesteps` 对 entropy 衰减的影响 |
| 曾误用旧 conda 环境 `aerialgym` 做命令规划 | 已锁定后续全部命令使用 `aerialgym_v2`；涉及 `ninja` 时必须保证 `aerialgym_v2/bin` 在 `PATH` 中 |
| 早期把“全过程可见”误读为任意 1 帧丢失即失败 | 已修正为“快速追击中允许短暂丢失并学习恢复”；success 只要求终点可见，长时间不可见通过累计 penalty 和 recovery stats 暴露 |
