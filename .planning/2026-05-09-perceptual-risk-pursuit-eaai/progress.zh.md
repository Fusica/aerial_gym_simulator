# 进展记录

## 全局运行环境约束
- 后续所有仿真、PPO、LiDAR/Warp sensor、Isaac Gym/gymtorch、rollout export、risk label 和测试命令必须使用 conda 环境 `aerialgym_v2`。
- 推荐命令：`conda run -n aerialgym_v2 python ...`。
- 若直接调用解释器，使用 `/home/ubuntu/miniconda3/envs/aerialgym_v2/bin/python`；涉及 Isaac Gym/gymtorch C++ extension 或 `ninja` 时同时设置 `PATH=/home/ubuntu/miniconda3/envs/aerialgym_v2/bin:$PATH`。
- 旧环境 `aerialgym` 不再用于本项目验证；下方历史记录中的 `aerialgym/bin/python` 命令不可作为后续复现实验命令。

## 2026-05-09
- 初始化 `.planning/2026-05-09-perceptual-risk-pursuit-eaai/`。
- 审计当前 pursuit task，确认其是使用 `apf_escape`、32D 几何观测、root-link 控制和 `thrust_bodyrate_control` 的特权全状态 PPO baseline。
- 在 `deep-research-output/perceptual-risk-pursuit-review/` 下完成 deep-research。
- 删除了不支持锁定 EAAI 声明的辅助风险旧指导。
- 历史记录：曾将论文 1 重新框架为 privileged-to-depth-only sensor-proxy transition；2026-05-14 已被 LiDAR/range-image 主线取代。
- 暂定主方法：降信息追逐 + 学习型动作条件可观测性风险 + 风险调制 PPO。
- 历史记录：曾锁定 `K = 3` depth 帧、`z_depth = 64`；2026-05-14 已切换为 `K = 3` LiDAR stack、`z_lidar = 64`。
- 锁定架构：离线预训练、冻结 LiDAR/range-image encoder 和 risk head、PPO 不反传进表示和修正模块。
- 锁定 B0-B4 baseline 阶梯和离线/在线指标。

## 2026-05-10
- 使用 novelty-assessment 的严苛评审口径重新分析。
- 判断宽泛的“risk-assisted partial-observation pursuit”不够强，因为已有 limited-FOV MARL、prediction-enhanced pursuit、risk-gated policy、sensor-compatible pursuit。
- 锁定更强的新颖性切口：从特权 rollout 学习动作条件 observability-risk，并在 LiDAR/range-image 降信息下用于调制 PPO 的动作分布与优势估计。
- 重写 `task_plan.zh.md` 和 `findings.zh.md`，删除弱中间路线。
- 将 `[@peng2025limitedvisual]`、`[@li2026safeintent]`、`[@li2025bearingonly]` 加入 deep-research 近邻比较材料。
- 实现 `aerial_gym/task/pursuit_guidance_task/risk_geometry.py`，包括：
  - `DetectionFrustum`
  - 追逐者机体系相对位置转换
  - 虚拟检测区域判定
  - `K_persist` 连续丢失起点检测
  - `H` 步 horizon 内 observability-loss 标签和 first-event offset
- 新增 `tests/test_pursuit_risk_geometry.py`，覆盖 FOV/range/behind target、姿态转换、连续丢失、horizon 标签和 pose-to-label 路径。
- 实现时保持 `risk_geometry.py` 不导入 `aerial_gym.utils.math` 或环境模块，避免离线单测触发 Isaac Gym/gymtorch 初始化。

## 2026-05-11

### E0-C1：e1 控制器参数（宽限制）——`runs/PE_20260511_000603`
- 完成 E0-C1：e1 控制器参数预验证实验。scale_input = [1.0, 3.0, 3.0, 1.5]，kOmega = [1.2, 1.2, 1.0]，torqueLimit = [2.4, 2.4, 0.2]。
- **结果：** 1.2B 步全状态 PPO，无课程学习，成功率 = 0.0000（统计上无异于 0，p=0.322）。熵降至 -4.28，pre-tanh OOB 61.6%，指令 100% 饱和。
- 四个训练阶段：避碰→接近→精细接近→停滞。但成功从未涌现。

### E0-C2：仓库默认控制器参数（窄限制）——`runs/PE_20260511_125039`
- 完成 E0-C2：仓库默认控制器参数预验证实验。scale_input = [1.0, 2.0, 2.0, 0.2]，kOmega = [1.0, 1.0, 0.8]，torqueLimit = [2.4, 2.4, 0.18]。
- **结果：成功率 0.97%（最高 2.33%），首次非零 SR 在 5.23 亿步。54/651 数据点 SR > 0.5%。**
- 最小距离 1.71m（vs e1 的 2.40m），1m 到达率 3.3%（vs e1 的 0%），3m 到达率 99.6%（vs e1 的 89%）。
- 熵仅降至 -1.71（vs e1 的 -4.28），pre-tanh OOB 48.5%（vs e1 的 61.6%）。

### E0-C1 vs E0-C2 对比结论
- **反直觉结论：更保守的控制器显著优于更激进的控制器。**
- 机制：宽限制→大动作空间→高方差梯度→早熵塌陷→tanh 饱和→梯度截断→无法学习精确接近→成功永不涌现。
- 窄限制→小动作空间→隐式正则化→维持探索→精度窗口打开时探索仍活跃→成功涌现→成功奖励持续提供正向梯度。
- **控制器限制是隐式正则化器，不是严格的物理约束。"更大 = 更好"对 RL 不成立。**
- 行动方向修正：不再考虑放大控制器限制。保持仓库默认值。未来可探索进一步缩小限制是否能加速学习。
- B0 上界可以在默认控制器 + 课程学习下建立。课程学习仍是 priority #1。

## 2026-05-13

### E0-C3：手动 vs 自动课程学习对比分析
- 分析了手动课程学习 run `PE_20260506_181131` 和自动课程学习 run `PE_20260513_102027` 的差异。
- 手动 run：分三次运行，每次 `total_timesteps=400M`，手动修改 success_threshold 并 resume。
- 自动 run：一次运行 `total_timesteps=1.2B`，使用 `--curriculum --no-early-stop`，`curriculum_stage_step_budgets=[400M, 400M]`。
- 两者 robot（`base_quad_root_link_control`）、controller 参数、阈值（5→3→1）、PPO 超参相同。
- **手动 run 最终 SR=77.2%（1m 阈值），自动 run 最终 SR=100%。**

### 根因分析：`total_timesteps` 通过 entropy 衰减影响性能
- 手动方式：每阶段 entropy 从 0.01 重新衰减。1m 精度阶段 entropy 噪声过高（0.01），策略难以收敛到高精度动作。
- 自动方式：entropy 在 1.2B 步内均匀衰减。进入 1m 阶段时 entropy 已自然降至 0.0033，适合精细控制。
- 次要因素：手动 run 的 `early_stop` 可能在 SR 暴跌期累积 patience 计数。
- 机器人模型不是差异来源：用户在旧 run 运行前已手动修改 `robot_name` 为 `base_quad_root_link_control`。

### 课程学习代码实现确认
- 审查了 `curriculum/curriculum_controller.py`：课程学习仅做阈值自动切换，无探索相关修改。
- 实质是把手动过程（到步数→改阈值→save checkpoint→继续）自动化，等价功能替换。
- 关键区别在于 `total_timesteps` 的设定方式影响到全局 entropy 衰减速度。

## 2026-05-13（续）

### 第 6.2 阶段动机
- 仅使用 B0 最终 100% SR checkpoint 进行 rollout 采集存在两个问题：
  1. **数据偏差：** 只有成功轨迹，缺少失败/边缘/风险累积场景，风险模型无法学到真实的风险分布。
  2. **数据量不足：** 100% SR 下目标几乎不丢失，正样本（`y_t=1`，可观测性丢失事件）极稀疏，风险预测器训练困难。
- 需要一个具备多维能力差异的策略池，覆盖"从不成功到总是成功"的行为谱系。

## 当前状态
- **计划 ID：** `2026-05-09-perceptual-risk-pursuit-eaai`
- **阶段：** 第 6.2 阶段——策略池构建与风险数据采集
- **总体状态：** active

## 关键判断
- 论文不做通用 pursuit RL、通用部分可观测、目标轨迹预测或完整 bilateral self-play。
- 当前唯一强主线是 `LiDAR/range-image 降信息 + 动作条件可观测性风险 + 风险调制 PPO`。
- B0 是上界参考和标签来源，不是本文方法。
- B4 是主方法。
- 如果主方法继续使用特权状态，视为 scope regression。
- **新增（E0-C1/C2 对比后修正）：** E0-C2 证明 B0 在当前奖励 + 默认控制器下可学习成功（~1-2%）。E0-C1 的失败源于控制器过度放宽导致的探索崩溃，而非奖励结构缺陷。B0 上界在课程学习加持下可以建立。

## 下一步（第 6.2 阶段）
1. 确定策略池来源维度：训练阶段、课程阈值、seed。
2. 从已保存的 stage checkpoint（`stage_0_thr_5.0m`、`stage_1_thr_3.0m`）和中间 training checkpoint 中筛选候选策略。
3. 验证 `risk_geometry` 标签在策略池 rollout 上的正确性和覆盖率。
4. 设计数据采集 pipeline：多策略 rollout → risk label 标注 → dataset manifest。
5. 确保正负样本平衡（`y_t=1` 和 `y_t=0` 均有足够样本）。

## 2026-05-13（策略池保存间隔）
- 确认 `ppo_guidance.py` 默认每个 update 对应 `512 * 3600 = 1,843,200` global steps。
- 自动课程学习 1.2B 训练约 651 updates；已保存阶段切换 checkpoint 为 update 218 和 update 436。
- 第 6.2 默认采用每 10 updates 保存一次中间策略，即约 18.4M steps；最多不应稀疏到超过 20-25 updates。
- PPO checkpoint 是主体数据来源，但第 6.2 需要补充少量规则/启发式/噪声行为来源，用于扩大目标丢失、边缘失败和动作条件风险覆盖。它们只作为数据采集策略，不作为 B0-B4 之外的方法 baseline。

## 2026-05-13（策略池保存机制实现）
- 修改 `aerial_gym/rl_training/cleanrl/ppo_guidance.py`，新增默认启用的策略池保存机制。
- 新增参数：`--no-save-policy-pool`、`--policy-pool-save-interval-updates`、`--policy-pool-dir-name`。
- 训练时在 `runs/<run>/policy_pool/` 下保存 `ppo_upd_<update>_step_<global_step>.pth`，默认每 10 updates 保存一次，额外保存首个 update 和课程切换 update。
- 同步写入 `policy_pool/manifest.jsonl`，每条记录包含 checkpoint 路径、来源类型、训练配置、课程状态、episode 指标、reach rate、score 和 update/action 诊断。

## 2026-05-14（运行环境修正）
- 用户明确指定本项目必须使用 conda 环境 `aerialgym_v2`，不是旧环境 `aerialgym`。
- 已将 `aerialgym_v2` 写入 `task_plan.zh.md`、`findings.zh.md`、`progress.zh.md` 和 `decision_log.zh.md` 作为强制运行环境约束。
- 验证 `conda run -n aerialgym_v2 python -c "import sys, shutil; print(sys.executable); print(shutil.which('ninja'))"` 返回：
  - Python：`/home/ubuntu/miniconda3/envs/aerialgym_v2/bin/python`
  - Ninja：`/home/ubuntu/miniconda3/envs/aerialgym_v2/bin/ninja`

## 2026-05-14（旧 depth 链路删除）
- 用户明确要求后续只考虑 LiDAR，不再保留旧 depth 专用代码链路。
- 已删除 pursuit 专用 depth config、depth smoke、depth rollout/export/action-query/common helper 脚本。
- 已删除旧 depth-camera root-link robot registry 和对应 config class。
- `risk_geometry.py` 保留，因为它是传感器无关的可观测性标签逻辑。

## 2026-05-14（真值输入约束）
- 旧 depth/camera 语境下的 segmentation 约束已上卷为通用真实性规则：LiDAR 主线不得使用仿真语义真值、目标 ID、真值 target-hit 或人工 mask 作为在线输入。
- 后续 `LiDAR + Action -> Risk probability` 主线的 `s_red`、`z_lidar`、risk head 和 PPO 风险调制分支只能使用自身状态、LiDAR range-image/point features 及其学习得到的风险量。
- 若未来使用 target-hit/bbox/mask，必须把它定义为真实 LiDAR 聚类/跟踪器或 detector 的 noisy output，并显式建模漏检、误检、延迟和噪声；这不属于当前 LiDAR raw/range-image 主线输入。

## 2026-05-14（主传感器路线切换：Depth -> LiDAR）
- 用户明确要求不再使用 depth camera 作为主传感器选项，改用 LiDAR。
- 新主线已收束为 `target_x500 + M3-like` 前向高分辨率 LiDAR；不再把 500m/100Hz 作为论文 1 的主实验目标。
- 已将 `task_plan.zh.md` 的主声明、不可妥协贡献、方法定义、B2 baseline、阶段 6.2/6.3 门槛从 `depth-only/z_depth/depth stack` 改为 `LiDAR/range-image/z_lidar/LiDAR stack`。
- 已在 `findings.zh.md` 和 `decision_log.zh.md` 记录：`target_x500 + M3-like` 是当前强制 sensor/target 约束，100m/150m/200m 均保留 2 个目标返回点。
- 当前仓库事实：LiDAR 走 Warp sensor 路线；`use_warp=False` 时启用 LiDAR 会报错，Warp 路线当前禁止 camera 与 LiDAR 同时启用。因此后续需要独立 LiDAR robot/config/exporter/static-smoke 链路。
- 既有 depth camera smoke/exporter 保留为历史验证和调试参考，不再作为论文 1 主感知路线。

## 2026-05-14（LiDAR OS2-64 static smoke）
- 新增严格 OS2 风格 pursuit LiDAR config：`PursuitForwardOS2_64_LidarConfig`，`64 x 512`、水平 `360°`、垂直 `22.5°`、`max_range=200m`、`segmentation_camera=False`。
- 新增 robot registry 名称：`base_quad_root_link_control_with_lidar`。
- 新增 `tests/smoke_pursuit_lidar_static_scene.py`，构造 Warp LiDAR 小静态场景，仅保留 target quad，测试 1/5/10/25/50/100/150/200m。
- 正前方结果：1m=408 pixels，5m=8，10m=4，25m 起为 0，200m=0。
- ray-aligned 结果：1m=419 pixels，5m=15，10m=4，25m 起仍为 0，200m=0。
- 结论：当前 OS2-64 `64 x 512` range image 对小型无人机目标远距返回过少，不能直接作为 100m-200m 甚至 500m 风险模型输入，需要更高角分辨率、ROI/跟踪式扫描、目标尺寸/反射建模或改用其他感知表示。

## 2026-05-14（高分辨率前向 LiDAR 配置与 smoke）
- 基于公开规格新增并保留 `PursuitForwardM3_120x25_UltraHighResLidarConfig`：`501 x 2401`、水平 `120°`、垂直 `25°`、`max_range=300m`、`segmentation_camera=False`。
- 当前 `base_quad_root_link_control_with_lidar` 默认使用 M3-like ultra-high-res profile。
- AT128P-like smoke：正前方 25m=4 pixels，50m 起为 0；ray-aligned 50m=2，100m/150m 仍为 0。
- M3-like smoke：正前方 50m=8 pixels，100m=2，150m=2，200m=0。
- 结论：若当前 target quad mesh 不放大、不增加反射/实体截面积，100-150m 小目标检测至少需要 `0.05°` 级别前向高密度扫描；即便如此返回点仍非常稀疏，后续必须加入性能 benchmark、目标点稀疏性处理或 tracking/temporal accumulation。
- 已删除未进入主线的 `PursuitForwardAT128_120x25_LidarConfig`；AT128P-like 只作为历史失败对照记录保留。

## 2026-05-14（目标 URDF 候选与 LiDAR 适配 smoke）
- 检查当前仓库 robot URDF：`quad`、`x500`、`lmf1/lmf2`、`octarotor`、`morphy`、`snakey`、`tinyprop` 等；当前 checkout 未发现 `F450.urdf`。
- 计算主要候选 visual/collision 尺寸：`quad/model.urdf` visual 约 `0.40 x 0.40 x 0.07m`，`x500/model.urdf` visual 约 `0.448 x 0.448 x 0.07m`；`octarotor/octarotor.urdf` 虽有更大垂向体积，但会改变任务对象语义，不进入当前主线。
- 新增 pursuit target asset：`target_x500`。2026-05-14 已按用户要求将默认 target 从 `target_quad` 切为 `target_x500`；`target_quad` 仅作为显式对照保留。`target_x500` 只覆盖 URDF 路径和 `controller_mass`，速度上限继续继承 pursuit `target_quad` 的 `max_angular_velocity=20/max_linear_velocity=40`。
- 扩展 `tests/smoke_pursuit_lidar_static_scene.py`，新增 `--target-asset-type {target_quad,target_x500}`。
- 验证 `conda run -n aerialgym_v2 python -m py_compile aerial_gym/config/asset_config/pursuit_guidance_asset_config.py aerial_gym/config/env_config/pursuit_guidance_env.py tests/smoke_pursuit_lidar_static_scene.py` 通过。
- 新增 smoke 输出：
  - `/tmp/pursuit_lidar_static_smoke_x500_at128/summary.json`：25m=6，50m 起为 0。
  - `/tmp/pursuit_lidar_static_smoke_x500_m3/summary.json`：50m=12，100m=2，150m=2，200m=2。
- 判断：x500 是当前仓库中最适合保持常规 quad 语义的目标候选。`target_x500 + M3-like` 已作为当前主线强制配置；100m/150m/200m 均保留 2 个目标返回点，点数稀疏性作为模型输入约束处理，而不是继续作为配置选择争论。
- 修正 target controller 质量假设：新增 `controller_mass` 到 `target_quad/target_x500` asset config，并在 `PursuitGuidanceTask._apply_target_controller()` 中使用 target mass 而不是 pursuer `robot_mass`。`target_x500 + M3-like` smoke 复跑通过，输出 `/tmp/pursuit_lidar_static_smoke_x500_m3_massfix/summary.json`，100/150/200m 均为 2 个有效返回点。
- 删除不进入主线的 pursuit `target_octarotor` 资产分支和 smoke 选项；仓库原生 `base_octarotor` robot/config 保留给其他任务使用。
- 扩展 `ppo_guidance.py --play`，新增 `--target-asset-type {target_quad,target_x500}`；该开关只切换 `target_*` asset，不影响 room/obstacle，回放文件按目标类型加后缀，避免覆盖原始 `play_trajectory.npz`。
- 使用 `runs/PE_20260513_175323/latest.pth` 做同 seed 单环境回放验证：
  - `target_quad`：966 steps，9.66s，final/min distance = `0.999m`，done=success；目标速度 mean/max = `7.24/20.52m/s`，target force z mean/max = `20.31/46.28N`。
  - `target_x500`：964 steps，9.64s，final/min distance = `0.994m`，done=success；目标速度 mean/max = `7.08/21.75m/s`，target force z mean/max = `17.19/39.66N`。
- 结论：在该 seed 的 latest-policy smoke 中，切换到 x500 URDF 和 `controller_mass=1.656` 没有导致任务失败或 target 动力学爆掉；正式切主实验前仍需做多 seed/多 env 成功率统计。
- 写作约束补充：当前 target 并非高保真多旋翼执行器模拟。`apf_escape` 生成期望位置/航向后，task 内 target controller 直接写 `target_force_tensor`/`target_torque_tensor`，跳过 allocator、motor model、ESC/prop 动态和单电机限制；已同步到 `task_plan.zh.md`、`findings.zh.md`、`decision_log.zh.md`。

## 2026-05-14（LiDAR x500 策略池 rollout exporter）
- 新增 `aerial_gym/rl_training/cleanrl/export_pursuit_lidar_rollouts.py`，作为独立 exporter，不改 PPO 训练或 play 路径。
- 默认从 `runs/PE_20260513_175323/policy_pool/manifest.jsonl` 选择 13 个 checkpoint：`1, 140, 150, 160, 170, 218, 220, 436, 440, 450, 460, 470, 640`。
- 每个 env 加载一个独立 PPO `Agent`，共享一个 13-env vectorized LiDAR env，统一拼接动作后 `env.step()`。
- 默认配置：`base_quad_root_link_control_with_lidar`、`target_x500`、`use_warp=True`、`max_steps=3600`、`num_episodes=1`。
- 输出目录：`runs/lidar_smoke/PE_20260513_175323_x500_13ckpt_smoke/`。
- 保存 `selected_checkpoints.json`、`manifest.json`、`index.jsonl`、13 个 episode `.npz`、LiDAR keyframe PNG 和 trajectory PNG。
- 验证命令：
  - `conda run -n aerialgym_v2 python -m py_compile aerial_gym/rl_training/cleanrl/export_pursuit_lidar_rollouts.py`
  - `conda run -n aerialgym_v2 python aerial_gym/rl_training/cleanrl/export_pursuit_lidar_rollouts.py --dry-run`
  - `conda run -n aerialgym_v2 python aerial_gym/rl_training/cleanrl/export_pursuit_lidar_rollouts.py --max-steps 3600 --num-episodes 1`
- 完整 smoke 验收：13 个 `.npz`、5397 个 PNG、`index.jsonl` 13 行、路径有效、risk label 长度与轨迹步数一致、输出约 `170M`。

## 2026-05-14（可见性保持 teacher 问题）
- 复查完整 LiDAR rollout 统计后确认：当前成功率接近 99%-100% 的全状态 PPO 并不等价于可见性保持 teacher。
- 成功 checkpoint 的可见率仍很低：
  - update `0460`：success，detectable rate `0.290`。
  - update `0470`：success，detectable rate `0.356`。
  - update `0640`：success，detectable rate `0.134`。
- 当前 reward 中 `forward_alignment` 不是 LiDAR FOV 可见性，且权重仅 `0.005`，无法约束全流程感知连续性。
- 下一步执行：最小修改 `pursuit_guidance_task` reward，加入可配置 visibility-aware shaping，权重可与 progress 同一量级；先不引入强硬 done，以保留临界丢失/恢复样本。

## 2026-05-14（visibility-aware teacher reward 最小实现）
- 修改 `aerial_gym/config/task_config/pursuit_guidance_task_config.py`：
  - 新增 visibility reward 权重参数；不设置 `enable_visibility_reward` 开关。
  - 可见性保持成为 pursuit task 默认训练目标，即使后续 teacher 不挂真实 LiDAR 也需要保持目标在前向可检测区域。
  - `weight_visibility=0.5`，与 `weight_progress=0.5` 同量级。
  - `weight_visibility_loss_penalty=0.25`，用于连续不可见的额外引导。
- 修改 `aerial_gym/task/pursuit_guidance_task/pursuit_guidance_task.py`：
  - 新增 `visibility_loss_steps` episode 状态。
  - 在 reward 中计算 target 是否位于 LiDAR sensor-frame frustum。
  - 若 robot 启用 LiDAR，直接读取 robot `lidar_config` 的 FOV/range/min_range/安装位姿；若未启用 LiDAR，使用当前主线 `PursuitForwardM3_120x25_UltraHighResLidarConfig` 作为 teacher 先验。
  - 加入连续 margin reward 和 persistent loss penalty。
  - 不修改 termination/reset 条件。
- 修改 `aerial_gym/rl_training/cleanrl/ppo_guidance.py`：
  - `RecordEpisodeStatisticsTorch.reward_keys` 增加 `visibility_reward`、`visibility_loss_penalty`、`contrib_visibility`，便于训练日志和策略池 manifest 记录。
- 验证通过：
  - `conda run -n aerialgym_v2 python -m py_compile aerial_gym/task/pursuit_guidance_task/pursuit_guidance_task.py aerial_gym/config/task_config/pursuit_guidance_task_config.py aerial_gym/rl_training/cleanrl/ppo_guidance.py`
  - `conda run -n aerialgym_v2 python aerial_gym/rl_training/cleanrl/export_pursuit_lidar_rollouts.py --max-steps 5 --num-episodes 1 --lidar-save-interval-steps 5 --output-dir /tmp/pursuit_visibility_reward_smoke`
- 根据用户反馈移除 `enable_visibility_reward` 开关，并将可见性几何改为从实际 robot `lidar_config` 读取；未启用 LiDAR 时 fallback 到 M3-like pursuit LiDAR 先验。
- 追加验证通过：
  - `conda run -n aerialgym_v2 python aerial_gym/rl_training/cleanrl/export_pursuit_lidar_rollouts.py --max-steps 5 --num-episodes 1 --lidar-save-interval-steps 5 --output-dir /tmp/pursuit_visibility_reward_config_smoke`
  - `conda run -n aerialgym_v2 python aerial_gym/rl_training/cleanrl/ppo_guidance.py --headless --play --play-steps 5 --checkpoint runs/PE_20260513_175323/latest.pth --target-asset-type target_x500`

## 2026-05-14（默认 target 切换为 x500）
- 按用户要求，默认目标无人机改为 `target_x500`，只影响 target asset，不修改追击无人机 robot/controller。
- 修改 `aerial_gym/config/env_config/pursuit_guidance_env.py`：
  - `target_quad: False`
  - `target_x500: True`
- 修改 `tests/smoke_pursuit_lidar_static_scene.py`：`--target-asset-type` 默认值从 `target_quad` 改为 `target_x500`。
- `ppo_guidance.py --target-asset-type` 保留为显式 override；不传该参数时使用 env config 默认的 x500。

## 2026-05-15（success-gated curriculum 与 visibility reward 渐进激活）
- 复盘 `runs/PE_20260514_221640` 的策略池 manifest：
  - stage 0 -> 1：update `218`，`global_step=401,817,600`，`transition_type=scheduled`，`succ_rate=0.0`。
  - stage 1 -> 2：update `436`，`global_step=803,635,200`，`transition_type=scheduled`，`succ_rate=0.0`。
  - 最后一条 update `650`：`success_rate=0.0`，`avg_min_relative_dist=34.00m`，`reach_rate_1m=0.0`。
- 修改 `aerial_gym/rl_training/cleanrl/curriculum/curriculum_controller.py`：
  - 默认课程切换模式改为 `success_stability`。
  - 每阶段需要连续 `10` 个 update 满足 `success_rate >= 0.95` 且结束 episode 数不少于 `64`，才进入下一阶段。
  - 保留 `step_budget` / `success_or_step_budget` 模式作为显式兼容模式，但当前 pursuit config 默认不再按固定 400M 步切换。
  - 课程 plan 改为两轮：第一轮无 visibility reward 的 5m -> 3m -> 1m；第二轮开启 visibility reward 后重新 5m -> 3m -> 1m。
  - 新增 `visibility_reward_weight_scale`，由 stage 固定档位设置；stage 3/4/5 默认分别为 `0.33/0.67/1.0`，同一 stage 内不再按 update ramp。
- 修改 `aerial_gym/config/task_config/pursuit_guidance_task_config.py`：
  - `reward.target_visibility_reward_active = False`，默认初始不计入新 reward。
  - `reward.visibility_reward_weight_scale = 0.0`，由 curriculum controller 动态设置。
  - 后续修正：删除 `task_config.curriculum` 中 pursuit PPO 课程学习参数，避免 task config 成为第二个课程参数入口。
- 修改 `aerial_gym/task/pursuit_guidance_task/pursuit_guidance_task.py`：
  - visibility reward 几何和日志字段保留。
  - 只有 `target_visibility_reward_active=True` 时才计算并计入 `contrib_visibility`；未激活时该项为 0，并清空 persistent loss 计数。
  - visibility contribution 实际为 penalty-only：`-visibility_reward_weight_scale * weight_visibility_loss_penalty * visibility_loss_penalty`。
- 修改 `aerial_gym/rl_training/cleanrl/ppo_guidance.py`：
  - 新增 curriculum visibility 阶段权重参数：`--curriculum-visibility-stage-weight-scales`；visibility pass 的 success threshold 自动继承最后一个 capture threshold，不再维护独立 threshold 参数。
  - 每个 PPO update 开始时调用 `curriculum_controller.begin_update(update)` 刷新 visibility weight scale。
  - policy-pool manifest 记录 transition mode、stable-success 配置、visibility stage scales、stage plan 和 visibility weight scale。
  - curriculum transition 日志显示 target visibility reward 是否被激活以及 visibility weight scale 切换。
  - 后续修正：`ppo_guidance.py` 解析后的 CLI/default 参数优先级最高，`task_config.curriculum` 不再在 parse 后覆盖 `args`；6 阶段默认通过 `--curriculum-target-visibility-reward-stage` 的 PPO 入口默认值直接启用。
  - resume 时 checkpoint 只恢复 curriculum 进度状态（stage、stage 起点、success streak、history），不再用 checkpoint 中旧的 SR 阈值、stable update 数、visibility 阶段配置覆盖当前 `ppo_guidance.py` 参数。
- 修改 `aerial_gym/rl_training/cleanrl/curriculum/curriculum_controller.py`：
  - 后续修正：去掉 controller 内 `getattr(args, ..., default)` fallback；controller 只消费 `ppo_guidance.py` 已解析出的课程字段，不再维护第二份默认参数入口。
- 新增 `tests/test_curriculum_controller.py`：
  - 覆盖固定步数已到但 SR 不达标时不切阶段。
  - 覆盖 1m 阶段达标后进入 visibility 5m 阶段，再继续到 visibility 1m final stage。
  - 覆盖 visibility weight scale 在 stage 3/4/5 之间按固定档位增强。
- 验证通过：
  - `conda run -n aerialgym_v2 python -m unittest discover -s tests -p 'test_curriculum_controller.py'`
  - `conda run -n aerialgym_v2 python -m unittest discover -s tests -p 'test_*.py'`
  - `conda run -n aerialgym_v2 python -m py_compile aerial_gym/rl_training/cleanrl/curriculum/curriculum_controller.py aerial_gym/rl_training/cleanrl/ppo_guidance.py aerial_gym/task/pursuit_guidance_task/pursuit_guidance_task.py aerial_gym/config/task_config/pursuit_guidance_task_config.py tests/test_curriculum_controller.py`
  - `git diff --check`

## 2026-05-18（3m visible-strike 与可恢复 visibility loss）
- 目标澄清：不是 3600 step 长时间跟踪，也不是任意 1 帧不可见即失败；当前 teacher 需要在快速追击过程中尽量保持目标可见，并保留短暂脱离视野后的恢复行为，为后续风险模型提供数据。
- 保持当前 5-stage curriculum：
  - `(5.0, 0.0)`
  - `(3.0, 0.0)`
  - `(3.0, 0.10)`
  - `(3.0, 0.20)`
  - `(3.0, 0.30)`
- 修改 `aerial_gym/config/task_config/pursuit_guidance_task_config.py`：
  - 默认 `success_threshold = 3.0`。
  - `weight_visibility_loss_penalty = 0.05`，从 `0.025` 小步提高。
- 修改 `aerial_gym/task/pursuit_guidance_task/pursuit_guidance_task.py`：
  - 将 LiDAR/FOV 可见性计算拆为 `_compute_target_detectable()`。
  - `success_mask = distance gate AND forward_alignment gate AND target_detectable`。
  - 不把中途不可见作为 termination；不可见时继续 episode 并累计 `visibility_loss_steps` 与 penalty。
  - 恢复可见时当前 `visibility_loss_steps` 清零。
  - 增加 episode visibility 统计：`visibility_episode_invisible_steps`、`visibility_episode_loss_segments`、`visibility_episode_recovered_segments`、`visibility_episode_max_loss_steps`。
- 修改 `aerial_gym/rl_training/cleanrl/ppo_guidance.py`：
  - 启动文案更新为 `5m,3m,3m+vis0.10,3m+vis0.20,3m+vis0.30`。
  - `RecordEpisodeStatisticsTorch` 透传 final target detectable 和 visibility recovery 统计。
  - TensorBoard/W&B/policy-pool manifest 增加 `final_target_detectable`、`visibility_loss_steps`、`visibility_episode_*` 指标。
- 修改 `aerial_gym/rl_training/cleanrl/export_pursuit_lidar_rollouts.py`：
  - `OUTPUT_SCHEMA_VERSION = 2`。
  - 新增 `compute_visibility_recovery_stats()`。
  - `.npz` 与 `index.jsonl` artifact 增加 `final_detectable`、`max_consecutive_invisible_steps`、`num_invisible_segments`、`mean_invisible_segment_steps`、`recovered_invisible_segments`、`recovery_rate_within_persist_steps`、`mean/max_recovery_steps`、`long_invisible_segments`、`detectable_rate_first_half/second_half`。
- 修改 `tests/test_curriculum_controller.py`：
  - 更新 5-stage curriculum 期望。
- 新增 `tests/test_pursuit_visibility_recovery.py`：
  - fake task 验证：不可见时即使到 3m 也不 success；恢复可见后 success；不可见计数和恢复段统计正确。
  - exporter 验证：实际调用 `write_episode_artifacts()`，检查 schema v2 和 recovery stats 落盘。
- 验证通过：
  - `conda run -n aerialgym_v2 python -m unittest discover -s tests -p 'test_curriculum_controller.py'`
  - `conda run -n aerialgym_v2 python -m unittest discover -s tests -p 'test_pursuit_risk_geometry.py'`
  - `conda run -n aerialgym_v2 python -m unittest discover -s tests -p 'test_pursuit_visibility_recovery.py'`
  - `conda run -n aerialgym_v2 python -m py_compile aerial_gym/task/pursuit_guidance_task/pursuit_guidance_task.py aerial_gym/rl_training/cleanrl/ppo_guidance.py aerial_gym/rl_training/cleanrl/export_pursuit_lidar_rollouts.py aerial_gym/rl_training/cleanrl/curriculum/curriculum_controller.py tests/test_curriculum_controller.py tests/test_pursuit_visibility_recovery.py`
  - `git diff --check`
  - `conda run -n aerialgym_v2 python aerial_gym/rl_training/cleanrl/export_pursuit_lidar_rollouts.py --dry-run`

## 2026-05-18（visibility 权重上调检查与计划同步）
- 检查结果：新成功条件已经生效，`success_mask` 包含 `target_detectable`，所以当前 `success_rate` 已经带终端可见要求。
- 基于 `runs/PE_20260518_133223/policy_pool/manifest.jsonl` 汇总后三个 visibility 阶段：
  - `vis=0.10`：平均 SR `0.998041`，平均 final detectable `0.998493`，平均 invisible steps `682.3`。
  - `vis=0.20`：平均 SR `0.999529`，平均 final detectable `0.999572`，平均 invisible steps `620.7`。
  - `vis=0.30`：平均 SR `0.999416`，平均 final detectable `0.999462`，平均 invisible steps `498.0`，最新 `451.5`。
- 结论：终端可见成功条件对当前模型偏简单，过程 visibility penalty 仍偏弱；支持将 visibility stage 权重提高到 `0.15/0.30/0.45` 作为下一轮 teacher 训练设置。
- 当前同步检查：
  - 核心逻辑方向合理：3m visible-strike + final detectable gate + recovery counters + visibility penalty 上调。
  - 发现阻塞同步项：`tests/test_curriculum_controller.py` 仍期望旧的 `0.10/0.20/0.30`，当前 `test_curriculum_controller.py` 失败 2 个断言。
  - 发现日志同步项：`ppo_guidance.py` 启动文案仍显示旧的 `3m+vis0.10,3m+vis0.20,3m+vis0.30`。
  - 设计提醒：默认 `success_threshold=3.0` 会影响非 curriculum 训练/评估；若当前目标是 3m visible-strike 则合理，若要 1m capture 需要单独恢复或拆分配置。
- 已运行检查：
  - `conda run -n aerialgym_v2 python -m unittest discover -s tests -p test_curriculum_controller.py`：失败，原因是测试期望未同步到 `0.15/0.30/0.45`。
  - `conda run -n aerialgym_v2 python -m unittest discover -s tests -p test_pursuit_visibility_recovery.py`：通过。
  - `conda run -n aerialgym_v2 python -m py_compile aerial_gym/rl_training/cleanrl/curriculum/curriculum_controller.py aerial_gym/task/pursuit_guidance_task/pursuit_guidance_task.py aerial_gym/rl_training/cleanrl/ppo_guidance.py tests/test_curriculum_controller.py tests/test_pursuit_visibility_recovery.py`：通过。
  - `git diff --check`：通过。
- 后续最小修复：同步 `tests/test_curriculum_controller.py` 的期望权重和 `ppo_guidance.py` 启动文案后，重跑 curriculum/visibility recovery tests，再启动新的 visibility-aware teacher 训练。

## 2026-05-19（APF 目标与双边博弈投稿强度判断）
- 结论：若只写“高速部分可观追击 + APF 目标轨迹生成”，创新强度不足以稳妥支撑 EAAI；APF 只能作为目标/场景分布，不应作为科学贡献本身。
- 当前强主线暂定为 `LiDAR/range-image 降信息 + 动作条件可观测性风险 + 风险调制 PPO`，并通过 B0-B4 证明风险变量对动作分布和优势估计的因果价值。
- 双机同样部分可观的追逃博弈不建议进入论文 1 主贡献；它会显著增加 self-play/non-stationary/opponent baseline 复杂度，且不自动增强方法创新。
- 若未来加入双边设置，应作为后续工作或扩展为 role-conditioned bilateral perceptual-risk framework，而不是简单把双方都改成部分可观。
- 对 APF 目标的最低要求：不要只用单一 APF 参数集；应做 APF 参数族、held-out APF、obstacle-aware APF 或至少一个非 APF scripted evader 的场景泛化检查，避免收益只在单一 APF 设置中成立。

## 2026-05-19（Baseline 阶梯收束：删除未来真值风险基线）
- 删除未来真值风险作为主表或 appendix 比较项；不再保留单独的不可部署风险上界 baseline。
- 动作条件风险调制 PPO 暂定为 B4，并明确为本文主方法方向。
- 当前主比较链条收束为 B0-B4：B0 全状态上界、B1 降信息无风险、B2 手工风险、B3 学习型状态风险、B4 学习型动作条件风险调制 PPO。
- 目的：避免额外真值风险输入削弱主叙事；B0 已承担 full-state upper-bound 角色，B4 的价值应直接体现在接近 B0 且显著优于 B1/B2/B3。

## 2026-05-19（主方法方向调整：风险调制 PPO）
- 基于 novelty-assessment 口径，简单 PPO 后处理容易被视为工程组合，创新强度不足。
- 暂定将 B4 主线调整为“动作条件可观测性风险调制 PPO 的动作分布与优势估计”。
- 当前不定型具体实现：actor risk gating、risk value/critic、risk-adjusted advantage、risk regularization 均作为后续候选。
- 执行动作应与 PPO logprob/ratio 对应动作保持一致，避免 credit assignment 不一致。

## 2026-05-19（阶段 1-5 复核结论）
- 结论：第 1-5 阶段不能沿用旧 complete 结论；旧状态是在 action correction/filter 和早期 sensor-proxy 语境下形成的，已经不能完全覆盖当前“动作条件风险调制 PPO”主线。完成本次 refresh 后，第 1/2/4/5 阶段已按当前窄 claim 重新同步，第 3 阶段仍待 B4 公式/接口定型。
- 第 0 阶段仍保持 complete：32D 全状态 B0、APF 目标路径、当前 pursuit 非 depth-driven 这些事实没有被新主线推翻。
- 第 1 阶段已完成定向复核：related work / novelty 新增或重审 risk-sensitive PPO、risk-aware actor-critic、actor risk gating、safe RL shield/filter、asymmetric actor-critic 等近邻，确认本文不是普通风险正则或安全过滤。
- 第 2 阶段已同步新主线：问题定义从“动作修正/动作过滤”改为“风险调制 PPO 的动作分布与优势估计”。
- 第 3 阶段改为 in_progress：`H/K/z_lidar` 已锁，但 B4 的 actor/critic/loss 结构尚未定型。
- 第 4 阶段已按 refresh 重审：B0-B4 主链保留，新增 PPO-Lagrangian、shield/filter、privileged critic、shuffled-action risk 与 LiDAR sparse-return sensitivity。
- 第 5 阶段已同步 deep-research 输出：新增 `perceptual-risk-modulated-ppo-eaai-refresh/` 作为当前有效版本，旧 `perceptual-risk-pursuit-review/` 标注为历史参考。

## 2026-05-19（Deep-Research Refresh：风险调制 PPO 新颖性与 EAAI Scope）
- 使用 `deep-research` 按 6 个 phase 新增当前有效输出：`deep-research-output/perceptual-risk-modulated-ppo-eaai-refresh/`。
- 生成文件：
  - `paper_db.jsonl`：41 篇。
  - `phase1_frontier/frontier.md`。
  - `phase2_survey/survey.md`。
  - `phase3_deep_dive/selection.md` 与 `deep_dive.md`：10 篇深读。
  - `phase4_code/code_repos.md`：7 个代码/工具生态条目。
  - `phase5_synthesis/synthesis.md` 与 `gaps.md`。
  - `phase6_report/report.md` 与 `references.bib`。
  - `cross_validation.md`：3 个 subagent 交叉验证摘要。
- 三个 subagent 独立结论一致：
  - 当前方向符合 EAAI 工程 AI scope。
  - 新颖性必须收窄为 “LiDAR/range-image reduced-information UAV pursuit 下的 action-conditioned observability-loss risk modulation of PPO”。
  - 不得声称通用 safe PPO、risk-sensitive PPO、risk-aware pursuit 或 action-conditioned risk gating 首创。
  - B4 必须对比 PPO-Lagrangian、shield/filter、risk concat、privileged critic、shuffled-action risk，并显著优于 B3。
  - `target_x500 + M3-like` 的 100/150/200m 返回点只能作为 sparse-return stress-test regime，不得写成真实长距部署保证。
- 已同步 `task_plan.zh.md`：
  - Phase 1 targeted refresh 标记为 complete。
  - Phase 2 problem definition 标记为 complete。
  - Phase 5 deep-research refresh 标记为 complete。
  - Phase 3 保持 in_progress，因为 B4 公式/网络接口尚未定型。
  - Phase 4 已按 refresh 重新收束为 complete；新增 ablation 已写入 `experiment_registry.zh.md`，后续进入实现阶段。

## 2026-05-19（Zotero 候选审计与重结论）
- 修正上一轮“批量扩充 Zotero pool”的处理方式：改为逐项审计用户提供论文是否服务当前主线。
- 用户提供 31 个候选；接受 11 个进入当前 active DB；18 个从当前主线排除；2 个直接映射为已有重复项。
- `paper_db.jsonl` 当前为 52 篇，而不是 70 篇。
- 直接重复项处理：
  - `33KIR7IV` 与已有 `[@shao2026peqmixer]` 重叠。
  - `F5LGULL9` 与已有 `[@chen2025open]` 重叠。
- 新增 `phase2_survey/zotero_candidate_audit.md`，记录 accepted/rejected 条目和每个条目服务或不服务主线的理由。
- 同步更新：
  - `phase1_frontier/frontier.md`
  - `phase2_survey/survey.md`
  - `phase3_deep_dive/selection.md`
  - `phase3_deep_dive/deep_dive.md`
  - `phase4_code/code_repos.md`
  - `phase5_synthesis/synthesis.md`
  - `phase5_synthesis/gaps.md`
  - `phase6_report/report.md`
  - `phase6_report/references.bib`
  - `cross_validation.md`
- 三个只读 subagent 交叉检查均确认：
  - EAAI scope 更强。
  - broad novelty 更弱。
  - 仍可辩护的 claim 是 `action-conditioned observability-loss risk-modulated PPO for LiDAR/range-image reduced-information UAV pursuit`。
  - B4 必须优于 B3、risk concat、PPO-Lagrangian、shield/filter、privileged critic 和 shuffled-action risk。
- 修正 `cross_validation.md` 中的旧状态：Phase 4 已因 ablation 写入实验注册表而可视为 design-complete；Phase 3 仍因 B4 公式/接口未定型保持 in_progress。
- 同步 planning files：
  - `task_plan.zh.md` 增加 Zotero 候选审计后的投稿约束和 Related Work 分组。
  - `findings.zh.md` 增加审计后重结论和保留强近邻表。
  - `decision_log.zh.md` 增加 D032。

## 2026-05-19（Related Work 四轴重组）
- 根据用户反馈，废弃“按材料来源或算法碎片堆列表”的外部比较方式。
- 新的 Intro / Related Work 顺序：
  1. 有限视场 / 有限可检测 / 传感器诚实追逐。
  2. UAV 追逃 / 拦截 / 在线规划 / CTBR。
  3. 约束下动作优化 / 风险门控控制。
  4. RL / privileged training / 工程 machinery。
- 明确 `[@chen2025open]` 未被排除：用户提供的 `F5LGULL9` 只是与 active DB 中已有 `[@chen2025open]` 重复。该论文现在作为 online planning / prediction-enhanced multi-UAV PE / CTBR 相关强近邻放入第二轴。
- 修正 citation key：`[@huh2026limited]` 统一为 active DB 中的 `[@huh2026limitedregion]`。
- 同步更新 `task_plan.zh.md`、`findings.zh.md`、`phase2_survey/survey.md`、`phase5_synthesis/synthesis.md`、`phase6_report/report.md` 和 `decision_log.zh.md`。

## 2026-05-19（必需引用清单）
- 新增 `deep-research-output/perceptual-risk-modulated-ppo-eaai-refresh/phase6_report/required_citations.md`。
- 清单按当前四轴 Related Work 结构组织：
  1. 有限视场 / 有限可检测 / 传感器诚实追逐。
  2. UAV 追逃 / 在线规划 / CTBR。
  3. 约束下动作优化 / 风险门控控制。
  4. RL / privileged training / 工程 machinery。
- 每条记录标注 `must/support`、为什么引用、是否已在 `paper_db.jsonl`、是否已在 `references.bib`。
- 当前建议正文主引用候选共 44 条；其中 `references.bib` 已有 19 条，另有 25 条需要补齐 BibTeX。
- 特别问题：`zhang2023gameofdrones` 当前在 plan 中被引用，但缺少 active DB 和 BibTeX 元数据；正式写作前必须核验并补齐。

## 2026-05-19（论文模板决策同步）
- 根据当前 `paper/main.tex` 已用 `IEEEtran` 编译通过的事实，将写作模板策略同步进 planning files。
- 当前论文草稿先按 `IEEEtran` / IEEE Transactions 双栏模板继续写作、引用和编译。
- 当前阶段不再寻找、安装或切换 `elsarticle.cls`；EAAI/Elsevier 官方模板只在投稿前最终格式化阶段处理。
- 已同步：
  - `task_plan.zh.md`：新增当前写作模板规则。
  - `decision_log.zh.md`：新增 D034，并写入已锁定写作规则。

## 2026-05-19（150-step recovery-aware visibility reward 实现）
- 基于 `runs/PE_20260519_103518` 结论，确认 `0.20/0.40/0.60` 单纯加权已经进入收益递减：final stage 仍有约 `340+` invisible steps 和 `300+` max loss steps。
- 实现新的 visibility penalty 形状：
  - 默认 `visibility_loss_persist_steps = 10`。
  - 新增 `visibility_recovery_horizon_steps = 150`。
  - 不可见时 penalty 从基础成本开始，超过 `K=10` 后继续线性增大，超过 `H=150` 后进一步增大，不再在 10 step 后饱和。
- 保持任务语义不变：
  - success 仍要求 `3m + forward_alignment + final target_detectable`。
  - 中途不可见不 hard reset，不把未来 risk label 直接塞进 reward。
- 新增训练/manifest 指标：
  - `visibility_episode_over_horizon_steps`
  - `visibility_episode_recovered_within_horizon_segments`
  - `visibility_episode_loss_area`
  - `visibility_episode_recovery_rate_within_horizon`
- curriculum controller 新增 visibility gate：
  - stage 2 推进要求 `max_loss <= 250` 且 `recovery_rate_within_horizon >= 0.80`。
  - stage 3 推进要求 `max_loss <= 180`、`invisible_mean <= 400` 且 `recovery_rate_within_horizon >= 0.90`。
- exporter recovery stats 增加 horizon 口径：
  - `recovery_rate_within_horizon_steps`
  - `over_horizon_invisible_segments`
- 交叉验证：
  - 第一轮 2 个 subagent 均发现 no-loss episode 的 recovery rate 被错误算为 `0.0`，已修为无 loss segment 时返回 `1.0`。
  - 第二轮 1 个 subagent 要求补生产默认 `K=10/H=150` 测试，已补。
  - 最终 2 个 subagent 复核均为 no blocking findings。
- 验证通过：
  - `conda run -n aerialgym_v2 python -m unittest discover -s tests -p test_pursuit_visibility_recovery.py`
  - `conda run -n aerialgym_v2 python -m unittest discover -s tests -p test_curriculum_controller.py`
  - `conda run -n aerialgym_v2 python -m py_compile aerial_gym/task/pursuit_guidance_task/pursuit_guidance_task.py aerial_gym/rl_training/cleanrl/ppo_guidance.py aerial_gym/rl_training/cleanrl/curriculum/curriculum_controller.py aerial_gym/rl_training/cleanrl/export_pursuit_lidar_rollouts.py tests/test_pursuit_visibility_recovery.py tests/test_curriculum_controller.py`
  - `git diff --check` targeted files

## 2026-05-20（PE_20260520_000809 visibility plateau 复盘与权重修正）
- 对比 `runs/PE_20260520_000809` 与旧 `runs/PE_20260519_103518`：
  - 新 run 末段仍停在 `stage=2, vis=0.20`；最新约 `success_rate=0.999`、`invisible_steps=403`、`max_loss=168`、`recovery_rate=0.77`、visibility contribution 绝对占比约 `14%`。
  - 旧 run final stage 为 `vis=0.60`；末段约 `invisible_steps=340`、`max_loss=305`、visibility contribution 绝对占比约 `17%`。
  - 因此新 run 更好地压低了最长连续丢失，但没有继续压低总不可见步数；低权重 stage 的 recovery gate 把训练长期卡在 `0.20`，导致后期 visibility pressure 下降。
- 设计结论：
  - 150-step recovery-aware penalty 不是完全无效，但目标偏向 `max_loss/recovery`，对总 `invisible_steps` 的持续压力不足。
  - `success_rate` 已不适合作为 visibility teacher 是否足够的主要指标；需要同时看 `invisible_steps`、`max_loss`、`recovery_rate` 和 `reward_contrib_ratio/visibility`。
  - 不改 success、FOV 几何、PPO loop 或 target motion；先做最小 reward/curriculum 调整。
- 实施修改：
  - 将 `weight_visibility_loss_penalty` 从 `0.05` 提高到 `0.08`，提高 stage 2 之后的 visibility penalty 占比。
  - 将 stage 2 的 `recovery_rate_within_horizon` 晋级 gate 从 `0.80` 放宽到 `0.75`，避免低权重阶段长期卡死；stage 3 更严格 gate 保持不变。
  - 同步 `test_curriculum_controller.py` 与 `test_pursuit_visibility_recovery.py`，新增 `contrib_visibility` 使用配置权重的集成断言。

## 2026-05-20（risk geometry 与 reward visibility 几何对齐）
- 修复 residual risk：离线 exporter / `risk_geometry.py` 原先只用机体系相对位置和对称 FOV 判断 target detectable，未对齐 reward 里的 `_compute_target_detectable()`。
- 最小实现：
  - `DetectionFrustum` 增加 sensor local position、sensor local quaternion、horizontal/vertical min-max FOV 边界。
  - `target_in_detection_frustum()` 先把 body-frame 相对位置转换到 sensor frame，再按 `forward > min_forward`、`range <= 150m`、FOV min/max 判断。
  - `export_pursuit_lidar_rollouts.py::frustum_from_lidar_cfg()` 从 lidar config 构造同 reward 口径的 150m detection frustum 和传感器外参。
- 不改变训练 reward、success 判定、FOV 配置或 PPO loop；只让后续离线 risk labels 与当前 reward visibility 逻辑一致。
- 验证通过：
  - `conda run -n aerialgym_v2 python -m unittest discover -s tests -p test_pursuit_risk_geometry.py`
  - `conda run -n aerialgym_v2 python -m unittest discover -s tests -p test_pursuit_visibility_recovery.py`
  - `conda run -n aerialgym_v2 python -m py_compile aerial_gym/task/pursuit_guidance_task/risk_geometry.py aerial_gym/rl_training/cleanrl/export_pursuit_lidar_rollouts.py tests/test_pursuit_risk_geometry.py tests/test_pursuit_visibility_recovery.py`

## 2026-05-21（PE_20260520_110828 B0 冻结与阶段状态更新）
- 使用 `data-analysis` 读取 `runs/PE_20260520_110828/policy_pool/manifest.jsonl`，确认该 run 是当前 visibility-loss penalty reward 下的最新完整训练记录。
- 训练最终保存点：
  - checkpoint：`runs/PE_20260520_110828/policy_pool/ppo_upd_001300_step_2396160000.pth`
  - update/global step：`1300 / 2,396,160,000`
  - progress fraction：`0.9984`
  - curriculum：`stage_idx=3`，`current_threshold=3.0`，`visibility_reward_weight_scale=0.4`
  - stage success streak：`20/100`
- 课程推进事实：
  - `update=349` 进入 `3m,vis0`。
  - `update=464` 进入 `3m,vis0.20`。
  - `update=754` 进入 `3m,vis0.40`。
  - 未进入最终 `3m,vis0.60`。
- 最后 10 个保存点统计：
  - `success_rate = 0.996887 ± 0.002338`
  - `reach_3m = 0.999265 ± 0.000601`
  - `final_target_detectable = 0.998244 ± 0.002047`
  - `visibility_episode_invisible_steps = 264.591 ± 6.793`
  - `visibility_episode_max_loss_steps = 120.250 ± 1.685`
  - `visibility_episode_over_horizon_steps = 7.083 ± 1.112`
  - `visibility_episode_recovery_rate_within_horizon = 0.901056 ± 0.008220`
- 指标解释已同步到 findings：
  - `invisible_steps` 是累计不可见步数。
  - `over_horizon_steps` 是连续不可见超过 `H=150` 后的严重尾部，不等同于累计不可见步数。
- 决策：冻结该 checkpoint 为 `B0: current vis-reward tracking baseline`。它适合作为后续 LiDAR rollout 和风险模型数据采集的 nominal strong-tracking teacher。
- 阶段状态更新：
  - 第 6.2 中“visibility-aware teacher 训练/筛选”标记为完成。
  - 第 6.2 中“在冻结 B0 上重新跑 LiDAR rollout export + label summary”和“dataset manifest”保持待办。

## 2026-05-23（Warp LiDAR 动态 target mesh stale bug 修复与清理）
- 根因确认：动态 pursuit 中 reward/exporter 的几何 visible/detectable 使用当前 `target_state`，但 Warp LiDAR raycast/semantic 使用未同步的旧 target mesh，导致 `detectable=True && target_pixel_count=0` 大量出现。
- 修复范围收窄：
  - 保留唯一 core fix：`EnvManager.render_sensors()` 在 `robot_manager.capture_sensors()` 前对 Warp env 做 mesh sync/refit。
  - 删除辅助诊断代码：terminal sensor snapshot hook、LiDAR center projection helper、center-ray semantic sample、aligned-detectable/debug-only exporter 字段、静态 smoke semantic bbox 检查。
  - 恢复 M3-like pursuit LiDAR 默认 `segmentation_camera=False`，避免长期把 debug-only semantic 输出带入主配置。
  - 删除辅助检查产物目录：`runs/lidar_smoke/*semantic*`、`*aligned_detectable*`、`*warp_sync_aligned_detectable*` 以及对应 `/tmp/pursuit_lidar_static_semantic_debug*`。
- 修复后诊断结论（产物已清理，摘要保留在 findings）：5 checkpoint 动态采样中 `visible|geometry=1.0`、`geom_pix0=0`，说明之前“几何可见但图像没有目标”的主因是 mesh stale，而不是 teacher 必然不可见。
- 下一步：重新用清洁 exporter 跑冻结 B0/policy-pool rollout，重算 `target_visible_frame_rate`、`visible|detectable`、可见帧平均像素和 recovery 指标，再判断是否需要 reward/curriculum 重训。

## 2026-05-23（三 agent 风险数据/模型/PPO 集成计划）
- 启动 3 个 agent 做只读方案 review：Agent A 负责数据采集与分布覆盖，Agent B 负责风险模型与标签设计，Agent C 负责 PPO 集成与验证路线。
- 统一结论：先把当前工作从可视化 smoke 推到正式 risk dataset，不再把 PNG/semantic image 当训练输入；正式数据只保存真实 stream 格式 `lidar_range_norm`，semantic/pixel/bbox 只作为 QA metadata。
- 统一结论：bool 脱视野标签只作为派生量，风险模型默认训练多头输出 `p_loss_H`、`severity_H`、`first_loss_offset`、`p_recover_H`。
- 统一结论：第一版风险模型采用离线冻结路线，先训练 K-frame CNN / CNN+GRU，再接入 frozen-risk PPO；不做 PPO 与 risk model 同步在线共训。
- 统一结论：在线数据采用 DAgger-like aggregation，但必须混合 offline policy-pool、current risk-PPO、action noise、hard cases、scripted stress cases，避免只在新策略窄分布上继续训练。
- 统一结论：lambda 先固定网格 `{0,0.02,0.05,0.1,0.2}` 做 scale 定标，再进入 adaptive dual lambda；约束指标使用 persistent loss / over-horizon risk。
- 本次只更新 planning files，未修改训练、环境或 exporter 代码。

## 错误
| 错误 | 处理 |
|------|------|
| 早期 guidance 保留弱中间路线 | 已改为单一 EAAI 强路线和显式失败条件 |
| 曾误用旧 conda 环境 `aerialgym` 规划验证命令 | 已更正为强制使用 `aerialgym_v2`；历史 `aerialgym/bin/python` 命令不再作为复现依据 |
| 沙箱内 `torch.cuda` device count 为 0，无法运行 Isaac Gym/Warp runtime smoke | 需要对 GPU/Isaac Gym runtime smoke 使用授权后的非沙箱命令 |

## 验证结果
| 检查 | 结果 | 状态 |
|------|------|------|
| `paper_db.jsonl` 解析 | 52 行 JSONL 全部可解析 | pass |
| 新近邻 key | `peng2025limitedvisual`、`li2026safeintent`、`li2025bearingonly` 已进入 DB/BibTeX/survey/synthesis/report | pass |
| 弱路线扫描 | active guidance 中不再保留旧的弱主线 | pass |
| `risk_geometry` 单测 | `/home/ubuntu/miniconda3/envs/aerialgym/bin/python -m unittest discover -s tests -p 'test_pursuit_risk_geometry.py'`，5 tests OK | pass |
| `risk_geometry` 编译检查 | `/home/ubuntu/miniconda3/envs/aerialgym/bin/python -m py_compile aerial_gym/task/pursuit_guidance_task/risk_geometry.py tests/test_pursuit_risk_geometry.py` | pass |
| E0-C1 数据分析 | 完整 TB event 序列分析，651 个记录点，10 个阶段过渡，对数线性收敛趋势 | pass |
| `ppo_guidance.py` 策略池保存语法检查 | `/home/ubuntu/miniconda3/envs/aerialgym/bin/python -m py_compile aerial_gym/rl_training/cleanrl/ppo_guidance.py` | pass |
| `aerialgym_v2` 环境检查 | `conda run -n aerialgym_v2 python -c "import sys, shutil; print(sys.executable); print(shutil.which('ninja'))"`，Python 和 Ninja 均解析到 `aerialgym_v2` | pass |

> 注意：上表中旧的 `aerialgym/bin/python` 记录是历史验证记录，不再作为后续命令模板。后续复现与新增验证必须使用 `aerialgym_v2`。
