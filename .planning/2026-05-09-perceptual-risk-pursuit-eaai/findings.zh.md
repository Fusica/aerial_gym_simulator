# 发现与锁定决策

本文件只保存对论文主线有长期价值的发现、证据和锁定约束。结构分为三类：
- **A. 当前置顶结论**：后续写作、实验和代码实现默认遵守的当前有效结论。
- **B. 证据归档：文献刷新与实验发现**：保留结论来源、实验事实和文献刷新过程，原则上只追加、不覆盖。
- **C. 工程归属与执行约束**：实现边界、策略池判断和当前下一步；若与 `task_plan.zh.md` / `progress.zh.md` 冲突，以最新同步后的 plan/progress 为准。

## A. 当前置顶结论

### 当前科学定位
首篇论文当前改为“privileged-to-LiDAR/range-image sensor-proxy transition”。此前 depth-only 路线已被用户在 2026-05-14 明确废弃为主传感器路线。

锁定机制是：

`LiDAR/range-image 降信息 + 动作条件可观测性风险预测 + 风险调制 PPO`

### 代码事实
- 当前 task config：`aerial_gym/config/task_config/pursuit_guidance_task_config.py`。
- 当前 task：`aerial_gym/task/pursuit_guidance_task/pursuit_guidance_task.py`。
- robot/control：`base_quad_root_link_control` + `thrust_bodyrate_control`。
- target baseline：`apf_escape`。
- 当前 32D 观测泄露目标相对位置、速度、closing geometry、墙体 clearance 和障碍物相对位置。
- 当前 robot config 默认不启用 camera。
- 强制运行环境：所有仿真、LiDAR/Warp sensor、Isaac Gym/gymtorch、PPO、rollout export 和风险数据采集命令必须使用 conda 环境 `aerialgym_v2`，不得使用旧环境 `aerialgym`。
- 可复现命令应优先写成 `conda run -n aerialgym_v2 python ...`；若直接调用 `/home/ubuntu/miniconda3/envs/aerialgym_v2/bin/python`，涉及 `ninja` 时必须同时设置 `PATH=/home/ubuntu/miniconda3/envs/aerialgym_v2/bin:$PATH`。
- 旧 depth/camera 路线中的仿真语义真值不属于 LiDAR/range-image 主线；在线策略和风险模型不得使用目标 ID、真值 target-hit 或人工 mask。
- 仓库 LiDAR 当前走 Warp sensor 路线：`use_warp=False` 时启用 LiDAR 会报错；Warp 路线当前禁止 camera 和 lidar 同时启用。LiDAR 主线需要单独的 robot/config/exporter/smoke-test 链路。
- 当前 target UAV 不算高保真多旋翼模拟：`apf_escape` 先生成 `target_command`，随后 task 内 `_old_lee_position_controller()` 计算 thrust/torque，并由 `_apply_target_controller()` 直接写 `target_force_tensor` / `target_torque_tensor`。该路径跳过分配矩阵、电机模型、ESC/prop 动态、单电机饱和和电机级噪声。

结论：现有策略是特权全状态 baseline，只能作为 B0 和标签生成器，不能作为主方法。

### 锁定问题陈述
如何利用特权全状态 rollout 学习短期可观测性风险，并在 LiDAR/range-image 传感器代理降信息条件下用动作条件风险调制 PPO 的动作分布与优势估计，从而恢复高速、大空间追逐决策质量？

### 新颖性判断
结论：**在收窄主张下具有中等偏强的新颖性，满足 EAAI 尝试条件；若写成通用 safe PPO / risk-aware pursuit / action-conditioned risk gating，则新颖性不足。**

不够新：
- full-state PPO 加一个风险标量
- 通用目标轨迹预测
- 有限 FOV reward shaping
- 通用 partial-observation MARL

足够支撑强 EAAI 尝试：
- 预测 observability-loss risk 而不是目标轨迹
- 特权 rollout 只用于标签和辅助监督
- 主实验在 reduced/LiDAR-only 或 LiDAR-primary 信息下进行
- 风险是动作条件的，并用于调制 PPO 的动作分布与优势估计

### 强近邻与区分点
| 论文 | 重合点 | 必须区分 |
|------|--------|----------|
| `[@huh2026limitedregion]` | limited detectable region、FOV reward、APF evader | 本文学习并校准 observability-risk，而非只做 reward/action design |
| `[@peng2025limitedvisual]` | limited visual field、search/pursuit/reacquisition MARL | 本文是 privileged-to-LiDAR/range-image bridge + 动作条件风险调制 PPO，而非 graph-attention MARL |
| `[@li2026safeintent]` | risk-gated hierarchical policy、安全投影、部分可观测 | 本文直接预测 observability-loss risk，并在 1v1 LiDAR/range-image sensor-proxy 下调制 PPO 的动作分布与优势估计 |
| `[@chen2025open]` | prediction-enhanced pursuit、CTBR deployment | 本文预测 target-loss risk 而非 target trajectory |
| `[@zhang2023gameofdrones]` | 目标预测与在线规划 | 本文单独评估风险变量的离线校准和在线控制作用 |
| `[@li2025bearingonly]` | sensor-compatible pursuit、target-loss robustness | 本文使用 LiDAR/range-image 风险学习和风险调制 PPO，而非 bearing-only filter |

### 最终贡献写法
1. **降信息追逐定义**：移除策略输入中的目标和环境真实值。
2. **动作条件可观测性风险学习**：从特权 rollout 学习 `p_lost` 和 `R_obs(s_red, u)`。
3. **风险调制 PPO**：动作条件可观测性风险暂定用于调制 PPO 的动作分布与优势估计；具体 actor/critic/loss 结构后续继续优化，不在当前阶段定型。
4. **因果实验阶梯**：B0-B4 分离全状态上界、降信息退化、启发式风险、学习型状态风险和动作条件风险调制。

### 失败条件
若出现以下情况，论文主张必须重写：
- learned risk 离线不优于 heuristic risk
- B3 在线不优于 B1/B2
- B4 只是通过过度保守换来 target retention，而非学到风险调制下的有效追逐策略
- 收益只在单一 APF 参数设置中存在
- 主线评估仍使用目标或环境真实值

### 写作规则
- 不说“首次 partial-observation pursuit”。
- 不说“完整 onboard visual pursuit”。
- 不把 B0 + risk scalar 描述成本文方法。
- 不把 bilateral self-play 放进论文 1 主贡献。
- 使用 “LiDAR/range-image sensor-proxy reduced-information pursuit” 作为当前主线，不再使用 “depth-only sensor-proxy” 作为主传感器声明。
- 旧 depth 专用代码链路已删除；后续不维护 pursuit depth config/smoke/exporter。
- 不把目标 ID、真值 target-hit、人工 mask 或仿真语义真值写成真实传感器可用输入；若后续使用 bbox/mask/target-hit，只能表述为额外 detector/cluster-tracker 的 noisy output，而非 LiDAR raw/range-image 主线输入。
- 论文 1 不再把 500m/100Hz 作为主实验目标；主线固定为 `target_x500 + M3-like` 前向高分辨率 LiDAR，并以 100m/150m/200m 均保留 2 个目标返回点的静态 smoke 结果作为传感器假设依据。100Hz 若出现，只能作为仿真同步或 ROI/局部更新假设单独标注。
- 不把 target 逃逸者写成高保真 motor-level UAV；只能写成 PhysX 刚体 + 高层位置控制/直接 force-torque 注入的规则逃逸目标。若要声称 target 高保真，必须新增 matched allocator/motor/drag/delay/identified actuator model。

## B. 证据归档：文献刷新与实验发现

以下内容保留为当前置顶结论的来源。文献/定位刷新放在前面，实验与工程发现放在后面；各块保留日期标签用于追溯，不作为当前结论的阅读入口。日期块可以互相修正，但不应直接覆盖删除，除非该证据已经被证明错误并在新日期块中说明。

### E0-C1/E0-C2 控制器对比实验发现（2026-05-11）

#### 实验矩阵
| 实验 | 运行 ID | 控制器参数 | scale_input | kOmega | torqueLimit |
|------|--------|-----------|-------------|--------|-------------|
| E0-C1 | `runs/PE_20260511_000603` | e1（用户修改） | [1.0, 3.0, 3.0, 1.5] | [1.2, 1.2, 1.0] | [2.4, 2.4, 0.2] |
| E0-C2 | `runs/PE_20260511_125039` | 仓库默认 | [1.0, 2.0, 2.0, 0.2] | [1.0, 1.0, 0.8] | [2.4, 2.4, 0.18] |

两者均为 1.2B 步全状态 PPO，无课程学习，所有其他超参一致。

#### 发现 1（修正）：更保守的控制器显著优于激进的控制器
- **事实：** E0-C2（更保守的默认参数）成功率达到 0.97%（最高 2.33%），而 E0-C1（更激进的 e1 参数）为 0%。默认参数下的最小距离（1.71m vs 2.40m）、1m 到达率（3.3% vs 0%）和 3m 到达率（99.6% vs 89%）均更优。
- **含义：** 激进的控制器限制适得其反。将 scale_input 从 [2.0, 2.0, 0.2] 放宽到 [3.0, 3.0, 1.5] 并未增强机动能力——它放大了探索空间，使梯度质量下降，导致策略更早陷入确定性困局。
- **行动：** 保持仓库默认控制器参数，不进一步放宽。考虑未来实验探索更窄的限制是否能进一步加速学习。

#### 发现 2：动作空间放大引发三重失败机制
- **事实：** E0-C1 的熵降至 -4.28（完全确定性），E0-C2 维持在 -1.71（仍保留一些探索）。E0-C1 的 pre-tanh OOB 为 61.6%，E0-C2 为 48.5%。E0-C1 的 tanh 饱和率为 6.4%，E0-C2 为 4.7%。
- **机制链：** 更宽的限制 → 更大的动作搜索空间 → 策略梯度中更多方差 → 更早的熵塌陷 → pre-tanh 动作漂移至更大值 → 更多 tanh 饱和 → 更差的梯度质量 → 无法学习精确最后接近 → 成功永远无法涌现。
- **含义：** "更大 = 更好"的控制器设计直觉对于 RL 是错误的。动作空间的正则化效果比动作空间的大小更重要。
- **行动：** 控制器限制应被视为一种隐式正则化器，而非严格的物理约束。选择能维持探索的最小可行限制。

#### 发现 3：成功涌现需要探索窗口与精度窗口对齐
- **事实：** E0-C2 的首次成功出现在 5.23 亿步（熵≈-0.54，演员标准差≈1.23，最小距离≈3.80m）。在这个时间点，策略仍保留显著探索能力，且已掌握进入 3.8m 的接近精度。两者交叉后，成功奖励开始提供持续的正向梯度。
- **含义：** 成功涌现不是一个改进量的问题，而是一个时机问题——探索必须在策略的接近精度足以达到成功阈值时仍然活跃。E0-C1 在接近精度窗口打开之前（~3 亿–4 亿步）就失去了探索能力。
- **行动：** 通过保持熵系数 ≥ 0.001（而非衰减至几乎为零）来主动管理探索窗口。课程学习通过将精度门槛向前移动来缩短窗口。这两者协同作用。

#### 发现 4（修正）：奖励结构不是阻断因素，但不理想
- **事实：** E0-C2 尽管时间惩罚（-17.93）+ 超时惩罚（-5.92）= 总固定惩罚 -23.85，仍能学习成功。成功奖励（+0.033，最高 0.084）相对于惩罚权重来说微不足道，但它仍然提供了可学习的梯度。
- **含义：** 之前关于"奖励结构导致成功无法学习"的结论太强——在正确的控制器设置下，即使微弱的成功信号也足够。但当成功信号弱时，控制器/探索设置成为决定性能的关键。
- **行动：** 课程学习仍然是 priority #1——在更容易的条件下增大成功频率，为策略创造更强的成功信号以锁定成功行为。

### E0-C3：手动 vs 自动课程学习对比实验（2026-05-13）

#### 实验矩阵
| 实验 | 运行 ID | 课程方式 | total_timesteps | early_stop | 阈值切换步数 |
|------|--------|---------|-----------------|-----------|------------|
| E0-C3-手动 | `runs/PE_20260506_181131` | 手动 resume + 改阈值 | 每次 400M（共三次） | True | ~400M, ~800M |
| E0-C3-自动 | `runs/PE_20260513_102027` | `--curriculum` 自动 | 1.2B（单次） | False（`--no-early-stop`） | ~400M, ~800M |

两者的 robot（`base_quad_root_link_control`）、controller 参数、阈值（5→3→1）、PPO 超参（lr=0.0003, γ=0.999, ent_coef=0.01→0, vf_coef=0.5, clip_coef=0.2）均相同。用户在旧 run 运行前已手动将 robot_name 改为 `base_quad_root_link_control`。

#### 实验细节
- **手动课程方式：** 分三次运行。每次 `total_timesteps=400M`，每阶段结束后手动修改 success_threshold，加载 latest.pth 并 resume。entropy 每阶段重新从 0.01 线性衰减到 0。
- **自动课程方式：** 一次运行 `total_timesteps=1.2B`，`curriculum_stage_step_budgets=[400M, 400M]`。entropy 在整个 1.2B 步内从 0.01 线性衰减到 0。

#### 手动 run 的 SR 轨迹
```
~400M (5m阈值): SR=98.2%, min_dist=4.67, reach1m=0.1%
~800M (3m阈值): SR=99.6%, min_dist=2.75, reach1m=0.2%  
~802M (刚切1m): SR=7.2%  ← 崩溃
~1200M(1m阈值): SR=77.2%, min_dist=0.98, reach1m=84.3%
```
每次阈值切换都造成 SR 瞬时暴跌（98%→65% 在 5→3m，99.6%→7.2% 在 3→1m），其中 3→1m 的切换后 SR 恢复极慢。

#### 自动 run 的最终状态
```
~1200M (1m阈值): SR=100%, reach_rate_1m=100%, min_dist=0.958, avg_len=1107, collisions=0
```
自动 run 在同样步数下达到了完全的成功率，且没有碰撞。

#### 关键发现：`total_timesteps` 通过 entropy 衰减速率影响性能

**这是主要原因。** 不是 robot name（两者相同），不是控制器参数（两者相同），不是 PPO 超参（两者相同）。

机制：
- **手动方式（total_timesteps=400M × 3）：** 每阶段 entropy 从 0.01 重新开始衰减。在 stage 3（1m 阈值）开始时，entropy=0.01，接近满探索水平。1m 精度阶段需要的是精细控制，过高的 entropy 噪声使得策略难以收敛到高精度动作，导致 SR 从 99.6% 崩溃到 7.2%，400M 步后仅恢复到 77.2%。
- **自动方式（total_timesteps=1.2B）：** entropy 在整个训练中均匀衰减。stage 2 开始时（~400M 步）entropy≈0.0067，stage 3 开始时（~800M 步）entropy≈0.0033。在需要高精度的 1m 阶段，entropy 已经自然降低到适合精细调优的水平，策略可以在已有精度基础上渐进改善。

**直观理解：** 每个 stage 重设 entropy=0.01 相当于每进入新难度就"失忆"所有已学到的精细控制，重新开始随机探索。1.2B 统一衰减则让 entropy 与难度自然匹配——在简单阶段探索、在困难阶段收敛。

#### 次要因素

1. **`--no-early-stop`：** 自动 run 禁用 early stop。在 stage 3 的 SR 暴跌后，best_score 仍保存着上一阶段的 99.6%，early_stop 的 patience 计数器可能持续累积，存在提前停止训练的风险。

2. **训练连续性：** 自动 run 不需要 checkpoint save/load，优化器状态和 obs_rms 在阈值切换时自然延续。手动 resume 虽然也恢复这些状态，但引入了重启开销。

#### 对课程学习设计的启示
- `total_timesteps` 不应每阶段重新设定——应与 `curriculum_stage_step_budgets` 的总和一致。
- 课程学习本身只做阈值自动切换，不涉及探索相关修改。但**熵衰减的全局步长感知**是课程学习打包配置的一部分。
- 后续课程学习配置只从 `ppo_guidance.py` 的 CLI/default 参数进入；不再通过 `task_config.curriculum` 维护第二份 thresholds/budgets/patience 默认值。

#### 对论文方法的影响
- E0-C2 证明 B0（全状态 PPO）在默认控制器下可以学习成功 → B0 上界可以建立。
- 控制器参数不是优化问题，而是正则化问题——默认仓库值很可能已经是最优点。
- 课程学习 + 默认控制器 + 保持熵系数应使 B0 能够在合理训练时间内达到 ≥ 0.8 成功率。
- B0-B4 阶梯在课程学习就绪后可以开始填写。

### 2026-05-14 LiDAR OS2-64 static smoke 结果

- 新增 `tests/smoke_pursuit_lidar_static_scene.py`，使用 `base_quad_root_link_control_with_lidar`、`use_warp=True`、headless、小静态场景，仅保留 target quad，不启用 segmentation。
- 当前 `PursuitForwardOS2_64_LidarConfig` 为严格 OS2 风格：`64 x 512`、水平 `360°`、垂直 `-11.25° ~ +11.25°`、`max_range=200m`、`segmentation_camera=False`。
- 正前方 body +X 放置结果：1m 有 408 个 return pixel，5m 有 8 个，10m 有 4 个，25/50/100/150/200m 均为 0。
- ray-aligned 诊断结果：将目标中心放到最近中心射线后，1m 有 419 个，5m 有 15 个，10m 有 4 个，25/50/100/150/200m 仍为 0。
- 结论：当前失败不是 200m `max_range` 的截断问题，而是 `64 x 512` OS2-style range image 对小型无人机目标的角分辨率/目标截获像素不足。该 config 不能直接支撑 100m-200m 乃至 500m 小目标风险模型输入。
- 输出文件：
  - `/tmp/pursuit_lidar_static_smoke_os2_64/summary.json`
  - `/tmp/pursuit_lidar_static_smoke_os2_64_ray_aligned/summary.json`
  - 每档距离保存 `*_range_raw.png`、`*_near_bright.png`、`*_return_mask.png`、`*_target_contrast.png`。

### 2026-05-14 高分辨率前向 LiDAR 自定义配置

- 市面公开规格审计后，100-200m 档可参考两类前向/长距高分辨率方案：
  - Hesai AT128P-like：`128` channels、水平 `120°`、垂直 `25.4°`、远场水平 `0.1°`、垂直 `0.2°`、约 `210m @ 10%`。
  - RoboSense M3-like：水平 `120°`、垂直 `25°`、`0.05° x 0.05°` ROI 角分辨率、等效 `500` 线、`300m @ 10%`。
- 当前 `pursuit_forward_lidar_config.py` 只保留 `PursuitForwardM3_120x25_UltraHighResLidarConfig`：`501 x 2401`，约 `0.05° x 0.05°`，作为论文 1 主线 pursuit LiDAR。
- `base_quad_root_link_control_with_lidar` 默认接入 M3-like ultra-high-res profile；AT128P-like 因 100m/150m 仍无目标返回点，已不再作为可选 config 保留。
- smoke 结果：
  - AT128P-like 正前方：5m=142、10m=31、25m=4、50m 起为 0。
  - AT128P-like ray-aligned：50m=2、100/150/200m=0。
  - M3-like 正前方：5m=1115、10m=262、25m=43、50m=8、100m=2、150m=2、200m=0。
- 结论：对于当前 target quad 模型，100-150m 小目标可见性至少需要接近 `0.05°` 级别的前向高密度扫描；即便如此，100m/150m 也只有 2 个有效点，不能声称远距目标特征丰富。M3-like 配置适合 feasibility probe 和小 env 数采集，不能默认承诺高 env 数 PPO 或 100Hz full-frame 训练吞吐。

### 2026-05-14 目标 URDF 候选与 LiDAR 适配

- 当前仓库可用的多旋翼/近似多旋翼 URDF 包括：`quad/quad.urdf`、`quad/model.urdf`、`x500/model.urdf`、`lmf1/model.urdf`、`lmf2/model.urdf`、`octarotor/octarotor.urdf`。本 checkout 未发现旧记忆中另一份仓库提到的 `F450.urdf`。
- 当前 pursuit target 默认不是最小 `quad.urdf`，而是 `resources/robots/quad/model.urdf`；其 visual 尺寸约 `0.40 x 0.40 x 0.07m`，质量约 `1.94kg`。
- `x500/model.urdf` 是最合适的“大一点但仍是常规 quad”候选：visual 约 `0.448 x 0.448 x 0.07m`，质量约 `1.656kg`。当前 `target_x500` 只覆盖 URDF 路径和 `controller_mass`，速度上限继续继承 pursuit `target_quad` 的 `max_angular_velocity=20/max_linear_velocity=40`，避免把目标几何变化和物理速度上限变化混在一起。
- `octarotor/octarotor.urdf` 的 visual 约 `0.398 x 0.398 x 0.392m`，比 x500 更有体积；但它不是同一类 quad target，作为主设定会改变任务对象语义，已从当前 pursuit target 分支删除。
- 已将 `target_x500` 设置为 pursuit 默认目标资产；`target_quad` 仅作为显式对照保留。LiDAR static smoke 支持 `--target-asset-type target_quad/target_x500` 切换。
- 关键 smoke 结果（return pixel count）：
  - `target_quad + AT128P-like`：25m=4，50/100/150/200m=0。
  - `target_quad + M3-like`：50m=8，100m=2，150m=2，200m=0。
  - `target_x500 + AT128P-like`：25m=6，50/100/150/200m=0。
- `target_x500 + M3-like`：50m=12，100m=2，150m=2，200m=2。
- 结论：主线 sensor/target 组合已锁定为 `target_x500 + M3-like`。该组合在 100m/150m/200m 均保留 2 个目标返回点，可作为论文 1 风险模型输入和可见性几何的固定实验约束；AT128P-like 与 OS2-64 仅保留为历史失败对照，不再作为候选配置继续讨论。
- 工程修正：target 侧 Lee controller 不能继续用 pursuer `robot_mass` 作为 target thrust 标定。`target_quad` 质量约 `1.94kg`，`target_x500` 约 `1.656kg`；已为 target asset 添加 `controller_mass`，并让 target controller 使用该质量换算 hover/加速度力。

### 2026-05-14 策略池 LiDAR rollout exporter 与可见性缺口

- 新增独立 exporter：`aerial_gym/rl_training/cleanrl/export_pursuit_lidar_rollouts.py`。
- 默认读取 `runs/PE_20260513_175323/policy_pool/manifest.jsonl`，选择 13 个 checkpoint：`1, 140, 150, 160, 170, 218, 220, 436, 440, 450, 460, 470, 640`。
- 默认配置为 `base_quad_root_link_control_with_lidar`、`target_x500`、`use_warp=True`、`num_envs=13`、`num_episodes=1`、`max_steps=3600`。
- 输出目录：`runs/lidar_smoke/PE_20260513_175323_x500_13ckpt_smoke/`。
- 完整 smoke 输出：
  - `13` 个 `.npz` episode 文件。
  - `5397` 个 PNG 人工核查图。
  - `index.jsonl` 共 `13` 行，`.npz` 和 PNG 路径均可打开。
  - `obs/action/detectable/observability_loss_label/first_loss_offset` 长度均与轨迹步数对齐。
  - 输出体积约 `170M`。
- 完整 smoke episode 结果：
  - update `0001`：`1596` steps，`collision`。
  - update `0140` 到 `0450`：`3600` steps，`timeout`。
  - update `0460`：`2426` steps，`success`。
  - update `0470`：`2295` steps，`success`。
  - update `0640`：`1211` steps，`success`。
- 关键可见性发现：当前全状态 PPO teacher 即使能成功捕获，也没有学到持续把目标保持在 LiDAR FOV 内。
  - update `0460` 成功轨迹可见率约 `29.0%`，observability-loss label rate 约 `86.6%`。
  - update `0470` 成功轨迹可见率约 `35.6%`，observability-loss label rate 约 `72.1%`。
  - update `0640` 成功轨迹可见率约 `13.4%`，observability-loss label rate 约 `99.3%`。
- 代码根因：当前 reward 只有很弱的 `forward_alignment` shaping，`weight_alignment=0.005`；success 只要求最终 `forward_alignment >= 0.95`，不要求目标在真实 LiDAR frustum 中持续可见。
- 结论：现有 B0 checkpoint 可以作为追击上界和失败/边缘数据来源，但不是合适的可见性保持 teacher。第 6.2 需要新增 visibility-aware teacher reward，使用真实 LiDAR config 的 frustum 生成 soft shaping，而不是立刻把不可见作为 hard done。
- 训练原则：visibility reward 权重可以提升到与 progress 同一量级，以明显改变 teacher 行为；但不应做过强终止，否则会丢失从可见到不可见、从临界到恢复的风险模型训练样本。

### 2026-05-14 visibility-aware teacher reward 实现

- 最小修改范围：
  - `aerial_gym/config/task_config/pursuit_guidance_task_config.py`
  - `aerial_gym/task/pursuit_guidance_task/pursuit_guidance_task.py`
  - `aerial_gym/rl_training/cleanrl/ppo_guidance.py`
- 可见性保持不设置开关，成为 pursuit task 的默认 reward 语义；即使后续不显式添加传感器，teacher 也需要长期保持目标在前向可检测区域内。
- 新增 reward 权重参数：
  - `visibility_loss_persist_steps = 10`
  - `weight_visibility = 0.5`
  - `weight_visibility_loss_penalty = 0.25`
- 可见性几何不再维护一份手写 FOV/range 常量：若当前 robot 启用 LiDAR，则直接读取 robot `lidar_config`；若未启用 LiDAR，则默认使用当前 pursuit 主线的 `PursuitForwardM3_120x25_UltraHighResLidarConfig` 作为任务先验。
- 对当前 `base_quad_root_link_control_with_lidar`，reward frustum 与 M3-like 前向 LiDAR 主线同源：水平 `[-60°, 60°]`、垂直 `[-12.5°, 12.5°]`、`max_range=300m`、`min_range=1m`。
- reward 还会使用 LiDAR config 的安装均值位姿；当前 M3-like 配置为位置 `[0.10, 0.0, 0.03]`、姿态 `[0,0,0]`，因此比早期按机体原点计算更贴近真实传感器坐标。
- reward 形式：
  - `visibility_reward`：连续 FOV/range margin，范围 clamp 到 `[-1, 1]`。
  - `visibility_loss_penalty`：目标不在 frustum 内时按连续不可见步数递增，`K=10` 后达到最大值。
  - `contrib_visibility = 0.5 * visibility_reward - 0.25 * visibility_loss_penalty`。
- 没有新增 hard done，也没有把不可见直接作为 termination；`done_success/done_timeout/done_far/done_collision` 逻辑保持不变。
- `RecordEpisodeStatisticsTorch.reward_keys` 已加入 `visibility_reward`、`visibility_loss_penalty` 和 `contrib_visibility`，后续训练日志和 policy-pool manifest 可以看到该项量级。
- 验证：
  - `conda run -n aerialgym_v2 python -m py_compile aerial_gym/task/pursuit_guidance_task/pursuit_guidance_task.py aerial_gym/config/task_config/pursuit_guidance_task_config.py aerial_gym/rl_training/cleanrl/ppo_guidance.py`
  - `conda run -n aerialgym_v2 python aerial_gym/rl_training/cleanrl/export_pursuit_lidar_rollouts.py --max-steps 5 --num-episodes 1 --lidar-save-interval-steps 5 --output-dir /tmp/pursuit_visibility_reward_smoke`
- 后续训练目标：获得一个比原 B0 更适合数据采集的 visibility-aware teacher，不把它作为论文主方法；它仍是数据源/上界/teacher 机制。

### 2026-05-15 visibility-aware run 失败复盘与课程修正

- 复盘 `runs/PE_20260514_221640/policy_pool/manifest.jsonl`：
  - 共 `68` 条策略池记录。
  - stage 0 -> 1 发生在 update `218`、global_step `401,817,600`，transition type 为 `scheduled`，切换时 `succ_rate=0.0`。
  - stage 1 -> 2 发生在 update `436`、global_step `803,635,200`，transition type 为 `scheduled`，切换时 `succ_rate=0.0`。
  - 最后一条 update `650` 仍为 `success_rate=0.0`，`avg_min_relative_dist=34.00m`，`avg_final_forward_alignment=0.939`，`reach_rate_1m=0.0`。
- 结论：这次失败不是“需要更快进入 1m”，而是课程推进条件错误。当前 fixed-step 逻辑会在 SR 仍为 0 时强行增加难度，visibility reward 又从第一阶段直接加入，使策略同时面对追近、捕获门槛和长期可见性约束。
- 修正原则：
  - 课程推进应由连续 update 的真实 `success_rate` 达标触发，而不是固定 400M 步触发。
  - 默认门槛：当前阶段 `success_rate >= 0.95` 且连续 `10` 个 PPO update 达标。
  - 初始 5m/3m/1m 阶段不计入 visibility reward，让策略先恢复 capture skill。
  - 1m capture 稳定达标后，不直接进入最难的 1m+visibility；应开启 visibility reward 后重新走 5m -> 3m -> 1m。
  - visibility reward 权重不应在同一 stage 内持续 ramp；应在 stage 3/4/5 使用固定低/中/满三个档位，避免同一难度阶段内优化目标漂移。

### 2026-05-18 3m visible-strike teacher 与恢复型可见性统计

- 新的 teacher 语义不再是“3600 step 长时间跟踪”，而是 **快速进入 3m 打击终点，并在追击过程中尽量保持/快速恢复目标可见性**。
- 硬件约束改变了最终成功定义：前向 LiDAR 有 `min_range=1m` 且安装点有前向偏移，1m 终点会和近场不可见冲突；因此当前任务默认 success threshold 改为 `3m`。
- 当前课程固定为 5 阶段：`(5.0,0.0)`、`(3.0,0.0)`、`(3.0,0.10)`、`(3.0,0.20)`、`(3.0,0.30)`。visibility 阶段只增强 penalty，不再改变成功距离阈值。
- success 语义更新为：`relative_dist <= success_threshold` 且 `forward_alignment >= success_forward_alignment_cos` 且 `target_detectable=True`。因此“最后冲进 3m 但目标不在 LiDAR/FOV 内”不再拿 success bonus，也不会作为 success done。
- 中途不可见不是 hard failure：不可见时 `visibility_loss_steps` 连续累计，恢复可见时当前 loss 计数清零；这保留了“脱离视野后如何恢复”的轨迹，对后续 risk model 是必要数据。
- 长时间不可见当前选择“强惩罚/显式统计”，不是立即 reset：`visibility_loss_penalty` 在 `visibility_loss_persist_steps=10` 后饱和，权重从 `0.025` 小步提高到 `0.05`。
- 新增 episode 级 recovery 指标后，训练日志和策略池 manifest 可以区分：
  - final detectable 是否满足；
  - episode 内累计不可见步数；
  - 不可见段数量；
  - 恢复段数量；
  - 最长连续不可见长度。
- exporter schema v2 增加恢复质量指标：
  - `max_consecutive_invisible_steps`
  - `num_invisible_segments`
  - `mean_invisible_segment_steps`
  - `recovered_invisible_segments`
  - `recovery_rate_within_persist_steps`
  - `mean_recovery_steps`
  - `max_recovery_steps`
  - `long_invisible_segments`
  - `detectable_rate_first_half` / `detectable_rate_second_half`
- 这些指标用于区分“高可见且能快速恢复”的轨迹与“最后成功但中途长时间丢视野”的轨迹。
- 验证补充：新增 `tests/test_pursuit_visibility_recovery.py`，用 fake task 覆盖 `_compute_reward_and_dones()` 的 final-detectable gate 和 recovery counter，用临时目录覆盖 `write_episode_artifacts()` 的 schema v2 recovery 字段落盘。

### 2026-05-18 visibility 阶段权重上调依据

- 当前成功条件已经生效：`success_mask = distance gate AND forward_alignment gate AND target_detectable`，因此 `success_rate` 已经不是纯距离成功，而是终端可见成功。
- 第三个阶段开始加入的是过程 visibility penalty，而不是首次加入终端可见成功条件。终端 `target_detectable` 条件对当前模型偏简单，因为成功时已经要求 `forward_alignment >= 0.95`，天然容易让目标回到前向视野。
- `runs/PE_20260518_133223/policy_pool/manifest.jsonl` 支持上调 visibility 权重：
  - stage 2：`vis=0.10`，平均 `success_rate=0.998041`，平均 `final_target_detectable=0.998493`，平均 invisible steps 约 `682.3`。
  - stage 3：`vis=0.20`，平均 `success_rate=0.999529`，平均 `final_target_detectable=0.999572`，平均 invisible steps 约 `620.7`。
  - stage 4：`vis=0.30`，平均 `success_rate=0.999416`，平均 `final_target_detectable=0.999462`，平均 invisible steps 约 `498.0`，最新约 `451.5`。
- 现象解释：当前 teacher 已经学会“最终可见并成功”，但还没有学会“过程中持续可见”。因此提高过程 visibility penalty 是合理的下一步。
- 当前建议档位：将 visibility stages 从 `0.10/0.20/0.30` 提到 `0.15/0.30/0.45`，保留 3m visible-strike 和可恢复轨迹，不把中途不可见改为 hard failure。
- 风险控制：新权重仍需用训练验证。如果 `success_rate`、`reach_rate_current_threshold` 或 collision/timeout 分布恶化，需要回退到更低档位或延长阶段稳定条件。
- 实现检查发现的同步问题：
  - `tests/test_curriculum_controller.py` 仍期望旧的 `0.10/0.20/0.30`，当前会失败。
  - `ppo_guidance.py` 启动文案仍显示旧权重，需同步到 `3m+vis0.15,3m+vis0.30,3m+vis0.45`。
  - 默认 `success_threshold=3.0` 是 3m visible-strike 语义的一部分；如果后续需要 1m capture，对非 curriculum 训练和默认评估会产生语义变化。

### 2026-05-19 Related Work 范围重检与 Deep-Research Refresh

当前有效输出目录：`deep-research-output/perceptual-risk-modulated-ppo-eaai-refresh/`。本次 05-19 research refresh 的目的是重新确认本文相关工作范围、EAAI scope、创新边界和必须比较的近邻。

综合结论：
- **EAAI scope 匹配。** Engineering Applications of Artificial Intelligence 接收 robotics、perception、real-time intelligent automation、safety/reliability 和 UAV/RL 工程应用；但摘要必须清楚区分 AI contribution 与 engineering application。
- **可投稿 claim 必须收窄。** 可写为 “action-conditioned observability-loss risk for LiDAR/range-image reduced-information UAV pursuit, used to modulate PPO”。不能写成 “first safe PPO”、 “first risk-aware pursuit”、 “first action-conditioned risk gating under partial observability”。
- **广义创新更弱，主线更窄。** 不能把高速 UAV 追逃、DRL 追逃、多机追逃、敏捷四旋翼追逃、sensor-based capture 或 shielded high-speed flight 写成本文创新。唯一可辩护核心仍是 `LiDAR/range-image reduced-information UAV pursuit` 下的 `action-conditioned observability-loss risk modulation of PPO`。
- **传感器 claim 需要降级。** `target_x500 + M3-like` 的 100/150/200m 返回点应写为 sparse-return stress-test regime，不得写成真实长距 LiDAR 部署保证。必须报告 return count、occupancy、dropout、longest invisible streak 和传感器扰动敏感性。

Related Work 必须按论文科学约束组织：
1. **有限视场 / 有限可检测 / 传感器诚实追逐。** 覆盖 limited FOV、limited detectable region、target loss/reacquisition、bearing-only/vision-compatible capture。核心不是“有限感知追逐首创”，而是将 target observability loss 建模为动作条件、可校准风险。
2. **UAV 追逃 / 拦截 / 在线规划 / CTBR。** `[@chen2025open]` 必须作为强近邻；用户提供的 `F5LGULL9` 与 active DB 中 `[@chen2025open]` 对应同一方向。本文不声称 online planning、prediction-enhanced PE 或 CTBR 部署新。
3. **约束下动作优化 / 风险门控控制。** 这是 B4 的主要方法比较家族，必须对应 PPO-Lagrangian、Recovery RL、shield/filter、risk concat、privileged critic、shuffled-action risk 等对照。
4. **RL / privileged training / 工程 machinery。** PPO、MARL、curriculum、asymmetric actor-critic、EAAI UAV RL 和 LiDAR policy 只作为实现/venue/scope 支撑，不作为科学核心。

必须保留的比较压力：
- **高压近邻类别：** limited-FOV UAV pursuit、Recovery RL/action-conditioned safety critic、CPO/PPO-Lagrangian、shielded RL、asymmetric actor-critic、EAAI reduced-measurement DRL guidance。
- **Zotero/tag pool 中服务主线的代表性近邻：** `[@yan2024lsrctd3]`、`[@yang2025rlpereview]`、`[@zhao2024autonomousuavpe]`、`[@xiang2025cihrl]`、`[@roncero2025agilecontrollers]`、`[@giral2026intercept]`、`[@luo2024improvedmadrl]`、`[@tan2026scalablefixedwing]`、`[@zhang2026safetyshieldedflight]`、`[@feroskhan2024multipursuitevasion]`、`[@mavcapturingmav2024]`。这些用于真实比较压力和 Related Work 分组，不作为单独分支扩展论文主线。
- **B4 成立条件：** 必须显著优于 B3 `p_lost(s_red)`、risk concat、PPO-Lagrangian 和 shield/filter baseline；否则会被认为只是已有 safe RL 范式的应用。
- **新增必须对照：** risk concat only、PPO-Lagrangian using same risk as cost、shield/filter using same risk as post-hoc action intervention、privileged critic baseline、shuffled-action risk、matched-state action-risk ranking、LiDAR sparse-return/dropout sensitivity。

结尾差异句：已有工作分别覆盖有限可检测、UAV 追逃/在线规划和 safe/risk-aware action optimization；本文的缺口是 **在 sensor-honest LiDAR/range-image reduced inputs 下，学习动作条件可观测性丢失风险并将其调制 PPO**。

## C. 工程归属与执行约束

### 实现归属
- `risk_geometry.py`：可检测区域几何和标签。
- `risk_dataset.py` / `risk_offline.py`：rollout 存储、manifest、离线标签。
- `risk_model.py` / `risk_training.py`：LiDAR/range-image encoder、risk heads、校准。
- `risk_policy.py`：降信息观测和风险注入。
- `risk_closed_loop.py`：风险调制 PPO 或相关闭环风险机制。
- `risk_diagnostics.py`：离线和在线指标。
- `pursuit_guidance_task.py`：只做编排。

### 第 6.2 策略池设计判断
- 当前默认 PPO 设置下，`num_envs=512`、`num_steps=3600`，因此 1 个 PPO update = 1,843,200 global steps。
- 1.2B 自动课程学习约为 651 updates；已保存阶段切换点为 update 218（约 401.8M steps）和 update 436（约 803.6M steps）。
- 为了直接重训时捕获成功率从 0 到 1 的快速爬升段，默认中间 checkpoint 间隔应设为 **每 10 updates 保存一次**，即约 18.4M global steps。
- 如果只允许更稀疏保存，最多不应超过 20-25 updates（约 36.9M-46.1M steps）；100M 级别保存会错过关键跃迁段。
- checkpoint 选择不应只按训练时 `success_rate`，因为课程阶段的成功阈值不同。后续应对候选 checkpoint 用统一评估协议重算 `reach_rate_1m/3m/5m`、`success_rate`、`min_relative_dist`、目标丢失标签覆盖率和动作分布。
- 策略池更准确地应称为行为策略池：PPO checkpoint 是主体来源，但还需要少量启发式/规则/噪声行为用于覆盖失败、边缘、离视野和动作条件风险样本。这些来源只服务数据覆盖，不构成论文方法轴。
- 已在 `ppo_guidance.py` 中添加训练期策略池保存：`runs/<run>/policy_pool/ppo_upd_*.pth` 保存 full training state，`policy_pool/manifest.jsonl` 同步 checkpoint 路径、update/global_step、课程阶段、成功率/reach rate、done 分布、reward 摘要和动作分布诊断，便于后续筛选策略池。

### 当前下一步
1. 使用 5-stage visible-strike curriculum 重新训练 visibility-aware teacher。
2. 用 exporter schema v2 重新导出 LiDAR rollout，并检查 recovery 指标分布。
3. 在训练任何 predictor 前实现 dataset manifest 检查。
4. 实现 B1-B4 降信息模式和动作条件风险调制。
