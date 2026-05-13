# 发现与锁定决策

## 当前科学定位
首篇论文只有在被表述为“privileged-to-depth-only sensor-proxy transition”时，才具备强 EAAI 投稿说服力。

锁定机制是：

`depth-only 降信息 + 动作条件可观测性风险预测 + CTBR 动作修正`

## 代码事实
- 当前 task config：`aerial_gym/config/task_config/pursuit_guidance_task_config.py`。
- 当前 task：`aerial_gym/task/pursuit_guidance_task/pursuit_guidance_task.py`。
- robot/control：`base_quad_root_link_control` + `thrust_bodyrate_control`。
- target baseline：`apf_escape`。
- 当前 32D 观测泄露目标相对位置、速度、closing geometry、墙体 clearance 和障碍物相对位置。
- 当前 robot config 默认不启用 camera。

结论：现有策略是特权全状态 baseline，只能作为 B0 和标签生成器，不能作为主方法。

## 锁定问题陈述
如何利用特权全状态 rollout 学习短期可观测性风险，并在 depth-only 传感器代理降信息条件下用动作条件风险修正 CTBR 动作，从而恢复追逐决策质量？

## 新颖性判断
结论：**在锁定主张下具有中等置信度的新颖性**。

不够新：
- full-state PPO 加一个风险标量
- 通用目标轨迹预测
- 有限 FOV reward shaping
- 通用 partial-observation MARL

足够支撑强 EAAI 尝试：
- 预测 observability-loss risk 而不是目标轨迹
- 特权 rollout 只用于标签和辅助监督
- 主实验在 reduced/depth-only 信息下进行
- 风险是动作条件的，并用于 CTBR 动作修正

## 强近邻与区分点
| 论文 | 重合点 | 必须区分 |
|------|--------|----------|
| `[@huh2026limited]` | limited detectable region、FOV reward、APF evader | 本文学习并校准 observability-risk，而非只做 reward/action design |
| `[@peng2025limitedvisual]` | limited visual field、search/pursuit/reacquisition MARL | 本文是 privileged-to-depth bridge + 动作条件风险修正，而非 graph-attention MARL |
| `[@li2026safeintent]` | risk-gated hierarchical policy、安全投影、部分可观测 | 本文直接预测 observability-loss risk，并在 1v1 depth-only sensor-proxy 下修正 CTBR 动作 |
| `[@chen2025open]` | prediction-enhanced pursuit、CTBR deployment | 本文预测 target-loss risk 而非 target trajectory |
| `[@zhang2023gameofdrones]` | 目标预测与在线规划 | 本文单独评估风险变量的离线校准和在线控制作用 |
| `[@li2025bearingonly]` | sensor-compatible pursuit、target-loss robustness | 本文使用 depth-only 风险学习和 CTBR 动作修正，而非 bearing-only filter |

## E0-C1/E0-C2 控制器对比实验发现（2026-05-11）

### 实验矩阵
| 实验 | 运行 ID | 控制器参数 | scale_input | kOmega | torqueLimit |
|------|--------|-----------|-------------|--------|-------------|
| E0-C1 | `runs/PE_20260511_000603` | e1（用户修改） | [1.0, 3.0, 3.0, 1.5] | [1.2, 1.2, 1.0] | [2.4, 2.4, 0.2] |
| E0-C2 | `runs/PE_20260511_125039` | 仓库默认 | [1.0, 2.0, 2.0, 0.2] | [1.0, 1.0, 0.8] | [2.4, 2.4, 0.18] |

两者均为 1.2B 步全状态 PPO，无课程学习，所有其他超参一致。

### 发现 1（修正）：更保守的控制器显著优于激进的控制器
- **事实：** E0-C2（更保守的默认参数）成功率达到 0.97%（最高 2.33%），而 E0-C1（更激进的 e1 参数）为 0%。默认参数下的最小距离（1.71m vs 2.40m）、1m 到达率（3.3% vs 0%）和 3m 到达率（99.6% vs 89%）均更优。
- **含义：** 激进的控制器限制适得其反。将 scale_input 从 [2.0, 2.0, 0.2] 放宽到 [3.0, 3.0, 1.5] 并未增强机动能力——它放大了探索空间，使梯度质量下降，导致策略更早陷入确定性困局。
- **行动：** 保持仓库默认控制器参数，不进一步放宽。考虑未来实验探索更窄的限制是否能进一步加速学习。

### 发现 2：动作空间放大引发三重失败机制
- **事实：** E0-C1 的熵降至 -4.28（完全确定性），E0-C2 维持在 -1.71（仍保留一些探索）。E0-C1 的 pre-tanh OOB 为 61.6%，E0-C2 为 48.5%。E0-C1 的 tanh 饱和率为 6.4%，E0-C2 为 4.7%。
- **机制链：** 更宽的限制 → 更大的动作搜索空间 → 策略梯度中更多方差 → 更早的熵塌陷 → pre-tanh 动作漂移至更大值 → 更多 tanh 饱和 → 更差的梯度质量 → 无法学习精确最后接近 → 成功永远无法涌现。
- **含义：** "更大 = 更好"的控制器设计直觉对于 RL 是错误的。动作空间的正则化效果比动作空间的大小更重要。
- **行动：** 控制器限制应被视为一种隐式正则化器，而非严格的物理约束。选择能维持探索的最小可行限制。

### 发现 3：成功涌现需要探索窗口与精度窗口对齐
- **事实：** E0-C2 的首次成功出现在 5.23 亿步（熵≈-0.54，演员标准差≈1.23，最小距离≈3.80m）。在这个时间点，策略仍保留显著探索能力，且已掌握进入 3.8m 的接近精度。两者交叉后，成功奖励开始提供持续的正向梯度。
- **含义：** 成功涌现不是一个改进量的问题，而是一个时机问题——探索必须在策略的接近精度足以达到成功阈值时仍然活跃。E0-C1 在接近精度窗口打开之前（~3 亿–4 亿步）就失去了探索能力。
- **行动：** 通过保持熵系数 ≥ 0.001（而非衰减至几乎为零）来主动管理探索窗口。课程学习通过将精度门槛向前移动来缩短窗口。这两者协同作用。

### 发现 4（修正）：奖励结构不是阻断因素，但不理想
- **事实：** E0-C2 尽管时间惩罚（-17.93）+ 超时惩罚（-5.92）= 总固定惩罚 -23.85，仍能学习成功。成功奖励（+0.033，最高 0.084）相对于惩罚权重来说微不足道，但它仍然提供了可学习的梯度。
- **含义：** 之前关于"奖励结构导致成功无法学习"的结论太强——在正确的控制器设置下，即使微弱的成功信号也足够。但当成功信号弱时，控制器/探索设置成为决定性能的关键。
- **行动：** 课程学习仍然是 priority #1——在更容易的条件下增大成功频率，为策略创造更强的成功信号以锁定成功行为。

## E0-C3：手动 vs 自动课程学习对比实验（2026-05-13）

### 实验矩阵
| 实验 | 运行 ID | 课程方式 | total_timesteps | early_stop | 阈值切换步数 |
|------|--------|---------|-----------------|-----------|------------|
| E0-C3-手动 | `runs/PE_20260506_181131` | 手动 resume + 改阈值 | 每次 400M（共三次） | True | ~400M, ~800M |
| E0-C3-自动 | `runs/PE_20260513_102027` | `--curriculum` 自动 | 1.2B（单次） | False（`--no-early-stop`） | ~400M, ~800M |

两者的 robot（`base_quad_root_link_control`）、controller 参数、阈值（5→3→1）、PPO 超参（lr=0.0003, γ=0.999, ent_coef=0.01→0, vf_coef=0.5, clip_coef=0.2）均相同。用户在旧 run 运行前已手动将 robot_name 改为 `base_quad_root_link_control`。

### 实验细节
- **手动课程方式：** 分三次运行。每次 `total_timesteps=400M`，每阶段结束后手动修改 success_threshold，加载 latest.pth 并 resume。entropy 每阶段重新从 0.01 线性衰减到 0。
- **自动课程方式：** 一次运行 `total_timesteps=1.2B`，`curriculum_stage_step_budgets=[400M, 400M]`。entropy 在整个 1.2B 步内从 0.01 线性衰减到 0。

### 手动 run 的 SR 轨迹
```
~400M (5m阈值): SR=98.2%, min_dist=4.67, reach1m=0.1%
~800M (3m阈值): SR=99.6%, min_dist=2.75, reach1m=0.2%  
~802M (刚切1m): SR=7.2%  ← 崩溃
~1200M(1m阈值): SR=77.2%, min_dist=0.98, reach1m=84.3%
```
每次阈值切换都造成 SR 瞬时暴跌（98%→65% 在 5→3m，99.6%→7.2% 在 3→1m），其中 3→1m 的切换后 SR 恢复极慢。

### 自动 run 的最终状态
```
~1200M (1m阈值): SR=100%, reach_rate_1m=100%, min_dist=0.958, avg_len=1107, collisions=0
```
自动 run 在同样步数下达到了完全的成功率，且没有碰撞。

### 关键发现：`total_timesteps` 通过 entropy 衰减速率影响性能

**这是主要原因。** 不是 robot name（两者相同），不是控制器参数（两者相同），不是 PPO 超参（两者相同）。

机制：
- **手动方式（total_timesteps=400M × 3）：** 每阶段 entropy 从 0.01 重新开始衰减。在 stage 3（1m 阈值）开始时，entropy=0.01，接近满探索水平。1m 精度阶段需要的是精细控制，过高的 entropy 噪声使得策略难以收敛到高精度动作，导致 SR 从 99.6% 崩溃到 7.2%，400M 步后仅恢复到 77.2%。
- **自动方式（total_timesteps=1.2B）：** entropy 在整个训练中均匀衰减。stage 2 开始时（~400M 步）entropy≈0.0067，stage 3 开始时（~800M 步）entropy≈0.0033。在需要高精度的 1m 阶段，entropy 已经自然降低到适合精细调优的水平，策略可以在已有精度基础上渐进改善。

**直观理解：** 每个 stage 重设 entropy=0.01 相当于每进入新难度就"失忆"所有已学到的精细控制，重新开始随机探索。1.2B 统一衰减则让 entropy 与难度自然匹配——在简单阶段探索、在困难阶段收敛。

### 次要因素

1. **`--no-early-stop`：** 自动 run 禁用 early stop。在 stage 3 的 SR 暴跌后，best_score 仍保存着上一阶段的 99.6%，early_stop 的 patience 计数器可能持续累积，存在提前停止训练的风险。

2. **训练连续性：** 自动 run 不需要 checkpoint save/load，优化器状态和 obs_rms 在阈值切换时自然延续。手动 resume 虽然也恢复这些状态，但引入了重启开销。

### 对课程学习设计的启示
- `total_timesteps` 不应每阶段重新设定——应与 `curriculum_stage_step_budgets` 的总和一致。
- 课程学习本身只做阈值自动切换，不涉及探索相关修改。但**熵衰减的全局步长感知**是课程学习打包配置的一部分。
- 未来课程学习配置建议直接在 task config 中指定 thresholds 和 budgets（已在 `956955d` 后的代码中通过 `task_config.curriculum` 实现），确保 `total_timesteps` 与 budgets 总和一致。

### 对论文方法的影响
- E0-C2 证明 B0（全状态 PPO）在默认控制器下可以学习成功 → B0 上界可以建立。
- 控制器参数不是优化问题，而是正则化问题——默认仓库值很可能已经是最优点。
- 课程学习 + 默认控制器 + 保持熵系数应使 B0 能够在合理训练时间内达到 ≥ 0.8 成功率。
- B0-B5 阶梯在课程学习就绪后可以开始填写。

## 最终贡献写法
1. **降信息追逐定义**：移除策略输入中的目标和环境真实值。
2. **动作条件可观测性风险学习**：从特权 rollout 学习 `p_lost` 和 `R_obs(s_red, u)`。
3. **CTBR 风险约束动作修正**：用约束投影层将 PPO 原始动作修正为风险感知动作。
4. **因果实验阶梯**：B0-B5 分离上界、退化、启发式风险、学习风险、oracle 风险和动作条件修正。

## 失败条件
若出现以下情况，论文主张必须重写：
- learned risk 离线不优于 heuristic risk
- B3 在线不优于 B1/B2
- B5 只是通过过度保守换来 target retention
- 收益只在单一 APF 参数设置中存在
- 主线评估仍使用目标或环境真实值

## 写作规则
- 不说“首次 partial-observation pursuit”。
- 不说“完整 onboard visual pursuit”。
- 不把 B0 + risk scalar 描述成本文方法。
- 不把 bilateral self-play 放进论文 1 主贡献。
- 除非真实 camera processing 已实现并验证，否则使用 “depth-only sensor-proxy reduced-information pursuit”。

## 实现归属
- `risk_geometry.py`：可检测区域几何和标签。
- `risk_dataset.py` / `risk_offline.py`：rollout 存储、manifest、离线标签。
- `risk_model.py` / `risk_training.py`：depth encoder、risk heads、校准。
- `risk_policy.py`：降信息观测和风险注入。
- `risk_closed_loop.py`：CTBR 动作修正。
- `risk_diagnostics.py`：离线和在线指标。
- `pursuit_guidance_task.py`：只做编排。

## 当前下一步
1. 首先实现 risk geometry 和标签生成。
2. 在训练任何 predictor 前实现 dataset manifest 检查。
3. 实现 B1-B5 降信息模式。
4. 大规模 PPO 前实现动作条件风险修正。
