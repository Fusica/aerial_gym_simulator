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
