# 任务计划：EAAI 强投稿版可观测性风险追逐

## 目标
围绕当前 `pursuit_guidance_task` 分支形成一条单一、强约束的 EAAI 投稿路线：

`特权全状态追逐 -> depth-only 传感器代理降信息追逐 -> 动作条件短期可观测性风险预测 -> CTBR 风险约束动作修正`

本文不再以通用 pursuit RL、通用部分可观测、通用目标预测或完整双边 self-play 作为首篇论文主线。

## 活动计划
- **计划 ID：** `2026-05-09-perceptual-risk-pursuit-eaai`
- **当前阶段：** 第 6 阶段——实现与验证
- **总体状态：** in_progress
- **投稿规则：** 所有方法、实验和写作都必须服务于下面锁定的 EAAI 声明。

## 锁定 EAAI 声明
通过特权全状态 rollout 学习动作条件短期可观测性风险，并用该风险对 CTBR 动作进行约束投影修正，可以让 UAV 追逐者在 depth-only 传感器代理降信息条件下恢复追逐决策质量。

## 不可妥协的贡献
1. **Depth-only 传感器代理降信息追逐定义**
   - 主策略输入为 `self-state + z_depth + p_lost`。
   - 主线策略输入中必须移除直接目标状态、真实值衍生 bearing/range/closing-speed、墙体真实值和障碍物真实值。
   - 全状态 PPO 只作为上界参考，不是本文方法。

2. **动作条件短期可观测性风险模型**
   - 主标签是 `H = 150`、`K_persist = 10` 下的目标丢失概率。
   - 主预测器必须支持动作条件风险评分：`R_obs(s_red, u)` 或等价的候选动作风险评估。
   - 特权状态只允许用于离线标签和辅助监督，不允许在策略推理时使用。

3. **CTBR 风险约束动作修正**
   - PPO 原始动作 `u_rl = [c, p, q, r]` 通过一步约束投影修正为 `u*`。
   - 动作修正层是必须实现的核心方法组件。
   - 论文 1 不让 PPO 梯度反传进冻结的 depth encoder、risk head 或动作修正层。

## 仓库事实
- 当前配置：`aerial_gym/config/task_config/pursuit_guidance_task_config.py`。
- 当前控制路径：`base_quad_root_link_control` + `thrust_bodyrate_control`。
- 当前目标运动 baseline：`apf_escape`。
- 当前 32D 观测是全状态几何观测，泄露目标相对状态和环境真实值。
- 当前 root-link robot 默认没有启用 camera；任何 depth-only 声明前必须实现 depth 输入路径。

## 必需 Baseline 阶梯
| ID | 方法 | 作用 |
|----|------|------|
| B0 | 原始 32D 全状态 PPO | 上界参考 |
| B1 | 降信息 PPO，无风险 | 主退化 baseline |
| B2 | 降信息 PPO + depth-honest 启发式风险 | 手工风险 baseline |
| B3 | 降信息 PPO + 学习型状态风险 `p_lost(s_red)` | 只含 bridge 的学习 baseline |
| B4 | 降信息 PPO + oracle 未来风险 | 风险变量价值上界 |
| B5 | 降信息 PPO + 学习型动作条件风险修正 | 本文主方法 |

## 必需外部比较类别
- 有限视场 / 有限可检测区域 MARL：`[@peng2025limitedvisual]`、`[@huh2026limited]`。
- 预测增强追逐和 CTBR 部署相关工作：`[@chen2025open]`、`[@zhang2023gameofdrones]`。
- 风险门控或安全投影的部分可观测追逐：`[@li2026safeintent]`。
- bearing-only 或传感器兼容追逐：`[@li2025bearingonly]`、`[@zheng2025visioncapture]`。
- 双边/self-play 强对手路线仅作为后续工作：`[@roncero2025amspb]`、`[@yao2026warning]`。

## 方法定义
### 可观测性标签
- 虚拟可检测区域位于追逐者机体系，由 `R_det`、`alpha_h`、`alpha_v` 定义。
- 若目标在 `H = 150` 步内离开该区域并持续 `K_persist = 10` 步，则 `y_t = 1`。
- 必须做 `H in {100, 150, 200}` 以及至少一组 FOV/range 扰动敏感性分析。

### Depth 与风险模型
- 输入：自身线速度、自身角速度、姿态、上一时刻 CTBR 动作、`K = 3` 帧 depth stack。
- 冻结 latent：`z_depth = 64`。
- 输出：`p_lost(s_red)` 和动作条件风险 `R_obs(s_red, u)`。
- 训练期辅助输出：由特权监督得到的 range/bearing。
- 推理期规则：无目标真实值、无墙体真实值、无障碍物真实值、无真实值衍生 bearing/range/closing-speed。

### 动作修正
`u* = argmin_u ||u - u_rl||_2^2 + lambda_risk R_obs(s_red, u) + lambda_fov L_fov(s_red, u) + lambda_obs L_obs(s_red, u) + lambda_rate ||u - u_prev||_2^2`

硬约束：
- `u_min <= u <= u_max`
- `||u - u_prev||_inf <= Delta_u_max`

实现规则：
- 使用 clamped projected-gradient 或小候选动作集合最小化。
- 论文 1 不做 MPC 内环。

## 阶段门槛
### 第 0 阶段：Baseline 审计
- [x] 确认 32D 全状态观测和 APF 目标路径。
- [x] 确认当前 pursuit 路径不是 depth-driven。
- **状态：** complete

### 第 1 阶段：文献与新颖性锁定
- [x] 完成 deep-research。
- [x] 完成 novelty assessment。
- [x] 将强近邻比较项加入 deep-research guidance。
- **状态：** complete

### 第 2 阶段：问题定义
- [x] 锁定为 privileged-to-depth-only sensor-proxy transition。
- [x] 排除通用目标预测、通用部分可观测和完整双边 self-play 作为论文 1 表述。
- **状态：** complete

### 第 3 阶段：方法设计
- [x] 锁定 `H = 150`、`K_persist = 10`、`K = 3`、`z_depth = 64`。
- [x] 将动作条件风险和 CTBR 动作修正提升为主方法。
- **状态：** complete

### 第 4 阶段：实验设计
- [x] 锁定 B0-B5。
- [x] 锁定离线/在线指标和统计规则。
- **状态：** complete

### 第 5 阶段：Deep-Research 同步
- [x] 加入近邻比较论文和修订后的 gap 表述。
- **状态：** complete

### 第 6 阶段：实现与验证
- [ ] 实现 risk geometry 和标签生成。
- [ ] 实现 dataset manifest 和离线风险训练。
- [ ] 实现 depth stack 输入和冻结 `z_depth` 路径。
- [ ] 实现 B1-B5 观测/控制模式。
- [ ] 实现动作条件风险修正。
- [ ] 大规模训练前运行 smoke test 和小 rollout 诊断。
- **状态：** in_progress

## 明确推迟
- 完整 RGB 视觉追逐。
- 完整机载 detector/tracker 声明。
- 双边 self-play 或学习型逃逸者作为中心贡献。
- 除非已经实现并验证，否则 full occlusion-aware sensing 不作为主声明。

## 错误记录
| 错误 | 处理 |
|------|------|
| planning 中保留了较弱中间路线 | 已重写为单一 EAAI 强主线 |
