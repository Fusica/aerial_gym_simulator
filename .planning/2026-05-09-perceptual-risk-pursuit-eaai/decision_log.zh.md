# 决策日志

## 决策记录

### 2026-05-09

| 决策 ID | 决策 | 被排除方案 | 理由 | 影响 |
|---------|------|------------|------|------|
| D001 | 首篇论文保持在现有 PPO 追逐者 + APF/规则逃逸者结构内 | 双边 self-play、端到端视觉追逐 | 耦合最低，适合形成可验证首篇论文 | 论文 1 范围 |
| D002 | 论文 1 主线使用 depth-only sensor-proxy | RGB 主线、完整 onboard visual pursuit | Depth 路径更接近当前仓库可落地能力，措辞更诚实 | 感知策略 |
| D003 | 论文 1 必须移除策略输入中的目标和环境真实值 | full-state + risk scalar | 否则无法证明信息桥梁作用 | 方法与实验 |
| D004 | 使用短期可观测性丢失风险作为主标签 | 目标轨迹预测、capture-denial 风险 | 与 sensing continuity 直接相关，区别于 prediction-enhanced pursuit | 标签定义 |
| D005 | 风险模型必须支持动作条件评分 `R_obs(s_red, u)` | 仅状态风险 `p_lost(s_red)` | 强化新颖性，并支撑动作修正层 | 方法 |
| D006 | 动作修正层是主方法组件 | 只作为观测侧风险 bridge | 将风险从被动特征变成主动控制变量 | 方法 |
| D007 | 动作修正使用一步约束投影，非 MPC | 完整 MPC 内环 | 当前实现成本可控，且足以表达 CTBR correction | 实现 |
| D008 | 使用 `H = 150`、`K_persist = 10` | `H = 40`、未锁定窗口 | 更匹配高速追逐的提前量 | 标签定义 |
| D009 | 使用 `K = 3` depth 帧和 `z_depth = 64` | 单帧 depth、较小 latent | 单帧 depth 对目标运动和遮挡变化表达不足 | 表示学习 |
| D010 | 使用 offline pretrain + freeze + stop-gradient | PPO 端到端共同训练 encoder/risk head | 保持模块可审计，避免 PPO 梯度污染风险模型 | 架构 |
| D011 | B0-B5 全部作为主实验阶梯 | 只做 B0/B3 或弱 ablation | EAAI 强投稿需要可证伪和因果归因 | 实验 |
| D012 | 将 `[@peng2025limitedvisual]`、`[@li2026safeintent]`、`[@li2025bearingonly]` 加入 near-neighbor 比较 | 只比较旧 deep-research 种子文献 | novelty assessment 显示这些是强近邻 | Related Work |

## 已锁定实验规则
- B0 是全状态上界参考和标签来源，不是本文方法。
- B1-B5 是主比较链条。
- B2 depth-honest heuristic risk 必须实现为手工 baseline；若其效果弱，也保留为弱基线而非从主表删除。
- B4 oracle risk 是诊断上界，默认保留。
- B5 是本文主方法。

## 已锁定写作规则
- 不再写“full-state PPO + extra scalar”路线。
- 不再写“完整 onboard visual pursuit”。
- 不再把 bilateral self-play 放入论文 1 主贡献。
- 不再把 generic target prediction 当作核心创新。

## 待实现决策
| 决策 ID | 问题 | 默认答案 | 阶段 |
|---------|------|----------|------|
| I001 | `R_det`、`alpha_h`、`alpha_v` 的默认值 | 以相机 FOV/像素可检测距离为依据，并做敏感性分析 | Phase 6 |
| I002 | B2 heuristic risk 具体公式 | 只能使用 depth-honest proxy，不使用目标/环境真实值 | Phase 6 |
| I003 | B5 动作候选生成方式 | 优先小候选集或 1-2 步 clamped projected-gradient | Phase 6 |
