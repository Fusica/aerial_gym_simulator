# 进展记录

## 2026-05-09
- 初始化 `.planning/2026-05-09-perceptual-risk-pursuit-eaai/`。
- 审计当前 pursuit task，确认其是使用 `apf_escape`、32D 几何观测、root-link 控制和 `thrust_bodyrate_control` 的特权全状态 PPO baseline。
- 在 `deep-research-output/perceptual-risk-pursuit-review/` 下完成 deep-research。
- 删除了不支持锁定 EAAI 声明的辅助风险旧指导。
- 将论文 1 重新框架为 privileged-to-depth-only sensor-proxy transition。
- 锁定主方法：降信息追逐 + 学习型短期可观测性风险 + CTBR 风险约束动作修正。
- 锁定默认值：`H = 150`、`K_persist = 10`、`K = 3` depth 帧、`z_depth = 64`。
- 锁定架构：离线预训练、冻结 depth encoder 和 risk head、PPO 不反传进表示和修正模块。
- 锁定 B0-B5 baseline 阶梯和离线/在线指标。

## 2026-05-10
- 使用 novelty-assessment 的严苛评审口径重新分析。
- 判断宽泛的“risk-assisted partial-observation pursuit”不够强，因为已有 limited-FOV MARL、prediction-enhanced pursuit、risk-gated policy、sensor-compatible pursuit。
- 锁定更强的新颖性切口：从特权 rollout 学习动作条件 observability-risk，并在 depth-only 降信息下用于 CTBR 动作修正。
- 重写 `task_plan.zh.md` 和 `findings.zh.md`，删除弱中间路线。
- 将 `[@peng2025limitedvisual]`、`[@li2026safeintent]`、`[@li2025bearingonly]` 加入 deep-research 近邻比较材料。

## 当前状态
- **计划 ID：** `2026-05-09-perceptual-risk-pursuit-eaai`
- **阶段：** 第 6 阶段——实现与验证
- **总体状态：** active

## 关键判断
- 论文不做通用 pursuit RL、通用部分可观测、目标轨迹预测或完整 bilateral self-play。
- 唯一强主线是 `depth-only 降信息 + 动作条件可观测性风险 + CTBR 修正`。
- B0 是上界参考和标签来源，不是本文方法。
- B5 是主方法。
- 如果主方法继续使用特权状态，视为 scope regression。

## 下一步
1. 实现并单测 `H = 150`、`K_persist = 10` 下的可检测区域标签。
2. 策略重训练前实现 dataset manifest 和离线 predictor 指标。
3. 通过显式 config switch 实现 B1-B5。

## 错误
| 错误 | 处理 |
|------|------|
| 早期 guidance 保留弱中间路线 | 已改为单一 EAAI 强路线和显式失败条件 |

## 验证结果
| 检查 | 结果 | 状态 |
|------|------|------|
| `paper_db.jsonl` 解析 | 65 行 JSONL 全部可解析 | pass |
| 新近邻 key | `peng2025limitedvisual`、`li2026safeintent`、`li2025bearingonly` 已进入 DB/BibTeX/survey/synthesis/report | pass |
| 弱路线扫描 | active guidance 中不再保留旧的弱主线 | pass |
