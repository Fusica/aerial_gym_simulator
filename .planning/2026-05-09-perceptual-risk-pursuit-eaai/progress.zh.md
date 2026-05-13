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
- 唯一强主线是 `depth-only 降信息 + 动作条件可观测性风险 + CTBR 修正`。
- B0 是上界参考和标签来源，不是本文方法。
- B5 是主方法。
- 如果主方法继续使用特权状态，视为 scope regression。
- **新增（E0-C1/C2 对比后修正）：** E0-C2 证明 B0 在当前奖励 + 默认控制器下可学习成功（~1-2%）。E0-C1 的失败源于控制器过度放宽导致的探索崩溃，而非奖励结构缺陷。B0 上界在课程学习加持下可以建立。

## 下一步（第 6.2 阶段）
1. 确定策略池来源维度：训练阶段、课程阈值、seed。
2. 从已保存的 stage checkpoint（`stage_0_thr_5.0m`、`stage_1_thr_3.0m`）和中间 training checkpoint 中筛选候选策略。
3. 验证 `risk_geometry` 标签在策略池 rollout 上的正确性和覆盖率。
4. 设计数据采集 pipeline：多策略 rollout → risk label 标注 → dataset manifest。
5. 确保正负样本平衡（`y_t=1` 和 `y_t=0` 均有足够样本）。

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
| `risk_geometry` 单测 | `/home/ubuntu/miniconda3/envs/aerialgym/bin/python -m unittest discover -s tests -p 'test_pursuit_risk_geometry.py'`，5 tests OK | pass |
| `risk_geometry` 编译检查 | `/home/ubuntu/miniconda3/envs/aerialgym/bin/python -m py_compile aerial_gym/task/pursuit_guidance_task/risk_geometry.py tests/test_pursuit_risk_geometry.py` | pass |
| E0-C1 数据分析 | 完整 TB event 序列分析，651 个记录点，10 个阶段过渡，对数线性收敛趋势 | pass |
