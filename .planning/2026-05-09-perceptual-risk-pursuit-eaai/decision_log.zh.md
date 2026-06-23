# 决策日志

## 决策记录

### 2026-05-09

| 决策 ID | 决策 | 被排除方案 | 理由 | 影响 |
|---------|------|------------|------|------|
| D001 | 首篇论文保持在现有 PPO 追逐者 + APF/规则逃逸者结构内 | 双边 self-play、端到端视觉追逐 | 耦合最低，适合形成可验证首篇论文 | 论文 1 范围 |
| D002 | 历史决策：论文 1 主线使用 depth-only sensor-proxy | RGB 主线、完整 onboard visual pursuit | Depth 路径更接近当时仓库可落地能力，措辞更诚实；2026-05-14 已被 D015 取代 | 感知策略 |
| D003 | 论文 1 必须移除策略输入中的目标和环境真实值 | full-state + risk scalar | 否则无法证明信息桥梁作用 | 方法与实验 |
| D004 | 使用短期可观测性丢失风险作为主标签 | 目标轨迹预测、capture-denial 风险 | 与 sensing continuity 直接相关，区别于 prediction-enhanced pursuit | 标签定义 |
| D005 | 风险模型必须支持动作条件评分 `R_obs(s_red, u)` | 仅状态风险 `p_lost(s_red)` | 强化新颖性，并支撑风险调制 PPO | 方法 |
| D006 | 暂定主方法为动作条件风险调制 PPO 的动作分布与优势估计 | 简单 observation concat；PPO 后处理 | 风险应作为结构化调制信号进入策略学习，而不是只作为输入特征或执行后处理 | 方法 |
| D007 | 风险调制 PPO 不做 MPC 内环，环境执行动作与 PPO logprob 对应动作保持一致 | 完整 MPC 内环；独立执行前后处理层 | 当前控制链路是 PPO 输出 CTBR 命令后进入 `thrust_bodyrate_control`、分配器和电机模型；主方法应避免造成 PPO credit assignment 不一致 | 实现 |
| D008 | 使用 `H = 150`、`K_persist = 10` | `H = 40`、未锁定窗口 | 更匹配高速追逐的提前量 | 标签定义 |
| D009 | 历史决策：使用 `K = 3` depth 帧和 `z_depth = 64` | 单帧 depth、较小 latent | 2026-05-14 已被 D015 的 LiDAR/range-image 主线取代，保留为历史记录 | 表示学习 |
| D010 | 使用 offline pretrain + freeze + stop-gradient | PPO 端到端共同训练 encoder/risk head | 保持模块可审计，避免 PPO 梯度污染风险模型 | 架构 |
| D011 | B0-B4 作为主实验阶梯 | 只做 B0/B3、弱 ablation 或额外真值风险上界 | EAAI 强投稿需要可证伪和因果归因；B0 已承担 full-state upper-bound 角色 | 实验 |
| D012 | 将 `[@peng2025limitedvisual]`、`[@li2026safeintent]`、`[@li2025bearingonly]` 加入 near-neighbor 比较 | 只比较旧 deep-research 种子文献 | novelty assessment 显示这些是强近邻 | Related Work |
| D013 | 所有本项目命令强制使用 conda 环境 `aerialgym_v2` | 旧环境 `aerialgym` | 用户指定且 Isaac Gym/gymtorch/ninja 依赖必须在正确环境内解析 | 全部实现与验证 |
| D014 | 仿真语义真值、目标 ID 和真值 target-hit 不作为模型或策略输入 | 将 Isaac Gym segmentation 或目标真值 mask 作为风险模型输入 | 这些信号不是真实 LiDAR/range-image 的原生输入，直接使用会造成不可部署的真值泄漏 | 数据与模型输入 |
| D015 | 主传感器路线从 depth-only 改为 LiDAR/range-image | 继续使用 RealSense-like depth camera 作为主输入 | 用户要求高速、大空间、500m 以内探测；depth work range 过短，无法支撑不降级任务 | 感知策略 |
| D016 | 主线 LiDAR/target 约束锁定为 `target_x500 + M3-like` 前向高分辨率配置 | 继续把 500m/100Hz 作为开放传感器选型问题 | 静态 smoke 已验证该组合在 100m/150m/200m 均保留 2 个目标返回点，足以作为论文 1 风险模型输入假设；500m/100Hz 不再作为主线目标 | 实验前置 |
| D017 | 严格 OS2-64 `64 x 512` 不能作为远距小目标主输入的默认方案 | 直接采用 `max_range=200m` OS2-64 range image 开始大规模采集 | static smoke 显示 10m 仅 4 个目标返回点，25m 起到 200m 均为 0；瓶颈是角分辨率/目标截获像素，不是 max range | LiDAR 配置需重新选型或上采样/ROI 设计 |
| D018 | 当前 pursuit 默认且唯一主线 LiDAR config 是 M3-like `501 x 2401` 前向 ultra-high-res profile | 使用或保留 AT128P-like `128 x 1201` 作为主输入/对照 config | AT128P-like 在 50m+ 不可用；`target_x500 + M3-like` 在 100m/150m/200m 均有 2 个目标返回点 | 删除 AT128P-like config；M3-like 作为固定主线 sensor，后续只做吞吐和稀疏点鲁棒性检查 |
| D019 | 默认 target 使用 `target_x500`，`target_quad` 只作为显式对照保留；不在主线引入 octarotor target | 直接把 octarotor 作为主 target；继续默认使用 `target_quad`；继续把 x500 做成二选一默认关闭项 | 当前 pursuit 默认曾用 `quad/model.urdf`，不是最小 `quad.urdf`；`x500` 是仓库内更大且已有高速配置的常规 quad，M3-like 下 200m 仍有 2 个点；octarotor 改变目标类型和任务语义。用户要求 x500 成为默认 target，且只改目标不改追击无人机 | env config 默认启用 `target_x500`、关闭 `target_quad`；保留 CLI override 供对照；不把更大体积目标带来的改善误写成传感器本身能力 |
| D020 | target asset 切换时 target controller 必须使用 target 质量，而不是 pursuer `robot_mass` | 继续复用 pursuer 质量标定 target thrust | `target_quad` 与 pursuer 质量接近时该假设不显著；`target_x500` 为 `1.656kg`，继续用 `1.94kg` 会造成目标控制力偏大 | target asset config 增加 `controller_mass`，后续新增目标 URDF 必须同步质量 |
| D021 | 删除旧 pursuit depth 专用代码链路，后续只维护 LiDAR/range-image 主线 | 保留 depth config/smoke/exporter 作为历史调试工具 | 用户明确要求后续只考虑 LiDAR；保留旧 depth 链路会造成采集入口和实验叙述混乱 | 删除 pursuit depth config、depth robot registry、depth smoke 和旧 depth rollout/export helper |
| D022 | 删除不进入主线的 pursuit `target_octarotor` 分支 | 保留 octarotor 作为 smoke/test 可选目标 | octarotor 会引入“目标类型变大/变高”的额外变量，和当前 LiDAR lost-risk 主线目标语义不一致 | 只保留 `target_quad` 与 `target_x500`；仓库原生 `base_octarotor` robot/config 不删除 |
| D023 | 当前 target 逃逸者不得表述为高保真 motor-level UAV 模拟 | 将 `target_x500` URDF + `controller_mass` 修正写成高保真 target 动力学 | target 由 APF/轨迹生成器给出期望位置/航向，再由 task 内 controller 直接写 `target_force_tensor`/`target_torque_tensor`，跳过分配矩阵、电机模型、ESC/prop 动态和单电机饱和 | 写作中只称为 PhysX 刚体 + 高层位置控制/直接 force-torque 注入的规则逃逸目标；高保真声明需另建 actuator/drag/delay/SysID 链路 |
| D024 | 需要训练 visibility-aware full-state teacher，但先使用 soft reward 而非 hard done | 继续直接用原 99%-100% SR checkpoint 采集主数据；或一不可见就 episode done；或给 visibility reward 设置默认关闭开关 | 第一轮 x500 + LiDAR rollout 显示成功 checkpoint 仍只有约 13%-36% 几何可见率；原 reward 的 `forward_alignment` 不等于 LiDAR FOV 可见性。hard done 会丢失临界丢失/恢复样本，不利于风险模型。长期保持可见应成为 pursuit task 的默认语义，即使 teacher 不显式挂载传感器 | 在 `pursuit_guidance_task` 中加入基于 LiDAR config 的 visibility reward/penalty，权重可与 progress 同一量级；不设置 enable 开关；有 LiDAR 时跟随实际 config，无 LiDAR 时使用 M3-like pursuit LiDAR 作为 teacher 先验 |
| D025 | visibility reward 训练必须晚于 capture curriculum，且分阶段增强 | 从第一阶段直接加入 visibility reward；或 1m 达标后直接进入满权重 1m+visibility；或在同一 visibility stage 内持续 ramp 权重；或继续按固定 400M 步强制切换阈值 | `PE_20260514_221640` 两次固定步数切换时 SR 均为 0，最终 SR 仍为 0；说明策略尚未恢复 capture skill 时直接加长期可见性约束会压垮课程。同一 stage 内持续 ramp 会让优化目标漂移，不利于判断该难度是否已稳定 | 默认课程切换改为连续 10 个 update 的 SR >= 95%；先无 visibility reward 完成 5m/3m/1m，再开启 visibility reward 重新走 5m/3m/1m；stage 3/4/5 分别使用固定低/中/满 visibility 权重档位 |
| D026 | `ppo_guidance.py` 是 pursuit PPO 课程学习参数的唯一入口 | 继续让 `task_config.curriculum` 维护第二份默认值；或让 `curriculum_controller.py` 自己提供 fallback 默认值；或 resume 时让 checkpoint 中旧 curriculum 参数覆盖当前参数 | 用户调参时主要编辑/传入 `ppo_guidance.py` 参数；多入口会导致实际 patience/SR 阈值与启动命令或文件默认值不一致，难以解释训练曲线 | `ppo_guidance.py` 解析后的 args 是唯一参数来源；task config 不再定义 pursuit curriculum 参数；checkpoint 只恢复课程进度，不覆盖当前 PPO 入口参数；curriculum controller 只消费已解析参数，不维护默认入口 |
| D027 | visibility-aware teacher 的最终任务定义为 `3m visible strike`，允许短暂丢视野并学习恢复 | 继续使用 1m success；任意 1 帧不可见即 failure；要求最近 K 步全可见才 success；把 episode 目标解释为 3600 step 长时跟踪 | 1m success 与 LiDAR `min_range=1m` 和传感器前向偏移存在近场可见性冲突；风险模型需要看到稳定可见、短暂脱离后恢复、持续脱离失败三类片段，硬终止会删除恢复数据 | 默认 `success_threshold=3.0`；success/success bonus 要求最终 `target_detectable=True`；中途不可见继续 episode 并累计 penalty，恢复可见后清零当前 loss；导出和训练日志记录 episode 内 recovery 统计 |
| D028 | 将 3 个 visibility 阶段权重从 `0.10/0.20/0.30` 提高到 `0.15/0.30/0.45` | 维持原权重；直接大幅提高到更高档位；把中途不可见改为 hard failure | `PE_20260518_133223` 在后三个 visibility 阶段保持约 `0.998-1.000` SR，final detectable 接近 1，但 final stage 平均 invisible steps 仍约 `498`、最新约 `451.5`；说明终端可见成功条件已生效但过程可见性约束偏弱 | 提高过程 visibility penalty，仍保留 3m visible-strike 成功定义和可恢复轨迹；实施后必须同步 curriculum 单测和 `ppo_guidance.py` 启动文案，并用新 run 检查 SR、invisible steps、max loss steps 和 done 分布 |
| D029 | 删除未来真值风险 baseline，动作条件风险调制 PPO 暂定为 B4 主方法 | 放入主表或 appendix 作为单独风险上界 | 不可部署真值风险输入会分散创新叙事，且不能证明动作条件风险调制优于可实现模型；全状态 B0 已提供性能上界参考 | 主比较链条固定为 B0-B4 |
| D030 | 阶段 1-5 按 risk-modulated PPO 主线重新打开复核 | 继续保持 1-5 全部 complete | 主方法已经从动作修正/过滤转为调制 PPO 动作分布与优势估计，旧 related work、问题定义和实验 ablation 不再完全匹配当前 claim | 第 1/2/3/4/5 阶段状态同步为 in_progress/需复核；第 0 阶段保持 complete |
| D031 | 当前 EAAI claim 收窄为 LiDAR/range-image reduced-information UAV pursuit 中的 action-conditioned observability-loss risk modulation of PPO | 通用 safe PPO、risk-sensitive PPO、risk-aware pursuit、action-conditioned risk gating 首创 claim | deep-research refresh 与 3 个 subagent 交叉验证均认为广义 claim 会被 Recovery RL、CPO/PPO-Lagrangian、shielded RL、asymmetric actor-critic 和 limited-FOV pursuit 近邻击穿 | B4 必须对比 risk concat、PPO-Lagrangian、shield/filter、privileged critic、shuffled-action risk；LiDAR 100/150/200m 只写 sparse-return stress-test |
| D032 | Zotero/tag list 只做候选审计，不做批量扩充 | 把用户给出的所有论文都无差别加入 active DB | active DB 只保留能服务 B4 设计、真实 comparison/ablation 压力或 Related Work 分组的条目；低相关、平台不匹配、未核验且不直接影响主线的条目只保留为候选审计记录 | paper DB 调整为 52；31 个候选中接受 11 个、排除 18 个、重复 2 个；Related Work 按主线需要分组写 |
| D033 | 外部比较按四个科学轴组织 | 继续按“有限视场/MARL、预测增强、风险门控、Zotero 候选”等来源或算法碎片分类 | 用户明确核心是有限视场、追逐和约束下动作优化，RL 是支撑；Intro/Related Work 需要形成清晰问题链，而不是材料来源列表 | 必需外部比较改为：有限可检测追逐、UAV 追逃/在线规划/CTBR、约束动作优化/风险门控、RL/privileged training/工程 machinery；`chen2025open` 明确列为强近邻 |
| D034 | 当前论文草稿先使用 `IEEEtran` / IEEE Transactions 双栏模板撰写和编译 | 当前阶段继续寻找、安装或切换 `elsarticle.cls` | 本地已能用 `paper/main.tex` 的 `IEEEtran` 外壳完成正文、BibTeX 和 PDF 编译；当前重点是收束论文主线、引用和内容结构，模板切换不应阻塞写作 | EAAI/Elsevier 官方模板只在投稿前最终格式化阶段处理；后续写作和引用更新默认直接改 `paper/main.tex` 与 `paper/main.bib` |
| D035 | visibility teacher 改用 150-step recovery-aware penalty 和 visibility-gated curriculum | 继续只提高 `0.20/0.40/0.60` 权重；把中途不可见改为 hard failure；把未来 H=150 risk label 直接塞进 reward | `PE_20260519_103518` 显示继续加权收益递减，final stage 仍有约 `340+` invisible steps 和 `300+` max loss steps；旧 penalty 在 `K=10` 后饱和，不能区分快速恢复和长期丢失 | 新增 `visibility_recovery_horizon_steps=150`；visibility penalty 在 `K=10` 后继续增大、`H=150` 后进一步增大；训练和 exporter 记录 horizon recovery 指标；visibility stages 推进需同时满足 success、max_loss 和 recovery_rate gate |
| D036 | 提高 recovery-aware visibility teacher 的后期 penalty 占比 | 维持 `weight_visibility_loss_penalty=0.05` 和 stage 2 `recovery_rate>=0.80`；退回旧饱和 penalty；改 success 或 FOV 几何 | `PE_20260520_000809` 末段 `max_loss` 已降到约 `168`，但 `invisible_steps` 仍约 `403`、recovery rate 约 `0.77`、visibility contribution 占比约 `14%`；旧 `PE_20260519_103518` final stage 约 `340` invisible steps 和约 `17%` contribution，说明新 run 低权重 stage 后期 visibility pressure 不足且被 gate 卡住 | 将 `weight_visibility_loss_penalty` 提高到 `0.08`；stage 2 recovery gate 放宽为 `0.75`；保持 3m visible-strike success、FOV 几何和 PPO loop 不变；后续用 `success_rate/reach_rate_3m/invisible_steps/max_loss/recovery_rate/reward_contrib_ratio_visibility` 联合验收 |
| D037 | 离线 risk geometry 必须复用 reward visibility 几何语义 | 保持 body-frame 对称 FOV 标签；只在文档中说明差异；把 exporter label 当作近似 | 后续 risk model 训练会依赖 exporter 的 `detectable` 和 observability-loss labels；如果不应用 sensor local translation/quaternion、FOV min/max 和 150m detection range，离线标签会偏离当前 success/reward 的 target-detectable 语义 | `DetectionFrustum` 保存 sensor 外参和 FOV min/max；exporter 从 lidar config 构造 reward 同口径 frustum；不改训练 reward、success 或 PPO loop |
| D038 | 冻结 `PE_20260520_110828` 的 `upd_1300` checkpoint 为当前 B0 | 等待最终 `vis0.60` stage；使用不稳定的 `latest.pth`；继续把 B0 写成抽象“最新 run” | 该 run 结束在 `stage_idx=3` (`3m + vis0.40`)，最后 10 个保存点保持约 `99.7%` success、`99.9%` reach_3m、`99.8%` final visible，同时仍有约 `265` invisible steps 和少量 over-horizon 尾部；足以作为强跟踪 teacher，且不是不可学习的完美可见上限 | B0 固定为 `runs/PE_20260520_110828/policy_pool/ppo_upd_001300_step_2396160000.pth`；后续 LiDAR rollout、risk labels 和 B1-B4 对比均以该 checkpoint 作为当前 nominal baseline，直到明确训练出更强 B0 替代版 |
| D039 | Warp LiDAR sensor capture 前必须同步/重拟合 Warp mesh；修复前 rollout 统计不得用于 reward/curriculum 结论 | 继续用修复前 `target_pixel_count=0` 统计判断 teacher 不可见；保留投影/center-ray/terminal-snapshot debug 分支作为长期代码 | 已确认动态 target mesh stale 会造成 geometry detectable 与 semantic pixels 错帧；修复后同一 5 checkpoint 的 `visible|geometry` 变为 `1.0` 且 `geom_pix0=0` | 只保留 `EnvManager.render_sensors()` 中的 mesh sync core fix；删除辅助 debug 检查代码和产物；下一步用干净 exporter 重新统计可见率/像素后再决定是否重训 B0 |
| D040 | 风险模型推进采用离线冻结-在线聚合的 DAgger-like 周期，而不是 PPO/risk model 同步在线共训 | 第一版直接将 risk model 与 PPO 同步更新；只做单次离线训练后永久冻结；只用最新 risk-PPO 数据继续训练 risk model | 同步共训会造成闭环偏差：PPO 被当前 risk 改变分布，risk model 又只在变窄的新分布上更新；单次离线冻结又无法覆盖 B4 访问的新状态。DAgger-like 聚合能纳入当前策略分布，同时保留旧策略和 hard cases | 路线固定为 `D0 offline policy-pool -> R0 frozen risk -> risk PPO -> D1 online aggregation -> R1 offline retrain`；每轮数据必须混合 offline/current/noise/hard-case/scripted 来源 |
| D041 | 风险模型标签升级为概率/时间/严重度/恢复多头，bool 只作为派生标签 | 只训练 `observability_loss_label` 二分类；把 target semantic/mask 作为模型输入；直接训练 trajectory prediction | bool 不能区分 1 step 后丢失与 140 step 后丢失、短暂恢复与长期脱离，也不能给 action 足够区分度；semantic 是 sim-only oracle，不能部署 | 默认输出 `p_loss_H`、`severity_H`、`first_loss_offset`、`p_recover_H`；semantic/pixel/bbox 只保留 QA metadata |
| D042 | PPO 风险惩罚 lambda 先固定网格定标，再使用 adaptive dual lambda | 只做固定网格；从第一版就 adaptive；按单帧 invisible rate 快速调 lambda | 固定网格能确认 reward scale 不压死追击；adaptive 需要已校准 risk 与稳定 constraint 指标；单帧短暂丢失不应驱动 lambda 大幅变化 | 第一轮 lambda 取 `{0,0.02,0.05,0.1,0.2}`；第二轮按 persistent/over-horizon 指标慢速更新 lambda |
| D043 | schema7 formal risk label 使用 semantic target pixel visibility，而不是 privileged geometry detectable | 继续用几何 frustum detectable 直接生成 label；把 semantic target mask 作为模型输入；完全删除 geometry detectable | 训练输入是 deployable LiDAR stream，label 应以实际 sensor-space 可见像素为准；geometry detectable 只反映特权状态和传感器几何，不表达遮挡、raycast 和实际像素命中。semantic ID 是 sim-only oracle，不能作为模型输入，但可作为离线标签和 QA 真值 | `label_source=semantic_target_pixel_count`，默认 `target_pixel_count >= 3` 为 visible；`label_detectable` 保留为 QA/audit metadata，不进入 risk-model input |
| D044 | 可视化 semantic 审计拆成独立脚本，formal exporter 只写可训练 shard 与 metadata | 给 exporter 增加 `--semantic-debug-*` 参数并混合保存 JPG/PNG；继续用临时 smoke 脚本手工检查 | formal dataset 路径必须保持轻量和可复现，避免 debug image 继续膨胀采集代码；同时 semantic 正确性需要可独立复核的 range/mask/overlay/target-pixel artifact | 新增 `audit_pursuit_lidar_semantics.py`：完整单 episode 保存 `range.jpg`、`target_mask.jpg`、`overlay.jpg` 和 `target_pixels.npz`；formal exporter 不保存 debug 图像 |
| D045 | schema7 500 episode 首次正式采集采用 `pilot + qa_baseline_output`，不复用旧 baseline | 直接 `collect` 并拿旧 schema6 baseline；没有 baseline 时关闭 QA gate；先大规模 collect 再补 baseline | schema7 label source、threshold 和 metadata 字段均已改变，旧 baseline 不可比；pilot 模式仍运行基础 QA gate，并能在采集完成后生成后续 collect 可用的 schema7 baseline | 首次 500 episode 命令使用 `--qa-mode pilot --qa-baseline-output qa_baseline_schema7.json`；后续重复采集再用 `--qa-mode collect --qa-baseline-path ...` |
| D046 | B4 第一版定型为 mean-baseline risk-adjusted advantage PPO | 连续动作策略分布重塑 `pi(u|o) exp(-beta rho)`；第一版 actor latent/mean/variance modulation；第一版 risk value critic；minimum baseline 主方法 | 连续动作分布重塑需要处理归一化、logprob/ratio 和 PPO credit assignment，工程风险高；minimum baseline 容易变成保守 shield/filter；advantage 修正能保持 PPO 执行动作与 logprob 一致，同时把动作条件风险作为结构化学习信号 | B4 v0 使用 `rho_bar=mean_i rho(o,u_i)` 和 `A_tilde=A_task-lambda*clip(norm(rho(o,u_exec)-rho_bar),-3,3)`；actor/critic 主结构不改，lambda 网格 `{0,0.02,0.05,0.1,0.2}` |
| D047 | 风险头输入/输出契约定型，且不使用 `policy_mean` 作为动作 reference | `R(o,u-policy_mean)`；把 PPO mean 作为行为克隆式动作锚点；只预测单一 `H=150` binary loss | 用户曾观察到直接使用 PPO action mean 行为克隆效果很差；风险模型需要学习实际动作对短期可观测性的影响，而不是依赖 mean action 作为伪参考。单一 binary 标签无法区分短时严重脱视野、长期轻微脱视野、first loss 和 recovery | 输入为 `concat(s_red,z_lidar,u,u-prev_action,abs(u),u^2)`；输出为 `p_loss_50/severity_50/first_loss_50/p_loss_150/severity_150/p_recover_150`，PPO v0 只使用两个 horizon 的 loss probability 与 severity |
| D048 | 离线 branch 候选动作加入 thrust 扰动并保留 yaw 扰动 | 只采 roll/pitch/yaw；只依赖 policy sample 随机覆盖 thrust；在线 PPO 第一版强制纳入 thrust 候选 | 当前 pilot 已有 yaw±，但原始 8 候选没有显式 thrust±。thrust 会影响闭合速度、距离变化、垂向运动、过冲和可见性恢复；若训练集中缺少受控 thrust 变化，风险头可能低估速度/减速带来的 observability risk 差异。在线强制纳入 thrust 则可能因饱和和保守减速引入不必要风险 | 离线 branch 默认 `M_train=16`，roll/pitch/yaw/thrust 单轴扰动均参与并做 L2 diversity check；PPO 在线 baseline 默认 `M_ppo=8`，thrust 候选作为后续 ablation |

## 已锁定实验规则
- B0 是全状态上界参考和标签来源，不是本文方法。
- B1-B4 是主比较链条。
- B2 LiDAR-honest heuristic risk 必须实现为手工 baseline；若其效果弱，也保留为弱基线而非从主表删除。
- B4 是本文主方法。
- B4 必须显著优于 B3、risk concat、PPO-Lagrangian 和 shield/filter baseline，才能支撑“动作条件风险调制 PPO”的主贡献。

## 已锁定写作规则
- 不再写“full-state PPO + extra scalar”路线。
- 不再写“完整 onboard visual pursuit”。
- 不再把 bilateral self-play 放入论文 1 主贡献。
- 不再把 generic target prediction 当作核心创新。
- 不再把仿真语义真值、目标 ID、真值 target-hit 或人工 mask 作为主线模型输入；需要 mask/bbox 时必须以真实检测器 noisy output 另行定义。
- 不再把 depth-only sensor-proxy 作为论文 1 主传感器声明；当前主线改为 LiDAR/range-image reduced-information pursuit。
- 不再把当前 target 逃逸者称作高保真多旋翼执行器模拟；只能称为规则/APF 逃逸目标的刚体级高层控制近似。
- 不把 100Hz LiDAR 全帧刷新写成真实硬件事实；若用于仿真同步，必须单独标注为 simulator-side sensor update 或 ROI/局部扫描假设。
- 不把 `target_x500 + M3-like` 在 100/150/200m 的 sparse returns 写成真实长距 LiDAR 部署保证；只能写成 sparse-return stress-test regime，除非后续加入真实 LiDAR 日志或硬件校准。
- 当前草稿先使用 `IEEEtran` / IEEE Transactions 双栏模板；不要在当前阶段把 `elsarticle.cls` 缺失作为写作或编译阻塞。EAAI/Elsevier 官方模板切换留到投稿前格式化。

## 待实现决策
| 决策 ID | 问题 | 默认答案 | 阶段 |
|---------|------|----------|------|
| I001 | `R_det`、`alpha_h`、`alpha_v` 的默认值 | 以相机 FOV/像素可检测距离为依据，并做敏感性分析 | Phase 6 |
| I002 | B2 heuristic risk 具体公式 | 只能使用 LiDAR-honest proxy，不使用目标/环境真实值 | Phase 6 |
| I003 | B4 风险调制 PPO 具体结构 | 已定型：B4 v0 采用 mean-baseline risk-adjusted advantage；actor 分布重塑、risk value critic、minimum baseline 和 shield/filter 作为 ablation | Phase 3 complete / Phase 6.4 implementation |
| I004 | LiDAR 主配置 | 已锁定：`target_x500 + M3-like` ultra-high-res 前向 profile；AT128P-like/OS2-64 仅作为历史失败对照记录 | Phase 6 |
| I005 | 主线目标 URDF 是否从 `target_quad` 切到 `target_x500` | 已切换：`target_x500` 是默认 target，`target_quad` 仅作显式对照 | Phase 6 |
| I006 | visibility-aware teacher 是否应把不可见作为终止条件 | 当前默认不终止；短暂丢视野保留恢复轨迹，长时间丢失通过 penalty 和 recovery stats 暴露。若后续 teacher 仍无法恢复，再考虑 persistent-loss termination 作为单独 ablation | Phase 6 |
| I007 | 提高 visibility 权重后是否继续使用单纯加权路线 | 已否决继续只加权；`0.20/0.40/0.60` 作为当前上限参考，下一轮使用 150-step recovery-aware penalty 和 visibility-gated curriculum | Phase 6 |
| I008 | B4 必需近邻 ablation | risk concat、PPO-Lagrangian、shield/filter、privileged critic、shuffled-action risk、matched-state action-risk ranking | Phase 6.4 |
