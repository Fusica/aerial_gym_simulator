# Phase 2 Survey: Landscape and Paper Clusters

Date: 2026-05-19

Master DB: `paper_db.jsonl` with 52 records after auditing the user-provided candidate papers.

## Intro / Related Work Taxonomy

The related-work structure should be organized by the paper's scientific constraints, not by source list or by RL algorithm family. The core is **limited field/detectability + pursuit-evasion + constrained action optimization**; reinforcement learning is supporting machinery.

| Axis | Representative papers | Role in the paper |
|---|---|---|
| Limited field / detectability / sensor-honest pursuit | `[@peng2025limitedvisual]`, `[@huh2026limitedregion]`, `[@shao2026peqmixer]`, `[@wang2025viper]`, `[@zhou2025visibilityocclusion]`, `[@li2025bearingonly]`, `[@zheng2025visioncapture]`, `[@mavcapturingmav2024]` | Establish that limited-FOV, limited detectable region, reacquisition and sensor-compatible pursuit are active; our distinction is calibrated action-conditioned observability-loss risk under LiDAR/range-image reduced inputs. |
| UAV pursuit-evasion / interception / online planning / CTBR | `[@chen2025open]`, `[@zhang2023gameofdrones]`, `[@roncero2025agilecontrollers]`, `[@yan2024lsrctd3]`, `[@giral2026intercept]`, `[@zhao2025msmar]`, `[@zhao2024autonomousuavpe]`, `[@luo2024improvedmadrl]`, `[@tan2026scalablefixedwing]`, `[@xiang2025cihrl]` | Defines real domain pressure: online planning, prediction-enhanced PE, CTBR/body-rate-thrust control, high-speed/agile pursuit and multi-UAV PE are not our novelty. `[@chen2025open]` is a strong neighbor, not excluded. |
| Constrained action optimization / risk-gated control | `[@achiam2017cpo]`, `[@thananjeyan2021recoveryrl]`, `[@alshiekh2018shielding]`, `[@zhang2026safetyshieldedflight]`, `[@li2026safeintent]`, `[@liu2026actionriskgating]`, `[@kim2024predictivecvar]`, `[@safetygym2019]`, `[@grape2019riskaware]` | This is the main methodological comparison family. It forces PPO-Lagrangian, shield/filter, risk concat, privileged critic and shuffled-action risk baselines. |
| RL / privileged training / engineering machinery | `[@schulman2017ppo]`, `[@pinto2017aac]`, `[@baisero2022adqn]`, `[@ebi2026iaac]`, `[@yang2025rlpereview]`, `[@ren2025losrate]`, `[@liu2026autopilot]`, `[@gao2026mentalstates]`, `[@park2023lidardrone]`, `[@applsci2026lidarbelief]` | Supports implementation choices and EAAI scope. PPO/MARL/DRL and privileged learning are not the scientific contribution. |

The section should close with this contrast: prior work covers limited detectability, pursuit-evasion and safe/risk-aware policy optimization separately; this paper isolates **action-conditioned observability-loss risk** and uses it to modulate PPO under strict LiDAR/range-image reduced inputs.

## Cluster A: EAAI UAV/RL Engineering Applications

Representative papers: `[@ren2025losrate]`, `[@liu2026autopilot]`, `[@gao2026mentalstates]`, `[@anonymous2026situationaware]`, `[@anonymous2026cooperativeuav]`, `[@liu2025evtol]`, `[@liu2026physicsuav]`, `[@andres2025windquad]`, `[@hou2023subtaskmasked]`, `[@nugroho2023poweredlanding]`.

Findings:

- EAAI accepts UAV guidance/control/navigation papers using PPO, DDPG, MAPPO, curriculum learning, world models and explainable/robust control components.
- The journal values engineering framing: measurable application performance, robustness, reliability and reproducibility.
- EAAI does not require a pure algorithmic RL breakthrough, but it does require the AI contribution to be explicit and non-trivial relative to application baselines.

Implication:

The current plan is in scope if the abstract says:

- AI contribution: action-conditioned observability-loss risk model and PPO modulation under sensor-honest reduced inputs.
- Engineering application: high-speed UAV pursuit with sparse LiDAR/range-image target observability.

## Cluster B: Limited-FOV / Partial-Observability UAV Pursuit

Representative papers: `[@peng2025limitedvisual]`, `[@huh2026limitedregion]`, `[@shao2026peqmixer]`, `[@li2025bearingonly]`, `[@chen2025open]`, `[@wang2025viper]`, `[@zhou2025visibilityocclusion]`, `[@cao2019droneschasing]`, `[@zhang2019coarsetofine]`.

Findings:

- Limited visual field, limited detectable region and target reacquisition are already active pursuit topics.
- Several papers already use APF or repulsive-force evaders, random evaders and RL evaders as scenario families.
- Existing work often emphasizes MARL coordination, phase switching, target prediction, FOV-dependent rewards, camera/vision tracking or bearing-only filtering.

Implication:

The paper must not claim first limited-FOV pursuit. It should instead claim a controlled single-pursuer or reduced-information setting where the scientific object is the **risk variable and PPO integration**, not MARL cooperation.

## Cluster C: Safe / Risk-Sensitive RL

Representative papers: `[@achiam2017cpo]`, `[@thananjeyan2021recoveryrl]`, `[@stooke2020pidlag]`, `[@alshiekh2018shielding]`, `[@ma2022cap]`, `[@kim2024predictivecvar]`, `[@ma2020dsac]`, `[@safetygym2019]`, `[@zhao2025msmar]`.

Findings:

- Learning a separate cost/risk critic is well-established.
- Risk-adjusted advantages and Lagrangian objectives are already standard safe-RL patterns.
- Action masking, shielded RL and recovery policies already intervene on unsafe actions.
- In high-speed pursuit, recovery/safety-aware RL is now a strong top-conference neighbor.

Implication:

B4 must not be described as a new safe PPO. If B4 is just `A_task - lambda A_risk`, the method is too close to PPO-Lagrangian/CPO. To be publishable, it must emphasize:

- observability-loss semantics rather than safety/collision semantics;
- deployable LiDAR/range-image reduced inputs;
- action-conditioned risk ranking;
- comparison against PPO-Lagrangian and shield/filter baselines using the same risk model.

## Cluster D: Privileged-to-Partial / Asymmetric Actor-Critic

Representative papers: `[@pinto2017aac]`, `[@baisero2022adqn]`, `[@ebi2026iaac]`.

Findings:

- Using privileged simulator state during training while restricting deployment observation is not new.
- This is common in robot RL and partial-observation settings.

Implication:

The plan should position privileged rollouts as a supervision and label-generation mechanism, not as the central novelty. The novelty should be in the observability-risk target, action conditioning and PPO modulation under LiDAR-reduced inputs.

## Cluster E: LiDAR / Range-Image Policies and Sparse Target Sensing

Representative papers: `[@park2023lidardrone]`, `[@applsci2026lidarbelief]`, `[@christian2024vtd3]`, `[@cao2019droneschasing]`, `[@grape2019riskaware]`.

Findings:

- LiDAR and depth policies are common for navigation/avoidance but less common for high-speed UAV-to-UAV pursuit.
- Vision target tracking papers show small target detection is a real bottleneck; LiDAR sparse returns for small UAVs should be treated as a stress condition.
- A “100/150/200m M3-like target_x500 return” claim is risky unless framed as simulation stress testing and backed with return-count/visibility sweeps.

Implication:

The engineering part of the paper needs sensor feasibility plots, return-count statistics, dropout/sparsity ablation and failure reporting. Do not write “sim-to-real” without real logs.

## Survey-Level Novelty Judgment

Conditionally sufficient for EAAI:

- The scope and application fit EAAI.
- The novelty is enough only if the method is scoped as **action-conditioned observability-risk modulation for LiDAR/range-image reduced-information UAV pursuit**.
- It is not enough if the paper becomes generic safe PPO, generic limited-FOV pursuit or generic privileged learning.

## 2026-05-19 Candidate Paper Integration

The user-provided candidate papers were audited against the current mainline instead of being bulk-added. The audit reviewed 31 candidates, accepted 11 active records, rejected 18 from the active DB, and skipped 2 direct duplicates already covered by the refresh:

- `33KIR7IV` overlaps with `[@shao2026peqmixer]`.
- `F5LGULL9` overlaps with `[@chen2025open]`.

Accepted records were retained only when they serve at least one current-paper role: real comparison pressure, B4 ablation/design pressure, or required Related Work coverage. The audit changes the qualitative judgment in two ways:

1. **Scope is stronger.** The expanded pool confirms that UAV pursuit-evasion, high-speed interception, cooperative pursuit, autonomous maneuver decision-making and RL-based aerial-robot interception are active engineering-AI topics.
2. **Broad novelty is weaker.** High-speed UAV PE, multi-UAV PE/MARL, agile quadrotor PE, vision-based capture, shielded high-speed flight and RL interception are crowded enough that none of them can be claimed as the main novelty.

High-pressure clusters retained from the audit:

| Cluster | Representative added keys | Meaning for this paper |
|---|---|---|
| High-speed UAV PE / intelligent maneuvering | `[@yan2024lsrctd3]`, `[@giral2026intercept]`, `[@roncero2025agilecontrollers]` | Do not claim high-speed pursuit, agile PE controllers or aerial interception as new. |
| Multi-UAV PE / MARL / assignment | `[@zhao2024autonomousuavpe]`, `[@luo2024improvedmadrl]`, `[@tan2026scalablefixedwing]`, `[@xiang2025cihrl]` | Keep paper 1 away from cooperation/scalability/self-play claims. |
| Safety shield / safe navigation | `[@zhang2026safetyshieldedflight]`, `[@feroskhan2024multipursuitevasion]` | PPO-Lagrangian and shield/filter baselines are mandatory. |
| Sensor/capture systems | `[@mavcapturingmav2024]` | Do not claim first sensor-based capture; claim observability-risk modeling under strict LiDAR inputs. |

Re-derived conclusion: the contribution remains conditionally sufficient only as **action-conditioned observability-loss risk modulation for LiDAR/range-image reduced-information UAV pursuit**. The paper must explicitly separate this from generic UAV PE, generic limited perception and generic safe RL.

✅ Phase 2 complete. Output: `phase2_survey/survey.md`, `paper_db.jsonl`. Proceeding to Phase 3.
