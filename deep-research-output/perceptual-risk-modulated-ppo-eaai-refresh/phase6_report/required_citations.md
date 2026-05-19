# Required Citation List

Date: 2026-05-19

Purpose: citation checklist for the planned EAAI paper. This is not the full active paper database. It lists the papers that should appear in the Introduction / Related Work / Method baseline discussion, organized by the four scientific axes in the current plan.

Legend:
- **Priority**: `must` = should appear in the main paper; `support` = cite if space allows or when making that specific point.
- **DB**: whether the key exists in `paper_db.jsonl`.
- **BibTeX**: whether the key currently exists in `phase6_report/references.bib`.

## Axis 1: Limited Field / Detectability / Sensor-Honest Pursuit

| Key | Priority | Paper | Why cite | DB | BibTeX |
|---|---|---|---|---|---|
| `peng2025limitedvisual` | must | Multi-UAV Cooperative Pursuit Strategy With Limited Visual Field in Urban Airspace | Direct limited-FOV UAV pursuit and reacquisition neighbor. | yes | yes |
| `huh2026limitedregion` | must | Multi-Agent Reinforcement Learning for Multi-UAV Pursuit with Full Planar Motion and a Limited Detectable Region | Direct limited-detectable-region/FOV-reward neighbor. | yes | yes |
| `shao2026peqmixer` | must | A perception-enhanced multi-agent deep reinforcement learning method for multi-UAV cooperative pursuit | Perception-enhanced cooperative pursuit pressure. | yes | missing |
| `wang2025viper` | must | ViPER: Visibility-based Pursuit-Evasion via Reinforcement Learning | Visibility-based pursuit-evasion pressure. | yes | missing |
| `zhou2025visibilityocclusion` | must | Control Strategies for Pursuit-Evasion Under Occlusion Using Visibility | Occlusion/visibility pursuit pressure. | yes | missing |
| `li2025bearingonly` | must | Cooperative Bearing-Only Target Pursuit via Multiagent Reinforcement Learning: Design and Experiment | Sensor-compatible pursuit without full target state. | yes | missing |
| `zheng2025visioncapture` | must | Vision-compatible UAV capture or target pursuit under onboard sensing constraints | Sensor-compatible UAV capture/pursuit pressure. | yes | missing |
| `mavcapturingmav2024` | support | Vision-Based Cooperative MAV-Capturing-MAV | Vision-based MAV capture system; useful to avoid sensor-capture overclaim. | yes | yes |
| `cao2019droneschasing` | support | Drones Chasing Drones: Reinforcement Learning and Deep Search Area Proposal | Early drone-chasing RL and target-search context. | yes | missing |
| `zhang2019coarsetofine` | support | Coarse-to-Fine UAV Target Tracking With Deep Reinforcement Learning | UAV target tracking with DRL background. | yes | missing |
| `christian2024vtd3` | support | A Vision-Based End-to-End Reinforcement Learning Framework for Drone Target Tracking | Vision-based drone target-tracking background. | yes | missing |

## Axis 2: UAV Pursuit-Evasion / Interception / Online Planning / CTBR

| Key | Priority | Paper | Why cite | DB | BibTeX |
|---|---|---|---|---|---|
| `chen2025open` | must | OPEN: Online Planning for Multi-UAV Pursuit-Evasion in Unknown Environments by Deep Reinforcement Learning | Strong near neighbor for online planning, prediction-enhanced multi-UAV PE and CTBR relevance. | yes | missing |
| `zhang2023gameofdrones` | must | Game of Drones: Multi-UAV Pursuit-Evasion Game With Online Motion Planning by Deep Reinforcement Learning | Strong online-motion-planning PE neighbor; metadata must be added. | missing | missing |
| `roncero2025agilecontrollers` | must | Learned Controllers for Agile Quadrotors in Pursuit-Evasion Games | Agile quadrotor PE and CTBR-like controller pressure. | yes | yes |
| `yan2024lsrctd3` | must | A Deep Reinforcement Learning-Based Intelligent Maneuvering Strategy for the High-Speed UAV Pursuit-Evasion Game | High-speed UAV PE / DRL maneuvering pressure. | yes | yes |
| `giral2026intercept` | must | Intercepting Unauthorized Aerial Robots in Controlled Airspace Using Reinforcement Learning | Aerial interception with RL; application-domain pressure. | yes | yes |
| `zhao2025msmar` | must | MSMAR-RL: Multi-Step Masked-Attention Recovery Reinforcement Learning for Safe Maneuver Decision in High-Speed Pursuit-Evasion Game | High-speed PE + recovery/safe maneuver RL; one of the closest threats. | yes | yes |
| `zhao2024autonomousuavpe` | support | Autonomous Decision Making for UAV Cooperative Pursuit-Evasion Game with Reinforcement Learning | Cooperative UAV PE with RL; helps delimit non-MARL claim. | yes | yes |
| `luo2024improvedmadrl` | support | Multi-UAV Cooperative Maneuver Decision-Making for Pursuit-Evasion Using Improved MADRL | Mature multi-UAV PE/MADRL comparison context. | yes | yes |
| `tan2026scalablefixedwing` | support | Scalable Pursuit-Evasion Game for Multi-Fixed-Wing UAV Based on Dynamic Target Assignment and Hierarchical Reinforcement Learning | Scalability/assignment pressure. | yes | yes |
| `xiang2025cihrl` | support | Decentralized Consensus Inference-Based Hierarchical Reinforcement Learning for Multi-Constrained UAV Pursuit-Evasion Game | Multi-constrained hierarchical UAV PE pressure. | yes | yes |
| `feroskhan2024multipursuitevasion` | support | Learning Multi-Pursuit Evasion for Safe Targeted Navigation of Drones | Drone multi-pursuit/evasion and real-flight-validation pressure. | yes | yes |

## Axis 3: Constrained Action Optimization / Risk-Gated Control

| Key | Priority | Paper | Why cite | DB | BibTeX |
|---|---|---|---|---|---|
| `achiam2017cpo` | must | Constrained Policy Optimization | Core constrained-policy-optimization baseline family. | yes | yes |
| `stooke2020pidlag` | must | Responsive Safety in Reinforcement Learning by PID Lagrangian Methods | PPO-Lagrangian / PID-Lagrangian comparison family. | yes | missing |
| `thananjeyan2021recoveryrl` | must | Recovery RL: Safe Reinforcement Learning with Learned Recovery Zones | Closest action-conditioned safety critic / recovery-policy analogy. | yes | yes |
| `alshiekh2018shielding` | must | Safe Reinforcement Learning via Shielding | Shield/filter baseline family. | yes | missing |
| `zhang2026safetyshieldedflight` | must | High-Speed Vision-Based Flight in Clutter with Safety-Shielded Reinforcement Learning | High-speed flight with safety shield; justifies shield/filter ablation. | yes | yes |
| `li2026safeintent` | must | Safe Cooperative Decision-Making for Multi-UAV Pursuit-Evasion Games via Opponent Intent Inference | Risk-gated / safe partial-observable PE neighbor. | yes | missing |
| `liu2026actionriskgating` | must | Action-Conditioned Risk Gating for Safety-Critical Control under Partial Observability | Generic action-conditioned risk-gating threat. | yes | missing |
| `kim2024predictivecvar` | support | Predictive CVaR Policy Gradient | Risk-sensitive policy-gradient background. | yes | missing |
| `ma2022cap` | support | Conservative and Adaptive Penalty for Model-Based Safe Reinforcement Learning | Penalty-based safe RL background. | yes | missing |
| `ma2020dsac` | support | Distributional Soft Actor Critic for Risk-Sensitive Reinforcement Learning | Risk-sensitive distributional RL background. | yes | missing |
| `safetygym2019` | support | Benchmarking Safe Exploration in Deep Reinforcement Learning | Safe-RL benchmark context. | yes | missing |
| `grape2019riskaware` | support | Geometric Risk-Aware Pursuit-Evasion for Drones with Uncertain Onboard Sensing | Risk-aware PE with uncertain sensing; useful bridge to sensing-risk framing. | yes | missing |

## Axis 4: RL / Privileged Training / Engineering Machinery

| Key | Priority | Paper | Why cite | DB | BibTeX |
|---|---|---|---|---|---|
| `schulman2017ppo` | must | Proximal Policy Optimization Algorithms | Base PPO method. | yes | yes |
| `pinto2017aac` | must | Asymmetric Actor Critic for Image-Based Robot Learning | Privileged critic / asymmetric training precedent. | yes | yes |
| `baisero2022adqn` | support | Asymmetric DQN for Partially Observable Reinforcement Learning | Privileged-to-partial RL background. | yes | missing |
| `ebi2026iaac` | support | Informed Asymmetric Actor-Critic: Leveraging Privileged Signals Beyond Full-State Access | Newer privileged-signal actor-critic background. | yes | missing |
| `yang2025rlpereview` | must | A review of reinforcement learning approaches for pursuit-evasion games | Broad PE/RL survey anchor. | yes | yes |
| `ren2025losrate` | must | Maneuvering target interception via deep reinforcement learning guidance using only line-of-sight rate measurement | EAAI reduced-measurement DRL guidance scope support. | yes | yes |
| `liu2026autopilot` | must | A curriculum-guided and explainable reinforcement learning framework for fixed-wing unmanned aerial vehicle autopilots | EAAI UAV PPO/curriculum engineering scope support. | yes | missing |
| `gao2026mentalstates` | must | Hierarchical mental-states reasoning with dynamic fusion world models for imperfect multi-unmanned aerial vehicle cooperative-competitive environments | EAAI imperfect-information multi-UAV scope support. | yes | missing |
| `park2023lidardrone` | support | LiDAR-based drone navigation with reinforcement learning | LiDAR policy background. | yes | missing |
| `applsci2026lidarbelief` | support | Deep Reinforcement Learning for Navigation via Multi-Modal Belief State Representation from LiDAR and Depth Sensors | LiDAR/depth belief-state policy background. | yes | missing |

## BibTeX Completion Checklist

Already present in `references.bib`: 19 keys.

Missing and should be added before paper writing:

- `shao2026peqmixer`
- `wang2025viper`
- `zhou2025visibilityocclusion`
- `li2025bearingonly`
- `zheng2025visioncapture`
- `cao2019droneschasing`
- `zhang2019coarsetofine`
- `christian2024vtd3`
- `chen2025open`
- `zhang2023gameofdrones`
- `stooke2020pidlag`
- `alshiekh2018shielding`
- `li2026safeintent`
- `liu2026actionriskgating`
- `kim2024predictivecvar`
- `ma2022cap`
- `ma2020dsac`
- `safetygym2019`
- `grape2019riskaware`
- `baisero2022adqn`
- `ebi2026iaac`
- `liu2026autopilot`
- `gao2026mentalstates`
- `park2023lidardrone`
- `applsci2026lidarbelief`

Special action:

- `zhang2023gameofdrones` is cited in the plan but is currently missing from both `paper_db.jsonl` and `references.bib`; add verified metadata before using it in the manuscript.
