# Deep Dive Notes: Perceptual-Risk Pursuit Review
Date: 2026-05-09

### [@peng2025limitedvisual] Multi-UAV Cooperative Pursuit Strategy With Limited Visual Field in Urban Airspace

**Metadata**
- Authors: Zhe Peng, Guohua Wu, Biao Luo, Ling Wang
- Year: 2025 | Venue: IEEE/CAA Journal of Automatica Sinica
- DOI: 10.1109/JAS.2024.124965
- Code: supplementary material reported on the journal page; no direct repository confirmed in this review

**Problem**
How to coordinate multiple UAV pursuers in urban airspace when visual field is limited by perception distance, viewing angle, and building obstruction.

**Key Contributions**
1. Defines a multi-UAV pursuit-evasion setting with limited visual field and target reacquisition phases.
2. Proposes NAGC, combining a normalizing-flow actor with an obstacle-target graph attention critic.
3. Evaluates search, pursuit, and reacquisition behavior in a high-precision simulator.

**Why It Matters For This Project**
- It is a strong high-tier warning that limited-FOV pursuit alone is not novel.
- The current project must therefore differentiate through action-conditioned observability-risk learning and CTBR correction, not only by adding a field-of-view constraint.

**Limitations For Our Claim**
- It does not learn a separately calibrated observability-loss risk label.
- It does not use privileged rollout labels to bridge toward depth-only reduced-information control.
- It does not define CTBR action correction as the central mechanism.

### [@li2026safeintent] Safe Cooperative Decision-Making for Multi-UAV Pursuit-Evasion Games via Opponent Intent Inference

**Metadata**
- Authors: Wenxin Li, Yongxin Feng, Wenbo Zhang
- Year: 2026 | Venue: Sensors
- DOI: 10.3390/s26072243
- Code: not confirmed in this review

**Problem**
How to make cooperative multi-UAV pursuit-evasion safer under occlusion, sensor noise, intermittent observability, variable observation windows, and non-stationary evader tactics.

**Key Contributions**
1. Infers evader behavior mode and subgoal from short observation windows.
2. Distills intent/subgoal structure into a length-agnostic trajectory predictor with calibrated uncertainty.
3. Builds a belief-risk-gated hierarchical multi-agent SAC policy with a safety projection layer.

**Why It Matters For This Project**
- It is the closest risk-aware partial-observation pursuit neighbor.
- It makes generic "risk-gated pursuit" framing too weak for the current project.

**Limitations For Our Claim**
- Its risk features are derived from intent/subgoal and trajectory prediction, not directly from an observability-loss label.
- It is a cooperative hierarchical MARL framework, not a 1v1 privileged-to-depth sensor-proxy transition.
- It does not center CTBR action-conditioned observability-risk correction.

### [@li2025bearingonly] Cooperative Bearing-Only Target Pursuit via Multiagent Reinforcement Learning

**Metadata**
- Authors: Jianan Li, Zhikun Wang, Susheng Ding, Shiliang Guo, Shiyu Zhao
- Year: 2025 | Venue: arXiv
- arXiv: 2503.08740
- Code: not confirmed in this review

**Problem**
How to combine bearing-only target estimation and multiagent pursuit control when target state is not directly observable and limited FoV can cause target loss.

**Key Contributions**
1. Uses a uniform bearing-only information filter for 3D target estimation.
2. Addresses target-loss resilience under limited field of view.
3. Integrates estimation with multiagent RL pursuit control and experiments.

**Why It Matters For This Project**
- It is a strong sensor-compatible pursuit reference and should be cited when discussing why direct target state is unrealistic.
- It reinforces the need to remove truth-derived bearing/range from the mainline policy.

**Limitations For Our Claim**
- It uses bearing-only filtering rather than depth-only risk representation learning.
- It does not learn action-conditioned observability risk from privileged rollouts.
- It does not use CTBR risk-constrained action correction.

### [@huh2026limited] Multi-Agent Reinforcement Learning for Multi-UAV Pursuit with Full Planar Motion and a Limited Detectable Region

**Metadata**
- Authors: Soobin Huh, Sungwon Lim, Hyeokjae Jang, Woohyun Byun, Suhyeong Yu, Woochul Nam
- Year: 2026 | Venue: Machines
- DOI: 10.3390/machines14040413
- Code: not reported in the article page

**Problem**
How to perform cooperative multi-UAV pursuit when each pursuer has a limited detectable region defined by practical FOV and range constraints.

**Key Contributions**
1. Formalizes cooperative UAV pursuit with a limited detectable region rather than broad or idealized visibility.
2. Uses more realistic UAV motion assumptions, explicitly separating roles of yaw and lateral motion under sensing constraints.
3. Quantitatively studies the value of lateral motion under limited detectability and reports a large success-rate increase.

**Methodology**
- Approach type: MARL / MAPPO-style cooperative pursuit
- Key idea: model pursuit as a Dec-POMDP where each pursuer has a sector-shaped detectable region with practical range/FOV constraints.
- Key components:
  - limited detectable region defined from a camera-model approximation
  - reward shaping for approach, visibility maintenance, and capture
  - large-scale parallel simulation on Isaac-based infrastructure
  - action space that includes yaw and lateral motion
- Novel aspects: combines practical visual constraints with richer planar maneuver authority; prior work often considered only subsets of these assumptions.

**Experiments**
- Setup: three pursuers vs one evader; 30 m max detection range; episodes begin after first detection.
- Comparisons: configurations with and without lateral motion, plus comparisons against prior assumptions summarized in the paper.
- Main results:
  - success rate increased to 99.2% when lateral motion was enabled.
  - the paper argues this shows lateral motion is critical under limited-detectable-region constraints.
- Ablations:
  - action-space analysis highlights distinct roles of yaw for visibility and lateral motion for spatial repositioning.

**Limitations**
- Acknowledged by authors:
  - scope centers on cooperative capture under fixed limited-detectable-region assumptions.
- Observed by reader:
  - the paper mostly relies on reward shaping and observation constraints, not a separate learned observability predictor.
  - this makes it an important baseline, but not a direct solution to action-conditioned observability-risk correction.

**Connections**
- Builds on: limited-FOV pursuit literature and APF-style evader motion.
- Related approaches: `[@li2024limitedusv]`, `[@sun2022pomdp]`, `[@liqin2026perception]`

**Code & Resources**
- Repository: not clearly linked on the article page
- Datasets released: none reported
- Models released: none reported

### [@xu2025general] A DRL Framework for Autonomous Pursuit-Evasion: From Multi-Spacecraft to Multi-Drone Scenarios

**Metadata**
- Authors: Zhenyang Xu, Shuyi Shao, Zengliang Han
- Year: 2025 | Venue: Drones
- DOI: 10.3390/drones9090636
- Code: not reported in the article page

**Problem**
How to build a single pursuit-evasion DRL framework that generalizes across very different domains and still handles complex terminal constraints.

**Key Contributions**
1. Proposes a unified pursuit-evasion framework used in both spacecraft and drone scenarios.
2. Introduces dynamics-agnostic curriculum learning for training efficiency and robustness.
3. Introduces a prediction-based reward to provide dense guidance toward state-dependent terminal conditions.

**Methodology**
- Approach type: self-play PPO / curriculum learning / reward design
- Key idea: keep the control-learning framework modular across domains while changing dynamics and environment settings.
- Key components:
  - PPO policy optimization
  - curriculum learning over scenario difficulty
  - prediction-based reward with forward-state projection
  - validation on both spacecraft and multi-drone pursuit
- Novel aspects: cross-domain generalization emphasis and transferable prediction-based reward.

**Experiments**
- Tasks: spacecraft pursuit-evasion and multi-drone pursuit-evasion, including obstacle-rich environments and larger team settings.
- Baselines: PPO and SAC are explicitly mentioned as main baselines.
- Main results:
  - 90.7% success in the primary spacecraft validation.
  - only 8.3% performance drop under stochastic perturbations, versus more than 18% for baselines.
  - successful transfer to multi-drone and obstacle-rich settings.
- Ablations:
  - curriculum learning and prediction-based reward are described as the main causal components.

**Limitations**
- Acknowledged by authors:
  - focus is on a general framework, not specifically sensing realism.
- Observed by reader:
  - prediction enters as reward shaping rather than as an explicit standalone decision-support representation.
  - therefore it is relevant to reward design but less directly to a deployable risk-interface story.

**Connections**
- Builds on: PPO, curriculum learning, dense reward shaping.
- Related approaches: `[@chen2025open]`, `[@tan2025scalable]`, `[@roncero2025amspb]`

**Code & Resources**
- Repository: not linked on the article page
- Datasets released: none reported
- Models released: none reported

### [@tan2025scalable] Scalable Pursuit-Evasion Game for Multi-Fixed-Wing UAV Based on Dynamic Target Assignment and Hierarchical Reinforcement Learning

**Metadata**
- Authors: Mulai Tan, Haocheng Sun, Dali Ding, Huan Zhou, Yongli Liu
- Year: 2025 | Venue: Drones
- DOI: 10.3390/drones10010005
- Code: PDF available; no clear code repository linked in the snippet

**Problem**
How to scale pursuit-evasion to larger multi-vs-multi fixed-wing UAV settings without retraining monolithic MARL from scratch for each scale.

**Key Contributions**
1. Decomposes scalable pursuit-evasion into target allocation, maneuver decision-making, and flight control layers.
2. Uses dynamic target assignment to reduce a multi-vs-multi game into several one-vs-one confrontations.
3. Uses trajectory prediction and stable auxiliary gradients to improve maneuver quality and suppress oscillatory flight behavior.

**Methodology**
- Approach type: hierarchical RL + task allocation
- Key idea: separate global allocation from local maneuver decision making and low-level flight control.
- Key components:
  - dynamic target assignment based on situation advantage and threat level
  - hierarchical maneuver decision-making with trajectory prediction
  - stable auxiliary gradient flight controller with angular-acceleration constraints
- Novel aspects: scalability emphasis and explicit decomposition of game layers.

**Experiments**
- Scales: 3v3, 6v6, 9v9, and 12v12.
- Baselines: other scalable algorithms and MARL variants.
- Main results:
  - high win rates across diverse scales.
  - improved training efficiency and scalability relative to baseline approaches.
- Ablations:
  - the paper attributes gains to trajectory prediction and stable auxiliary gradient design.

**Limitations**
- Acknowledged by authors:
  - fixed-wing setting and hierarchical architecture designed for scale.
- Observed by reader:
  - scale and assignment are central; observability risk and onboard sensing are not the main object of study.
  - useful as a contrast for what paper 1 should explicitly defer.

**Connections**
- Builds on: hierarchical RL and scalable target-assignment methods.
- Related approaches: `[@zhao2024cooperative]`, `[@luo2024commnet]`, `[@yao2026warning]`

**Code & Resources**
- Repository: not clearly linked in the surfaced snippet
- Datasets released: none reported
- Models released: none reported

### [@yan2024lsrc] A Deep Reinforcement Learning-Based Intelligent Maneuvering Strategy for the High-Speed UAV Pursuit-Evasion Game

**Metadata**
- Authors: Tian Yan, Can Liu, Mengjing Gao, Zijian Jiang, Tong Li
- Year: 2024 | Venue: Drones
- DOI: 10.3390/drones8070309
- Code: not reported

**Problem**
How an evader can execute effective intelligent maneuvers against high-speed, high-dynamic pursuers under strong trajectory and energy constraints.

**Key Contributions**
1. Proposes LSRC-TD3, integrating LOS angle-rate correction with TD3.
2. Designs reward functions that combine terminal/process and strong/weak incentive guidance.
3. Targets particularly hard high-speed evasion regimes where performance margins are small.

**Methodology**
- Approach type: TD3-based single-agent maneuver strategy
- Key idea: improve sensitivity to line-of-sight rate changes so the evader times orbit changes more effectively.
- Key components:
  - LOS angle-rate correction factor
  - reward design combining terminal and shaping incentives
  - explicit consideration of evasion, trajectory constraints, and energy consumption
- Novel aspects: closer coupling between guidance geometry and deep RL maneuver strategy.

**Experiments**
- Evaluation: Monte Carlo simulations in challenging high-speed pursuit-evasion scenarios.
- Baselines: other maneuver strategies and DRL comparisons referenced in the paper.
- Main results:
  - paper reports high evasion performance with energy-aware behavior.
  - improved effectiveness and adaptive capability in high-speed scenarios.
- Ablations:
  - LOS-rate correction is presented as the key performance-improving ingredient.

**Limitations**
- Acknowledged by authors:
  - focused on maneuver strategy for one side.
- Observed by reader:
  - does not address limited sensing or transition away from privileged-state observations.
  - more useful as a strong evader-maneuver reference than as a direct baseline for paper 1.

**Connections**
- Builds on: TD3 and guidance-inspired evasion design.
- Related approaches: `[@lei2026causal]`, `[@xu2025general]`

**Code & Resources**
- Repository: none reported
- Datasets released: none reported
- Models released: none reported

### [@roncero2025amspb] Learned Controllers for Agile Quadrotors in Pursuit-Evasion Games

**Metadata**
- Authors: Alejandro Sanchez Roncero, Yixi Cai, Olov Andersson, Petter Ogren
- Year: 2025 | Venue: arXiv (preprint)
- arXiv: 2506.02849
- Code: no official repository surfaced in the available search results

**Problem**
How to train robust 1v1 quadrotor pursuit-evasion controllers when adversarial co-training is unstable and agents forget earlier opponents.

**Key Contributions**
1. Learns both pursuer and evader body-rate-and-thrust policies for agile 1v1 quadrotor pursuit-evasion.
2. Proposes AMSPB to mitigate nonstationarity and catastrophic forgetting by sampling from a growing pool of past and current opponents.
3. Shows rate-based controllers outperform velocity-based controllers in both capture rate and achievable agility.

**Methodology**
- Approach type: PPO-based adversarial RL in a high-fidelity simulator
- Key idea: alternate training between pursuer and evader while freezing the opponent sampled from a policy population.
- Key components:
  - high-fidelity quadrotor dynamics in Isaac Sim at 62.5 Hz
  - observations include time, relative opponent positions, self state, and role-specific terms
  - rate/thrust and velocity-level policy variants
  - AMSPB training loop with opponent sampling probability over older policies
- Novel aspects: direct body-rate-and-thrust adversarial pursuit-evasion plus population-based asynchronous stabilization.

**Experiments**
- Simulation: 256 parallel environments; 10 m x 10 m x 4 m arena; 0.5 m capture threshold; 600-step episodes.
- Baselines:
  - AMS-DRL without old-policy sampling
  - velocity-level controllers
  - benchmark heuristic opponents such as Hover, Circular, and Repel
- Main results:
  - older-policy sampling preserves strong performance against early opponents, while no-sampling degrades sharply.
  - in unseen cross-modality matchups, rate-based pursuer vs velocity-based evader catches the evader in 57.03% of trials, versus 13.67% in the reverse setup.
  - peak linear speed reaches 12.90 m/s for rate-based policies versus 10.49 m/s for velocity-based.
- Ablations:
  - AMSPB versus AMS demonstrates forgetting mitigation.
  - rate-based versus velocity-based control demonstrates the benefit of lower-level action realism.

**Limitations**
- Acknowledged by authors:
  - long-term goal is broader urban-airspace security, but current validation is still simulator-based.
- Observed by reader:
  - observation assumes opponent positions are observed; the paper is not about sensing-constrained pursuit.
  - this is therefore a crucial later-stage bilateral-training reference, not the right first-paper baseline target.

**Connections**
- Builds on: `[@zhang2023gameofdrones]` and asynchronous adversarial training ideas.
- Related approaches: `[@yao2026warning]`, `[@xu2025general]`, `[@chen2025open]`

**Code & Resources**
- Repository: not linked in available results
- Datasets released: none reported
- Models released: none reported

### [@zheng2025visioncapture] Vision-Based Cooperative MAV-Capturing-MAV

**Metadata**
- Authors: Canlun Zheng, Yize Mi, Hanqing Guo, Huaben Chen, Shiyu Zhao
- Year: 2025 | Venue: arXiv (preprint)
- arXiv: 2503.06412
- Code: no official repository surfaced in current search results

**Problem**
How to build a complete onboard-vision cooperative capture system that can detect, estimate, pursue, and physically capture a moving MAV target.

**Key Contributions**
1. Presents a complete multi-MAV capture system with visual perception, estimation, control, and autonomous net-launch decisions.
2. Proposes a lightweight net-motion approximation for real-time capture triggering.
3. Demonstrates real-world autonomous capture with cooperating pursuer MAVs.

**Methodology**
- Approach type: system paper combining perception, estimation, MPC, and low-level control
- Key idea: distributed visual sensing and collaborative state estimation provide enough target information for formation pursuit and net launch.
- Key components:
  - global-local YOLOv5 detection pipeline
  - neighbor elimination to avoid false detections of fellow pursuers
  - Spatial-Temporal Triangulation cooperative estimation
  - formation-based pursuit with MPC-SO(3) control
  - simplified flying-net capture-zone approximation and 0.5 s persistence requirement
- Novel aspects: integrated onboard-vision cooperative physical capture rather than only simulation pursuit.

**Experiments**
- Simulation:
  - four pursuer MAVs and one target MAV
  - target follows a 10 m radius circular path at 3 m/s
  - noisy bearing measurements and neighbor communication are included
- Real-world:
  - three DJI M300 pursuers with onboard computers, gimbal cameras, Zigbee, and net-gun devices
  - target moves at 4 m/s on a 10 m radius circular path
  - 11 successful captures in 17 trials, for a 64.7% success rate
- Ablations:
  - qualitative demonstration of simplified net capture zone versus complex net simulation

**Limitations**
- Acknowledged by authors:
  - physical capture system complexity and target-state-estimation dependence.
- Observed by reader:
  - not an RL pursuit paper.
  - still highly relevant for making the user's "observability loss" concept physically meaningful.

**Connections**
- Builds on: visual detection, cooperative estimation, MPC interception.
- Related approaches: `[@zhang2026shielded]`, `[@chen2025open]`

**Code & Resources**
- Repository: not found from available sources
- Datasets released: none reported
- Models released: none reported

### [@zhang2026shielded] High-Speed Vision-Based Flight in Clutter with Safety-Shielded Reinforcement Learning

**Metadata**
- Authors: Jiarui Zhang, Chengyong Lei, Chengjiang Dai, Lijie Wang, Zhichao Han, Fei Gao
- Year: 2026 | Venue: arXiv (preprint)
- arXiv: 2602.08653
- Code: not linked in the surfaced results

**Problem**
How to achieve end-to-end vision-based high-speed quadrotor flight in clutter while retaining stronger safety guarantees than pure RL policies.

**Key Contributions**
1. Adds model-based safety shielding to an end-to-end RL navigation policy.
2. Uses physics-informed reward design during training and safe-set projection at deployment.
3. Demonstrates strong generalization in dense clutter and challenging outdoor environments.

**Methodology**
- Approach type: end-to-end vision RL with deployment-time safety filter
- Key idea: retain fast learned action generation while constraining actions online via a provably safe projection.
- Key components:
  - physics-informed reward for global guidance
  - real-time safety filter at deployment
  - benchmark comparison against planners and differentiable-physics obstacle-avoidance methods
- Novel aspects: explicit reconciliation of end-to-end RL speed with formal-ish safety constraints.

**Experiments**
- Tasks: dense clutter and challenging outdoor forest environments.
- Baselines: traditional planners and recent end-to-end obstacle avoidance methods based on differentiable physics.
- Main results:
  - strong generalization in dense clutter.
  - reliable flight up to 7.5 m/s.
- Ablations:
  - implicit contrast between shielded and unshielded behavior in the paper framing.

**Limitations**
- Acknowledged by authors:
  - focus is obstacle avoidance rather than adversarial pursuit.
- Observed by reader:
  - useful as a sensing-compatible safety reference, not as a pursuit baseline.
  - could later inform a deployment-time safety layer for pursuit policies near obstacles.

**Connections**
- Related approaches: `[@sun2026curriculum]`, `[@zhao2026dynamicflight]`, `[@ren2026realworld]`

**Code & Resources**
- Repository: request-code style listings only; no official GitHub linked in surfaced results
- Datasets released: none reported
- Models released: none reported

### [@sun2026curriculum] Curriculum Reinforcement Learning for Quadrotor Racing with Random Obstacles

**Metadata**
- Authors: Fangyu Sun, Fanxing Li, Yu Hu, Linzuo Zhang, Yueqian Liu, Wenxian Yu, Danping Zou
- Year: 2026 | Venue: arXiv (preprint)
- arXiv: 2602.24030
- Code: https://github.com/SJTU-ViSYS-team/CRL-Drone-Racing

**Problem**
How to train a robust vision-based quadrotor controller that handles random obstacles while maintaining aggressive flight performance.

**Key Contributions**
1. Proposes a multi-stage curriculum RL framework for obstacle-rich quadrotor racing.
2. Combines curriculum learning, domain randomization, and multi-scene updating.
3. Demonstrates real-world aggressive flight with random obstacles.

**Methodology**
- Approach type: end-to-end vision RL
- Key idea: separate and progressively mix obstacle-avoidance and gate-traversal difficulty during training.
- Key components:
  - multi-stage curriculum
  - domain randomization
  - multi-scene updating
  - single end-to-end control network
- Novel aspects: practical curriculum design for aggressive vision-based flight under obstacle uncertainty.

**Experiments**
- Tasks: obstacle-rich quadrotor racing with unseen obstacles.
- Baselines: existing racing and obstacle-avoidance approaches.
- Main results:
  - faster lap times and higher success rates than existing methods.
  - real-world flight with random obstacles at speeds up to 8 m/s.
- Ablations:
  - reported ablations on GRU removal, reward terms, and one-step learning in the surfaced summary.

**Limitations**
- Acknowledged by authors:
  - task is racing rather than pursuit.
- Observed by reader:
  - still useful for curriculum design once the user's project moves from full-state pursuit toward image-based pursuit.

**Connections**
- Related approaches: `[@zhang2026shielded]`, `[@ren2026realworld]`

**Code & Resources**
- Repository: `SJTU-ViSYS-team/CRL-Drone-Racing`
- Datasets released: none explicitly noted in surfaced results
- Models released: code and video are reported
