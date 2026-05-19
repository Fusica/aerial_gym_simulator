# Phase 3 Deep Dive Notes

Date: 2026-05-19

## 1. `[@ren2025losrate]` Maneuvering target interception via DRL using only LOS-rate measurement

Problem: Intercepting maneuvering targets with reduced seeker information. The paper explicitly argues that traditional APN needs target acceleration and optimal/differential-game guidance can be computationally or information intensive.

Method: DRL guidance with limited LOS-rate measurements; uses RL to map reduced measurements into guidance commands. It is in EAAI, so it is important for scope.

Experiments: Simulation experiments compare guidance performance and unknown scenario behavior. The contribution is framed around engineering applicability under limited measurement.

Limitations for our comparison: It focuses on interception guidance, not UAV-to-UAV LiDAR/range-image pursuit, not target observability-loss prediction, and not PPO-internal risk modulation.

Relevance: Strong evidence that EAAI will accept reduced-information RL guidance if the engineering constraint and AI novelty are both explicit.

## 2. `[@zhao2025msmar]` MSMAR-RL high-speed pursuit-evasion

Problem: High-speed pursuit-evasion with strict safety constraints and obstacle avoidance. Existing RL has difficulty balancing safety and reward in diverse environments.

Method: Multi-step reach-avoid theory plus masked-attention recovery RL; emphasizes zero-constraint-violation recovery and early danger detection.

Experiments: Obstacle-rich high-speed UAV pursuit-evasion scenarios with ablation and comparison.

Limitations for our comparison: Safety is about constraint violation/obstacles, not sensing continuity; the mechanism is recovery/safety control, not LiDAR observability-risk calibration.

Relevance: A strong warning that “high-speed pursuit + safe/risk RL” is not enough as a novelty claim. Our method must avoid generic safe-RL language and focus on observability risk under sensor-reduced inputs.

## 3. `[@peng2025limitedvisual]` Multi-UAV pursuit with limited visual field

Problem: Multi-UAV pursuit in urban airspace with limited visual fields and possible target loss/reacquisition.

Method: Multi-agent RL with search/pursuit/reacquisition behavior and scenario generalization across evasion strategies and environments.

Experiments: Multiple evader types including random/repulsive/RL evasion, mission-time metrics, lost-time and success/capture metrics.

Limitations for our comparison: Multi-agent coordination is central; risk is not separately predicted, calibrated, or action-conditioned. The method does not isolate reduced LiDAR representation versus full truth leakage.

Relevance: We cannot claim limited-FOV or target reacquisition is new. We can still claim a separate risk-variable study if B0-B4 isolate state risk, heuristic risk and action-conditioned risk modulation.

## 4. `[@huh2026limitedregion]` Limited detectable region MAPPO pursuit

Problem: Cooperative pursuit with practical range/FOV constraints and different planar action structures.

Method: MAPPO with FOV-dependent reward and capturability reward; analyzes forward-yaw versus forward-lateral-yaw controls.

Experiments: Limited detectable region; ablation around motion command structures and rewards.

Limitations for our comparison: It relies on reward shaping and MARL action-space design, not learned observability-risk prediction. It does not evaluate risk calibration or action-conditioned risk ranking.

Relevance: This is a direct baseline family for B2-like FOV heuristic and reward shaping. Our B4 must outperform or at least be compared against a limited-FOV reward/heuristic equivalent.

## 5. `[@thananjeyan2021recoveryrl]` Recovery RL

Problem: Safe robot RL needs to avoid constraint-violating regions while preserving task performance.

Method: Offline data trains a safety critic that estimates action-conditioned violation risk; a recovery policy intervenes when risk is high.

Experiments: Simulation and physical-robot tasks including image-based navigation and obstacle avoidance; compares against safe RL alternatives.

Limitations for our comparison: It is a safety/recovery method and includes decision-time intervention. If our method filters or projects actions after PPO sampling, it becomes too close to this family.

Relevance: This is the closest structural prior to `R_obs(s_red,u)`. Our distinguishing point must be observability-loss semantics and PPO-internal modulation rather than recovery switching.

## 6. `[@achiam2017cpo]` Constrained Policy Optimization

Problem: Policy search with reward and constraints rather than reward shaping alone.

Method: Optimizes expected return subject to cost constraints with theoretical near-constraint satisfaction guarantees. Uses cost value/advantage machinery.

Experiments: Simulated robot locomotion constraints.

Limitations for our comparison: It is general safe RL, not sensor observability risk or UAV pursuit.

Relevance: Any B4 formulation using a risk-cost advantage or Lagrangian term needs to compare against CPO/PPO-Lagrangian style baselines. Otherwise reviewers can call the method a domain-specific cost constraint.

## 7. `[@pinto2017aac]` Asymmetric Actor-Critic

Problem: Robot policies must act from deployable observations, but simulators have privileged full state during training.

Method: Train the actor on partial image observations and the critic on privileged full state; combines with domain randomization for real robot transfer.

Experiments: Simulated and real robot manipulation tasks.

Limitations for our comparison: It does not learn an explicit observability-loss risk variable and does not score candidate actions by target-loss risk.

Relevance: Privileged rollout supervision is not a core novelty. It is a legitimate training pattern that supports our implementation, but the novelty must be the risk target and modulation.

## 8. `[@liu2026actionriskgating]` Action-conditioned risk gating under partial observability

Problem: Safety-critical control under partial observability where finite histories approximate state and candidate actions differ in near-term risk.

Method: Action-conditioned risk predictor and gating/value learning under partial observability.

Experiments: Reported as method-level preprint; details need verification before final citation.

Limitations for our comparison: Not UAV pursuit-specific, not LiDAR/range-image target observability, and may not use PPO action-distribution/advantage modulation.

Relevance: Strongest novelty threat. Our paper should not claim action-conditioned risk gating as a general first. It should scope to LiDAR-reduced UAV pursuit and prove observability-risk-specific benefits.

## 9. `[@gao2026mentalstates]` EAAI imperfect multi-UAV cooperative-competitive environments

Problem: Multi-UAV coordination/competition under imperfect information and opponent intent uncertainty.

Method: Hierarchical world-opponent modeling with dynamic fusion for opponent intent and environment prediction.

Experiments: Close-range multi-UAV engagements and SMAC-style benchmarks.

Limitations for our comparison: Focuses on opponent/world modeling rather than onboard sensor observability and target-loss risk.

Relevance: EAAI scope supports imperfect multi-UAV decision intelligence, but our first paper should avoid opponent intent modeling and keep the APF/scripted evader as scenario distribution.

## 10. `[@park2023lidardrone]` LiDAR-based drone navigation with PPO

Problem: Drone navigation using LiDAR observations and reinforcement learning.

Method: PPO policy with LiDAR inputs for navigation.

Experiments: Navigation task rather than pursuit.

Limitations for our comparison: Does not address small-target LiDAR returns, target reacquisition, action-conditioned observability risk or high-speed pursuit.

Relevance: Supports the plausibility of LiDAR-PPO policies but also makes “LiDAR + PPO” alone clearly insufficient as novelty.

## Supplemental Deep-Dive Notes from Accepted Zotero Candidates

### `[@yan2024lsrctd3]` High-speed UAV PE intelligent maneuvering

Problem: high-speed UAV pursuit-evasion maneuvering with DRL under simplified high-speed vehicle dynamics and line-of-sight geometry.

Method: DRL maneuvering strategy with line-of-sight-rate correction and process/terminal reward design.

Relevance: This blocks any broad claim that high-speed UAV pursuit-evasion with DRL maneuvering is new. It does not model onboard LiDAR observability loss, target return sparsity or action-conditioned sensing-risk calibration.

### `[@giral2026intercept]` RL interception of unauthorized aerial robots

Problem: intercepting unauthorized aerial robots in controlled airspace using RL.

Method: RL-based autonomous interception/control framework for aerial security scenarios.

Relevance: Strong application-neighbor evidence that aerial interception with RL is an active engineering topic. It supports scope, but also means our contribution cannot be "RL for interception" itself.

### `[@roncero2025agilecontrollers]` Learned agile quadrotor pursuit-evasion controllers

Problem: agile quadrotor pursuit-evasion controller learning.

Method: learned controllers for agile quadrotor PE.

Relevance: Strong domain neighbor. It pressures any agile-pursuit claim, but does not appear to isolate LiDAR/range-image observability-loss risk or PPO-internal risk modulation.

### `[@luo2024improvedmadrl]` Multi-UAV cooperative maneuver decision-making

Problem: cooperative UAV pursuit-evasion maneuver decision-making.

Method: improved MADRL with communication/recurrent memory for cooperative maneuvering.

Relevance: Shows mature multi-UAV PE/MARL baselines. Paper 1 should avoid making cooperation or multi-agent game strategy its main novelty.

### `[@tan2026scalablefixedwing]` Scalable multi-fixed-wing UAV PE

Problem: multi-fixed-wing pursuit-evasion scalability and target assignment.

Method: hierarchical framework with target assignment, maneuver decision-making and flight control.

Relevance: Blocks claims around scalable multi-UAV PE. Our paper should remain single-pursuer/reduced-information risk modulation unless the experiment scope is explicitly expanded.

### `[@zhang2026safetyshieldedflight]` High-speed flight with safety shield

Problem: high-speed vision-based flight in clutter with safety assurance.

Method: RL policy augmented by a safety filter that projects outputs into a safe set.

Relevance: This makes post-hoc action projection/filtering a weak main method for this paper. Shield/filter should be a baseline, while B4 should keep PPO logprob/executed action consistency.

### `[@feroskhan2024multipursuitevasion]` Multipursuit evasion for drones

Problem: drone evasion from multiple pursuers while navigating safely to a target.

Method: asynchronous multi-stage DRL and adversarial agent evolution, with reported real-time flight validation.

Relevance: This raises the bar for any real-world PE claim. Unless this project adds real logs, it should be written as deployment-aware simulation, not real-world validated pursuit.

### `[@mavcapturingmav2024]` Vision-based cooperative MAV-capturing-MAV

Problem: cooperative MAV capture with onboard vision.

Method: vision-based cooperative capture system.

Relevance: It blocks "first vision/sensor-based MAV capture" framing. Our sensor contribution must be the strict no-truth LiDAR/range-image reduced-information setup and observability-risk variable.

## Deep-Dive Verdict

The plan is defensible only under a narrowed claim:

> action-conditioned observability-loss risk, learned from privileged rollouts, used as a deployable LiDAR/range-image reduced-information modulation signal inside PPO for high-speed UAV pursuit.

The plan is weak if it becomes:

- safe PPO with a cost critic;
- recovery/shielded action filtering;
- FOV reward shaping;
- privileged asymmetric actor-critic;
- generic limited-FOV pursuit.

✅ Phase 3 complete. Output: `phase3_deep_dive/selection.md`, `phase3_deep_dive/deep_dive.md`. Proceeding to Phase 4.
