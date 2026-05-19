# Phase 1 Frontier: Risk-Modulated PPO for LiDAR-Reduced UAV Pursuit

Date: 2026-05-19

## Frontier Question

Current plan claim:

> Learn an action-conditioned short-horizon observability-loss risk from privileged rollouts, then use it to modulate PPO action distribution / value or advantage estimation under LiDAR/range-image reduced-information UAV pursuit.

The refresh specifically tests whether this is novel enough for Engineering Applications of Artificial Intelligence (EAAI) and where the claim must be narrowed.

## Scope Check: EAAI

The EAAI journal scope explicitly accepts practical AI methods in engineering, including deep learning real-world applications, distributed AI/control, perception, safety/reliability, real-time intelligent automation, industrial case studies and robotics. It also requires the abstract to distinguish the AI contribution and the engineering application, and says submitted papers should report novel AI aspects for real-world engineering applications with reproducible validation.

Implication: the topic is in scope if framed as an engineering AI method for onboard-sensing-limited UAV pursuit, not as a generic RL benchmark or a purely metaphorical optimizer.

## Latest / High-Pressure Frontier Papers

1. `[@zhao2025msmar]` IJCAI 2025: high-speed pursuit-evasion, safe maneuver decision, recovery RL, obstacle constraints. It is a top-conference close neighbor for high-speed PE and safety.
2. `[@peng2025limitedvisual]` IEEE/CAA JAS 2025: limited visual field, search-pursuit-reacquisition phases, MARL with multiple evader strategies.
3. `[@huh2026limitedregion]` Machines 2026: limited detectable region and FOV-dependent/capturability rewards for multi-UAV pursuit.
4. `[@ren2025losrate]` EAAI 2025: DRL guidance with limited LOS-rate measurement, directly supporting EAAI scope for reduced-information guidance/interception.
5. `[@gao2026mentalstates]` EAAI 2026: imperfect multi-UAV cooperative-competitive environments and opponent/world modeling.
6. `[@liu2026autopilot]` EAAI 2026: curriculum-guided PPO framework for UAV autopilots, supporting scope for PPO variants in UAV engineering.
7. `[@wang2025viper]` 2025: visibility-based pursuit-evasion via RL; strong warning against claiming visibility PE is new.
8. `[@li2026safeintent]` 2026: risk-gated hierarchical multi-UAV PE under intermittent observability; closest PE-specific risk-gating neighbor.
9. `[@liu2026actionriskgating]` 2026 preprint: action-conditioned risk gating under partial observability; closest method-level neighbor if accepted as valid.
10. `[@thananjeyan2021recoveryrl]` Recovery RL: action-conditioned safety critic and policy intervention; strongest analogy to `R_obs(s,u)`.
11. `[@achiam2017cpo]` CPO: cost value/cost advantage in constrained policy optimization; makes ordinary risk-adjusted advantage non-novel by itself.
12. `[@pinto2017aac]` Asymmetric Actor-Critic: simulator privileged training with deployable partial observation actor; directly overlaps privileged-to-reduced learning.
13. `[@park2023lidardrone]` LiDAR-based drone navigation with PPO; supports LiDAR-policy feasibility but not pursuit novelty.
14. `[@shao2026peqmixer]` perception-enhanced multi-UAV cooperative pursuit under partial observability; confirms the broader field is active in adjacent journals.

## Candidate Paper Integration

The user-provided candidate papers were audited rather than bulk-added. Only papers that directly support B4 design, real comparison pressure or required Related Work coverage were retained in the active DB. The retained additions are not stronger method-level novelty support; they are stronger novelty pressure:

- `[@yan2024lsrctd3]`: high-speed UAV pursuit-evasion with DRL intelligent maneuvering.
- `[@giral2026intercept]`: reinforcement learning for unauthorized aerial-robot interception in controlled airspace.
- `[@roncero2025agilecontrollers]`: learned agile quadrotor pursuit-evasion controllers.
- `[@luo2024improvedmadrl]`, `[@zhao2024autonomousuavpe]`, `[@tan2026scalablefixedwing]`, `[@xiang2025cihrl]`: multi-UAV pursuit-evasion, cooperative maneuver decision-making, hierarchical RL and scalable assignment.
- `[@zhang2026safetyshieldedflight]`: high-speed vision-based flight with a safety shield, strengthening the need for a shield/filter baseline.
- `[@feroskhan2024multipursuitevasion]`: multipursuit evasion with real-time flight validation, strengthening the warning against overclaiming real-world PE validation.

Revised frontier implication: the topic is clearly within the broader engineering-AI/UAV-RL landscape, but high-speed UAV PE, cooperative PE, agile pursuit and shielded high-speed flight are now too crowded to be the novelty. The novelty must stay on observability-loss risk semantics, action conditioning and PPO-internal modulation under strict LiDAR/range-image reduced inputs.

## Frontier Verdict

The current direction is not novel if stated as:

- risk-aware RL;
- action-conditioned risk gating;
- limited-FOV UAV pursuit;
- safe PPO or shielded PPO;
- privileged-to-partial actor-critic.

It remains conditionally novel if stated as the intersection of:

1. **LiDAR/range-image reduced-information high-speed UAV pursuit** with strict no target-truth/no semantic-mask/no oracle-bearing input.
2. **Short-horizon observability-loss risk** rather than collision risk, intent prediction, trajectory prediction or generic safety cost.
3. **Action-conditioned risk** `R_obs(s_red, u)` with matched-state action-risk ranking evidence.
4. **PPO-internal risk modulation** whose executed action remains the PPO log-prob action, avoiding post-hoc action filters.
5. **Causal baseline chain** B0/B1/B2/B3/B4, showing that B4 is not just risk concat, reward shaping, Lagrangian safe PPO or shielded action correction.

✅ Phase 1 complete. Output: `phase1_frontier/frontier.md`, `phase1_frontier/paper_finder_config.yaml`. Proceeding to Phase 2.
