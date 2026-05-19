# Phase 6 Final Report: EAAI Novelty and Scope Refresh

Date: 2026-05-19

## Executive Verdict

After integrating the user-provided candidate papers, the revised plan is **more clearly EAAI-scope compatible but more strictly conditional in novelty**.

The added UAV pursuit-evasion, high-speed maneuvering, agile quadrotor PE, multi-UAV PE and shielded-flight papers strengthen the application fit. They also make broad claims weaker. The paper should not claim novelty in high-speed PE, limited perception, multi-UAV PE, agile capture, shielded flight, or generic risk-aware RL.

The strongest defensible claim is:

> We study LiDAR/range-image reduced-information high-speed UAV pursuit and learn an action-conditioned short-horizon observability-loss risk from privileged rollouts to modulate PPO learning and action distribution under sensor-honest inputs.

The current contribution is not defensible as a generic new safe-PPO or risk-sensitive-RL method.

## Why It Fits EAAI

EAAI explicitly covers practical AI methods in engineering, including deep learning real-world applications, robotics, perception, safety/reliability and real-time intelligent automation. It expects the abstract to specify the AI contribution and the engineering application.

This project can satisfy that if:

- AI contribution: `R_obs(s_red,u)` action-conditioned observability-loss risk and PPO-internal modulation.
- Engineering application: sparse LiDAR/range-image onboard-sensing-limited high-speed UAV pursuit.
- Validation: not just success rate, but risk calibration, visibility/recovery metrics and sensor sparsity stress tests.

## Strongest Neighbor Set

| Neighbor | Threat | Required distinction |
|---|---|---|
| `[@zhao2025msmar]` | high-speed PE + safe/recovery RL | We are not zero-constraint safety recovery; we model observability loss. |
| `[@peng2025limitedvisual]` | limited visual field pursuit + reacquisition | We learn calibrated action-conditioned risk rather than relying on MARL phase/reward design. |
| `[@huh2026limitedregion]` | limited detectable region + FOV reward | We must compare against FOV heuristic/reward and show risk model value. |
| `[@ren2025losrate]` | EAAI reduced-measurement DRL guidance | Supports scope; our task is UAV LiDAR pursuit rather than LOS-rate interception. |
| `[@thananjeyan2021recoveryrl]` | action-conditioned safety critic | Our risk is target observability loss and should modulate PPO rather than switch to recovery/shield. |
| `[@achiam2017cpo]` | cost advantage and constraints | B4 must beat PPO-Lagrangian/CPO-like baseline. |
| `[@pinto2017aac]` | privileged-to-partial robot RL | Privileged rollout is a training pattern, not central novelty. |
| `[@liu2026actionriskgating]` | action-conditioned risk in POMDP | Do not claim general action-risk gating novelty. |
| `[@chen2025open]` | online planning / prediction-enhanced multi-UAV PE in unknown environments with CTBR relevance | We are not claiming online planning, prediction-enhanced PE or CTBR pursuit novelty; our distinction is LiDAR/range-image reduced inputs, calibrated observability-loss risk and PPO-internal action-conditioned risk modulation. |
| `[@yan2024lsrctd3]` | high-speed UAV PE + DRL maneuvering | Do not claim high-speed PE or DRL maneuvering novelty. |
| `[@roncero2025agilecontrollers]` | agile 1v1 quadrotor PE with learned CTBR-like controllers | Our claim is sensing/observability-risk modulation, not agile PE controllers. |
| `[@giral2026intercept]` | RL aerial-robot interception | Supports application relevance; not a reduced LiDAR observability-risk method. |
| `[@luo2024improvedmadrl]` | multi-UAV PE/MADRL | Keep paper 1 out of cooperative MARL novelty. |
| `[@tan2026scalablefixedwing]` | scalable multi-fixed-wing PE + assignment | Do not claim scalability/assignment novelty. |
| `[@zhang2026safetyshieldedflight]` | high-speed flight + safety shield | Shield/filter is a baseline family, not the main method. |

## Updated Novelty Position

### Sufficiently novel if all are true

- Online inputs are strictly LiDAR/range-image reduced and do not contain target truth, target ID, semantic mask or oracle bearing/range/closing speed.
- Risk label is target observability loss over `H=150`, `K_persist=10`, not generic safety/collision.
- `R_obs(s_red,u)` is action-conditioned and validated by matched-state action-risk ranking.
- B4 changes PPO internals while preserving PPO action/logprob/execution consistency.
- B4 beats B3, PPO-Lagrangian and shield/filter baselines.

### Not sufficiently novel if any dominate

- B4 only concatenates risk to observation.
- B4 only penalizes reward with risk.
- B4 filters/projects sampled actions after PPO.
- The paper only shows success-rate gain without risk calibration and action-risk evidence.
- The paper claims real long-range LiDAR deployability without sensor validation.

## Required Experiment Updates

Main baselines:

- B0 full-state PPO upper bound.
- B1 reduced PPO without risk.
- B2 LiDAR-honest heuristic risk.
- B3 learned state risk.
- B4 action-conditioned risk-modulated PPO.

Mandatory internal ablations:

- risk concat only.
- action-conditioned risk vs state-only risk.
- shuffled-action risk.
- actor modulation only.
- risk-conditioned advantage only.
- actor + advantage.
- PPO-Lagrangian using same risk as cost.
- shield/filter using same risk as post-hoc intervention.
- privileged critic baseline.

Mandatory sensor validation:

- return-count distributions at 50/100/150/200m.
- target point/pixel occupancy.
- dropout and longest loss streak.
- noise/latency/angular-resolution/reflectivity sensitivity.
- sparse-point ablation by 0/1/2/5/10 target returns.

## Writing Guidance

Organize Related Work in this order:

1. Limited field / detectability / sensor-honest pursuit.
2. UAV pursuit-evasion / interception / online planning / CTBR.
3. Constrained action optimization / risk-gated control.
4. RL / privileged training / engineering machinery.

This order matches the paper's scientific logic: finite observability and pursuit are the problem, constrained action optimization is the method family, and RL is implementation support.

Use:

- "observability-loss risk"
- "sensor-honest LiDAR/range-image reduced-information pursuit"
- "action-conditioned risk-modulated PPO"
- "deployment-aware simulation study"

Avoid:

- "first safe PPO"
- "first risk-aware pursuit"
- "real-world validated long-range LiDAR pursuit"
- "full onboard visual pursuit"
- "generic partial-observation RL"

## Final Recommendation

Proceed with the current mainline, with the stricter post-supplement conclusion:

1. Keep the claim narrow and engineering-specific.
2. Add safe-RL and shield/filter baselines to B4 internal ablation.
3. Add matched-state action-risk ranking and calibration gates.
4. Add LiDAR sparsity/dropout validation before writing EAAI sensor claims.
5. Treat B4 as the paper's main method only if it beats B3 and PPO-Lagrangian/shield variants.

Final wording to use:

> Action-conditioned observability-loss risk modulation of PPO for sensor-honest LiDAR/range-image reduced-information UAV pursuit.

Final wording to avoid:

> A new high-speed UAV pursuit algorithm, a new safe PPO method, a first limited-perception pursuit method, or real-world validated long-range LiDAR pursuit.

✅ Phase 6 complete. Output: `phase6_report/report.md`, `phase6_report/references.bib`.
