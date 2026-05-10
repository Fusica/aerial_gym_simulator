# Findings And Locked Decisions

## Current Scientific Position
The first paper is a strong EAAI submission only if it is framed as a privileged-to-depth-only sensor-proxy transition method for UAV pursuit-evasion.

The publishable mechanism is not generic prediction and not generic partial observability. The locked mechanism is:

`depth-only reduced information + action-conditioned observability-risk prediction + CTBR action correction`

## Code-Grounded Baseline Facts
- Active task config: `aerial_gym/config/task_config/pursuit_guidance_task_config.py`.
- Active task: `aerial_gym/task/pursuit_guidance_task/pursuit_guidance_task.py`.
- Robot/control path: `base_quad_root_link_control` with `thrust_bodyrate_control`.
- Target-motion baseline: `apf_escape`.
- Current observation dimension: 32.
- Current observation leaks target-relative position, target-relative velocity, closing geometry, wall clearances, and obstacle-relative positions.
- Current robot config does not enable camera by default.

Consequence: the existing policy is a privileged full-state baseline. It must be used as B0 and as a label generator, not as the main proposed sensing-compatible controller.

## Locked Problem Statement
How can a privileged full-state UAV pursuit policy be converted into a depth-only sensor-proxy reduced-information controller by learning short-horizon observability risk from privileged rollouts and using action-conditioned risk to correct CTBR actions before execution?

## Novelty Assessment
Decision: **novel with medium confidence**, under the locked claim.

Not novel:
- full-state PPO plus an extra learned risk scalar
- generic target prediction before pursuit
- limited-FOV reward shaping
- generic partial-observation MARL

Novel enough for a strong EAAI attempt:
- learning observability-loss risk instead of target trajectory
- using privileged rollouts only for labels and auxiliary supervision
- evaluating under reduced/depth-only information rather than full state
- making the risk action-conditioned and using it to correct CTBR actions

## Strong Near Neighbors And Required Differentiation
| Paper | Overlap | Required Differentiation |
|-------|---------|--------------------------|
| `[@huh2026limited]` | limited detectable region, FOV-aware reward, APF evader | Our method learns and audits observability-risk, instead of only shaping rewards or action space |
| `[@peng2025limitedvisual]` | limited visual field, search/pursuit/reacquisition MARL | Our method is a privileged-to-depth bridge with action-conditioned risk correction, not a graph-attention MARL policy |
| `[@li2026safeintent]` | risk-gated hierarchical policy, safety projection, partial observability | Our method predicts observability-loss risk directly from privileged pursuit rollouts and corrects CTBR actions in a depth-only sensor-proxy 1v1 setting |
| `[@chen2025open]` | prediction-enhanced pursuit, CTBR deployment | Our method predicts target-loss risk rather than target trajectory and treats CTBR correction as the core mechanism |
| `[@zhang2023gameofdrones]` | target prediction and online planning | Our method evaluates a separately trained risk variable with offline calibration and online reduced-information control impact |
| `[@li2025bearingonly]` | sensor-compatible pursuit and resilience to target loss | Our method uses depth-only risk learning and CTBR action correction rather than bearing-only filtering |

## Final Contribution Wording
1. **Reduced-information pursuit formulation**: define a depth-only sensor-proxy observation regime that removes direct target and environment truth from policy input.
2. **Action-conditioned observability-risk learning**: learn `p_lost` and `R_obs(s_red, u)` from privileged full-state rollouts with offline calibration.
3. **CTBR risk-constrained action correction**: convert raw PPO CTBR actions into risk-aware corrected actions with a constrained projection layer.
4. **Causal experiment ladder**: B0-B5 separates privileged-state upper reference, reduced-information degradation, heuristic risk, learned risk, oracle risk, and action-conditioned correction.

## Method Details
### Label
- `y_t = 1` when the target exits the virtual detectable region within `H = 150` and remains outside for `K_persist = 10`.
- The virtual detectable region is parameterized by `R_det`, `alpha_h`, and `alpha_v`.
- Sensitivity is mandatory for `H in {100, 150, 200}` and at least one FOV/range perturbation.

### Predictor
- Inputs: ego linear velocity, ego angular velocity, orientation, previous CTBR action, stacked depth frames.
- Depth stack: `K = 3`.
- Frozen latent: `z_depth = 64`.
- Main outputs: `p_lost(s_red)` and action-conditioned `R_obs(s_red, u)`.
- Auxiliary outputs during training only: range and bearing supervision from privileged labels.
- Inference rule: no target truth, no wall truth, no obstacle truth, no truth-derived bearing/range/closing-speed.

### Action Correction
`u* = argmin_u ||u - u_rl||_2^2 + lambda_risk R_obs(s_red, u) + lambda_fov L_fov(s_red, u) + lambda_obs L_obs(s_red, u) + lambda_rate ||u - u_prev||_2^2`

Hard constraints:
- `u_min <= u <= u_max`
- `||u - u_prev||_inf <= Delta_u_max`

Paper-1 implementation:
- clamped projected-gradient or small candidate-set minimization
- no MPC inner loop
- no PPO gradient through frozen encoder, risk head, or correction layer

## Required Experiments
| ID | Experiment | Acceptance Standard |
|----|------------|--------------------|
| E1 | Offline risk prediction | learned risk beats heuristic risk on AUROC/AUPRC and has reportable Brier/ECE |
| E2 | Reduced-information bridge | B3 beats B1 and B2 on target-loss and pursuit metrics |
| E3 | Action correction | B5 beats B3 on target retention/recovery without collapsing success rate |
| E4 | Oracle upper bound | B4 shows whether risk variable has remaining headroom |
| E5 | Full-state reference | B0 remains an upper reference; proposed method must recover a meaningful fraction of B0 performance |
| E6 | Generalization | ID/APF, held-out APF parameters, scripted motions, and stronger target policies when available |
| E7 | Sensitivity | horizon/FOV/range choices are not arbitrary |

## Failure Conditions
The paper claim is rejected or must be rewritten if:
- learned risk does not beat heuristic risk offline
- B3 does not improve over B1/B2 online
- B5 improves target retention only by making pursuit overly conservative
- gains vanish outside one APF parameter setting
- mainline evaluation uses privileged target or environment truth

## Writing Rules
- Do not say "first partial-observation pursuit".
- Do not say "full onboard visual pursuit".
- Do not describe B0 plus risk scalar as the proposed method.
- Do not present bilateral self-play as Paper 1.
- Use "depth-only sensor-proxy reduced-information pursuit" unless true onboard camera processing is implemented and validated.

## Implementation Ownership
- `risk_geometry.py`: detectable-region geometry and labels.
- `risk_dataset.py` / `risk_offline.py`: rollout storage, manifests, offline labeling.
- `risk_model.py` / `risk_training.py`: depth encoder, risk heads, calibration.
- `risk_policy.py`: reduced-information observation and risk injection.
- `risk_closed_loop.py`: CTBR action correction.
- `risk_diagnostics.py`: offline and online metrics.
- `pursuit_guidance_task.py`: orchestration only.

## Current Next Tasks
1. Implement risk geometry and label generation first.
2. Implement dataset manifest checks before training any predictor.
3. Implement reduced-information observation modes B1-B5.
4. Implement action-conditioned risk correction before large PPO training.
