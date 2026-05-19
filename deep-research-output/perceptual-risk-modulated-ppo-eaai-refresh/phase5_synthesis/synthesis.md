# Phase 5 Synthesis

Date: 2026-05-19

## Taxonomy

| Family | What it already covers | What remains open for this plan |
|---|---|---|
| Limited field / detectability / sensor-honest pursuit | Limited FOV, limited detectable region, target loss/reacquisition, bearing-only/vision-compatible capture and FOV rewards | Calibrated action-conditioned target-observability-loss risk from strict LiDAR/range-image reduced inputs. |
| UAV pursuit-evasion / interception / online planning / CTBR | Multi-UAV PE, prediction-enhanced online planning, CTBR/body-rate-thrust deployment, high-speed/agile PE and aerial interception | Not online planning or agile PE control; the open point is preserving observability under reduced sensing during pursuit. |
| Constrained action optimization / risk-gated control | CPO, PPO-Lagrangian, Recovery RL, shields, safety filters, risk-sensitive policy gradients and action-conditioned risk gating | Observability-loss risk as the constrained variable, with action-conditioned ranking and PPO-internal modulation rather than post-hoc filtering. |
| RL / privileged training / engineering machinery | PPO/MARL/curriculum, asymmetric actor-critic, LiDAR/depth policies and EAAI UAV RL applications | RL is supporting machinery; novelty must be the risk target, action conditioning, sensor honesty and evaluation package. |

## Novelty Evaluation

### Strong if stated narrowly

The contribution is plausibly EAAI-level if the paper claims:

1. A LiDAR/range-image reduced-information UAV pursuit formulation with strict no-truth-input rules.
2. A short-horizon observability-loss risk label from privileged rollouts.
3. An action-conditioned risk model `R_obs(s_red, u)` with calibration and matched-state action-ranking evidence.
4. Risk-modulated PPO where risk changes policy learning/distribution/advantage while the executed action remains the PPO-modeled action.
5. A B0-B4 causal experiment chain proving action-conditioned risk adds value beyond reduced PPO, heuristic risk, and learned state risk.

### Weak if stated broadly

The contribution is not novel if claimed as:

- first limited-FOV UAV pursuit;
- first risk-aware pursuit-evasion;
- first action-conditioned risk gating in POMDPs;
- new safe PPO;
- new privileged-to-partial actor-critic;
- LiDAR + PPO policy.

## EAAI Scope Fit

Fit is positive:

- EAAI scope includes robotics, perception, real-time automation, distributed AI/control and safety/reliability.
- Recent EAAI papers include UAV PPO/autopilot, reduced-measurement interception guidance, cooperative UAV decision-making and imperfect multi-UAV competitive modeling.

Main risk:

- EAAI expects the AI contribution and engineering application to be clear. A purely simulated RL benchmark with weak sensor realism would be vulnerable.

## Required Baselines

Main:

- B0 full-state PPO upper bound.
- B1 reduced-state PPO without risk.
- B2 reduced PPO + LiDAR-honest heuristic risk.
- B3 reduced PPO + learned state risk `p_lost(s_red)`.
- B4 reduced PPO + learned action-conditioned risk-modulated PPO.

B4 internal:

- risk concat only.
- action-conditioned risk vs state-only risk.
- shuffled-action risk.
- risk-conditioned advantage.
- actor latent/mean/variance modulation.
- risk critic/value.
- PPO-Lagrangian baseline using same risk as cost.
- shield/filter baseline using same risk as post-hoc action intervention.

## Required Metrics

Offline:

- AUROC, AUPRC, Brier, ECE, reliability diagram.
- matched-state action-risk ranking accuracy.
- risk-bin future loss rate.
- calibration under held-out APF and non-APF target behaviors.

Online:

- success rate, time-to-capture, collision/near-collision.
- final detectable, target lost time, max consecutive invisible steps.
- recovery count/rate/delay.
- control smoothness, action saturation, entropy/KL.
- visible-strike score rather than success alone.

Sensor:

- target return count per frame at 50/100/150/200m.
- pixel/point occupancy distribution.
- longest dropout, visibility duty cycle.
- range noise, angular resolution, reflectivity/dropout, latency and scan-rate sensitivity.

## Final Synthesis Verdict

After integrating the user-provided candidate papers, the current plan is still **conditionally sufficient and potentially strong for EAAI**, but the conclusion is stricter.

EAAI scope is stronger because the literature confirms a dense engineering-AI context around UAV pursuit, online planning, interception, cooperative maneuvering, high-speed flight and RL-based aerial autonomy. Broad novelty is weaker for the same reason: the paper cannot claim novelty in high-speed UAV PE, multi-UAV PE, agile pursuit, CTBR deployment, sensor-based capture, shielded flight or generic risk-aware RL.

The title-level contribution should be:

> Action-conditioned observability-risk-modulated PPO for LiDAR/range-image reduced-information UAV pursuit.

Not:

> A new safe PPO method, generic risk-aware pursuit, or first partial-observation UAV pursuit.

Required final framing:

> We isolate target observability loss as a sensing-continuity risk, learn an action-conditioned risk model from privileged rollouts, and use it as a PPO-internal modulation signal under sensor-honest LiDAR/range-image inputs.

✅ Phase 5 synthesis complete. Output: `phase5_synthesis/synthesis.md`. Proceeding to gaps.
