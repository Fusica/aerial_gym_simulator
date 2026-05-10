# Task Plan: EAAI-Strong Observability-Risk Pursuit

## Goal
Produce a strong EAAI submission from the current `pursuit_guidance_task` checkout by solving one precise problem:

`privileged full-state pursuit -> depth-only sensor-proxy reduced-information pursuit -> action-conditioned short-horizon observability-risk prediction -> CTBR risk-constrained action correction`

The paper must not be framed as generic pursuit RL, generic partial observability, generic target prediction, or full bilateral self-play.

## Active Plan
- **Plan ID:** `2026-05-09-perceptual-risk-pursuit-eaai`
- **Current Phase:** Phase 6 - Implementation And Validation
- **Overall Status:** in_progress
- **Submission Rule:** all method, experiment, and writing decisions must support the locked EAAI claim below.

## Locked EAAI Claim
A UAV pursuer trained from a privileged full-state baseline can recover decision quality under depth-only sensor-proxy reduced information by learning an action-conditioned short-horizon observability-risk model from privileged rollouts and using that risk to correct CTBR actions through a constrained projection layer.

## Non-Negotiable Contribution Set
1. **Reduced-information depth-only sensor-proxy pursuit formulation**
   - Main policy input is `self-state + z_depth + p_lost`.
   - Direct target state, truth-derived bearing/range/closing-speed, wall truth, and obstacle truth are removed from the mainline policy input.
   - Full-state PPO is an upper reference only, not the proposed method.

2. **Action-conditioned short-horizon observability-risk model**
   - Primary target is probability of target loss from the virtual detectable region within horizon `H = 150` with persistence `K_persist = 10`.
   - Main predictor must support action-conditioned scoring: `R_obs(s_red, u)` or an equivalent candidate-action risk evaluation.
   - Privileged state is allowed for offline labels and auxiliary supervision only; it is not available at policy inference.

3. **CTBR risk-constrained action correction**
   - Raw PPO action `u_rl = [c, p, q, r]` is corrected to `u*` by a one-step constrained projection.
   - The correction layer is a required core method component.
   - Paper 1 does not backpropagate PPO gradients through the frozen depth encoder, risk head, or action-correction layer.

## Repo Ground Truth To Preserve
- Current task config: `aerial_gym/config/task_config/pursuit_guidance_task_config.py`.
- Current task uses `robot_name = "base_quad_root_link_control"` and `controller_name = "thrust_bodyrate_control"`.
- Current target motion baseline is `target_motion.type = "apf_escape"`.
- Current observation is a 32D full-state geometric observation and leaks target-relative state plus environment truth.
- Current root-link robot inherits `enable_camera = False` from `BaseQuadCfg.sensor_config`; depth ingestion must be explicitly implemented before any depth-only claim is made.

## Required Baseline Ladder
| ID | Method | Role |
|----|--------|------|
| B0 | Full-state PPO, original 32D observation | Upper reference only |
| B1 | Reduced-information PPO, no risk | Main degradation baseline |
| B2 | Reduced-information PPO + depth-honest heuristic observability risk | Hand-crafted risk baseline |
| B3 | Reduced-information PPO + learned state risk `p_lost(s_red)` | Bridge-only learned baseline |
| B4 | Reduced-information PPO + oracle future risk label | Diagnostic upper bound for risk usefulness |
| B5 | Reduced-information PPO + learned action-conditioned risk correction | Main proposed method |

## Required External Comparison Classes
- Limited visual field / limited detectable region MARL: `[@peng2025limitedvisual]`, `[@huh2026limited]`.
- Prediction-enhanced pursuit with CTBR/deployment relevance: `[@chen2025open]`, `[@zhang2023gameofdrones]`.
- Risk-gated or safety-projected partial-observation pursuit: `[@li2026safeintent]`.
- Bearing-only or sensor-compatible target pursuit: `[@li2025bearingonly]`, `[@zheng2025visioncapture]`.
- Bilateral/self-play strong-adversary work for future scope only: `[@roncero2025amspb]`, `[@yao2026warning]`.

## Method Specification
### Observability Label
- Virtual detectable region in pursuer body frame: range `R_det`, horizontal FOV `alpha_h`, vertical FOV `alpha_v`.
- Label `y_t = 1` if the target exits the region within `H = 150` steps and remains outside for `K_persist = 10` consecutive steps.
- Sensitivity sweep: `H in {100, 150, 200}` and at least one FOV/range perturbation set.

### Depth And Risk Model
- Input: ego linear velocity `3`, ego angular velocity `3`, orientation representation `9`, previous CTBR action `4`, stacked depth frames `K = 3`.
- Frozen latent: `z_depth = 64`.
- Outputs:
  - `p_lost(s_red)`.
  - action-conditioned risk score `R_obs(s_red, u)` for the executed action and candidate correction actions.
  - auxiliary range/bearing targets during offline training only.
- Training: offline pretrain, freeze encoder and risk head, stop gradient during PPO.

### Action Correction
Use a one-step penalty projection:

`u* = argmin_u ||u - u_rl||_2^2 + lambda_risk R_obs(s_red, u) + lambda_fov L_fov(s_red, u) + lambda_obs L_obs(s_red, u) + lambda_rate ||u - u_prev||_2^2`

Hard constraints:
- `u_min <= u <= u_max`
- `||u - u_prev||_inf <= Delta_u_max`

Implementation rule:
- Use clamped projected-gradient or small candidate-set minimization.
- No MPC inner loop in Paper 1.

## Required Metrics
### Offline Risk Metrics
- AUROC, AUPRC, F1, Brier score, ECE.
- Event lead time before target loss.
- ID and OOD degradation across target-motion regimes.
- Learned risk must beat B2 heuristic risk on AUROC and AUPRC.

### Online Control Metrics
- Success rate, timeout rate, collision rate, episode length.
- Target-loss event rate, target-loss recovery rate, recovery latency.
- Fraction of time target remains inside the virtual detectable region.
- Control smoothness and action-correction magnitude.

### Statistical Rule
- At least 5 seeds for main online comparisons.
- Fixed training budget across B1-B5.
- Report confidence intervals and significance tests for B3 vs B1/B2 and B5 vs B3.

## Phase Gates
### Phase 0: Baseline Audit
- [x] Confirm full-state 32D observation and APF target path.
- [x] Confirm current pursuit path is not depth-driven.
- **Status:** complete

### Phase 1: Literature And Novelty Lock
- [x] Deep-research completed.
- [x] Novelty assessment completed.
- [x] Strong near-neighbor comparison set added to deep-research guidance.
- **Status:** complete

### Phase 2: Problem Definition
- [x] Lock problem as privileged-to-depth-only sensor-proxy transition.
- [x] Reject generic target prediction, generic partial observation, and full bilateral self-play as Paper 1 framing.
- **Status:** complete

### Phase 3: Method Design
- [x] Lock `H = 150`, `K_persist = 10`, `K = 3`, `z_depth = 64`.
- [x] Promote action-conditioned risk and CTBR correction to the main method.
- **Status:** complete

### Phase 4: Experiment Design
- [x] Lock B0-B5 ladder.
- [x] Lock offline/online metrics and statistical rules.
- **Status:** complete

### Phase 5: Deep-Research Synchronization
- [x] Add near-neighbor comparison papers and revised gap language.
- **Status:** complete

### Phase 6: Implementation And Validation
- [ ] Implement risk geometry and label generation.
- [ ] Implement dataset manifest and offline risk training.
- [ ] Implement depth stack ingestion and frozen `z_depth` path.
- [ ] Implement B1-B5 observation/control modes.
- [ ] Implement action-conditioned risk correction.
- [ ] Run smoke tests and small-rollout diagnostics before large training.
- **Status:** in_progress

### Phase 7: Paper Writing
- [ ] Write around the locked claim only.
- [ ] Keep weak intermediate ideas out of abstract, introduction, method, and experiment sections.
- **Status:** pending

## Explicitly Deferred
- Full RGB visual pursuit.
- Full onboard detector/tracker claims.
- Bilateral self-play or learned evader as the central contribution.
- Full occlusion-aware sensing as a main claim unless it is already implemented and validated.

## Errors Encountered
| Error | Resolution |
|-------|------------|
| Weak intermediate guidance remained in planning files | Rewrote guidance around the locked EAAI claim and removed full-state-plus-scalar framing |
