# Three-Subagent Cross Validation Summary

Date: 2026-05-19

## Subagent 1: Pursuit / Partial-Observability Novelty

Verdict: EAAI potential is real, but claim must be narrowed. Limited-FOV pursuit, visibility-based pursuit, safe/risk-aware PE and action-conditioned risk gating already exist. The strongest surviving contribution is LiDAR/range-image reduced UAV pursuit with action-conditioned observability-loss risk and PPO-internal modulation.

Required additions:

- action-conditioned ablation;
- risk concat vs modulation;
- sensor truth-leakage controls;
- risk calibration and stress tests.

## Subagent 2: RL Method Novelty

Verdict: B4 cannot be sold as a new safe PPO or risk-sensitive PPO. It overlaps strongly with Recovery RL, CPO/PPO-Lagrangian, shielded RL and asymmetric actor-critic.

Required additions:

- compare against PPO-Lagrangian and shield/filter baselines;
- prove B4 beats B3 and cost-advantage variants;
- keep PPO logprob/executed action consistent;
- define B4 narrowly as observability-risk modulation for this engineering problem.

## Subagent 3: Sensor / Engineering Credibility

Verdict: EAAI scope matches, but sensor assumptions are the weakest part. The 100/150/200m target return setting should be written as sparse-return stress testing, not real long-range deployability.

Required additions:

- LiDAR return-count and dropout sweeps;
- range/reflectivity/angular-resolution/latency sensitivity;
- avoid sim-to-real claims without real logs;
- add tracking-before-policy baseline if feasible.

## Combined Decision

Proceed with the narrowed claim after candidate-paper integration:

- Phase 1 and 5 are refreshed only for the narrowed risk-modulated PPO claim.
- Phase 3 remains in progress until B4 chooses a concrete formula/interface.
- Phase 4 can be treated as design-complete because the required PPO-Lagrangian, shield/filter, risk concat, privileged critic, shuffled-action risk and matched-state action-risk tests have been added to the experiment registry.

## 2026-05-19 Candidate Paper Cross-Check

Three read-only follow-up checks after candidate-paper integration reached the same conclusion:

- EAAI scope is strengthened by the expanded UAV/RL/interception literature.
- Broad novelty is weakened because high-speed UAV PE, multi-UAV PE, agile quadrotor PE, sensor capture and shielded high-speed flight are crowded.
- The defensible contribution remains `R_obs(s_red,u)` as an action-conditioned observability-loss risk, learned from privileged rollouts and used as a PPO-internal modulation signal under sensor-honest LiDAR/range-image inputs.
- No fatal contradiction was found in the planning files. The only stale item was the earlier Phase 4 wording above, now corrected.

Required defense package remains:

- B0-B4 main chain.
- B4 ablations: risk concat, state-risk vs action-risk, shuffled-action risk, actor modulation, risk-conditioned advantage, actor+advantage, PPO-Lagrangian, shield/filter, privileged critic.
- Offline proof: AUROC/AUPRC/Brier/ECE, reliability, risk-bin future loss rate and matched-state candidate-action risk ranking.
- Online proof: success/time-to-capture plus final detectable, lost time, max invisible streak, recovery rate/delay, action saturation, KL/entropy and visible-strike score.
- Sensor proof: return-count/occupancy at 50/100/150/200m, dropout, sparse-point ablation, latency/noise/angular-resolution/reflectivity sensitivity.
