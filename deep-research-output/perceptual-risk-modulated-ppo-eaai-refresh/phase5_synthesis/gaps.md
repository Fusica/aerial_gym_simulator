# Phase 5 Gap Analysis

Date: 2026-05-19

## Key Gaps That Still Support the Paper

1. **Observability-loss risk is distinct from safety risk.**
   Existing safe RL usually models collision/constraint violations. The paper can focus on target-loss and reacquisition risk under onboard sensing.

2. **Action-conditioned observability risk remains under-isolated.**
   Several works use FOV rewards or risk gates, but few isolate whether different CTBR actions from the same reduced state have calibrated different probabilities of future target loss.

3. **PPO-internal risk modulation has not been clearly evaluated for LiDAR-reduced pursuit.**
   Existing safe PPO/CPO/Lagrangian methods provide nearby mechanisms, but not this sensing-specific risk target and B0-B4 causal chain.

4. **Sensor-honest pursuit evaluation is still sparse.**
   Many pursuit papers use local observations but still assume geometric target availability or simplified detection. A strict LiDAR/range-image no-truth-input protocol is useful.

5. **Engineering reliability metrics are often missing.**
   Lost-time, recovery latency, risk calibration, return-count distributions and action saturation can make the paper stronger for EAAI than success-only plots.

## Main Novelty Threats

1. Action-conditioned risk gating under partial observability may already exist as a generic method.
2. Recovery RL already learns action-conditioned safety critics.
3. CPO/PPO-Lagrangian already use cost/risk advantages.
4. Limited-FOV UAV pursuit and target reacquisition are already active research topics.
5. Asymmetric actor-critic already covers privileged training for partial-observation policies.
6. The accepted candidate papers show high-speed UAV PE, agile quadrotor PE, multi-UAV PE, scalable assignment, drone interception and shielded high-speed flight are all crowded.

## Required Claim Narrowing

Allowed:

- "We introduce an action-conditioned observability-loss risk model for LiDAR/range-image reduced-information UAV pursuit and use it to modulate PPO learning under deployable inputs."

Disallowed:

- "First action-conditioned risk-gated RL under partial observability."
- "First risk-aware UAV pursuit."
- "A new safe PPO algorithm."
- "Real-world validated 100-200m small-UAV LiDAR pursuit."

## Required Plan Updates

- Treat Phase 1/5 as refreshed only for the narrowed claim after candidate-paper integration.
- Add PPO-Lagrangian and shield/filter to B4 internal ablations.
- Add matched-state action-risk ranking as an offline validation gate.
- Add LiDAR return-count/dropout/sparsity sensitivity as an engineering validation gate.
- Treat 100/150/200m sparse target returns as a stress-test regime, not a real hardware guarantee.
- Write a dedicated related-work split: UAV PE/MARL, high-speed/agile/interception, limited-perception sensing, safe/shielded/risk RL, and this paper's observability-risk PPO modulation.

✅ Phase 5 complete. Output: `phase5_synthesis/synthesis.md`, `phase5_synthesis/gaps.md`. Proceeding to Phase 6.
