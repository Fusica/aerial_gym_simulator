# Phase 3 Selection
Date: 2026-05-09

## Selection Principle
Selected papers maximize coverage over:
- direct UAV pursuit-evasion under sensing constraints
- prediction-augmented or knowledge-enhanced pursuit
- 1v1 agile quadrotor adversarial learning
- adjacent vision / safety / real-world control works that inform sensing-compatible pursuit implementation

## Selected Papers
1. `[@peng2025limitedvisual]`
   Why selected: high-tier limited-visual-field multi-UAV pursuit paper with search, pursuit, and reacquisition phases; strongest direct warning that limited-FOV pursuit alone is not novel.
2. `[@li2026safeintent]`
   Why selected: closest risk-gated partial-observation pursuit neighbor; uses opponent intent inference, trajectory prediction, risk features, hierarchical policy, and safety projection.
3. `[@li2025bearingonly]`
   Why selected: sensor-compatible target pursuit reference focused on bearing-only information and resilience to target loss under limited FoV.
4. `[@huh2026limited]`
   Why selected: most directly aligned limited-detectable-region UAV pursuit paper; uses APF evader and realistic sensing geometry.
5. `[@xu2025general]`
   Why selected: self-play PPO framework spanning multi-spacecraft to multi-drone pursuit-evasion with prediction-based reward.
6. `[@tan2025scalable]`
   Why selected: strong representative of hierarchical scalable pursuit-evasion and target assignment.
7. `[@yan2024lsrc]`
   Why selected: high-speed UAV pursuit-evasion / evasion strategy with dense reward engineering and strong maneuver focus.
8. `[@roncero2025amspb]`
   Why selected: strongest 1v1 agile quadrotor pursuit-evasion reference with body-rate-and-thrust control and asynchronous adversarial training.
9. `[@zheng2025visioncapture]`
   Why selected: strongest onboard-vision cooperative capture system among the seed papers, useful for realistic sensing and capture-zone logic.
10. `[@zhang2026shielded]`
   Why selected: not a pursuit paper, but highly relevant for vision-based control plus deployment-time safety filter.
11. `[@sun2026curriculum]`
   Why selected: not a pursuit paper, but directly relevant to vision-based curriculum design and obstacle-rich aggressive flight.

## Why These 8 Are Sufficient For Synthesis
- They jointly cover the closest research lines to the current repo:
  1. limited visual field / limited detectable region pursuit
  2. prediction-enhanced and risk-gated partial-observation pursuit
  3. agile 1v1 or adversarial quadrotor training
  4. sensing-compatible high-speed flight and real-world deployment
- They are enough to test the stricter Paper-1 claim: action-conditioned observability-risk correction under depth-only reduced information.
