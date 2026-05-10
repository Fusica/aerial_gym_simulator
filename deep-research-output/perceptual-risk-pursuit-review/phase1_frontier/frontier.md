# Frontier: UAV Pursuit-Evasion RL Under Sensing Constraints
Date: 2026-05-09
Topic slug: `perceptual-risk-pursuit-review`

## Search Basis
- Seed set from user-provided BibTeX entries, focused on 2024-2026 UAV pursuit-evasion, perception-enhanced pursuit, and related agile flight papers.
- Web expansion used because the local Semantic Scholar/arXiv scripts could not access the network inside the current sandbox.

## Recent Key Papers (latest 1-2 cycles)
1. `[@liqin2026perception]` A perception-enhanced multi-agent deep reinforcement learning method for multi-UAV cooperative pursuit. ESWA 2026.
   Local-observation pursuit with graph attention and Apollonius-circle pursuit judgment.
2. `[@lei2026causal]` Causal reinforcement learning for UAV pursuit-evasion games with sparse rewards. Journal of Supercomputing 2026.
   Sparse-reward pursuit with causal intrinsic reward plus attraction-based prioritized replay.
3. `[@yuan2026cbpe]` CBPE-based assignment policy for multiple player pursuit-evasion game by D3QN. IEEE TIE 2026.
   Game-theoretic assignment and approximate Nash equilibrium for multi-player pursuit-evasion.
4. `[@yao2026warning]` Pursuit-evasion game of unmanned tracked vehicles based on MATD3. EAAI 2026.
   Warning-area state compression and simultaneous pursuer/evader training.
5. `[@huh2026limited]` Multi-Agent Reinforcement Learning for Multi-UAV Pursuit with Full Planar Motion and a Limited Detectable Region. Machines 2026.
   Limited-detectable-region formulation with APF evader and FOV-aware reward design.
6. `[@li2026safeintent]` Safe Cooperative Decision-Making for Multi-UAV Pursuit-Evasion Games via Opponent Intent Inference. Sensors 2026.
   Opponent intent/subgoal prediction, belief-risk-gated hierarchical policy, and safety projection under intermittent observability.
7. `[@zhang2026shielded]` High-speed vision-based flight in clutter with safety-shielded reinforcement learning. arXiv 2026 (preprint).
   End-to-end vision policy with deployment-time safety filter.
8. `[@sun2026curriculum]` Curriculum reinforcement learning for quadrotor racing with random obstacles. arXiv 2026 (preprint).
   Vision-based curriculum RL for obstacle-rich high-speed quadrotor control.
9. `[@ren2026realworld]` Learning agile quadrotor flight in the real world. arXiv 2026 (preprint).
   Real-world online adaptation for aggressive quadrotor flight without heavy sim-to-real dependence.
10. `[@peng2025limitedvisual]` Multi-UAV Cooperative Pursuit Strategy With Limited Visual Field in Urban Airspace. IEEE/CAA JAS 2025.
    Limited visual field, obstruction, target reacquisition, normalizing-flow actor, and graph-attention critic.
11. `[@li2025bearingonly]` Cooperative Bearing-Only Target Pursuit via Multiagent Reinforcement Learning. arXiv 2025.
    Bearing-only target pursuit and target-loss resilience under limited FoV.
12. `[@chen2025open]` Online planning for multi-UAV pursuit-evasion in unknown environments using deep reinforcement learning. RA-L 2025.
   Prediction-enhanced cooperative pursuit with CTBR-compatible deployment emphasis.
13. `[@sun2025kematd3]` Emergent cooperative strategies for pursuit-evasion in cluttered environments. IROS 2025.
    Knowledge-enhanced MATD3 using APF-derived cues in cluttered environments.
14. `[@xu2025general]` A DRL framework for autonomous pursuit-evasion: from multi-spacecraft to multi-drone scenarios. Drones 2025.
    Self-play PPO with curriculum and prediction-based dense reward.
15. `[@roncero2025amspb]` Learned controllers for agile quadrotors in pursuit-evasion games. arXiv 2025 (preprint).
    Asynchronous population-based 1v1 quadrotor pursuit-evasion with body-rate-and-thrust control.
16. `[@zheng2025visioncapture]` Vision-based cooperative MAV-capturing-MAV. arXiv 2025 (preprint).
    Onboard-vision cooperative capture system with distributed estimation and MPC.
17. `[@tan2025scalable]` Scalable Pursuit-Evasion Game for Multi-Fixed-Wing UAV Based on Dynamic Target Assignment and Hierarchical RL. Drones 2025.
    Large-scale scalable hierarchical pursuit-evasion with target assignment.

## Frontier Themes

### Theme A: Perception- or prediction-enhanced pursuit
- Key papers: `[@liqin2026perception]`, `[@chen2025open]`, `[@zheng2025visioncapture]`
- Trend: teams are moving away from purely privileged-state pursuit and adding prediction, attention, or onboard sensing.

### Theme B: Partial observability and limited detectable regions
- Key papers: `[@peng2025limitedvisual]`, `[@huh2026limited]`, `[@li2025bearingonly]`, `[@li2024limitedusv]`, `[@sun2022pomdp]`
- Trend: observability constraints are now treated as first-class modeling assumptions, including limited visual fields, target loss, reacquisition, and bearing-only sensing.

### Theme B2: Risk-gated decision support under intermittent observability
- Key papers: `[@li2026safeintent]`
- Trend: risk features and safety projection are entering pursuit-evasion pipelines, so a new paper must differentiate through the specific risk target and action-level correction mechanism.

### Theme C: Stronger adversaries and bilateral optimization
- Key papers: `[@roncero2025amspb]`, `[@yao2026warning]`, `[@xu2025general]`
- Trend: training both pursuer and evader, frozen-population opponents, and self-play are becoming more common.

### Theme D: High-fidelity low-level control and real-world transfer
- Key papers: `[@chen2025open]`, `[@ren2026realworld]`, `[@zhang2026shielded]`
- Trend: body-rate/thrust commands, safety filters, online adaptation, and deployment claims matter more than abstract point-mass pursuit.

## Why This Frontier Matters For The Current Repo
- The current repo already has a working CTBR/body-rate pursuit stack with APF evader motion.
- The nearest frontier gap is not generic pursuit RL, limited-FOV pursuit, or generic risk-gated pursuit.
- Among the latest works, no seed paper cleanly matches the exact transition problem: `full-state successful pursuit -> depth-only reduced information -> action-conditioned observability-risk model -> CTBR action correction`.

## Active Groups / Directions
- Multi-UAV cooperative MARL with partial observability and graph communication.
- High-fidelity quadrotor pursuit-evasion with body-rate-and-thrust control.
- Vision-based interception and obstacle-rich autonomous flight.
- Bilateral adversarial training and population-based stabilization.

## Initial Frontier Judgment
- The hottest directly relevant subspace is the intersection of pursuit-evasion, partial observability, and deployable control.
- The current repo can plausibly contribute by attacking action-conditioned observability-risk correction instead of competing head-on with full vision pursuit, generic limited-FOV MARL, or full bilateral self-play.
