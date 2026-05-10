# Survey: Pursuit-Evasion RL For UAVs Under Sensing Constraints
Date: 2026-05-09
Curated papers in DB: 65

## Scope
This survey focuses on pursuit-evasion and adjacent UAV autonomy work that is useful for the current `aerial_gym_simulator` project:
- 1v1 or multi-agent pursuit-evasion
- partial observability / limited detectable region
- prediction-augmented or perception-enhanced control
- high-fidelity quadrotor control, body-rate / thrust interfaces, and transfer concerns
- auxiliary works on onboard vision and obstacle-rich agile flight when they inform sensing-compatible pursuit

## Themes Identified

### Theme 1: Multi-UAV cooperative pursuit with MARL
Representative papers:
- `[@liqin2026perception]` perception-enhanced cooperative pursuit
- `[@zhao2024cooperative]` autonomous decision making for cooperative pursuit-evasion
- `[@sun2025kematd3]` knowledge-enhanced cooperative pursuit in clutter
- `[@luo2024commnet]` improved MADRL with communication GRUs

Key pattern:
- Centralized training with decentralized execution dominates.
- Coordination is improved via graph attention, explicit communication, or shaped team rewards.

### Theme 2: Partial observability and limited perception
Representative papers:
- `[@peng2025limitedvisual]` limited visual field pursuit with search/pursuit/reacquisition phases
- `[@huh2026limited]` limited detectable region for UAV pursuit
- `[@li2024limitedusv]` limited perception MAPPO for USV swarm pursuit
- `[@sun2022pomdp]` scalable DRL for partially observable pursuit-evasion
- `[@horak2016pointbased]` one-sided partially observable PEG foundations

Key pattern:
- Many works acknowledge sensing limits, but most solve them with reward shaping, recurrent encoders, or local observations rather than a dedicated risk-prediction interface.

### Theme 3: Prediction-enhanced or knowledge-enhanced pursuit
Representative papers:
- `[@zhang2023gameofdrones]` target prediction network in cooperative pursuit
- `[@chen2025open]` evader prediction-enhanced network
- `[@li2026safeintent]` intent/subgoal prediction with belief-risk-gated policy and safety projection
- `[@sun2025kematd3]` APF-derived knowledge enhancement
- `[@xu2025general]` prediction-based reward shaping

Key pattern:
- Prediction is already a known lever.
- Risk-gated decision layers now exist, so the current project must go beyond a generic risk feature.
- What remains less explored is action-conditioned *observability failure* prediction for CTBR correction under depth-only reduced information.

### Theme 4: Bilateral training, self-play, and strong-adversary learning
Representative papers:
- `[@roncero2025amspb]` asynchronous population-based 1v1 quadrotor training
- `[@yao2026warning]` simultaneous pursuer/evader optimization with compressed state encoding
- `[@xu2025general]` self-play PPO framework

Key pattern:
- These methods strengthen opponents and improve robustness, but they significantly increase coupling and experimental burden.
- They are better treated as later-stage expansion rather than first-paper core scope for the current repo.

### Theme 5: High-fidelity control and real-world sensing-compatible autonomy
Representative papers:
- `[@chen2025open]` body-rate and thrust deployment
- `[@zheng2025visioncapture]` onboard-vision capture system
- `[@ren2026realworld]` real-world adaptive aggressive flight
- `[@zhang2026shielded]` vision-based shielded RL in clutter

Key pattern:
- Realistic low-level control and sensing matter.
- However, not every paper that uses vision or body-rate control is a pursuit-evasion paper; these should support the realism argument rather than define the core method comparison.

## Venue Distribution
- Peer-reviewed journals and conferences: 27
- arXiv / SSRN preprints: 8

## Time Distribution
- 2025-2026: 19 papers
- 2023-2024: 12 papers
- 2002-2022: 4 papers

## Key Observations
1. Pursuit-evasion RL has moved from simple fully observable games toward clutter, stronger evaders, limited detection, and deployment-oriented control interfaces.
2. Prediction modules already exist, but mostly for target motion forecasting or dense reward shaping.
3. A narrower contribution centered on *short-horizon observability-risk prediction* is better differentiated than a generic "prediction-enhanced pursuit" claim.
4. The current repo is especially well positioned for a modular bridge paper because it already has a successful full-state pursuit baseline and a CTBR/body-rate-compatible low-level path.

## Most Relevant Works To The Current Repo
- `[@chen2025open]`: because it already uses CTBR/body-rate control and prediction-enhanced pursuit in unknown environments.
- `[@peng2025limitedvisual]`: because it is a high-tier limited-visual-field MARL baseline with search, pursuit, and reacquisition logic.
- `[@li2026safeintent]`: because it is the closest risk-gated partial-observation pursuit neighbor and must be explicitly differentiated.
- `[@li2025bearingonly]`: because it is a sensor-compatible target-pursuit reference focused on limited sensing and target-loss resilience.
- `[@huh2026limited]`: because it formalizes limited detectable region and uses APF evader motion.
- `[@zhang2023gameofdrones]`: because it is a strong prior example of prediction-augmented pursuit.
- `[@roncero2025amspb]`: because it is the strongest 1v1 agile quadrotor adversarial training reference.
- `[@liqin2026perception]`: because it addresses partial observability and local observation in cooperative UAV pursuit.

## Survey-Level Research Gap
The gap most aligned with the current repo is not "yet another pursuit RL policy" and not "full end-to-end vision pursuit." The gap is:

`How to derive an action-conditioned, learnable observability-risk signal from a successful privileged-state pursuit system, and show that this signal improves depth-only reduced-information CTBR pursuit through explicit action correction rather than only adding a passive risk feature.`
