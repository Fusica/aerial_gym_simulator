# Synthesis: Perceptual-Risk-Guided Pursuit-Evasion
Date: 2026-05-09

## Taxonomy

Pursuit-Evasion RL for UAVs
├── Observation model
│   ├── Full-state / privileged-state pursuit: `[@zhang2023gameofdrones]`, `[@roncero2025amspb]`, `[@xu2025general]`
│   ├── Local observation / limited visual field / limited detectable region: `[@peng2025limitedvisual]`, `[@huh2026limited]`, `[@li2025bearingonly]`, `[@li2024limitedusv]`, `[@liqin2026perception]`, `[@sun2022pomdp]`
│   └── Onboard-vision / sensor-driven systems: `[@zheng2025visioncapture]`, `[@zhang2026shielded]`, `[@sun2026curriculum]`
├── Decision support mechanism
│   ├── Reward shaping and hand-crafted geometry: `[@huh2026limited]`, `[@yan2024lsrc]`
│   ├── Prediction-enhanced policy or reward: `[@zhang2023gameofdrones]`, `[@chen2025open]`, `[@xu2025general]`
│   ├── Risk-gated / safety-projected hierarchy: `[@li2026safeintent]`
│   └── Proposed gap: action-conditioned observability-risk correction under depth-only reduced information
├── Multi-agent interaction structure
│   ├── Cooperative MARL: `[@liqin2026perception]`, `[@sun2025kematd3]`, `[@luo2024commnet]`
│   ├── Hierarchical / target allocation: `[@tan2025scalable]`
│   └── Bilateral adversarial co-training / self-play: `[@roncero2025amspb]`, `[@yao2026warning]`, `[@xu2025general]`
└── Action realism
    ├── Kinematic or simplified command policies: many cooperative MARL papers
    ├── Velocity-level policies: older pursuit works and some partial-observation baselines
    └── Body-rate / thrust / realistic deployment: `[@chen2025open]`, `[@roncero2025amspb]`, `[@zheng2025visioncapture]`

## Comparative Table
| Method class | Paper | Observability assumption | Core mechanism | Action realism | Key result / role |
|-------------|-------|--------------------------|----------------|----------------|-------------------|
| Prediction-enhanced cooperative pursuit | `[@zhang2023gameofdrones]` | mostly full-state / prediction-aided | target prediction network | mid/high | strongest prior prediction-augmented pursuit reference |
| Unknown-environment pursuit | `[@chen2025open]` | partial, prediction-enhanced | evader prediction + adaptive env generation | high | direct CTBR/body-rate deployment relevance |
| Limited visual field MARL | `[@peng2025limitedvisual]` | explicit visual field with obstruction | normalizing-flow actor + graph attention critic | moderate | strongest high-tier limited-FOV pursuit neighbor |
| Risk-gated partial-observation pursuit | `[@li2026safeintent]` | occlusion, sensor noise, intermittent observability | intent/subgoal prediction + belief-risk gate + safety projection | moderate | closest risk-gated decision-support neighbor |
| Bearing-only target pursuit | `[@li2025bearingonly]` | bearing-only sensing under limited FoV | information filter + MARL pursuit control | moderate | strongest sensor-compatible target-loss reference |
| Limited detectable region pursuit | `[@huh2026limited]` | explicit limited FOV/range | reward shaping + lateral/yaw motion design | moderate | strongest observability-constraint baseline |
| Perception-enhanced cooperative pursuit | `[@liqin2026perception]` | local observations | graph attention among pursuers | moderate | strong cooperative partial-observation reference |
| Cross-domain self-play pursuit | `[@xu2025general]` | mostly privileged pursuit state | prediction-based reward + curriculum | moderate/high | strong generalization and reward-design reference |
| Scalable hierarchical pursuit | `[@tan2025scalable]` | assignment + hierarchy | target allocation + trajectory prediction | high | scale-oriented but less aligned to first-paper scope |
| Agile 1v1 adversarial quadrotor RL | `[@roncero2025amspb]` | privileged opponent positions | AMSPB + rate/thrust control | high | strongest 1v1 later-stage adversarial benchmark |
| Vision-based capture system | `[@zheng2025visioncapture]` | onboard vision | cooperative estimation + MPC + net capture | very high | strongest real sensing/capture systems reference |

## Timeline
- **2002**: `[@vidal2002probabilistic]` establishes probabilistic pursuit-evasion games and sensing-capability tradeoffs.
- **2016-2022**: partial-observable PEGs and approximate algorithms begin to formalize sensing uncertainty, e.g. `[@horak2016pointbased]`, `[@sun2022pomdp]`.
- **2021-2024**: cooperative MARL and limited-perception swarm pursuit expand, e.g. `[@luo2024commnet]`, `[@li2024limitedusv]`.
- **2023**: `[@zhang2023gameofdrones]` makes prediction-enhanced multi-UAV pursuit a visible line of work.
- **2025-2026**: the field splits into several stronger sub-lines:
  - deployable prediction-enhanced pursuit `[@chen2025open]`
  - limited-visual-field and limited-detectable-region pursuit `[@peng2025limitedvisual]`, `[@huh2026limited]`
  - risk-gated partial-observation pursuit `[@li2026safeintent]`
  - scalable hierarchy `[@tan2025scalable]`
  - agile 1v1 body-rate adversarial RL `[@roncero2025amspb]`
  - vision-based real systems and high-speed flight `[@zheng2025visioncapture]`, `[@zhang2026shielded]`, `[@sun2026curriculum]`

## Cross-Cutting Insights
1. **Prediction is already accepted, but mostly as motion prediction.**
   The literature already supports adding predictive components to pursuit pipelines. What is still underdeveloped is predicting *failure of observability or tactical viability* rather than future target coordinates alone.

2. **Partial observability is usually handled implicitly, not as a standalone risk interface.**
   Most papers address sensing limits through local observations, recurrence, reward shaping, or limited-FOV environment design. Few isolate a separate learned observability-risk signal that can be audited offline and then injected into the policy.

3. **Realistic action spaces and deployment paths matter.**
   Works like `[@chen2025open]`, `[@roncero2025amspb]`, and `[@zheng2025visioncapture]` show that body-rate/thrust control and onboard sensing compatibility are becoming decisive for credibility.

4. **Full bilateral co-training is powerful but too entangled for this repo’s first paper.**
   `[@roncero2025amspb]`, `[@yao2026warning]`, and `[@xu2025general]` show the value of stronger opponents and self-play. But they also raise training complexity and confound causal interpretation. For the current project, they are better reserved for paper 2 or later.

5. **The current repo has a specific comparative advantage.**
   It already has a successful full-state pursuit stack, APF evader motion, and a CTBR/body-rate-friendly path. This makes it unusually well suited to study a *transition interface* from privileged-state pursuit to sensing-compatible pursuit.

## Synthesis-Level Positioning For The Current Project
The strongest first-paper position is:

`Use the current successful full-state pursuit system as a supervision source, define a short-horizon observability-risk label, learn an action-conditioned risk model from depth-only reduced information, and show that CTBR action correction recovers pursuit robustness lost under sensing constraints.`

This position is stronger than:
- a generic "prediction-enhanced pursuit" claim, because prediction already exists in the literature
- a generic "partial-observability pursuit" claim, because limited-perception pursuit already exists
- a full vision pursuit claim, because the current repo does not yet expose a clean visual pursuit stack
- a generic risk-gated policy claim, because risk-gated partial-observation pursuit already exists

It is also narrower and more defensible than:
- full bilateral co-training
- full end-to-end visual pursuit
- pure reward shaping around FOV without a learned predictor
- passive risk-as-observation without action-conditioned correction
