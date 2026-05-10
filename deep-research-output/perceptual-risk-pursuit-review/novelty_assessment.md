# Novelty Assessment: Action-Conditioned Observability-Risk-Corrected UAV Pursuit-Evasion
Date: 2026-05-10
Based on: 65-paper database plus novelty-assessment web expansion

## Assessment Result

```json
{
  "decision": "novel",
  "confidence": "medium",
  "search_rounds": 6,
  "total_papers_reviewed": 65,
  "justification": "The broad idea of risk-assisted or partial-observation pursuit is not novel. The narrowed idea remains novel: depth-only reduced-information pursuit using an action-conditioned short-horizon observability-risk model learned from privileged rollouts and used for CTBR action correction.",
  "most_similar_papers": [
    {"title": "Safe Cooperative Decision-Making for Multi-UAV Pursuit-Evasion Games via Opponent Intent Inference", "year": 2026, "overlap": "Risk-gated partial-observation pursuit with intent/subgoal prediction and safety projection. Differs: does not directly learn observability-loss risk from privileged rollouts for depth-only CTBR action correction."},
    {"title": "Multi-UAV Cooperative Pursuit Strategy With Limited Visual Field in Urban Airspace", "year": 2025, "overlap": "Limited visual field pursuit with search/pursuit/reacquisition and graph-attention MARL. Differs: no auditable learned observability-risk label and no CTBR risk correction."},
    {"title": "Multi-Agent Reinforcement Learning for Multi-UAV Pursuit with Full Planar Motion and a Limited Detectable Region", "year": 2026, "overlap": "Limited detectable region, FOV rewards, APF evader. Differs: reward/action design rather than learned action-conditioned risk."},
    {"title": "Online Planning for Multi-UAV Pursuit-Evasion in Unknown Environments Using Deep Reinforcement Learning", "year": 2025, "overlap": "Prediction-enhanced pursuit and CTBR deployment. Differs: target trajectory prediction, not observability-loss risk or action correction."},
    {"title": "Cooperative Bearing-Only Target Pursuit via Multiagent Reinforcement Learning", "year": 2025, "overlap": "Sensor-compatible target pursuit and target-loss resilience. Differs: bearing-only filtering, not depth-only risk learning and CTBR correction."}
  ],
  "differentiation": "The paper must combine four pieces as one mechanism: reduced/depth-only policy input, observability-loss label, action-conditioned risk scoring, and CTBR constrained action correction. Any version that drops action conditioning or returns to full-state-plus-risk is not sufficiently novel."
}
```

## Harsh-Critic Boundary

Not novel enough:
- full-state PPO plus a learned risk scalar
- risk as a passive observation side channel only
- generic target trajectory prediction
- generic limited-FOV or limited-detectable-region reward shaping
- generic risk-gated MARL or safety projection

Novel enough for a strong EAAI attempt:
- `R_obs(s_red, u)` action-conditioned observability-risk scoring
- privileged rollout labels used only offline
- depth-only sensor-proxy reduced-information policy input
- CTBR action correction as the main mechanism
- B0-B5 experiment ladder proving the effect is not just extra information

## Most Similar Papers For Related Work

| Paper | Shared Elements | Required Differentiation |
|-------|-----------------|--------------------------|
| `[@li2026safeintent]` | intermittent observability, risk features, safety projection | direct observability-loss label, depth-only sensor-proxy, CTBR action-conditioned correction |
| `[@peng2025limitedvisual]` | limited visual field, target loss, reacquisition | learned calibrated risk and correction layer rather than NAGC MARL |
| `[@huh2026limited]` | limited detectable region, APF evader, FOV reward | learned risk target and offline/online risk evaluation |
| `[@chen2025open]` | prediction-enhanced pursuit, CTBR relevance | risk prediction instead of trajectory prediction, action correction instead of policy input only |
| `[@li2025bearingonly]` | sensor-compatible pursuit under limited FoV | depth-only risk model instead of bearing-only filter |
| `[@dalal2018safe]` | action projection / safety layer | pursuit-specific observability-risk objective and CTBR semantics |

## Required Pre-Submission Novelty Checks
- Search exact phrase: `"observability risk" "pursuit-evasion"`.
- Search exact phrase: `"belief-risk" "pursuit-evasion"`.
- Search exact phrase: `"limited visual field" "pursuit-evasion"`.
- Search exact phrase: `"action-conditioned risk" "pursuit" UAV`.
- Re-check arXiv and Semantic Scholar within one month before submission.
