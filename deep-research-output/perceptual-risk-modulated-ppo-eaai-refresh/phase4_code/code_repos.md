# Phase 4 Code and Tool Landscape

Date: 2026-05-19

This phase records implementation ecosystems relevant to the planned work. It is not a code adoption recommendation; it is a reproducibility and baseline-awareness survey.

## Repositories / Tool Families

| Repo / Tool | URL | Relevance | Notes |
|---|---|---|---|
| MSMAR-RL project | https://msmar-rl.github.io | High-speed UAV pursuit-evasion safe/recovery RL | Linked by IJCAI 2025 paper; useful for how a high-speed PE paper presents baselines, safety metrics and ablations. |
| Safety Gym / safe RL baselines | https://openai.com/index/safety-gym/ | PPO-Lagrangian/CPO style baselines | Useful conceptual baseline: if B4 risk is treated as cost, compare against PPO-Lagrangian/CPO-like implementation. |
| Recovery RL | https://github.com/abalakrishna123/recovery-rl | Learned safety critic / recovery policy | Closest code family to action-conditioned risk critic with intervention. Compare against shield/filter baseline rather than using as main method. |
| CleanRL PPO | https://github.com/vwxyzjn/cleanrl | Minimal PPO implementation | Current repo already follows CleanRL-style PPO path; useful for clean integration and ablation discipline. |
| Safe Policy Optimization / safe-control-gym family | https://github.com/utiasDSL/safe-control-gym | Safe RL with control constraints | Useful for cost-critic, CBF/shield and constraint-reporting conventions. |
| RLlib / Stable-Baselines3 | https://github.com/ray-project/ray ; https://github.com/DLR-RM/stable-baselines3 | Standard RL baselines | Useful only as reference for PPO and mask/filter baselines; not needed for main implementation. |
| Isaac Gym / Aerial Gym local stack | local repo | Simulation and CTBR execution chain | Main implementation environment; critical because B4 must keep PPO-sampled action aligned with executed action. |

## Candidate Paper Code Implication

The accepted candidate papers did not reveal a directly reusable open-source implementation for action-conditioned observability-loss PPO modulation under LiDAR-reduced pursuit. They did, however, strengthen three code-facing requirements:

- Include safe-RL/shield-style references as baselines because high-speed shielded flight is now a direct adjacent family.
- Treat multi-UAV/MARL PE repos as comparison context only; adopting them would expand paper 1 beyond the current single-pursuer risk-variable scope.
- Keep the local implementation minimal and auditable inside the current CleanRL PPO path; the main engineering risk is action/logprob/execution inconsistency, not lack of external code.

## Code-Landscape Implications

- There are enough existing safe-RL tools to make a generic safe-PPO claim weak.
- A clean local B4 implementation should stay close to the current CleanRL PPO path and add only the risk-conditioned terms needed for the ablation.
- A fair baseline should include at least one PPO-Lagrangian-like variant and one post-hoc shield/filter variant using the same risk model.
- The code evidence strengthens the plan’s decision to avoid MPC/action-projection as the main method.

✅ Phase 4 complete. Output: `phase4_code/code_repos.md`. Proceeding to Phase 5.
