# Phase 3 Deep-Dive Selection

Date: 2026-05-19

Selected papers are chosen to cover the strongest novelty threats and the EAAI scope argument.

| Key | Why selected |
|---|---|
| `[@ren2025losrate]` | EAAI paper showing reduced-measurement DRL guidance/interception is in scope. |
| `[@zhao2025msmar]` | High-speed pursuit-evasion safe/recovery RL from IJCAI; strongest high-speed PE safety neighbor. |
| `[@peng2025limitedvisual]` | Limited-FOV UAV pursuit with search/reacquisition and multiple evader strategies. |
| `[@huh2026limitedregion]` | Limited detectable region and FOV reward/capturability reward in UAV pursuit. |
| `[@thananjeyan2021recoveryrl]` | Learned action-conditioned safety critic; closest structural analogy to `R_obs(s,u)`. |
| `[@achiam2017cpo]` | Cost value and cost advantage baseline for any risk-adjusted PPO claim. |
| `[@pinto2017aac]` | Privileged full-state training for partial-observation robot policies. |
| `[@liu2026actionriskgating]` | Direct method-level threat: action-conditioned risk gating in POMDPs. |
| `[@gao2026mentalstates]` | EAAI imperfect multi-UAV competitive environment; scope and opponent-modeling contrast. |
| `[@park2023lidardrone]` | LiDAR + PPO drone navigation; sensor-policy feasibility but not pursuit novelty. |

## Supplemental Candidate Paper Selection

The following accepted candidate papers are not all deep-read candidates for method transfer, but each is retained because it creates concrete comparison, ablation or Related Work obligations:

| Key | Why added to the pressure set |
|---|---|
| `[@yan2024lsrctd3]` | High-speed UAV PE with DRL maneuvering; blocks broad "high-speed PE + DRL" novelty. |
| `[@giral2026intercept]` | Aerial-robot interception with RL; supports aerospace relevance and pressures interception claims. |
| `[@roncero2025agilecontrollers]` | Agile quadrotor pursuit-evasion learned controllers; close domain pressure. |
| `[@luo2024improvedmadrl]` | Multi-UAV PE with improved MADRL; blocks generic cooperative PE novelty. |
| `[@tan2026scalablefixedwing]` | Scalable multi-fixed-wing UAV PE with hierarchical RL; blocks scalability/assignment claims. |
| `[@zhang2026safetyshieldedflight]` | High-speed flight with a safety shield; makes shield/filter a mandatory baseline family. |
| `[@feroskhan2024multipursuitevasion]` | Drone multipursuit evasion with real-time flight validation; warns against real-world validation overclaiming. |
| `[@mavcapturingmav2024]` | Vision-based cooperative MAV capture; blocks first sensor/capture-system framing. |

These papers re-weight the deep-dive conclusion: the paper should not compete on "better UAV PE" broadly. It should compete on the isolated risk variable, action-conditioned risk ranking and PPO-internal modulation.

✅ Phase 3 selection complete. Proceeding to deep reading notes.
