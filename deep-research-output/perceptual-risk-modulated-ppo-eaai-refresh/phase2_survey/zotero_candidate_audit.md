# Zotero Candidate Audit

Date: 2026-05-19

Purpose: audit the user-provided Zotero/tag list against the paper's current mainline. This is deliberately **not** a bulk database expansion. A paper is accepted into the active DB only if it directly serves at least one of these roles:

- real comparison or near-neighbor pressure for UAV pursuit-evasion / interception;
- design inspiration for B4, especially shield/filter, PPO modulation, body-rate/thrust policy structure, curriculum or sensing evaluation;
- required Related Work coverage for the narrowed claim.

## Audit Summary

- User-provided candidates reviewed: 31.
- Direct duplicates already covered: 2.
- Accepted into active `paper_db.jsonl`: 11.
- Rejected from active DB for this paper: 18.
- Active DB after audit: 52 records.

Direct duplicates:

- `33KIR7IV` -> existing `shao2026peqmixer`.
- `F5LGULL9` -> existing `chen2025open`.

## Accepted Into Active DB

| Zotero key | Active key | Role in current paper | How it serves the mainline |
|---|---|---|---|
| `DZ9XEJ3V` | `yan2024lsrctd3` | near-neighbor pressure | High-speed UAV PE with DRL maneuvering; blocks broad "high-speed DRL pursuit" novelty. |
| `NB27S9HK` | `yang2025rlpereview` | survey anchor | Provides the broad PE/RL landscape; supports Related Work framing. |
| `QRK4FJ2P` | `zhao2024autonomousuavpe` | comparison pressure | UAV cooperative PE with RL; helps delimit paper 1 away from cooperative/MARL claims. |
| `B4JYNMIE` | `xiang2025cihrl` | comparison pressure | Multi-constrained UAV PE with hierarchical RL; strengthens need to avoid generic constrained-PE novelty. |
| `VY3RF5XZ` | `zhang2026safetyshieldedflight` | ablation driver | High-speed flight with safety shielding; makes shield/filter a mandatory B4 baseline. |
| `BDR6PTXJ` | `giral2026intercept` | scope + comparison pressure | RL aerial-robot interception; supports application fit while blocking "RL interception" novelty. |
| `M744BDI3` | `roncero2025agilecontrollers` | strongest domain neighbor | Agile quadrotor PE with learned controllers/body-rate-thrust relevance; important comparison and controller-policy context. |
| `UQC3ZYQM` | `feroskhan2024multipursuitevasion` | evaluation pressure | Drone multipursuit/evasion with real-time validation; prevents overclaiming deployment. |
| `PDW84WKY` | `luo2024improvedmadrl` | comparison pressure | Multi-UAV PE/MADRL; required to distinguish from cooperative maneuver decision-making. |
| `4G7JXTX7` | `tan2026scalablefixedwing` | comparison pressure | Scalable multi-fixed-wing PE; blocks scalability/assignment claims. |
| `I2KS8X4Z` | `mavcapturingmav2024` | sensor/capture-system pressure | Vision-based MAV capture; prevents "first sensor-based capture" framing. |

## Rejected From Active DB

These are not deleted from the user's Zotero library. They are simply not part of the current paper's active evidence set because they do not directly support B4 design, mandatory baselines, or the narrowed Related Work.

| Zotero key | Reason not active for paper 1 |
|---|---|
| `SQ3KTSFA` | Broad spacecraft-to-drone DRL PE framework; too general for B4 or sensor-risk comparison. |
| `GJ8BKKXJ` | UAV navigation/collision-aware memory; not pursuit-specific enough and unverified. |
| `V45HPIK4` | Potentially relevant sparse-reward UAV PE, but metadata/source is weak; keep as verify-before-use, not active evidence. |
| `KR7FL7ND` | Assignment/D3QN for generic uncrewed vehicles; too far from LiDAR observability-risk PPO. |
| `2R38VLEI` | Quadrotor racing curriculum can inspire training intuition, but not a pursuit/sensor-risk comparison. |
| `PXRRQJI9` | Limited-perception USV MAPPO; non-UAV and unverified. |
| `MEK3YYXY` | Cooperative PE in clutter; too generic and unverified for current claim. |
| `TVKHHUZI` | Collaborative AAV navigation, not pursuit-evasion or observability-loss risk. |
| `XVNLF5CG` | Fixed-wing pursuit in complex terrain may become relevant if verified, but current metadata is insufficient. |
| `LJRQYWKK` | High-speed obstacle avoidance by imitation learning; navigation/avoidance, not pursuit-risk modulation. |
| `28KC4DTQ` | General agile real-world quadrotor flight; useful background but not needed for the active DB. |
| `DDW2NH4Z` | Generic drone swarm MARL; too broad. |
| `3C6WJA23` | Model-based multi-agent PE; unverified and not directly tied to LiDAR observability-risk PPO. |
| `UZTLFF7D` | USV chasing; non-UAV, obstacle-assistance focus. |
| `6UXQHEA8` | Vision-based navigation memory; not pursuit-specific and unverified. |
| `7PN6PRPX` | Probabilistic PE theory; not connected to B4 implementation or required baselines yet. |
| `9QFGR88U` | Tracked-vehicle MATD3 PE; platform and method mismatch. |
| `LPMDLTIN` | Generic vision-based drone navigation; not pursuit-specific. |

## Design Lessons Kept

The accepted papers create concrete obligations:

- `zhang2026safetyshieldedflight` -> include a shield/filter baseline using the same risk model.
- `roncero2025agilecontrollers` -> discuss CTBR/body-rate-thrust policy control and avoid claiming agile PE controller novelty.
- `yan2024lsrctd3`, `giral2026intercept` -> compare against high-speed PE/interception framing, not just limited-FOV work.
- `luo2024improvedmadrl`, `tan2026scalablefixedwing`, `zhao2024autonomousuavpe` -> keep multi-agent/cooperative/scalable PE out of the main contribution and treat it as Related Work.
- `mavcapturingmav2024` -> separate sensor/capture-system work from LiDAR observability-loss risk modeling.
- `feroskhan2024multipursuitevasion` -> do not write real-world validation unless real logs are added.

## Corrected Conclusion

The Zotero list does not justify a larger paper scope. It sharpens the current scope. The active paper should cite/compare only the accepted set above and use the rejected set only as future query leads if needed.
