# Progress Log

## Session: 2026-05-09
- Initialized planning files under `.planning/2026-05-09-perceptual-risk-pursuit-eaai/`.
- Audited the active pursuit task and confirmed the current code is a privileged full-state PPO pursuit baseline using `apf_escape`, 32D geometric observation, root-link control, and `thrust_bodyrate_control`.
- Completed local deep-research outputs under `deep-research-output/perceptual-risk-pursuit-review/`.
- Removed rejected auxiliary-risk guidance that did not support the locked EAAI claim.
- Reframed Paper 1 as a privileged-to-depth-only sensor-proxy transition problem.
- Locked the main method as reduced-information pursuit plus learned short-horizon observability risk plus CTBR risk-constrained action correction.
- Locked default planning values: `H = 150`, `K_persist = 10`, `K = 3` depth frames, `z_depth = 64`.
- Locked architecture: offline pretrain, freeze depth encoder and risk head, stop PPO gradients through the representation and correction modules.
- Locked the B0-B5 baseline ladder and the required offline/online metrics.
- Minimized workspace planning hooks to `SessionStart` and `UserPromptSubmit`.

## Session: 2026-05-10
- Ran novelty assessment with a harsh-critic lens.
- Found that broad "risk-assisted partial-observation pursuit" is not strong enough because limited-FOV MARL, prediction-enhanced pursuit, risk-gated policies, and sensor-compatible pursuit already exist.
- Identified the stronger novelty cut: action-conditioned observability-risk prediction learned from privileged rollouts and used for CTBR action correction under depth-only reduced information.
- Rewrote `task_plan.md` and `findings.md` to remove weak intermediate guidance and make the locked EAAI claim the only active route.
- Updated deep-research artifacts with new near-neighbor comparison items: `[@peng2025limitedvisual]`, `[@li2026safeintent]`, and `[@li2025bearingonly]`.

## Current Status
- **Plan ID:** `2026-05-09-perceptual-risk-pursuit-eaai`
- **Phase:** Phase 6 - Implementation And Validation
- **Overall Status:** active

## Key Judgments
- The paper is not about generic pursuit RL, generic partial observability, target trajectory prediction, or full bilateral self-play.
- The only strong Paper-1 claim is `depth-only reduced information + action-conditioned observability-risk + CTBR correction`.
- B0 is an upper reference and label source, not the proposed method.
- B5 is the main method.
- B4 oracle risk is required as a diagnostic upper bound if implementation cost remains moderate.
- Any implementation or writing path that uses privileged state in the proposed method should be treated as a failed scope regression.

## Next 3 Tasks
1. Implement and unit-test detectable-region labels with `H = 150` and `K_persist = 10`.
2. Implement dataset manifest checks and offline predictor metrics before policy retraining.
3. Implement B1-B5 reduced-information and action-correction modes behind explicit config switches.

## Test Results
| Test | Expected | Actual | Status |
|------|----------|--------|--------|
| Planning catchup | No blocking unsynced context | No output, no blocker | pass |
| Guidance cleanup | Weak full-state-plus-scalar route removed | Active English guidance rewritten around locked EAAI claim | pass |
| Deep-research JSONL | `paper_db.jsonl` parses after adding near neighbors | 65 rows parsed successfully | pass |
| Near-neighbor citation keys | New keys appear in DB, BibTeX, survey, synthesis, and report | `peng2025limitedvisual`, `li2026safeintent`, `li2025bearingonly` present | pass |
| Weak-route scan | No active weak-route wording remains in planning/deep-research guidance | Only locked-claim wording remains | pass |

## Errors
| Error | Resolution |
|-------|------------|
| Earlier guidance retained weak intermediate research paths | Replaced with a single locked EAAI route and explicit failure conditions |
