# Gap Analysis: Perceptual-Risk Pursuit Review
Date: 2026-05-09

## Open Problems
1. **Bridge from privileged-state pursuit to sensing-compatible CTBR action correction**
   Existing works either stay privileged, move directly to local observation / vision, or use risk as a high-level gate. Few define an intermediate, auditable learned observability-risk signal that is action-conditioned and directly corrects low-level CTBR pursuit commands.

2. **What to predict under sensing constraints**
   Prediction modules typically forecast target motion, infer intent/subgoals, or provide dense reward hints. The stricter gap is predicting *observability-loss risk for candidate actions* as a separately calibrated learning target.

3. **Fair evaluation under observability stress**
   Many papers compare different observation models and different learning systems at once, making it difficult to isolate whether gains come from better representation, richer information, or easier tasks.

4. **Interaction between sensing realism and low-level agility**
   High-fidelity body-rate / thrust control is rising in importance, but few pursuit papers combine strong low-level control realism with explicit sensing degradation and auxiliary learned risk signals.

## Contradictions / Tensions
1. `[@huh2026limited]` and `[@peng2025limitedvisual]` suggest much of the sensing-constrained benefit can be unlocked by limited-FOV formulation, action-space design, reward shaping, and graph-attention MARL, while prediction-augmented works like `[@zhang2023gameofdrones]` and `[@chen2025open]` suggest explicit predictive structure is the key.
2. `[@li2026safeintent]` shows that risk-gated hierarchical decision making under intermittent observability is already a close neighboring idea, so the current project must emphasize observability-loss labels, depth-only reduced information, and CTBR action-conditioned correction rather than generic risk gating.
3. `[@roncero2025amspb]` and `[@yao2026warning]` argue that stronger bilateral training is central to robustness, while the current repo’s first-paper path benefits from freezing the evader side to keep causal attribution clear.
4. Vision-based system papers like `[@zheng2025visioncapture]` demonstrate practical sensing stacks, but they are not RL-centric. RL pursuit papers often remain more privileged than those systems.

## Missing Evaluations
- Few papers compare:
  - no auxiliary signal vs heuristic observability signal vs learned state risk vs action-conditioned observability-risk correction
  - full-state baseline vs reduced-state baseline vs reduced-state + learned risk
  - offline predictor quality metrics together with online pursuit outcomes
- There is little standardized evaluation for:
  - visibility-loss events
  - track-recovery latency
  - robustness under held-out evader maneuver families when sensing is constrained

## Under-Explored Directions
1. **Action-conditioned short-horizon observability-risk prediction**
   A label/model that predicts whether a candidate CTBR action will increase the chance of target loss in horizon `H` is the strongest underexplored bridge objective.

2. **State-proxy to sensor-proxy distillation**
   A staged path where the first paper learns a risk module from geometric/state proxy inputs, and later work replaces or augments those inputs with actual sensor latents, is structurally compatible with the current repo and underrepresented in the surveyed literature.

3. **Auditable auxiliary modules**
   Most auxiliary components in pursuit RL are folded into end-to-end architectures. A predictor that is separately trained, separately evaluated, and then integrated into policy learning has stronger scientific interpretability.

4. **CTBR/body-rate pursuit under sensing degradation**
   High-fidelity pursuit stacks are becoming more common, but there is still room for papers that explicitly connect realistic low-level control with observability-aware decision support.

## Concrete Research Questions
1. **RQ1**: Can a learned action-conditioned observability-risk predictor outperform depth-honest heuristic risk in forecasting imminent target loss?
2. **RQ2**: Does learned state risk recover performance under reduced/depth-only information compared with no-risk and heuristic-risk baselines?
3. **RQ3**: Does CTBR action-conditioned risk correction improve target retention and recovery beyond risk-as-observation only?
4. **RQ4**: Do gains persist across held-out target-motion regimes such as fixed-rule, APF variants, and learned or stronger evaders?

## First-Paper Recommendation
- **Best paper-1 direction**: action-conditioned observability-risk-corrected CTBR pursuit under depth-only reduced information.
- **Not paper-1 direction**: full bilateral self-play, full raw-vision end-to-end pursuit, or large-scale multi-vs-multi hierarchy.

## What This Means For The Current Repo
- Keep the existing PPO pursuer + APF evader stack as the stable baseline.
- Use current full-state trajectories to define and collect observability-risk labels.
- Train a small, auditable predictor first, then extend it to score candidate CTBR actions.
- Compare B0-B5 so the paper distinguishes state risk, oracle risk, heuristic risk, and action-conditioned correction.
