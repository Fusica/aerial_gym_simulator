# Perceptual-Risk-Guided UAV Pursuit-Evasion: A Focused Literature Review for Sensing-Compatible RL

## 1. Introduction
Pursuit-evasion has become a central benchmark for autonomous aerial decision-making because it couples perception, control, cooperation, and adversarial adaptation in a single task. Recent reinforcement-learning work has shown that UAV pursuers can learn cooperative encirclement, target interception, and even agile 1v1 aerial maneuvering in simulated environments [@zhang2023gameofdrones; @roncero2025amspb]. At the same time, the literature has started to move beyond simplified point-mass or fully observable formulations. Partial observability, limited detectable regions, obstacle-rich environments, and real-world deployment constraints increasingly shape what counts as a meaningful result [@huh2026limited; @chen2025open; @zheng2025visioncapture].

For the current `aerial_gym_simulator` project, this trend creates a very specific research opportunity. The local repository already supports successful full-state pursuit with APF-generated evader motion and a CTBR/body-rate-compatible control path. The immediate question is therefore not whether pursuit-evasion RL works in general. The question is how to move from a successful privileged-state pursuit controller toward a sensing-compatible controller without taking on the full complexity of end-to-end vision or bilateral self-play from the outset.

This review surveys the literature most relevant to that transition problem. After novelty assessment against recent limited-visual-field and risk-gated pursuit work, the strongest first-paper direction is no longer a passive risk side channel. The defensible contribution is an action-conditioned observability-risk interface that compresses privileged pursuit information into a short-horizon risk model and uses it to correct CTBR actions under depth-only reduced information.

## 2. Background
Foundational pursuit-evasion work already highlighted that sensing capability is inseparable from capture performance. Vidal et al. framed probabilistic pursuit-evasion in terms of sensing, policy computation, and real-world UAV/UGV coordination [@vidal2002probabilistic]. Later work formalized one-sided or partially observable pursuit-evasion more explicitly, using approximate planning or scalable deep RL under limited information [@horak2016pointbased; @sun2022pomdp].

Modern UAV pursuit-evasion RL has evolved along several parallel lines. One line emphasizes cooperative MARL under local observations or explicit communication. Examples include graph-attention-based perception enhancement [@liqin2026perception], improved CommNet-style communication [@luo2024commnet], and clutter-aware cooperative pursuit using knowledge extracted from artificial potential fields [@sun2025kematd3]. A second line emphasizes prediction-enhanced pursuit. Game of Drones and OPEN both show that explicit predictive components can materially improve capture performance in complex environments [@zhang2023gameofdrones; @chen2025open]. A third line focuses on stronger adversaries and realistic low-level control, ranging from self-play PPO across domains [@xu2025general] to agile body-rate-and-thrust adversarial training in 1v1 quadrotor engagement [@roncero2025amspb].

At the same time, adjacent work outside strict pursuit-evasion has matured rapidly in onboard perception and aggressive flight. Vision-based cooperative MAV capture combines onboard detection, distributed estimation, MPC pursuit, and physical capture mechanisms [@zheng2025visioncapture]. Safety-shielded vision RL and curriculum-based aggressive flight in obstacle-rich environments show that sensor-driven end-to-end control can be made faster, safer, and more robust than earlier pipelines [@zhang2026shielded; @sun2026curriculum]. These works are not direct pursuit-evasion baselines, but they shape what a sensing-compatible aerial pursuit paper needs to take seriously.

## 3. Taxonomy of Approaches
The literature is best understood along four orthogonal axes.

First is the **observation model**. Some works remain fundamentally privileged-state or lightly abstracted, even when they add predictive modules [@zhang2023gameofdrones; @roncero2025amspb; @xu2025general]. Others explicitly constrain sensing through local observations, limited visual fields, limited detectable regions, bearing-only inputs, or sequence encoding [@peng2025limitedvisual; @huh2026limited; @li2025bearingonly; @li2024limitedusv; @sun2022pomdp]. A third set moves toward genuine onboard sensing, typically with cameras and distributed estimation [@zheng2025visioncapture].

Second is the **decision-support mechanism**. Some papers rely on reward shaping and geometry-aware incentives, for example via gaze, capturability, or LOS-rate correction [@huh2026limited; @yan2024lsrc]. Others add explicit predictive structure, such as target prediction networks or prediction-based reward functions [@zhang2023gameofdrones; @chen2025open; @xu2025general]. Recent work also uses intent/subgoal prediction, belief-risk gates, and safety projection under intermittent observability [@li2026safeintent]. The gap identified in this review is therefore stricter: a learned, calibrated, action-conditioned observability-risk model that directly corrects CTBR actions rather than only gating a high-level policy.

Third is the **interaction structure**. Cooperative MARL dominates the multi-pursuer setting [@liqin2026perception; @sun2025kematd3]. Hierarchical methods focus on scale, target assignment, and decomposed control [@tan2025scalable]. Bilateral self-play and co-training emphasize robustness against adaptive adversaries [@roncero2025amspb; @yao2026warning]. For the current repository, this third axis matters because it determines what should be deferred. Bilateral co-training is scientifically interesting, but it is not the lowest-risk route to a first paper.

Fourth is **action realism**. Many pursuit papers still operate at higher-level kinematic or velocity-command layers. More recent works demonstrate the importance of body-rate/thrust interfaces, calibrated dynamics, and deployment-oriented controllers [@chen2025open; @roncero2025amspb; @zheng2025visioncapture]. This is an area where the current repo is comparatively strong: it already lives close to a realistic low-level control interface, which makes representation and sensing questions more meaningful.

## 4. Detailed Analysis
Among directly relevant papers, `[@peng2025limitedvisual]` and `[@huh2026limited]` are the strongest limited-FOV / limited-detectable-region pursuit baselines. They show that visual-field constraints, target loss, reacquisition, reward shaping, action-space design, and graph-attention MARL are already serious neighboring ideas. However, their mechanisms do not learn a separately calibrated action-conditioned observability-risk model for CTBR correction.

`[@li2026safeintent]` is the closest risk-aware decision-support neighbor. It couples opponent intent/subgoal inference, trajectory prediction, belief-risk gating, hierarchical MARL, and a safety projection layer. This makes a generic "risk-gated pursuit" claim too weak for the current project. The required differentiation is to predict observability-loss risk itself, learn it from privileged rollouts for a depth-only sensor-proxy setting, and use it to correct CTBR actions.

`[@zhang2023gameofdrones]` and `[@chen2025open]` occupy the closest neighboring prediction line. Both suggest that prediction-before-decision is effective in pursuit-evasion. OPEN is especially relevant because it pairs prediction enhancement with adaptive environment generation and a real deployment path using body-rate and collective thrust. Yet these methods are still best understood as motion- or planning-prediction systems. They do not isolate observability-risk as a separately trained, separately evaluated signal.

`[@liqin2026perception]` shows that local-observation cooperative pursuit can benefit from richer inter-agent feature extraction, here through graph attention. This is strong evidence that there is publishable value in representation learning under partial observability. But the representation in that paper is communication-enhanced state fusion rather than a tactically meaningful risk forecast.

`[@roncero2025amspb]` is the strongest 1v1 agile quadrotor reference for later stages. It establishes that body-rate-and-thrust policies outperform velocity-level baselines and that population-based asynchronous opponent sampling mitigates forgetting. For a future bilateral-pursuit paper, it is a major anchor. For a first observability-risk paper, however, it is too far downstream: the paper still assumes privileged opponent position observations and focuses on adversarial co-training stability.

The adjacent vision and deployment literature sharpens the boundary of the first-paper scope. `[@zheng2025visioncapture]` shows what a serious onboard sensing and real-world capture system looks like: visual detection, cooperative estimation, formation control, and an explicit capture condition derived from physical net behavior. `[@zhang2026shielded]` and `[@sun2026curriculum]` demonstrate two complementary lessons from vision-based flight: end-to-end sensing stacks benefit from strong inductive structure, and carefully designed training curricula are often necessary when obstacles or real-world complexity enter the loop. These insights matter because they suggest that a later transition from state-proxy risk to camera-latent risk is plausible, but should not be forced into the first paper.

## 5. Applications
For the current repo, the most useful application of this literature is not reproducing the entire state of the art. It is selecting the right abstraction layer for the first contribution. The strongest immediate application is:

1. Treat the existing full-state PPO pursuer plus APF evader as a **supervision-generating baseline**.
2. Define a **virtual detectable region** consistent with future onboard sensing assumptions.
3. Label each sampled state by whether the target will leave that region within horizon `H`, or remain outside long enough to count as track loss.
4. Train a **compact action-conditioned risk predictor** from depth-only reduced information.
5. Compare online policy variants:
   - reduced-information no-risk PPO
   - depth-honest heuristic risk
   - learned state risk
   - oracle risk
   - learned action-conditioned risk correction

This application strategy is well grounded in the existing literature. It inherits observability realism from limited-detectable-region work [@huh2026limited], modular predictive structure from prediction-enhanced pursuit [@zhang2023gameofdrones; @chen2025open], and deployment-aware caution from body-rate and sensing-oriented quadrotor work [@roncero2025amspb; @zheng2025visioncapture].

## 6. Open Problems and Future Directions
The main unresolved issue is the **transition problem** itself. Existing papers usually choose one of two extremes: stay privileged, or move directly to limited local observations or raw perception. The surveyed literature provides very little support for an intermediate, auditable learned signal that preserves tactical information while reducing privileged dependence.

A second open problem is **what exactly to predict**. Target-motion prediction is common. Reward shaping around FOV or capture conditions is also common. What remains underexplored is predicting *observability failure* or *short-horizon tactical risk* as a task-level variable. This is where the current repository can plausibly make a differentiated contribution.

A third open problem is **evaluation discipline**. Many papers conflate changes in observation model, environment difficulty, opponent intelligence, and low-level control realism. A strong first paper in this repo should explicitly avoid that. It should keep the evader side simple, freeze the control interface, and isolate the causal effect of the risk signal itself.

Longer term, the literature suggests two natural expansion paths. One is to replace state-proxy risk inputs with real camera latents while keeping the same risk target. The other is to introduce stronger evaders and eventually bilateral asynchronous co-training, borrowing ideas from AMSPB-like training or zero-sum compressed-state systems [@roncero2025amspb; @yao2026warning].

## 7. Conclusion
The literature on UAV pursuit-evasion RL is now broad enough that a generic pursuit paper is hard to position. Cooperative MARL, prediction-enhanced pursuit, limited-detectable-region modeling, self-play, and real-world sensing systems all already have strong representatives [@liqin2026perception; @zhang2023gameofdrones; @huh2026limited; @roncero2025amspb; @zheng2025visioncapture].

The current `aerial_gym_simulator` project should therefore avoid overly broad claims. Its strongest first-paper opportunity is narrower and more concrete: use the existing successful full-state pursuit stack to supervise a learned short-horizon, action-conditioned observability-risk predictor, then use that predictor to correct CTBR actions under depth-only reduced information. This positions the work not as another pursuit controller or a generic risk-gated MARL system, but as a principled bridge from privileged-state pursuit toward sensing-compatible aerial pursuit control.
