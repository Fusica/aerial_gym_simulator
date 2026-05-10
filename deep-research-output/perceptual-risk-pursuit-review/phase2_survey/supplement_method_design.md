# Supplement: Method-Design Literature for Three Technical Directions
Date: 2026-05-10
Based on: Phase 1-2 problem-positioning results (35 papers)
Purpose: Fill the method-design gap for the three technical contributions

## Context

The existing Phase 1-2 survey (35 papers) adequately covers **problem positioning**: pursuit-evasion RL,
partial observability, prediction-enhanced pursuit, and sensing constraints. What is missing is
**method-design-level** literature for the three specific technical mechanisms proposed:

1. **Direction 1**: Depth-only sensor-proxy reduced-information pursuit formulation
2. **Direction 2**: Learned short-horizon observability risk prediction bridge
3. **Direction 3**: Risk-constrained CTBR action optimization / correction

This supplement adds **27 papers** targeting these three directions. Together with the existing
35 papers, the combined database supports both Related Work positioning and method-design justification.

---

## Direction 1: Depth-Only Sensor-Proxy Reduced-Information Pursuit (9 papers)

### D1.1 Depth-to-Control End-to-End Learning for Agile Quadrotor Flight

**[@loquercio2021learning]** Loquercio, A., Kaufmann, E., Ranftl, R., Dosovitskiy, A., Koltun, V., & Scaramuzza, D. (2021).
"Learning High-Speed Flight in the Wild." *Science Robotics*, 6(59), eabg5810.
- **Core idea**: End-to-end RL policy mapping depth images to body-rate + thrust commands for agile
  quadrotor flight through natural forest at up to 10 m/s. Uses 4-frame depth stack, trained with
  PPO purely in simulation with domain randomization, zero-shot sim-to-real.
- **Relevance (5/5)**: Directly demonstrates depth-only → CTBR control mapping. The depth frame
  stacking (K=4) standard validates the project's K=3 design choice. Strongest single prior for
  depth-only agile flight feasibility.
- **Key distinction from our work**: End-to-end visual policy without privileged-state teacher,
  no risk prediction, no observability-aware action correction, no pursuit setting.

**[@kaufmann2020deep]** Kaufmann, E., Loquercio, A., Ranftl, R., Dosovitskiy, A., Koltun, V., & Scaramuzza, D. (2020).
"Deep Drone Racing: From Simulation to Reality with Domain Randomization." *IEEE Transactions on Robotics*,
36(3), 1-12.
- **Core idea**: CNN policy from stacked 4-frame depth images for autonomous drone racing.
  Imitation learning from expert MPC, then RL fine-tuning. Domain randomization on depth rendering.
- **Relevance (5/5)**: Establishes depth frame stacking as standard practice for aggressive
  quadrotor control. The pretrain (IL) + fine-tune (RL) pipeline parallels our
  offline pretrain + freeze + online PPO paradigm.
- **Key distinction**: Imitation-from-MPC rather than privileged-to-risk; racing not pursuit; no
  observability-risk intermediate representation.

**[@kaufmann2023champion]** Kaufmann, E., Bauersfeld, L., Loquercio, A., Muller, M., Koltun, V., & Scaramuzza, D. (2023).
"Champion-level Drone Racing using Deep Reinforcement Learning." *Nature*, 620, 982-987.
- **Core idea**: Vision-based end-to-end RL defeating human world champions in physical drone
  racing. Combines learned visual encoder, memory module, and control module. Trained in simulation.
- **Relevance (4/5)**: Strongest full-stack demonstration of vision-based quadrotor autonomy.
  Temporal visual input. Architecture decomposition (encoder → memory → controller) is relevant.
- **Key distinction**: RGB-based system, no privileged-teacher distillation, racing not pursuit,
  no explicit risk prediction module.

### D1.2 Depth Encoding with Auxiliary Tasks (Same Simulator Stack)

**[@kulkarni2024collision]** Kulkarni, M., & Alexis, K. (2024).
"Reinforcement Learning for Collision-free Flight Exploiting Deep Collision Encoding."
*IEEE ICRA 2024*.
- **Core idea**: Deep Collision Encoder (DCE) compressing depth images into collision-aware latent
  via self-supervised auxiliary collision prediction. Encoder pretrained + frozen, latent fed to RL
  policy. Built on the **Aerial Gym Simulator** — same codebase as this project.
- **Relevance (5/5)**: Same simulator, depth-only input, auxiliary supervised learning from depth,
  pretrain + freeze paradigm. The DCE architecture is a direct template for our depth encoder
  design. Shows that depth → auxiliary-task → frozen-latent → RL is viable in this simulator.
- **Key distinction**: Auxiliary task is collision prediction (binary), not range/bearing regression.
  Navigation task, not pursuit. No observability-risk concept.

### D1.3 Privileged-to-Sensor Distillation for Robot Control

**[@lee2020learning]** Lee, J., Hwangbo, J., Wellhausen, L., Koltun, V., & Hutter, M. (2020).
"Learning Quadrupedal Locomotion over Challenging Terrain." *Science Robotics*, 5(47), eabc5986.
- **Core idea**: Teacher-student framework: teacher trained with privileged terrain info (elevation,
  friction, etc.), student learns from depth + proprioception. Zero-shot sim-to-real.
- **Relevance (5/5)**: **Seminal privileged-to-depth distillation paper.** The teacher-student
  paradigm (teacher sees privileged info, student sees depth + proprioception) is the closest
  architectural analog to our approach: privileged full-state teacher → offline labels →
  depth-only student risk predictor → online RL.
- **Key distinction**: Locomotion not flight; student imitates teacher actions, not risk labels;
  no action correction layer.

**[@chen2020learning]** Chen, D., Zhou, B., Koltun, V., & Kragic, D. (2020).
"Learning by Cheating." *CoRL 2020*.
- **Core idea**: Privileged agent (full state + map) trained first; vision-based sensor agent
  distilled via online DAgger-style imitation. The "cheating" formulation explicitly frames
  the problem as privileged → sensor transition.
- **Relevance (5/5)**: **Key privileged-to-vision distillation paper.** The two-stage approach
  (train privileged expert → distill to sensor student) directly parallels our two-stage
  design: privileged trajectory labels → depth-only risk predictor → frozen module → online PPO.
- **Key distinction**: Autonomous driving, not pursuit; student imitates actions, not risk;
  online DAgger distillation, not offline pretrain + freeze.

**[@pinto2018asymmetric]** Pinto, L., Andrychowicz, M., Welinder, P., Zaremba, W., & Abbeel, P. (2018).
"Asymmetric Actor Critic for Image-Based Robot Learning." *RSS 2018*.
- **Core idea**: Asymmetric actor-critic: actor sees only images (no privileged state), critic
  sees full privileged state during training. Critic provides better gradient signals to actor
  without privileged info at deployment time.
- **Relevance (4/5)**: Establishes asymmetry as a principled way to use privileged information
  during training while deploying with sensor-only input. Relevant to our design: privileged
  supervision only during offline pretraining, sensor-only at inference.
- **Key distinction**: Critic sees privileged state throughout online training; we freeze the
  encoder before online training entirely. No explicit risk prediction.

**[@kumar2021rma]** Kumar, A., Fu, Z., Pathak, D., & Malik, J. (2021).
"RMA: Rapid Motor Adaptation for Legged Robots." *RSS 2021*.
- **Core idea**: Two-stage: (1) teacher trained with privileged env info, (2) student adaptation
  module infers privileged latent from proprioception history. No depth/vision; uses proprioception
  and action history for online adaptation.
- **Relevance (4/5)**: Alternative distillation architecture — student learns to *infer* privileged
  latent rather than directly imitating teacher actions. This directly parallels our auxiliary
  supervision design: the depth encoder learns to predict privileged quantities (range, bearing)
  as auxiliary tasks, forcing the latent to encode target-relevant features.
- **Key distinction**: Proprioception-based adaptation, not depth-based; locomotion, not pursuit.

### D1.4 Depth-Based Visual Navigation Foundations

**[@loquercio2018dronet]** Loquercio, A., Maqueda, A. I., del-Blanco, C. R., & Scaramuzza, D. (2018).
"DroNet: Learning to Fly by Driving." *IEEE RA-L*, 3(4), 3223-3230.
- **Core idea**: Imitation learning from car driving data for drone collision avoidance. Uses
  single camera (RGB), not depth-only. Demonstrates cross-domain transfer (cars → drones).
- **Relevance (3/5)**: Foundational for learning drone control from visual input. The methodology
  of training control from non-drone data is relevant conceptually. Lower priority for depth-only
  focus.
- **Key distinction**: RGB-based, not depth; collision avoidance, not pursuit; no privileged teacher.

---

## Direction 2: Learned Short-Horizon Observability Risk Prediction Bridge (7 papers)

### D2.1 Auxiliary Task Learning and Representation Learning in RL

**[@jaderberg2017reinforcement]** Jaderberg, M., Mnih, V., Czarnecki, W. M., Schaul, T., Leibo, J. Z.,
Silver, D., & Kavukcuoglu, K. (2017).
"Reinforcement Learning with Unsupervised Auxiliary Tasks." *ICLR 2017*.
- **Core idea**: UNREAL agent learns auxiliary control and reward prediction tasks alongside the
  main RL objective. Auxiliary tasks shape the shared representation, improving data efficiency
  and final performance. Pixel control and value replay as auxiliary losses.
- **Relevance (5/5)**: **Foundational paper for auxiliary tasks in RL.** Our approach directly
  follows this paradigm: the depth encoder learns auxiliary tasks (range estimation, bearing
  estimation) alongside the primary risk prediction task (p_lost). Multi-task training of
  a shared encoder with task-specific heads is exactly the UNREAL pattern.
- **Key distinction**: Auxiliary tasks are self-supervised (pixel control, reward prediction),
  not privileged-supervised (range/bearing from ground truth). No offline pretrain + freeze.

**[@srinivas2020curl]** Srinivas, A., Laskin, M., & Abbeel, P. (2020).
"CURL: Contrastive Unsupervised Representations for Reinforcement Learning." *ICML 2020*.
- **Core idea**: Contrastive learning (SimCLR-style) for visual RL representations. Learns image
  representations via instance discrimination before/alongside RL training.
- **Relevance (3/5)**: Represents the self-supervised representation learning approach (alternative
  to our supervised auxiliary task approach). Useful as a methodological contrast.
- **Key distinction**: Self-supervised (contrastive), not privileged-supervised; no risk prediction.

### D2.2 Risk and Safety Prediction in RL

**[@thananjeyan2021recovery]** Thananjeyan, B., Balakrishna, A., Rosenthal, S., Grice, R., Hwang, M.,
Gonzalez, J., & Goldberg, K. (2021).
"Recovery RL: Safe Reinforcement Learning with Learned Recovery Zones."
*CoRL 2021*.
- **Core idea**: Learns to predict "unsafe" states (recovery zones) from offline data, then uses
  a separate recovery policy when the agent enters predicted unsafe regions. Binary safety
  classifier trained offline, integrated into online RL via policy switching.
- **Relevance (5/5)**: **Strongest architectural analog for our risk prediction approach.**
  Binary classifier predicting future undesirable events, trained offline from privileged data,
  integrated into online RL control. The "predict-then-intervene" structure is directly parallel.
- **Key distinction**: Recovery by policy switching, not action correction; safety-focused
  (collision, constraint violation), not observability-focused (target loss). No CTBR-specific
  design.

**[@kahn2017uncertainty]** Kahn, G., Villaflor, A., Pong, V., Abbeel, P., & Levine, S. (2017).
"Uncertainty-Aware Reinforcement Learning for Collision Avoidance." *arXiv:1702.01182*.
- **Core idea**: Uses MC dropout to estimate predictive uncertainty of a dynamics model, then
  uses uncertainty as a collision risk signal for action selection. Uncertainty → risk proxy → control.
- **Relevance (4/5)**: Demonstrates the uncertainty → risk → action modification pipeline.
  The concept of converting model uncertainty into a risk signal for control is directly analogous
  to our p_lost → action correction pipeline.
- **Key distinction**: MC dropout uncertainty, not learned event probability. No offline
  privileged supervision. Collision prediction, not observability-loss prediction.

**[@clements2020estimating]** Clements, W. R., Van Delft, B., Robaglia, B.-M., Slaoui, R. B., & Toth, S. (2020).
"Estimating Risk and Uncertainty in Deep Reinforcement Learning." *ICML Workshop on Uncertainty
and Robustness in Deep Learning*, 2020. arXiv:1905.09638.
- **Core idea**: Framework for disentangling epistemic and aleatoric uncertainty in learned
  Q-values. Derives unbiased uncertainty estimators; proposes uncertainty-aware DQN.
- **Relevance (3/5)**: Relevant for the uncertainty estimation aspect of risk prediction.
  Distinguishing reducible (epistemic) vs irreducible (aleatoric) uncertainty in risk estimates
  could strengthen our predictor evaluation.
- **Key distinction**: Q-value uncertainty, not event probability prediction. DQN, not PPO.

### D2.3 Offline Pretraining for Online RL

**[@nair2020accelerating]** Nair, A., Dalal, M., Gupta, A., & Levine, S. (2020).
"Accelerating Online Reinforcement Learning with Offline Datasets." *arXiv:2006.09359*.
- **Core idea**: Uses offline data to pretrain policies via advantage-weighted regression (AWAR),
  then fine-tunes online. Shows dramatic sample efficiency gains from offline pretraining.
- **Relevance (4/5)**: Validates the offline pretrain → online integrate paradigm. The concept of
  using offline data to learn useful priors before online RL directly supports our design:
  offline pretrain risk predictor → freeze → online PPO integration.
- **Key distinction**: Pretrains the policy, not an auxiliary risk module. Online fine-tuning,
  not freeze + stop-gradient.

**[@srinivasan2020learning]** Srinivasan, K., Eysenbach, B., Ha, S., Tan, J., & Finn, C. (2020).
"Learning to be Safe: Deep RL with a Safety Critic." *arXiv:2010.14603*.
- **Core idea**: Learns a binary safety critic predicting probability of entering dangerous states,
  trained from offline trajectories collected in safe source environments. The safety critic
  constrains exploration during online RL on new tasks, enabling safe transfer. Tested on
  navigation, legged locomotion, and manipulation.
- **Relevance (5/5)**: **Direct architectural analog for our risk predictor.** Binary classifier
  (safe/unsafe) trained offline from trajectory data, frozen, used to constrain online RL action
  selection. The offline-pretrained safety-critic-to-online-constraint pipeline is exactly our
  p_lost → action correction pathway.
- **Key distinction**: General safety violations (collisions, falls), not observability-specific
  target loss events. No depth-only sensor input. No CTBR-specific action space.

**[@li2026failure]** Li, H., Lei, K., Zang, S., Hu, K., Liang, Y., An, B., Li, X., & Xu, H. (2026).
"Failure-Aware RL: Reliable Offline-to-Online Reinforcement Learning with Self-Recovery
for Real-World Manipulation." *arXiv:2601.07821*.
- **Core idea**: World-model-based safety critic trained offline to predict "Intervention-requiring
  Failures" (IR Failures). Recovery policy trained alongside. During online RL, the safety critic
  detects impending failures and triggers recovery. 73.1% reduction in IR failures on real robots.
- **Relevance (5/5)**: **Arguably the closest existing pipeline to our proposed approach.**
  Offline pretrain safety critic → freeze → online RL integration → failure detection →
  control intervention. The two-stage offline-to-online transfer with a learned failure
  predictor is the same architectural paradigm.
- **Key distinction**: World-model-based (not direct state → risk mapping); manipulation not pursuit;
  recovery by policy switching (not action correction); no observability-specific risk target.

**[@buhrer2023multiplicative]** Buhrer, N., Zhang, Z., Liniger, A., Yu, F., & Van Gool, L. (2023).
"A Multiplicative Value Function for Safe and Efficient Reinforcement Learning." *arXiv:2303.04118*.
- **Core idea**: Factorizes value function into safety critic (prediction of constraint violation
  probability) and reward critic (constraint-free returns). Multiplicative decomposition allows
  the policy to trade off risk against reward. Zero-shot sim-to-real on differential-drive robot.
- **Relevance (4/5)**: The risk-reward decomposition is a useful theoretical framework. Our
  approach implicitly does this via a post-hoc correction layer rather than multiplicative
  value factorization. Useful as an alternative design point to contrast against.
- **Key distinction**: Multiplicative value function during training, not post-hoc action
  correction. No observability-specific constraint. No depth-only input.

### D2.4 Visibility-Aware and Occlusion-Aware Pursuit-Evasion

**[@bajcsy2023learning]** Bajcsy, A., Loquercio, A., Kumar, A., & Malik, J. (2023).
"Learning Vision-based Pursuit-Evasion Robot Policies." *arXiv:2308.16185*.
- **Core idea**: Transforms pursuit-evasion into a teacher-student problem: fully-observable
  privileged teacher generates supervision for partially-observable vision-based student.
  Student learns to gather information when uncertain and anticipate to intercept. Deployed
  on physical quadruped with RGB-D camera.
- **Relevance (5/5)**: **The closest prior work combining privileged-to-sensor distillation
  with pursuit-evasion.** The teacher (privileged evader state) → student (RGB-D observations)
  structure is directly analogous to our privileged-full-state teacher → depth-only student →
  risk prediction pipeline. Shows that pursuit-specific distillation is viable.
- **Key distinction**: Student imitates teacher actions directly (no intermediate risk
  representation). Quadruped ground robot, not aerial CTBR pursuit. RGB-D, not depth-only.
  No action correction layer.

**[@zhou2024control]** Zhou, M., Shaikh, M., Chaubey, V., Haggerty, P., Koga, S., Panagou, D.,
& Atanasov, N. (2024).
"Control Strategies for Pursuit-Evasion Under Occlusion Using Visibility and Safety
Barrier Functions." *arXiv:2411.01321*.
- **Core idea**: Visibility-based CBFs to keep evader within pursuer's FoV despite occlusions.
  Signed distance functions of FoV as CBF constraints. Sampling-based kinodynamic planner for
  non-myopic pursuit under occlusion.
- **Relevance (4/5)**: Directly addresses visibility maintenance in pursuit-evasion. The
  visibility CBF formulation could provide mathematically grounded privileged labels for
  training a learned risk predictor. Shows that visibility constraints matter in PE.
- **Key distinction**: CBF-based (not learning-based). Uses SDF-based geometry, not learned
  risk prediction. No RL integration. Occlusion focus vs. general observability focus.

**[@schwarzer2021pretraining]** Schwarzer, M., Rajkumar, N., Noukhovitch, M., Anand, A., Charlin, L.,
Hjelm, R. D., Courville, A. (2021).
"Pretraining Representations for Data-Efficient Reinforcement Learning." *NeurIPS 2021*.
- **Core idea**: Self-supervised pretraining of visual representations for RL using temporal
  contrastive learning (SPR). The frozen or fine-tuned pretrained encoder dramatically improves
  sample efficiency in online RL.
- **Relevance (4/5)**: Direct evidence for the pretrain + freeze approach. Shows that frozen
  pretrained representations can be highly effective for online RL. Supports our design of
  freezing the depth encoder and risk head during PPO training.
- **Key distinction**: Self-supervised pretraining (temporal contrastive), not privileged-supervised.
  No risk prediction as a specific target.

---

## Direction 3: Risk-Constrained CTBR Action Optimization (6 papers)

### D3.1 Safety Layers and Action Projection for RL

**[@dalal2018safe]** Dalal, G., Dvijotham, K., Vecerik, M., Hester, T., Paduraru, C., & Tassa, Y. (2018).
"Safe Exploration in Continuous Action Spaces." *arXiv:1801.08757*.
- **Core idea**: "Safety layer" that analytically solves a QP to minimally perturb the RL policy's
  action to satisfy safety constraints. The safety layer is a differentiable projection:
  `argmin_u ||u - u_rl||^2` subject to `C(u) <= 0`. Trained safety critic predicts constraint values.
- **Relevance (5/5)**: **Directly inspires our action correction formulation.** The QP-based
  minimal-perturbation action projection with learned safety constraints is exactly the
  mathematical structure we adopt: `u* = argmin ||u - u_rl||^2 + penalty_terms` with hard
  and soft constraints.
- **Key distinction**: Learns constraint model online alongside policy; we use offline-trained
  risk predictor. Safety constraints are environment-level (collision, joint limits), not
  observability-aware (FOV maintenance, target-keeping).

**[@alshiekh2018safe]** Alshiekh, M., Bloem, R., Ehlers, R., Könighofer, B., Niekum, S., & Topcu, U. (2018).
"Safe Reinforcement Learning via Shielding." *AAAI 2018*.
- **Core idea**: "Shield" that monitors RL agent's action and overrides it when it would violate
  a temporal logic safety specification. The shield is pre-computed from safety specs.
- **Relevance (4/5)**: Establishes the shielding concept in RL. The pre-computed safety monitor
  → action override structure parallels our offline-trained risk predictor → action correction
  structure. Least-restrictive shield concept is relevant.
- **Key distinction**: Formal temporal logic specifications, not learned risk predictors.
  Discrete action spaces, not continuous CTBR. Hard override, not soft penalty projection.

### D3.2 Control Barrier Functions with Learning

**[@taylor2020learning]** Taylor, A. J., Singletary, A., Yue, Y., & Ames, A. D. (2020).
"Learning for Safety-Critical Control with Control Barrier Functions." *L4DC 2020*.
- **Core idea**: Learning-based framework for control barrier functions (CBFs): learns a model
  of the system and uses it to construct CBF safety filters. Combines model learning with
  barrier-certified safety.
- **Relevance (4/5)**: **Key paper bridging learned models and CBF-based safety filters.**
  The concept of learning a safety-relevant model offline and using it for online action
  filtering is the same pattern as our risk predictor → action correction pipeline.
- **Key distinction**: Learns system dynamics model for CBF construction, not risk event
  prediction. Full CBF framework (forward invariance), not QP penalty projection.
  General safety, not observability-aware control.

**[@cheng2019end]** Cheng, R., Orosz, G., Murray, R. M., & Burdick, J. W. (2019).
"End-to-End Safe Reinforcement Learning through Barrier Functions for Safety-Critical
Continuous Control Tasks." *AAAI 2019*.
- **Core idea**: Integrates CBFs directly into RL: the CBF-based safety controller intervenes
  during exploration and policy updates. Combines model-free RL with model-based CBFs for
  guaranteed safety during training.
- **Relevance (4/5)**: Direct integration of safety constraints with RL training. The barrier
  function acts as an action filter during online RL, paralleling our risk-constrained action
  optimization during deployment.
- **Key distinction**: Uses known dynamics + hand-designed CBFs, not learned risk predictors.
  Safety = collision avoidance, not observability maintenance. RL training is safety-constrained,
  not training-then-correction.

### D3.3 Safety-Shielded Vision-Based Drone Flight

**[@zhang2026shielded]** Zhang, Y., et al. (2026).
"High-Speed Vision-Based Flight in Clutter with Safety-Shielded Reinforcement Learning."
*arXiv preprint 2026*.
- **Core idea**: End-to-end vision policy with a deployment-time safety filter ("shield").
  The shield monitors the vision policy's output and modifies actions to prevent collisions.
  High-speed quadrotor flight in cluttered environments.
- **Relevance (4/5)**: Closest existing work for safety-shielded vision-based quadrotor control.
  The shield architecture (policy → shield → modified action) is the same topology as our
  PPO → risk-constrained correction → u* pipeline. Already in DB from Phase 1-2.
- **Key distinction**: Collision safety, not observability risk. Shield based on geometric
  checks, not learned risk prediction. Not pursuit context.

### D3.4 Constrained and Safe RL Foundations

**[@achiam2017constrained]** Achiam, J., Held, D., Tamar, A., & Abbeel, P. (2017).
"Constrained Policy Optimization." *ICML 2017*.
- **Core idea**: CPO — constrained policy optimization with trust-region guarantees. First
  principled algorithm for policy gradient under safety constraints (cost limits).
  `max J(π) s.t. D(π) <= d`.
- **Relevance (3/5)**: Foundational for constrained RL. Our action correction layer can be
  viewed as a post-hoc constraint enforcement mechanism — CPO integrates constraints into
  the policy gradient itself. Useful as a methodological contrast.
- **Key distinction**: Constraint enforcement during gradient-based policy optimization,
  not post-hoc action projection. General MDP constraints, not observability-specific.

**[@ray2019benchmarking]** Ray, A., Achiam, J., & Amodei, D. (2019).
"Benchmarking Safe Exploration in Deep Reinforcement Learning." *arXiv:1910.01708*.
- **Core idea**: Standardized benchmark suite (Safety Gym) for safe RL. Formalizes
  cost-constrained MDPs with collision avoidance and goal-reaching. Introduces constrained
  RL baselines including CPO, PPO-Lagrangian, TRPO-Lagrangian.
- **Relevance (3/5)**: Provides the standard safe RL evaluation framework. The cost-constrained
  MDP formalism and Lagrangian baselines are relevant for positioning our action correction
  approach within the broader safe RL landscape.
- **Key distinction**: General safe RL benchmark, not UAV-specific. Collision costs, not
  observability costs. No CTBR-specific action structure.

---

## Summary and Integration

### Coverage by Technical Direction

| Direction | Papers | Key Anchors |
|-----------|--------|-------------|
| D1: Depth-only sensor-proxy | 9 | Loquercio+ (2021), Kaufmann+ (2020), Kulkarni+ (2024), Lee+ (2020), Chen+ (2020) |
| D2: Risk prediction bridge | 12 | Jaderberg+ (2017), Srinivasan+ (2020), Li+ (2026), Thananjeyan+ (2021), Bajcsy+ (2023), Nair+ (2020) |
| D3: Action optimization | 6 | Dalal+ (2018), Alshiekh+ (2018), Taylor+ (2020), Cheng+ (2019), Achiam+ (2017) |

### How These Papers Plug Into the Paper's Narrative

**Related Work positioning** (using Phase 1-2 results):
- Pursuit-evasion RL, partial observability, prediction-enhanced pursuit → serve as the
  "what problem are we solving" backdrop
- The existing 35 papers demonstrate that while prediction + partial observability +
  pursuit exist, the specific combination of depth-only sensor-proxy + observability-risk +
  action correction is novel

**Method justification** (using this supplement):
- Direction 1 papers justify: (a) depth-only → control is viable (Loquercio+, Kaufmann+),
  (b) privileged-to-sensor distillation is a valid paradigm (Lee+, Chen+, Pinto+),
  (c) auxiliary tasks + pretrain + freeze is effective (Kulkarni+, Schwarzer+)
- Direction 2 papers justify: (a) auxiliary prediction in RL improves representations
  (Jaderberg+), (b) learning to predict unsafe events offline and integrating online
  is a valid architecture (Thananjeyan+), (c) offline pretraining accelerates online RL
  (Nair+, Schwarzer+)
- Direction 3 papers justify: (a) safety layers / QP action projection is a valid
  mechanism (Dalal+), (b) shielding / action filtering is accepted in RL (Alshiekh+),
  (c) combining learned models with safety filters is possible (Taylor+, Cheng+),
  (d) safety-shielded vision-based flight is emerging (Zhang+)

### Combined Database
- Phase 1-2 (problem positioning): 35 papers
- This supplement (method design): 27 papers
- **Total**: 62 papers in combined database

### Key Citations for Method Sections

In the paper's **Method** section, the following papers should be cited as direct
methodological precedents:

1. **Depth encoding + auxiliary supervision**: `[@kulkarni2024collision]` (same simulator),
   `[@jaderberg2017reinforcement]` (auxiliary tasks)
2. **Privileged-to-sensor distillation**: `[@lee2020learning]` (teacher-student),
   `[@chen2020learning]` (learning by cheating)
3. **Offline pretrain + freeze**: `[@schwarzer2021pretraining]` (pretrained representations),
   `[@nair2020accelerating]` (offline-to-online)
4. **Risk prediction from privileged data**: `[@srinivasan2020learning]` (safety critic),
   `[@li2026failure]` (failure-aware RL), `[@thananjeyan2021recovery]` (recovery zones)
5. **Privileged-to-sensor PE distillation**: `[@bajcsy2023learning]` (vision-based PE teacher-student)
6. **Action correction via QP projection**: `[@dalal2018safe]` (safety layer)
7. **Safety-shielded drone flight**: `[@zhang2026shielded]` (vision-based shield)

### What Remains Novel After This Review

After mapping these 22 papers against the three technical directions, the combined
contribution remains novel because:

1. No paper combines depth-only sensor-proxy + observability-risk prediction + action correction
   in a UAV pursuit-evasion context
2. The specific risk target (short-horizon target observability loss, H=150) is unique
3. The CTBR-specific constrained action optimization (FOV maintenance, target-keeping, obstacle
   clearance, rate bounds) differs from generic safety layers
4. The two-stage offline-pretrain + freeze + online-PPO architecture for UAV pursuit is novel
5. The same-simulator baseline stack (Aerial Gym) provides a unified evaluation platform
