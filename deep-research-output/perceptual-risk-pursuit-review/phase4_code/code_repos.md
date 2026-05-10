# Code Resources: Perceptual-Risk Pursuit Review
Date: 2026-05-09

## Key Repositories

### thu-uav/Multi-UAV-pursuit-evasion
- **URL**: https://github.com/thu-uav/Multi-UAV-pursuit-evasion
- **Paper**: `[@chen2025open]` Online planning for multi-UAV pursuit-evasion in unknown environments using deep reinforcement learning
- **Description**: Official implementation of OPEN, including adaptive environment generation, prediction-enhanced pursuit, and real-quadrotor deployment scripts.
- **Language / Stack**: Python, Isaac Sim, TorchRL-style stack, OmniDrones-derived environment code
- **Metadata**: MIT license; active repository with deployment and training scripts surfaced in the repository page
- **Why it matters**: closest public implementation to a CTBR/body-rate pursuit stack with prediction enhancement and sim-to-real intent.

### NTU-ICG/AMS-DRL-for-Pursuit-Evasion
- **URL**: https://github.com/NTU-ICG/AMS-DRL-for-Pursuit-Evasion
- **Paper**: related to `[@sychov2024multipursuit]` / asynchronous multistage pursuit-evasion line
- **Description**: Official repository for asynchronous multistage pursuit-evasion navigation with adversarial pursuers and real-flight validation.
- **Language / Stack**: Python
- **Metadata**: MIT license; 39 stars and 5 forks in the surfaced GitHub snapshot
- **Why it matters**: public reference implementation for asynchronous stage-wise adversarial training, useful when the current project later moves toward stronger evader training.

### ntnu-arl/aerial_gym_simulator
- **URL**: https://github.com/ntnu-arl/aerial_gym_simulator
- **Paper**: infrastructure underlying the current local project
- **Description**: high-fidelity Isaac-Gym-based simulator with low-level and high-level controllers, depth / segmentation sensors, and GPU-parallel simulation.
- **Language / Stack**: Python
- **Metadata**: 613 stars and 98 forks in the surfaced GitHub snapshot; BSD-3-Clause license; release `v2.0.0`
- **Why it matters**: the current research will likely live inside this simulator stack, and it already supports the sensor and control hooks needed for a later transition toward sensing-compatible pursuit.

### SJTU-ViSYS-team/CRL-Drone-Racing
- **URL**: https://github.com/SJTU-ViSYS-team/CRL-Drone-Racing
- **Paper**: `[@sun2026curriculum]` Curriculum reinforcement learning for quadrotor racing with random obstacles
- **Description**: vision-based curriculum RL for aggressive quadrotor control in obstacle-rich environments.
- **Language / Stack**: GitHub repository reported by the paper; exact language mix not verified in current search snapshot
- **Metadata**: official code release reported in the paper abstract
- **Why it matters**: useful reference for later-stage visual curriculum design if the current project replaces state-proxy risk inputs with image-driven features.

## Supporting Ecosystem Notes
- `[@chen2025open]` explicitly reports an official website and repository, which makes it the most actionable external baseline for reproduction ideas.
- `[@roncero2025amspb]` did not surface an official repository in the current search results; this weakens immediate reproducibility but not its importance as a conceptual baseline.
- `[@zheng2025visioncapture]` and `[@zhang2026shielded]` did not surface official repositories in the current search results; they remain important as system references.

## Datasets / Benchmarks
| Name | URL | Task | Notes |
|------|-----|------|-------|
| Aerial Gym Simulator benchmark tasks | https://github.com/ntnu-arl/aerial_gym_simulator | UAV control / navigation / sensing | Relevant infrastructure rather than a pursuit benchmark |
| OPEN unknown-environment pursuit scenarios | https://github.com/thu-uav/Multi-UAV-pursuit-evasion | Multi-UAV pursuit-evasion | Includes adaptive environment generation and deployment branches |
| CRL-Drone-Racing environments | https://github.com/SJTU-ViSYS-team/CRL-Drone-Racing | Vision-based aggressive flight | Adjacent benchmark for obstacle-rich visual control |

## Practical Takeaways For The Current Project
1. The strongest code-adjacent external baseline is OPEN because it combines prediction enhancement, real quadrotor deployment, and adversarial pursuit.
2. The strongest training-method reference for later 1v1 escalation is AMS-DRL / AMSPB-style asynchronous adversarial training.
3. The strongest infrastructure reference for the current paper is the existing Aerial Gym stack, because it already exposes low-level control and sensor simulation pathways.
