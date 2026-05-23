# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_action_isaacgympy

import os
import sys
import random
import time
import uuid
import json
from typing import Optional

try:
    import gymnasium as gym
except ImportError:
    import gym
import isaacgym  # noqa
from isaacgym import gymutil
from isaacgym import gymapi
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

# 尝试导入wandb并检查可用性
try:
    import wandb
    WANDB_AVAILABLE = hasattr(wandb, "init")
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

if not WANDB_AVAILABLE:
    wandb = None

from aerial_gym.registry.env_registry import env_config_registry
from aerial_gym.registry.task_registry import task_registry

WANDB_LOGGING_FAILED = False
WANDB_SYNCED_CODE_FILES = {
    "aerial_gym/rl_training/cleanrl/ppo_guidance.py",
    "aerial_gym/rl_training/cleanrl/curriculum/curriculum_controller.py",
    "aerial_gym/task/pursuit_guidance_task/pursuit_guidance_task.py",
    "aerial_gym/config/task_config/pursuit_guidance_task_config.py",
    "aerial_gym/control/controllers/thrust_bodyrate_control.py",
    "aerial_gym/robots/pursuit_quad_direct.py",
}
OUTPUT_ACTION_EXEC_NAMES = (
    "thrust_normalized",
    "force_z",
    "torque_x",
    "torque_y",
    "torque_z",
)
VISIBILITY_EPISODE_METRIC_KEYS = (
    "final_target_detectable",
    "visibility_loss_steps",
    "visibility_episode_invisible_steps",
    "visibility_episode_loss_segments",
    "visibility_episode_recovered_segments",
    "visibility_episode_max_loss_steps",
    "visibility_episode_over_horizon_steps",
    "visibility_episode_recovered_within_horizon_segments",
    "visibility_episode_loss_area",
    "visibility_episode_recovery_rate_within_horizon",
)
VISIBILITY_REWARD_INFO_KEYS = VISIBILITY_EPISODE_METRIC_KEYS[1:]
EPISODE_METRIC_MAP = {
    "r_final_relative_dist": ("episode_metrics/final_relative_dist", "final_relative_dist"),
    "r_final_forward_alignment": (
        "episode_metrics/final_forward_alignment",
        "final_forward_alignment",
    ),
    "r_min_relative_dist": ("episode_metrics/min_relative_dist", "min_relative_dist"),
    "r_episode_mean_closing_speed": (
        "episode_metrics/episode_mean_closing_speed",
        "episode_mean_closing_speed",
    ),
    "r_approach_fraction": ("episode_metrics/approach_fraction", "approach_fraction"),
    "r_episode_min_hazard_clearance": (
        "episode_metrics/episode_min_hazard_clearance",
        "episode_min_hazard_clearance",
    ),
}
EPISODE_METRIC_MAP.update(
    {
        f"r_{key}": (f"episode_metrics/{key}", key)
        for key in VISIBILITY_EPISODE_METRIC_KEYS
    }
)


def resolve_output_action_command_names(controller_name: str):
    controller_name = str(controller_name).lower()
    if controller_name == "lee_attitude_control":
        return ("thrust", "roll", "pitch", "yaw_rate")
    if controller_name == "thrust_bodyrate_control":
        return ("thrust", "p_rate", "q_rate", "r_rate")
    return tuple(f"cmd_{idx}" for idx in range(4))


def configure_target_asset_type(task_config, target_asset_type: Optional[str]):
    if target_asset_type is None:
        return None

    env_config = env_config_registry.get_env_config(task_config.env_name)
    include_asset_type = env_config.env_config.include_asset_type
    asset_map = env_config.env_config.asset_type_to_dict_map
    target_asset_types = [
        asset_type for asset_type in include_asset_type.keys()
        if str(asset_type).startswith("target_")
    ]
    if target_asset_type not in target_asset_types or target_asset_type not in asset_map:
        raise ValueError(
            f"Unknown target asset type {target_asset_type!r}. "
            f"Available target assets: {target_asset_types}"
        )

    for asset_type in target_asset_types:
        include_asset_type[asset_type] = asset_type == target_asset_type

    asset_config = asset_map[target_asset_type]
    return {
        "target_asset_type": target_asset_type,
        "asset_folder": getattr(asset_config, "asset_folder", None),
        "file": getattr(asset_config, "file", None),
        "controller_mass": getattr(asset_config, "controller_mass", None),
        "max_linear_velocity": getattr(asset_config, "max_linear_velocity", None),
        "max_angular_velocity": getattr(asset_config, "max_angular_velocity", None),
    }


def get_learning_rate(step, warmup_steps, total_steps, base_lr, min_lr_ratio=0.01):
    """
    学习率调度器：Warmup + 余弦衰减

    Args:
        step: 当前global step (从0开始)
        warmup_steps: warmup阶段的global steps
        total_steps: 总的global steps (通常等于total_timesteps)
        base_lr: 基础学习率（目标学习率）
        min_lr_ratio: 最小学习率相对于基础学习率的比例

    Returns:
        当前步数对应的学习率
    """
    if step < warmup_steps:
        # Warmup阶段：从 min_lr_ratio * base_lr 线性增长到 base_lr
        warmup_lr_start = base_lr * min_lr_ratio
        # 修正：确保在warmup_steps-1时达到base_lr
        if warmup_steps > 1:
            lr = warmup_lr_start + (base_lr - warmup_lr_start) * (step / (warmup_steps - 1))
        else:
            lr = base_lr
        lr = min(lr, base_lr)  # 确保不超过base_lr
    else:
        # 余弦衰减阶段：从 base_lr 衰减到 min_lr_ratio * base_lr
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        progress = min(progress, 1.0)  # 确保不超过1.0

        # 余弦衰减公式
        cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
        min_lr = base_lr * min_lr_ratio
        lr = min_lr + (base_lr - min_lr) * cosine_decay

    return lr

def get_entropy_coef(step: int, total_steps: int, start_coef: float, end_coef: float) -> float:
    if total_steps <= 0:
        return float(end_coef)
    progress = min(max(float(step) / float(total_steps), 0.0), 1.0)
    return float(start_coef + (end_coef - start_coef) * progress)

def is_better_score(candidate, best, success_delta, length_delta, return_delta: float = 1.0, min_dist_delta: float = 0.2):
    """Hybrid compare for checkpoint selection.

    In the low-success regime, prefer higher avg return and then smaller minimum
    relative distance so we do not save "shorter but failed faster" policies.
    Once success becomes meaningful, switch back to success_rate first and use
    shorter episode length as the main tie-breaker.
    """
    if candidate is None:
        return False
    if best is None:
        return True

    cand_success, cand_return, cand_min_dist, cand_length = candidate
    best_success, best_return, best_min_dist, best_length = best

    low_success_regime = (
        cand_success <= success_delta and best_success <= success_delta
    )

    if low_success_regime:
        if cand_return > best_return + return_delta:
            return True
        if cand_return < best_return - return_delta:
            return False

        if cand_min_dist < best_min_dist - min_dist_delta:
            return True
        if cand_min_dist > best_min_dist + min_dist_delta:
            return False

        return cand_length < best_length - length_delta

    if cand_success > best_success + success_delta:
        return True
    if cand_success < best_success - success_delta:
        return False

    # Success rate is effectively tied; use episode length as tie-breaker.
    return cand_length < best_length - length_delta


def format_score(score):
    if score is None:
        return "N/A"
    success_rate, avg_return, avg_min_dist, avg_length = score
    return (
        f"success_rate={success_rate:.3f}, avg_return={avg_return:.2f}, "
        f"avg_min_dist={avg_min_dist:.2f}, avg_len={avg_length:.1f}"
    )


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if torch.is_tensor(value):
        if value.numel() == 1:
            return json_safe(value.item())
        return json_safe(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def append_jsonl(path: str, record: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(json_safe(record), ensure_ascii=False, sort_keys=True) + "\n")


def score_to_dict(score):
    if score is None:
        return None
    success_rate, avg_return, avg_min_dist, avg_length = score
    return {
        "success_rate": float(success_rate),
        "avg_return": float(avg_return),
        "avg_min_relative_dist": float(avg_min_dist),
        "avg_episode_length": float(avg_length),
    }


def curriculum_metadata(curriculum_controller):
    if curriculum_controller is None:
        return None
    state = curriculum_controller.get_state_dict()
    return {
        "stage_idx": int(state["stage_idx"]),
        "current_threshold": float(curriculum_controller.current_threshold),
        "stage_success_streak": int(state["stage_success_streak"]),
        "visibility_reward_weight_scale": float(
            curriculum_controller.visibility_reward_weight_scale
        ),
    }


def should_save_policy_pool_checkpoint(args, update: int, start_update: int, curriculum_transition) -> bool:
    if not getattr(args, "save_policy_pool", False):
        return False
    if update == start_update:
        return True
    interval_updates = int(getattr(args, "policy_pool_save_interval_updates", 0))
    if interval_updates > 0 and update % interval_updates == 0:
        return True
    return curriculum_transition is not None


def policy_pool_save_reason(args, update: int, start_update: int, curriculum_transition):
    reasons = []
    if update == start_update:
        reasons.append("first_update")
    interval_updates = int(getattr(args, "policy_pool_save_interval_updates", 0))
    if interval_updates > 0 and update % interval_updates == 0:
        reasons.append("interval")
    if curriculum_transition is not None:
        reasons.append("curriculum_transition")
    return reasons


def build_policy_pool_metadata(
    args,
    run_name: str,
    run_dir: str,
    checkpoint_path: str,
    global_step: int,
    update: int,
    start_update: int,
    score,
    episode_metrics,
    update_metrics,
    curriculum_controller,
    curriculum_transition,
    reasons,
):
    step_per_update = int(args.num_envs * args.num_steps)
    return {
        "schema_version": 1,
        "source_family": "ppo",
        "source_kind": "ppo_curriculum_checkpoint" if curriculum_controller is not None else "ppo_checkpoint",
        "pool_role": "trajectory_data_source",
        "checkpoint_type": "full_training_state",
        "checkpoint_path": os.path.relpath(checkpoint_path, run_dir),
        "checkpoint_abspath": os.path.abspath(checkpoint_path),
        "run_name": run_name,
        "run_dir": run_dir,
        "update": int(update),
        "global_step": int(global_step),
        "start_update": int(start_update),
        "total_timesteps": int(args.total_timesteps),
        "progress_fraction": float(global_step) / float(max(args.total_timesteps, 1)),
        "step_per_update": step_per_update,
        "save_interval_updates": int(getattr(args, "policy_pool_save_interval_updates", 0)),
        "save_reasons": reasons,
        "score": score_to_dict(score),
        "episode_metrics": episode_metrics,
        "update_metrics": update_metrics,
        "curriculum": curriculum_metadata(curriculum_controller),
        "curriculum_transition": curriculum_transition,
        "training_config": {
            "task": args.task,
            "seed": int(args.seed),
            "num_envs": int(args.num_envs),
            "num_steps": int(args.num_steps),
            "batch_size": int(args.batch_size),
            "total_timesteps": int(args.total_timesteps),
            "learning_rate": float(args.learning_rate),
            "ent_coef": float(args.ent_coef),
            "ent_coef_final": float(args.ent_coef_final),
            "gamma": float(args.gamma),
            "gae_lambda": float(args.gae_lambda),
            "curriculum_enabled": bool(args.curriculum),
            "curriculum_stable_success_rate": float(args.curriculum_stable_success_rate),
            "curriculum_stable_success_updates": int(args.curriculum_stable_success_updates),
            "curriculum_stable_success_min_episodes": int(
                args.curriculum_stable_success_min_episodes
            ),
        },
    }


def make_run_name(task: str) -> str:
    from datetime import datetime, timezone, timedelta

    beijing_time = datetime.now(timezone(timedelta(hours=8)))
    time_str = beijing_time.strftime("%Y%m%d_%H%M%S")
    return f"PE_{time_str}"


def extract_agent_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "agent" in checkpoint:
        return checkpoint["agent"]
    return checkpoint


def infer_run_dir(checkpoint_path: str, checkpoint) -> str:
    if isinstance(checkpoint, dict):
        run_dir = checkpoint.get("run_dir")
        if run_dir:
            return run_dir
    return os.path.dirname(os.path.abspath(checkpoint_path))


def generate_wandb_run_id() -> str:
    if wandb is not None and hasattr(wandb, "util") and hasattr(wandb.util, "generate_id"):
        return wandb.util.generate_id()
    return uuid.uuid4().hex


def should_log_wandb_code(path: str) -> bool:
    root = os.path.abspath(".")
    abs_path = os.path.abspath(path)
    rel_path = os.path.normpath(os.path.relpath(abs_path, root))
    return rel_path in WANDB_SYNCED_CODE_FILES


def wandb_log(data, step: int, commit: bool = True):
    global WANDB_LOGGING_FAILED
    if WANDB_LOGGING_FAILED:
        return
    if wandb is None or getattr(wandb, "run", None) is None:
        return
    try:
        wandb.log(data, step=step, commit=commit)
    except Exception as exc:
        WANDB_LOGGING_FAILED = True
        print(f"WandB logging disabled after wandb.log failed: {exc}")


def move_optimizer_state_to_device(optimizer: optim.Optimizer, device: str):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def restore_rng_states(checkpoint):
    if not isinstance(checkpoint, dict):
        return
    if checkpoint.get("rng_python") is not None:
        random.setstate(checkpoint["rng_python"])
    if checkpoint.get("rng_numpy") is not None:
        np.random.set_state(checkpoint["rng_numpy"])
    if checkpoint.get("rng_torch") is not None:
        torch.set_rng_state(checkpoint["rng_torch"])
    if torch.cuda.is_available() and checkpoint.get("rng_cuda") is not None:
        torch.cuda.set_rng_state_all(checkpoint["rng_cuda"])


def build_training_checkpoint(
    agent: nn.Module,
    optimizer: optim.Optimizer,
    args,
    global_step: int,
    update: int,
    best_score,
    best_update,
    no_improve_updates: int,
    run_name: str,
    run_dir: str,
    wandb_run_id: Optional[str],
):
    return {
        "agent": agent.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": int(global_step),
        "update": int(update),
        "best_score": best_score,
        "best_update": best_update,
        "no_improve_updates": int(no_improve_updates),
        "run_name": run_name,
        "run_dir": run_dir,
        "wandb_run_id": wandb_run_id,
        "args": dict(vars(args)),
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }



def get_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "pursuit_guidance_task", "help": "Resume training or start testing from a checkpoint. Overrides config file if provided."},
        {"name": "--experiment_name", "type": str, "default": os.path.basename(__file__).rstrip(".py"), "help": "Name of the experiment to run or load. Overrides config file if provided."},
        {"name": "--checkpoint", "type": str, "default": None, "help": "Saved model checkpoint number."},        
        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
        {"name": "--num_envs", "type": int, "default": 512, "help": "Number of environments to create. Overrides config file if provided."},
        {"name": "--seed", "type": int, "default": 1, "help": "Random seed. Overrides config file if provided."},
        {"name": "--play", "required": False, "help": "only run network", "action": 'store_true'},
        {"name": "--play-steps", "type": int, "default": 3600, "help": "Maximum number of simulation steps to run in --play mode."},
        {"name": "--target-asset-type", "type": str, "default": None, "choices": ["target_quad", "target_x500"], "help": "Optional target asset override for pursuit runs."},
        {"name": "--resume", "action": "store_true", "default": False, "help": "Resume training from a latest checkpoint with optimizer/global_step state."},

        {"name": "--torch-deterministic-off", "action": "store_true", "default": False, "help": "if toggled, `torch.backends.cudnn.deterministic=False`"},

        {"name": "--track", "action": "store_true", "default": False,"help": "if toggled, this experiment will be tracked with Weights and Biases"},
        {"name": "--no-track", "action": "store_true", "default": False,"help": "if toggled, disable Weights and Biases tracking"},
        {"name": "--wandb-project-name", "type":str, "default": "ppo-guidance-strike", "help": "the wandb's project name"},
        {"name": "--wandb-entity", "type":str, "default": None, "help": "the entity (team) of wandb's project"},
        {"name": "--wandb-mode", "type": str, "default": "online", "help": "WandB mode: online, offline, or disabled."},

        # Algorithm specific arguments
        {"name": "--total-timesteps", "type":int, "default": 2400000000,
            "help": "total timesteps of the experiments"},
        {"name": "--learning-rate", "type":float, "default": 0.0003, # 降低学习率以适应稳定的reward范围
            "help": "the learning rate of the optimizer"},
        {"name": "--num-steps", "type":int, "default": 3600, # 修复：匹配环境episode长度(36s/0.01s=3600步)，避免数据不完整导致CUDA错误
            "help": "the number of steps to run in each environment per policy rollout"},
        {"name": "--anneal-lr", "action": "store_true", "default": False, # 逐渐降低学习率，提高后期训练的稳定性
            "help": "Toggle learning rate annealing for policy and value networks"},

        # 学习率调度参数
        {"name": "--use-lr-scheduler", "action": "store_true", "default": True, # 启用新的学习率调度器（warmup + 余弦衰减）
            "help": "Use warmup + cosine annealing learning rate scheduler"},
        {"name": "--warmup-steps", "type":int, "default": 10000000, # warmup阶段的global steps，从小学习率线性增长到目标学习率
            "help": "Number of warmup global steps for learning rate scheduler (default: 10M steps)"},
        {"name": "--min-lr-ratio", "type":float, "default": 0.01, # 最小学习率相对于初始学习率的比例
            "help": "Minimum learning rate as a ratio of the initial learning rate"},
        {"name": "--gamma", "type":float, "default": 0.999, # 折扣因子,衡量未来奖励的重要性。值越高，越重视长期奖励
            "help": "the discount factor gamma"},
        {"name": "--gae-lambda", "type":float, "default": 0.99, # 控制广义优势估计（GAE）的偏差和方差。值越高，方差越大，偏差越小。
            "help": "the lambda for the general advantage estimation"},
        {"name": "--num-minibatches", "type":int, "default": 15, # 修复：3600/15=240，保证整除避免batch不均匀导致的数值问题
            "help": "the number of mini-batches"},
        {"name": "--update-epochs", "type":int, "default": 12, # 每次更新中对数据的迭代次数。更多的迭代可以充分利用数据，但可能导致过拟合
            "help": "the K epochs to update the policy"},
        {"name": "--clip-coef", "type":float, "default": 0.2, # PPO算法中的重要性采样剪切系数，防止策略更新过大
            "help": "the surrogate clipping coefficient"},
        {"name": "--clip-vloss", "action": "store_true", "default": False, # 是否使用剪切损失函数。剪切损失函数可以提高训练的稳定性
            "help": "Toggles whether or not to use a clipped loss for the value function, as per the paper."},
        {"name": "--ent-coef", "type":float, "default": 0.01, # 鼓励探索的熵项系数。值越高，探索越多，但可能导致过分探索
            "help": "coefficient of the entropy"},
        {"name": "--ent-coef-final", "type":float, "default": 0.0, 
            "help": "final entropy coefficient after decay"},
        {"name": "--vf-coef", "type":float, "default": 0.5, # 降低value function权重，避免value loss主导训练
            "help": "coefficient of the value function"},
        {"name": "--max-grad-norm", "type":float, "default": 0.5,  # 增加梯度裁剪阈值，允许更大的梯度更新
            "help": "the maximum norm for the gradient clipping"},
        {"name": "--target-kl", "type":float, "default": None,
            "help": "the target KL divergence threshold"},
        # Best model saving
        {"name": "--save-best", "action": "store_true", "default": True, "help": "启用基于指标的best.pth保存（注意：某些解析器对store_true默认值处理不一致）"},
        {"name": "--no-save-best", "action": "store_true", "default": False, "help": "禁用best.pth保存（覆盖 --save-best）"},
        {"name": "--best-metric", "type": str, "default": "hybrid_success_then_return_fallback", "help": "保存最优策略的指标：低成功率阶段优先avg_return/avg_min_relative_dist；成功后优先success_rate，其次avg_episode_length（越短越好）"},
        {"name": "--save-policy-pool", "action": "store_true", "default": True, "help": "启用策略池中间checkpoint保存，默认每10个update保存一次"},
        {"name": "--no-save-policy-pool", "action": "store_true", "default": False, "help": "禁用策略池中间checkpoint保存（覆盖 --save-policy-pool）"},
        {"name": "--policy-pool-save-interval-updates", "type": int, "default": 10, "help": "策略池checkpoint保存间隔，单位为PPO update；默认10个update约18.4M steps"},
        {"name": "--policy-pool-dir-name", "type": str, "default": "policy_pool", "help": "run目录下的策略池子目录名称"},
        # Early stopping
        {"name": "--early-stop", "action": "store_true", "default": True, "help": "启用早停：主看success_rate，辅看avg_episode_length"},
        {"name": "--no-early-stop", "action": "store_true", "default": False, "help": "禁用早停（覆盖 --early-stop）"},
        {"name": "--early-stop-patience", "type": int, "default": 20, "help": "连续多少个update没有显著提升后停止训练"},
        {"name": "--early-stop-start-success-rate", "type": float, "default": 0.8, "help": "只有当best success_rate达到该阈值后，才开始累计early-stop patience"},
        {"name": "--early-stop-success-delta", "type": float, "default": 0.005, "help": "success_rate被视为显著提升的最小增量"},
        {"name": "--early-stop-length-delta", "type": float, "default": 10.0, "help": "当success_rate近似持平时，avg_episode_length被视为显著改善的最小下降步数"},
        {"name": "--early-stop-min-episodes", "type": int, "default": 64, "help": "单个update至少统计到多少个结束episode才参与best/early-stop判断"},
        # Curriculum learning
        {"name": "--curriculum", "action": "store_true", "default": False, "help": "Enable automatic multi-stage curriculum learning"},
        {"name": "--curriculum-stable-success-rate", "type": float, "default": 0.98, "help": "Success-rate threshold for success_stability curriculum transitions"},
        {"name": "--curriculum-stable-success-updates", "type": int, "default": 100, "help": "Consecutive PPO updates above the success-rate threshold before a curriculum transition"},
        {"name": "--curriculum-stable-success-min-episodes", "type": int, "default": 64, "help": "Minimum finished episodes in an update before it can count toward curriculum success stability"},
        ]

    # parse arguments
    args = gymutil.parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)
    
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)

    args.torch_deterministic = not args.torch_deterministic_off
    if args.wandb_mode not in ("online", "offline", "disabled"):
        raise ValueError("--wandb-mode must be one of: online, offline, disabled")

    # WandB tracking logic: 默认启用，除非显式禁用
    if args.no_track:
        args.track = False
    elif getattr(args, "wandb_mode", "online") == "disabled":
        args.track = False
    else:
        args.track = True  # 默认启用WandB

    # 统一处理best保存开关：默认启用，除非显式传入 --no-save-best
    if args.no_save_best:
        args.save_best = False
    else:
        args.save_best = True

    if args.no_save_policy_pool:
        args.save_policy_pool = False
    else:
        args.save_policy_pool = True
    if args.policy_pool_save_interval_updates < 0:
        raise ValueError("--policy-pool-save-interval-updates must be >= 0")
    args.policy_pool_dir_name = str(args.policy_pool_dir_name).strip() or "policy_pool"

    if args.no_early_stop:
        args.early_stop = False
    else:
        args.early_stop = True

    # name allignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device=='cuda':
        args.sim_device += f":{args.sim_device_id}"
    return args

class RecordEpisodeStatisticsTorch:
    def __init__(self, env, device):
        self.env = env
        self.num_envs = getattr(env, "num_envs", 1)
        self.num_obs = int(env.task_config.observation_space_dim)
        self.num_actions = int(env.task_config.action_space_dim)
        self.device = device
        self.episode_returns = None
        self.episode_lengths = None
        self.reward_keys = [
            "total",
            "progress",
            "success_bonus",
            "time_penalty",
            "timeout_penalty",
            "effort",
            "collision_penalty",
            "avoid_penalty",
            "visibility_loss_penalty",
            "contrib_progress",
            "contrib_success_bonus",
            "contrib_time_penalty",
            "contrib_timeout_penalty",
            "contrib_effort",
            "contrib_collision_penalty",
            "contrib_avoid_penalty",
            "contrib_alignment",
            "contrib_visibility",
        ]

    def __getattr__(self, name):
        return getattr(self.env, name)

    def close(self):
        return self.env.close()

    def reset(self, **kwargs):
        task_obs, _rewards, _terminations, _truncations, infos = self.env.reset()
        observations = task_obs["observations"]
        # self.episode_returns = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.episode_returns = {key: torch.zeros(self.num_envs, dtype=torch.float32, device=self.device) 
                                for key in self.reward_keys}
        self.episode_lengths = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        # self.returned_episode_returns = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.returned_episode_returns = {key: torch.zeros(self.num_envs, dtype=torch.float32, device=self.device) 
                                         for key in self.reward_keys}
        self.returned_episode_lengths = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.episode_min_relative_dist = torch.full(
            (self.num_envs,), float("inf"), dtype=torch.float32, device=self.device
        )
        self.returned_final_relative_dist = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.returned_final_forward_alignment = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.returned_min_relative_dist = torch.full(
            (self.num_envs,), float("inf"), dtype=torch.float32, device=self.device
        )
        self.episode_min_hazard_clearance = torch.full(
            (self.num_envs,), float("inf"), dtype=torch.float32, device=self.device
        )
        self.returned_episode_min_hazard_clearance = torch.full(
            (self.num_envs,), float("inf"), dtype=torch.float32, device=self.device
        )
        self.episode_closing_speed_sum = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.returned_episode_mean_closing_speed = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.episode_approach_steps = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.returned_approach_fraction = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        return observations, infos

    def step(self, action):
        task_obs, rewards, terminations, truncations, infos = self.env.step(action)
        observations = task_obs["observations"]
        dones = (terminations | truncations).float()
        reward_info = infos
        # pdb.set_trace()
        # 更新总奖励和各个组成部分的奖励
        self.episode_returns["total"] += rewards
        for key in self.reward_keys:
            if key != "total":
                # 检查reward_info中是否存在该键，如果不存在则使用0
                if key in reward_info:
                    self.episode_returns[key] += reward_info[key]
                else:
                    # 如果reward_info中没有该键，则不更新（保持为0）
                    pass

        if "relative_dist" in reward_info:
            self.returned_final_relative_dist[:] = reward_info["relative_dist"]
            self.episode_min_relative_dist = torch.minimum(
                self.episode_min_relative_dist, reward_info["relative_dist"]
            )
        if "forward_alignment" in reward_info:
            self.returned_final_forward_alignment[:] = reward_info["forward_alignment"]
        if "min_hazard_clearance" in reward_info:
            self.episode_min_hazard_clearance = torch.minimum(
                self.episode_min_hazard_clearance, reward_info["min_hazard_clearance"]
            )
        if "closing_speed" in reward_info:
            self.episode_closing_speed_sum += reward_info["closing_speed"]
            self.episode_approach_steps += (reward_info["closing_speed"] > 0.0).float()

        self.episode_lengths += 1

        done_mask = (1 - dones.float())

        # 保存完成episode的奖励和长度（在重置之前）
        for key in self.reward_keys:
            self.returned_episode_returns[key][:] = self.episode_returns[key]
        self.returned_episode_lengths[:] = self.episode_lengths
        self.returned_min_relative_dist[:] = self.episode_min_relative_dist
        self.returned_episode_min_hazard_clearance[:] = self.episode_min_hazard_clearance
        lengths_float = self.returned_episode_lengths.clamp_min(1).float()
        self.returned_episode_mean_closing_speed[:] = self.episode_closing_speed_sum / lengths_float
        self.returned_approach_fraction[:] = self.episode_approach_steps / lengths_float

        # 然后重置完成的episode
        for key in self.reward_keys:
            self.episode_returns[key] *= done_mask
        self.episode_lengths *= done_mask.int()
        done_bool = dones.bool()
        self.episode_min_relative_dist = torch.where(
            done_bool,
            torch.full_like(self.episode_min_relative_dist, float("inf")),
            self.episode_min_relative_dist,
        )
        self.episode_min_hazard_clearance = torch.where(
            done_bool,
            torch.full_like(self.episode_min_hazard_clearance, float("inf")),
            self.episode_min_hazard_clearance,
        )
        self.episode_closing_speed_sum *= done_mask
        self.episode_approach_steps *= done_mask

        # 更新 infos 字典
        infos["r"] = self.returned_episode_returns["total"]  # 保持原有的总奖励
        infos["l"] = self.returned_episode_lengths

        # 添加详细的奖励信息到 infos
        for key in self.reward_keys:
            if key != "total":
                infos[f"r_{key}"] = self.returned_episode_returns[key]

        if "relative_dist" in reward_info:
            infos["r_final_relative_dist"] = self.returned_final_relative_dist
            infos["r_min_relative_dist"] = self.returned_min_relative_dist
        if "forward_alignment" in reward_info:
            infos["r_final_forward_alignment"] = self.returned_final_forward_alignment
        if "closing_speed" in reward_info:
            infos["r_episode_mean_closing_speed"] = self.returned_episode_mean_closing_speed
            infos["r_approach_fraction"] = self.returned_approach_fraction
        if "min_hazard_clearance" in reward_info:
            infos["r_episode_min_hazard_clearance"] = self.returned_episode_min_hazard_clearance
        if "target_detectable" in reward_info:
            infos["r_final_target_detectable"] = reward_info["target_detectable"].float()
        for key in VISIBILITY_REWARD_INFO_KEYS:
            if key in reward_info:
                infos[f"r_{key}"] = reward_info[key].float()
        return (
            observations,
            rewards,
            dones,
            infos,
        )


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class RunningMeanStd(nn.Module):
    def __init__(self, obs_dim, epsilon=1e-4):
        super().__init__()
        self.register_buffer("mean", torch.zeros(obs_dim, dtype=torch.float32))
        self.register_buffer("var", torch.ones(obs_dim, dtype=torch.float32))
        self.register_buffer("count", torch.tensor(epsilon, dtype=torch.float32))

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        x = x.reshape(-1, self.mean.numel()).detach()
        if x.numel() == 0:
            return

        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = torch.tensor(float(x.shape[0]), dtype=torch.float32, device=x.device)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total_count
        new_var = m2 / total_count

        self.mean.copy_(new_mean)
        self.var.copy_(new_var)
        self.count.copy_(total_count)

    def normalize(self, x: torch.Tensor, clip: float = 10.0, eps: float = 1e-8):
        x_norm = (x - self.mean) / torch.sqrt(self.var + eps)
        return torch.clamp(x_norm, -clip, clip)


def atanh_clipped(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = torch.clamp(x, -1.0 + eps, 1.0 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()

        obs_dim = int(np.array(envs.num_obs).prod())
        act_dim = int(np.prod(envs.num_actions))

        self.log_std_min = -5.0
        self.log_std_max = 0.5
        self.tanh_eps = 1e-6
        self.obs_rms = RunningMeanStd(obs_dim)

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        # PPO直接输出4维姿态控制指令 [thrust, roll, pitch, yaw_rate] 完全替代制导律
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, act_dim), std=0.01),
        )

        self.actor_logstd = nn.Parameter(torch.full((1, act_dim), -1.0))

    @torch.no_grad()
    def update_obs_rms(self, x: torch.Tensor):
        self.obs_rms.update(x)

    def normalize_obs(self, x: torch.Tensor) -> torch.Tensor:
        return self.obs_rms.normalize(x)

    def get_value(self, x):
        x = self.normalize_obs(x)
        return self.critic(x)

    def get_action_and_value(self, x, action=None, return_aux: bool = False):
        x = self.normalize_obs(x)

        action_mean = self.actor_mean(x)
        action_logstd = torch.clamp(
            self.actor_logstd,
            self.log_std_min,
            self.log_std_max
        ).expand_as(action_mean)

        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if action is None:
            pre_tanh_action = probs.rsample()
            squashed_action = torch.tanh(pre_tanh_action)
        else:
            squashed_action = torch.clamp(action, -1.0 + self.tanh_eps, 1.0 - self.tanh_eps)
            pre_tanh_action = atanh_clipped(squashed_action, self.tanh_eps)

        log_prob = probs.log_prob(pre_tanh_action) - torch.log(
            1.0 - squashed_action.pow(2) + self.tanh_eps
        )
        log_prob = log_prob.sum(dim=1)

        base_entropy = probs.entropy().sum(dim=1)
        entropy_proxy = (
            base_entropy +
            torch.log(1.0 - squashed_action.pow(2) + self.tanh_eps).sum(dim=1)
        )

        value = self.critic(x)
        if return_aux:
            aux = {
                "action_mean": action_mean.detach(),
                "action_std": action_std.detach(),
                "pre_tanh_action": pre_tanh_action.detach(),
                "squashed_action": squashed_action.detach(),
            }
            return squashed_action, log_prob, entropy_proxy, value, aux
        return squashed_action, log_prob, entropy_proxy, value

    
if __name__ == "__main__":
    args = get_args()
    if args.resume and args.play:
        raise ValueError("--resume cannot be used together with --play.")
    if args.resume and args.checkpoint is None:
        raise ValueError("--resume requires --checkpoint to point to latest.pth.")

    loaded_checkpoint = None
    resume_global_step = 0
    start_update = 1
    best_score = None
    best_update = None
    no_improve_updates = 0
    run_name = None
    run_dir = None
    writer = None
    wandb_run_id = None
    wandb_run = None
    if args.checkpoint is not None:
        print("Loading checkpoint file...")
        loaded_checkpoint = torch.load(args.checkpoint, map_location="cpu")

    if args.resume:
        if not isinstance(loaded_checkpoint, dict) or "agent" not in loaded_checkpoint or "optimizer" not in loaded_checkpoint:
            raise RuntimeError(
                "--resume requires a full training checkpoint with agent/optimizer state, "
                "such as latest.pth."
            )
        saved_args = loaded_checkpoint.get("args", {})
        if isinstance(saved_args, dict):
            if not args.no_track and "track" in saved_args:
                args.track = bool(saved_args["track"])
            if "wandb_project_name" in saved_args:
                args.wandb_project_name = saved_args["wandb_project_name"]
            if "wandb_entity" in saved_args:
                args.wandb_entity = saved_args["wandb_entity"]
        resume_global_step = int(loaded_checkpoint.get("global_step", 0))
        start_update = int(loaded_checkpoint.get("update", 0)) + 1
        best_score = loaded_checkpoint.get("best_score")
        best_update = loaded_checkpoint.get("best_update")
        no_improve_updates = int(loaded_checkpoint.get("no_improve_updates", 0))
        run_dir = infer_run_dir(args.checkpoint, loaded_checkpoint)
        run_name = loaded_checkpoint.get("run_name") or os.path.basename(os.path.normpath(run_dir))
        wandb_run_id = loaded_checkpoint.get("wandb_run_id")

    base_task_config = task_registry.get_task_config(args.task)
    target_asset_metadata = configure_target_asset_type(
        base_task_config, args.target_asset_type
    )
    if target_asset_metadata is not None:
        print(
            "Target asset override: "
            f"{target_asset_metadata['target_asset_type']} | "
            f"urdf={target_asset_metadata['asset_folder']}/{target_asset_metadata['file']} | "
            f"controller_mass={target_asset_metadata['controller_mass']} | "
            f"max_linear_velocity={target_asset_metadata['max_linear_velocity']} | "
            f"max_angular_velocity={target_asset_metadata['max_angular_velocity']}"
        )

    if not args.play:
        if run_name is None:
            run_name = make_run_name(args.task)
        if run_dir is None:
            run_dir = os.path.join("runs", run_name)
        os.makedirs(run_dir, exist_ok=True)

        # WandB初始化和状态检查
        if args.track and WANDB_AVAILABLE:
            if args.resume and not wandb_run_id:
                print(
                    "Resume checkpoint is missing wandb_run_id; disabling WandB to avoid creating a new run."
                )
                args.track = False
            elif not args.resume:
                wandb_run_id = generate_wandb_run_id()
        if args.track and WANDB_AVAILABLE:
            wandb_resume_mode = "must" if args.resume else "allow"
            wandb_run = wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=False,
                config=vars(args),
                name=run_name,
                id=wandb_run_id,
                resume=wandb_resume_mode,
                dir=run_dir,
                mode=args.wandb_mode,
                monitor_gym=True,
                save_code=False,
                tags=["ppo", "guidance", "strike", "aerial"],
                settings=wandb.Settings(
                    init_timeout=30.0,
                    summary_timeout=15,
                    x_file_stream_timeout_seconds=15.0,
                    x_file_transfer_timeout_seconds=15.0,
                    x_graphql_timeout_seconds=15.0,
                    x_service_wait=15.0,
                ),
            )
            print(f"WandB run: {getattr(wandb_run, 'url', None)} | mode={args.wandb_mode}")
            wandb_run.log_code(
                root=os.path.abspath("."),
                include_fn=should_log_wandb_code,
            )
        elif args.track and not WANDB_AVAILABLE:
            args.track = False

        writer = SummaryWriter(log_dir=run_dir, purge_step=resume_global_step if args.resume else None)
        if args.resume:
            writer.add_text(
                "resume",
                "|key|value|\n|-|-|\n"
                f"|checkpoint|{args.checkpoint}|\n"
                f"|global_step|{resume_global_step}|\n"
                f"|start_update|{start_update}|",
            )
            print(
                f"Resume training: checkpoint={args.checkpoint} | start_update={start_update} "
                f"| global_step={resume_global_step} | run_dir={run_dir}"
            )
        else:
            writer.add_text(
                "hyperparameters",
                "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
            )
            print(
                f"Best checkpoint saving: {'ENABLED' if args.save_best else 'DISABLED'} "
                f"| metric=hybrid_success_then_return_fallback | early_stop={'ENABLED' if args.early_stop else 'DISABLED'} "
                f"| run_dir={run_dir}"
            )
            print(
                f"Curriculum: {'ENABLED' if args.curriculum else 'DISABLED'} "
                f"| stages=5m,3m,3m+vis0.20,3m+vis0.40,3m+vis0.60 "
                f"| transition=success_stability "
                f"| stable_sr={args.curriculum_stable_success_rate} "
                f"x{args.curriculum_stable_success_updates} "
                f"| min_episodes={args.curriculum_stable_success_min_episodes}"
            )
    else:
        if args.track:
            print("--play mode: skip WandB/TensorBoard run creation.")
        args.track = False

    # TRY NOT TO MODIFY: seeding
    if args.seed == -1:
        args.seed = int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
        print(f"Resolved runtime random seed: {args.seed}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = args.rl_device
    print("using device:", device)

    # env setup
    original_argv = sys.argv[:]
    sys.argv = [
        original_argv[0],
        "--headless",
        str(args.headless),
        "--num_envs",
        str(args.num_envs),
        "--use_warp",
        "False",
        "--sim_device",
        device,
    ]
    try:
        env = task_registry.make_task(
            task_name=args.task,
            seed=args.seed,
            num_envs=args.num_envs,
            headless=args.headless,
            use_warp=False,
            device=device,
        )
    finally:
        sys.argv = original_argv
    env_cfg = env.task_config
    envs = RecordEpisodeStatisticsTorch(env, device)
    base_env = envs.env
    output_action_command_names = resolve_output_action_command_names(
        getattr(env_cfg, "controller_name", "thrust_bodyrate_control")
    )

    print("num actions: ",envs.num_actions)
    print("num obs: ", envs.num_obs)
    
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    if args.play and args.checkpoint is None:
        raise ValueError("No checkpoint provided for testing.")

    # load checkpoint if needed
    if loaded_checkpoint is not None:
        print("Loading checkpoint...")
        checkpoint = loaded_checkpoint
        state_dict = extract_agent_state_dict(checkpoint)
        missing_keys, unexpected_keys = agent.load_state_dict(state_dict, strict=False)
        print("Loaded checkpoint")
        missing_obs_rms_keys = [key for key in missing_keys if key.startswith("obs_rms.")]
        missing_non_obs_keys = [key for key in missing_keys if not key.startswith("obs_rms.")]
        if missing_keys:
            print(f"Missing keys when loading checkpoint: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys when loading checkpoint: {unexpected_keys}")
        if missing_non_obs_keys or unexpected_keys:
            raise RuntimeError(
                "Checkpoint is incompatible with the current agent definition. "
                f"Missing non-obs keys: {missing_non_obs_keys}, unexpected keys: {unexpected_keys}"
            )
        if missing_obs_rms_keys:
            message = (
                "Checkpoint is missing observation normalization statistics "
                f"({missing_obs_rms_keys})."
            )
            if args.play:
                raise RuntimeError(
                    message +
                    " Refusing to run --play with default obs_rms stats because evaluation would be misleading."
                )
            print(
                "WARNING: " + message +
                " Resumed training will rebuild obs_rms online, so behavior may differ from the original run."
            )
        if args.resume:
            optimizer.load_state_dict(checkpoint["optimizer"])
            move_optimizer_state_to_device(optimizer, device)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs, envs.num_obs), dtype=torch.float).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs, envs.num_actions), dtype=torch.float).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float).to(device)
    values = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float).to(device)
    timeouts = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float).to(device)
    advantages = torch.zeros_like(rewards, dtype=torch.float).to(device)

    # TRY NOT TO MODIFY: start the game
    if args.resume and loaded_checkpoint is not None:
        restore_rng_states(loaded_checkpoint)
    global_step = resume_global_step
    start_time = time.time()
    next_obs,_info = envs.reset()
    next_done = torch.zeros(args.num_envs, dtype=torch.float).to(device)
    if not args.play:
        with torch.no_grad():
            agent.update_obs_rms(next_obs)

    num_updates = args.total_timesteps // args.batch_size

    if not args.play:
        # Curriculum controller initialization
        curriculum_controller = None
        if args.curriculum:
            from aerial_gym.rl_training.cleanrl.curriculum.curriculum_controller import (
                CurriculumController,
            )
            curriculum_controller = CurriculumController(
                task_config=env_cfg,
                args=args,
            )
            if args.resume and loaded_checkpoint is not None:
                if "curriculum" not in loaded_checkpoint:
                    raise ValueError(
                        "--resume --curriculum requires a checkpoint with curriculum state."
                    )
                curriculum_controller.load_state_dict(loaded_checkpoint["curriculum"])
                print(
                    f"Resumed curriculum: stage {curriculum_controller.stage_idx}, "
                    f"threshold {curriculum_controller.current_threshold:.1f}m"
                )

        last_update_score = None
        best_path = os.path.join(run_dir, "best.pth")
        latest_path = os.path.join(run_dir, "latest.pth")
        policy_pool_dir = os.path.join(run_dir, args.policy_pool_dir_name)
        policy_pool_manifest_path = os.path.join(policy_pool_dir, "manifest.jsonl")
        if args.save_policy_pool:
            os.makedirs(policy_pool_dir, exist_ok=True)
            print(
                f"Policy pool checkpointing: ENABLED | dir={policy_pool_dir} "
                f"| interval_updates={args.policy_pool_save_interval_updates}"
            )
        else:
            print("Policy pool checkpointing: DISABLED")
        reward_cfg = getattr(env_cfg, "reward", None)
        success_threshold = float(getattr(reward_cfg, "success_threshold", 1.0))
        raw_reward_component_keys = [
            key for key in envs.reward_keys
            if key != "total" and not key.startswith("contrib_")
        ]
        reward_contrib_keys = [
            key for key in envs.reward_keys if key.startswith("contrib_")
        ]
        for update in range(start_update, num_updates + 1):
            policy_pool_episode_metrics = None
            policy_pool_score = None
            policy_pool_curriculum_transition = None

            # 学习率调度
            if args.use_lr_scheduler:
                # 使用新的 warmup + 余弦衰减调度器（基于global_step）
                current_lr = get_learning_rate(
                    step=global_step,  # 使用global_step而不是update数量
                    warmup_steps=args.warmup_steps,
                    total_steps=args.total_timesteps,  # 使用total_timesteps而不是num_updates
                    base_lr=args.learning_rate,
                    min_lr_ratio=args.min_lr_ratio
                )
                optimizer.param_groups[0]["lr"] = current_lr
            elif args.anneal_lr:
                # 使用原有的线性退火调度器
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            current_ent_coef = get_entropy_coef(
                step=global_step,
                total_steps=args.total_timesteps,
                start_coef=args.ent_coef,
                end_coef=args.ent_coef_final,
            )

            # 初始化本轮数据收集的奖励统计
            episode_rewards_summary = {
                "count": 0,
                "returns": [],
                "lengths": [],
                "final_relative_dist": [],
                "final_forward_alignment": [],
                "min_relative_dist": [],
                "episode_mean_closing_speed": [],
                "approach_fraction": [],
                "episode_min_hazard_clearance": [],
                **{key: [] for key in VISIBILITY_EPISODE_METRIC_KEYS},
                "done_timeout": 0,
                "done_collision": 0,
                "done_success": 0,
                "done_far": 0,
                "raw_component_returns": {key: [] for key in raw_reward_component_keys},
                "contrib_component_returns": {key: [] for key in reward_contrib_keys},
            }
            rollout_diag = {
                "count": 0,
                "policy_mean_abs_mean": 0.0,
                "policy_mean_abs_max": 0.0,
                "effective_actor_std": 0.0,
                "pre_tanh_action_abs_mean": 0.0,
                "pre_tanh_action_abs_max": 0.0,
                "pre_tanh_action_std": 0.0,
                "pre_tanh_action_oob_ratio": 0.0,
                "squashed_action_sat_ratio": 0.0,
                "squashed_sat_per_dim": torch.zeros(envs.num_actions, dtype=torch.float32, device=device),
                "output_action_command_mean": torch.zeros(
                    len(output_action_command_names), dtype=torch.float32, device=device
                ),
                "output_action_command_abs_mean": torch.zeros(
                    len(output_action_command_names), dtype=torch.float32, device=device
                ),
                "output_action_command_abs_max": torch.zeros(
                    len(output_action_command_names), dtype=torch.float32, device=device
                ),
                "output_action_exec_mean": torch.zeros(
                    len(OUTPUT_ACTION_EXEC_NAMES), dtype=torch.float32, device=device
                ),
                "output_action_exec_abs_mean": torch.zeros(
                    len(OUTPUT_ACTION_EXEC_NAMES), dtype=torch.float32, device=device
                ),
                "output_action_exec_abs_max": torch.zeros(
                    len(OUTPUT_ACTION_EXEC_NAMES), dtype=torch.float32, device=device
                ),
            }

            for step in range(0, args.num_steps):
                global_step += 1 * args.num_envs
                obs[step] = next_obs
                dones[step] = next_done
                
                with torch.no_grad():
                    action, logprob, _, value, policy_aux = agent.get_action_and_value(
                        next_obs, return_aux=True
                    )
                    values[step] = value.flatten()

                rollout_diag["count"] += 1
                rollout_diag["policy_mean_abs_mean"] += policy_aux["action_mean"].abs().mean().item()
                rollout_diag["policy_mean_abs_max"] = max(
                    rollout_diag["policy_mean_abs_max"],
                    policy_aux["action_mean"].abs().max().item(),
                )
                rollout_diag["effective_actor_std"] += policy_aux["action_std"].mean().item()
                rollout_diag["pre_tanh_action_abs_mean"] += policy_aux["pre_tanh_action"].abs().mean().item()
                rollout_diag["pre_tanh_action_abs_max"] = max(
                    rollout_diag["pre_tanh_action_abs_max"],
                    policy_aux["pre_tanh_action"].abs().max().item(),
                )
                rollout_diag["pre_tanh_action_std"] += policy_aux["pre_tanh_action"].std().item()
                rollout_diag["pre_tanh_action_oob_ratio"] += (
                    policy_aux["pre_tanh_action"].abs() > 1.0
                ).float().mean().item()
                rollout_diag["squashed_action_sat_ratio"] += (
                    policy_aux["squashed_action"].abs() >= 0.999
                ).float().mean().item()
                rollout_diag["squashed_sat_per_dim"] += (
                    policy_aux["squashed_action"].abs() >= 0.999
                ).float().mean(dim=0)

                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, raw_rewards, next_done, info = envs.step(action)

                command_batch = base_env.last_actor1_attitude_command.detach()
                exec_batch = torch.cat(
                    (
                        base_env.last_actor1_output_thrust_normalized.detach().unsqueeze(1),
                        base_env.last_actor1_force_z.detach().unsqueeze(1),
                        base_env.last_actor1_output_torques.detach(),
                    ),
                    dim=1,
                )
                rollout_diag["output_action_command_mean"] += command_batch.mean(dim=0)
                rollout_diag["output_action_command_abs_mean"] += command_batch.abs().mean(dim=0)
                rollout_diag["output_action_command_abs_max"] = torch.maximum(
                    rollout_diag["output_action_command_abs_max"],
                    command_batch.abs().max(dim=0).values,
                )
                rollout_diag["output_action_exec_mean"] += exec_batch.mean(dim=0)
                rollout_diag["output_action_exec_abs_mean"] += exec_batch.abs().mean(dim=0)
                rollout_diag["output_action_exec_abs_max"] = torch.maximum(
                    rollout_diag["output_action_exec_abs_max"],
                    exec_batch.abs().max(dim=0).values,
                )

                # 记录本步是否为time-limit截断
                if "time_outs" in info:
                    # env返回的是bool张量，存为float方便与dones一致处理
                    timeouts[step] = info["time_outs"].float().to(device)
                else:
                    timeouts[step] = 0.0

                # 直接使用环境原始奖励
                rewards[step] = raw_rewards
                
                for idx, d in enumerate(next_done):
                    if d:
                            if "done_reason_timeout" in info and bool(info["done_reason_timeout"][idx].item()):
                                episode_rewards_summary["done_timeout"] += 1
                            if "done_reason_collision" in info and bool(info["done_reason_collision"][idx].item()):
                                episode_rewards_summary["done_collision"] += 1
                            if "done_reason_success" in info and bool(info["done_reason_success"][idx].item()):
                                episode_rewards_summary["done_success"] += 1
                            if "done_reason_far" in info and bool(info["done_reason_far"][idx].item()):
                                episode_rewards_summary["done_far"] += 1

                            episodic_return = info["r"][idx].item()
                            episode_length = info["l"][idx].item()
                            # 累积episode回报
                            episode_rewards_summary["returns"].append(episodic_return)

                            # 记录到TensorBoard
                            writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                            writer.add_scalar("charts/episodic_length", episode_length, global_step)
                            # 累积 episode 时长，后续用于 avg_episode_length 统计
                            episode_rewards_summary["lengths"].append(episode_length)

                            # 累积奖励数据用于汇总显示
                            episode_rewards_summary["count"] += 1

                            for component in raw_reward_component_keys:
                                if f"r_{component}" in info:
                                    reward_value = info[f"r_{component}"][idx].item()
                                    writer.add_scalar(f"rewards/raw/{component}", reward_value, global_step)
                                    episode_rewards_summary["raw_component_returns"][component].append(reward_value)

                            for component in reward_contrib_keys:
                                if f"r_{component}" in info:
                                    reward_value = info[f"r_{component}"][idx].item()
                                    writer.add_scalar(
                                        f"rewards/contrib/{component.replace('contrib_', '', 1)}",
                                        reward_value,
                                        global_step,
                                    )
                                    episode_rewards_summary["contrib_component_returns"][component].append(reward_value)

                            for key, (tb_tag, summary_key) in EPISODE_METRIC_MAP.items():
                                if key in info:
                                    metric_value = info[key][idx].item()
                                    writer.add_scalar(tb_tag, metric_value, global_step)
                                    episode_rewards_summary[summary_key].append(metric_value)

                            if "consecutive_successes" in info:
                                writer.add_scalar(
                                    "charts/consecutive_successes", info["consecutive_successes"].item(), global_step
                                )

                            # 刷新episode数据到TensorBoard
                            writer.flush()

            # 数据收集完成，显示本轮汇总奖励信息
            if episode_rewards_summary["count"] > 0:
                print(f"\nUpdate {update} 数据收集完成 - {episode_rewards_summary['count']} 个episode结束:")

                # 计算平均值并显示
                summary_items = []
                avg_return = None
                if episode_rewards_summary["returns"]:
                    avg_return = sum(episode_rewards_summary["returns"]) / len(episode_rewards_summary["returns"])
                    summary_items.append(f"avg_return: {avg_return:.3f}")
                avg_final_dist = None
                if episode_rewards_summary["final_relative_dist"]:
                    avg_final_dist = (
                        sum(episode_rewards_summary["final_relative_dist"]) /
                        len(episode_rewards_summary["final_relative_dist"])
                    )
                    summary_items.append(f"avg_final_dist: {avg_final_dist:.1f}m")
                avg_final_forward_alignment = None
                if episode_rewards_summary["final_forward_alignment"]:
                    avg_final_forward_alignment = (
                        sum(episode_rewards_summary["final_forward_alignment"]) /
                        len(episode_rewards_summary["final_forward_alignment"])
                    )
                    summary_items.append(f"avg_align_cos: {avg_final_forward_alignment:.3f}")
                avg_min_dist = None
                if episode_rewards_summary["min_relative_dist"]:
                    avg_min_dist = (
                        sum(episode_rewards_summary["min_relative_dist"]) /
                        len(episode_rewards_summary["min_relative_dist"])
                    )
                    summary_items.append(f"avg_min_dist: {avg_min_dist:.1f}m")
                avg_speed = None
                if episode_rewards_summary["episode_mean_closing_speed"]:
                    avg_speed = (
                        sum(episode_rewards_summary["episode_mean_closing_speed"]) /
                        len(episode_rewards_summary["episode_mean_closing_speed"])
                    )
                    summary_items.append(f"avg_speed: {avg_speed:.1f}m/s")
                avg_approach_fraction = None
                if episode_rewards_summary["approach_fraction"]:
                    avg_approach_fraction = (
                        sum(episode_rewards_summary["approach_fraction"]) /
                        len(episode_rewards_summary["approach_fraction"])
                    )
                    summary_items.append(f"approach_frac: {avg_approach_fraction:.1%}")
                avg_min_hazard_clearance = None
                if episode_rewards_summary["episode_min_hazard_clearance"]:
                    avg_min_hazard_clearance = (
                        sum(episode_rewards_summary["episode_min_hazard_clearance"]) /
                        len(episode_rewards_summary["episode_min_hazard_clearance"])
                    )
                    summary_items.append(f"min_clearance: {avg_min_hazard_clearance:.1f}m")
                avg_len = None
                if episode_rewards_summary["lengths"]:
                    avg_len = sum(episode_rewards_summary["lengths"]) / len(episode_rewards_summary["lengths"])
                    summary_items.append(f"avg_len: {avg_len:.1f} steps")
                avg_visibility_metrics = {}
                for key in VISIBILITY_EPISODE_METRIC_KEYS:
                    metric_values = episode_rewards_summary[key]
                    avg_visibility_metrics[key] = (
                        sum(metric_values) / len(metric_values) if metric_values else None
                    )
                if avg_visibility_metrics["final_target_detectable"] is not None:
                    summary_items.append(
                        f"final_visible: {avg_visibility_metrics['final_target_detectable']:.1%}"
                    )
                if avg_visibility_metrics["visibility_episode_max_loss_steps"] is not None:
                    summary_items.append(
                        f"vis_max_loss: {avg_visibility_metrics['visibility_episode_max_loss_steps']:.1f} steps"
                    )

                if summary_items:
                    print(f"   平均奖励: {' | '.join(summary_items)}")

                avg_raw_component_returns = {}
                for component in raw_reward_component_keys:
                    component_values = episode_rewards_summary["raw_component_returns"][component]
                    avg_raw_component_returns[component] = (
                        sum(component_values) / len(component_values) if component_values else 0.0
                    )

                avg_contrib_component_returns = {}
                for component in reward_contrib_keys:
                    component_values = episode_rewards_summary["contrib_component_returns"][component]
                    avg_contrib_component_returns[component] = (
                        sum(component_values) / len(component_values) if component_values else 0.0
                    )

                contrib_abs_sum = sum(abs(v) for v in avg_contrib_component_returns.values()) + 1e-6
                print("   Reward真实贡献:")
                for component in reward_contrib_keys:
                    component_avg = avg_contrib_component_returns[component]
                    component_ratio = abs(component_avg) / contrib_abs_sum
                    print(
                        f"      {component.replace('contrib_', '', 1)}: {component_avg:+.3f} "
                        f"({component_ratio:.1%})"
                    )
                done_total = max(episode_rewards_summary["count"], 1)
                print(
                    "   Done原因: "
                    f"timeout={episode_rewards_summary['done_timeout']} ({episode_rewards_summary['done_timeout'] / done_total:.1%}) | "
                    f"collision={episode_rewards_summary['done_collision']} ({episode_rewards_summary['done_collision'] / done_total:.1%}) | "
                    f"success={episode_rewards_summary['done_success']} ({episode_rewards_summary['done_success'] / done_total:.1%}) | "
                    f"far={episode_rewards_summary['done_far']} ({episode_rewards_summary['done_far'] / done_total:.1%})"
                )
                print()

                # log summary
                writer.add_scalar("metrics/finished_episodes", episode_rewards_summary["count"], global_step)
                if avg_return is not None:
                    writer.add_scalar("metrics/avg_return", avg_return, global_step)
                if avg_final_dist is not None:
                    writer.add_scalar("metrics/avg_final_relative_dist", avg_final_dist, global_step)
                if avg_final_forward_alignment is not None:
                    writer.add_scalar("metrics/avg_final_forward_alignment", avg_final_forward_alignment, global_step)
                if avg_min_dist is not None:
                    writer.add_scalar("metrics/avg_min_relative_dist", avg_min_dist, global_step)
                if avg_len is not None:
                    writer.add_scalar("metrics/avg_episode_length", avg_len, global_step)
                if avg_speed is not None:
                    writer.add_scalar("metrics/avg_episode_mean_closing_speed", avg_speed, global_step)
                if avg_approach_fraction is not None:
                    writer.add_scalar("metrics/avg_approach_fraction", avg_approach_fraction, global_step)
                if avg_min_hazard_clearance is not None:
                    writer.add_scalar("metrics/avg_episode_min_hazard_clearance", avg_min_hazard_clearance, global_step)
                for key, value in avg_visibility_metrics.items():
                    if value is not None:
                        writer.add_scalar(f"metrics/{key}", value, global_step)
                writer.add_scalar("metrics/done_timeout_count", episode_rewards_summary["done_timeout"], global_step)
                writer.add_scalar("metrics/done_collision_count", episode_rewards_summary["done_collision"], global_step)
                writer.add_scalar("metrics/done_success_count", episode_rewards_summary["done_success"], global_step)
                writer.add_scalar("metrics/done_far_count", episode_rewards_summary["done_far"], global_step)
                writer.add_scalar("metrics/done_timeout_rate", episode_rewards_summary["done_timeout"] / done_total, global_step)
                writer.add_scalar("metrics/done_collision_rate", episode_rewards_summary["done_collision"] / done_total, global_step)
                writer.add_scalar("metrics/done_success_rate", episode_rewards_summary["done_success"] / done_total, global_step)
                writer.add_scalar("metrics/done_far_rate", episode_rewards_summary["done_far"] / done_total, global_step)
                for component in raw_reward_component_keys:
                    writer.add_scalar(
                        f"metrics/reward_raw_components/{component}",
                        avg_raw_component_returns[component],
                        global_step,
                    )
                for component in reward_contrib_keys:
                    writer.add_scalar(
                        f"metrics/reward_contrib/{component.replace('contrib_', '', 1)}",
                        avg_contrib_component_returns[component],
                        global_step,
                    )
                    writer.add_scalar(
                        f"metrics/reward_contrib_ratio/{component.replace('contrib_', '', 1)}",
                        abs(avg_contrib_component_returns[component]) / contrib_abs_sum,
                        global_step,
                    )
                succ_rate = None
                if episode_rewards_summary["count"] > 0:
                    succ_rate = (
                        float(episode_rewards_summary["done_success"]) /
                        float(episode_rewards_summary["count"])
                    )
                    writer.add_scalar("metrics/success_rate", succ_rate, global_step)
                current_threshold_reach_rate = None
                reach_rates = {}
                if episode_rewards_summary["min_relative_dist"]:
                    for threshold_m, threshold_tag in ((10.0, "10m"), (5.0, "5m"), (3.0, "3m"), (1.0, "1m")):
                        reach_rate = float(
                            sum(1.0 if d <= threshold_m else 0.0 for d in episode_rewards_summary["min_relative_dist"])
                        ) / float(len(episode_rewards_summary["min_relative_dist"]))
                        reach_rates[threshold_tag] = reach_rate
                        writer.add_scalar(f"metrics/reach_rate_{threshold_tag}", reach_rate, global_step)
                    if curriculum_controller is not None:
                        threshold_m = float(curriculum_controller.current_threshold)
                        current_threshold_reach_rate = float(
                            sum(1.0 if d <= threshold_m else 0.0 for d in episode_rewards_summary["min_relative_dist"])
                        ) / float(len(episode_rewards_summary["min_relative_dist"]))
                        writer.add_scalar(
                            "metrics/reach_rate_current_threshold",
                            current_threshold_reach_rate,
                            global_step,
                        )
                if avg_len is not None and args.gamma > 0.0:
                    gamma_to_avg_len = float(np.exp(avg_len * np.log(args.gamma)))
                    writer.add_scalar("diagnostics/gamma_to_avg_len", gamma_to_avg_len, global_step)

                wandb_episode_log = {
                    "metrics/finished_episodes": episode_rewards_summary["count"],
                    "metrics/done_timeout_count": episode_rewards_summary["done_timeout"],
                    "metrics/done_collision_count": episode_rewards_summary["done_collision"],
                    "metrics/done_success_count": episode_rewards_summary["done_success"],
                    "metrics/done_far_count": episode_rewards_summary["done_far"],
                    "metrics/done_timeout_rate": episode_rewards_summary["done_timeout"] / done_total,
                    "metrics/done_collision_rate": episode_rewards_summary["done_collision"] / done_total,
                    "metrics/done_success_rate": episode_rewards_summary["done_success"] / done_total,
                    "metrics/done_far_rate": episode_rewards_summary["done_far"] / done_total,
                }
                optional_episode_values = {
                    "metrics/avg_return": avg_return,
                    "metrics/avg_final_relative_dist": avg_final_dist,
                    "metrics/avg_final_forward_alignment": avg_final_forward_alignment,
                    "metrics/avg_min_relative_dist": avg_min_dist,
                    "metrics/avg_episode_length": avg_len,
                    "metrics/avg_episode_mean_closing_speed": avg_speed,
                    "metrics/avg_approach_fraction": avg_approach_fraction,
                    "metrics/avg_episode_min_hazard_clearance": avg_min_hazard_clearance,
                    "metrics/success_rate": succ_rate,
                }
                optional_episode_values.update(
                    {
                        f"metrics/{key}": value
                        for key, value in avg_visibility_metrics.items()
                    }
                )
                wandb_episode_log.update(
                    {key: float(value) for key, value in optional_episode_values.items() if value is not None}
                )
                for component in raw_reward_component_keys:
                    wandb_episode_log[f"metrics/reward_raw_components/{component}"] = float(
                        avg_raw_component_returns[component]
                    )
                for component in reward_contrib_keys:
                    component_name = component.replace("contrib_", "", 1)
                    wandb_episode_log[f"metrics/reward_contrib/{component_name}"] = float(
                        avg_contrib_component_returns[component]
                    )
                    wandb_episode_log[f"metrics/reward_contrib_ratio/{component_name}"] = float(
                        abs(avg_contrib_component_returns[component]) / contrib_abs_sum
                    )
                if reach_rates:
                    for threshold_tag, reach_rate in reach_rates.items():
                        wandb_episode_log[f"metrics/reach_rate_{threshold_tag}"] = reach_rate
                    if current_threshold_reach_rate is not None:
                        wandb_episode_log["metrics/reach_rate_current_threshold"] = current_threshold_reach_rate
                if avg_len is not None and args.gamma > 0.0:
                    wandb_episode_log["diagnostics/gamma_to_avg_len"] = gamma_to_avg_len
                wandb_log(wandb_episode_log, global_step, commit=False)

                policy_pool_episode_metrics = {
                    "finished_episodes": int(episode_rewards_summary["count"]),
                    "success_rate": succ_rate,
                    "avg_return": avg_return,
                    "avg_final_relative_dist": avg_final_dist,
                    "avg_final_forward_alignment": avg_final_forward_alignment,
                    "avg_min_relative_dist": avg_min_dist,
                    "avg_episode_length": avg_len,
                    "avg_episode_mean_closing_speed": avg_speed,
                    "avg_approach_fraction": avg_approach_fraction,
                    "avg_episode_min_hazard_clearance": avg_min_hazard_clearance,
                    "visibility_metrics": avg_visibility_metrics,
                    "done_counts": {
                        "timeout": int(episode_rewards_summary["done_timeout"]),
                        "collision": int(episode_rewards_summary["done_collision"]),
                        "success": int(episode_rewards_summary["done_success"]),
                        "far": int(episode_rewards_summary["done_far"]),
                    },
                    "done_rates": {
                        "timeout": float(episode_rewards_summary["done_timeout"] / done_total),
                        "collision": float(episode_rewards_summary["done_collision"] / done_total),
                        "success": float(episode_rewards_summary["done_success"] / done_total),
                        "far": float(episode_rewards_summary["done_far"] / done_total),
                    },
                    "reach_rates": reach_rates,
                    "reach_rate_current_threshold": current_threshold_reach_rate,
                    "raw_reward_components": avg_raw_component_returns,
                    "reward_contrib": {
                        key.replace("contrib_", "", 1): value
                        for key, value in avg_contrib_component_returns.items()
                    },
                }

                writer.flush()

                # 计算用于选取最优策略的score：
                # 低成功率阶段优先avg_return/avg_min_relative_dist，
                # 成功率抬起来后再优先success_rate与更短episode。
                score = None
                if (
                    succ_rate is not None
                    and avg_return is not None
                    and avg_min_dist is not None
                    and avg_len is not None
                    and episode_rewards_summary["count"] >= args.early_stop_min_episodes
                ):
                    score = (
                        float(succ_rate),
                        float(avg_return),
                        float(avg_min_dist),
                        float(avg_len),
                    )
                policy_pool_score = score
                last_update_score = score

                # Curriculum transition check
                if curriculum_controller is not None:
                    transition = curriculum_controller.update(
                        succ_rate=succ_rate,
                        episode_rewards_summary=episode_rewards_summary,
                    )
                    if transition is not None:
                        policy_pool_curriculum_transition = transition
                        print(
                            f"\n{'='*60}\n"
                            f"CURRICULUM TRANSITION\n"
                            f"  Stage {transition['stage_idx']-1} -> Stage {transition['stage_idx']}\n"
                            f"  Threshold: {transition['old_threshold']:.1f}m -> {transition['new_threshold']:.1f}m\n"
                            f"  Visibility weight scale: {transition['old_visibility_reward_weight_scale']:.3f} -> "
                            f"{transition['new_visibility_reward_weight_scale']:.3f}\n"
                            f"  Success gate: sr={transition['succ_rate']:.3f} "
                            f"| episodes={transition['finished_episodes']} "
                            f"| required={transition['stable_success_rate']:.3f}"
                            f"x{transition['stable_success_updates']}\n"
                            f"{'='*60}\n"
                        )
                        no_improve_updates = 0
                    # Log curriculum metrics
                    curriculum_controller.log_curriculum_metrics(
                        writer, global_step
                    )

            # bootstrap value if not done
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done.float()
                        nextvalues = next_value
                        # TimeLimit引导：若因时间上限而done，则不视为终止，允许bootstrap
                        nextnonterminal = torch.where(timeouts[t] > 0.5, torch.ones_like(nextnonterminal), nextnonterminal)
                    else:
                        nextnonterminal = 1.0 - dones[t + 1].float()
                        nextvalues = values[t + 1]
                        # TimeLimit引导：若下一步是超时done，则不视为终止
                        nextnonterminal = torch.where(timeouts[t + 1] > 0.5, torch.ones_like(nextnonterminal), nextnonterminal)
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
                reward_mean = rewards.mean().item()
                reward_std = rewards.std(unbiased=False).item()

            # flatten the batch
            b_obs = obs.reshape((-1, envs.num_obs)) # 将观测数据 obs 重塑为二维张量，其中每一行是一个观测
            b_logprobs = logprobs.reshape(-1) # 将动作的对数概率 logprobs 展平成一维张量
            b_actions = actions.reshape((-1, envs.num_actions)) # 将动作数据 actions 重塑为二维张量
            b_advantages = advantages.reshape(-1) #  将优势函数 advantages 展平成一维张量
            b_returns = returns.reshape(-1) # 将回报 returns 展平成一维张量
            b_values = values.reshape(-1) # 将值函数 values 展平成一维张量
            adv_mean = b_advantages.mean().item()
            adv_std = b_advantages.std(unbiased=False).item()
            adv_abs_mean = b_advantages.abs().mean().item()
            return_mean = b_returns.mean().item()
            return_std = b_returns.std(unbiased=False).item()
            value_mean = b_values.mean().item()
            value_std = b_values.std(unbiased=False).item()
            returns_var = torch.var(b_returns, unbiased=False)
            if returns_var.item() < 1e-8:
                explained_variance = float("nan")
            else:
                explained_variance = (
                    1.0 - torch.var(b_returns - b_values, unbiased=False) / returns_var
                ).item()

            # Optimizing the policy and value network
            clipfracs = []
            grad_norm_sum = 0.0
            grad_norm_count = 0
            for epoch in range(args.update_epochs):
                b_inds = torch.randperm(args.batch_size, device=device) # 随机打乱批次索引。这有助于打破数据的顺序，防止过拟合
                for start in range(0, args.batch_size, args.minibatch_size): # 生成从 0 到 batch_size 的起始索引，索引以 minibatch_size 为步长递增
                    end = start + args.minibatch_size # 计算每个小批次的结束索引
                    mb_inds = b_inds[start:end] # 从打乱的批次索引中选择一个小批次的索引
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std(unbiased=False) + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - current_ent_coef * entropy_loss + v_loss * args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    grad_norm_sum += float(grad_norm)
                    grad_norm_count += 1
                    optimizer.step()

                if args.target_kl is not None:
                    if approx_kl > args.target_kl:
                        break

            if not args.play:
                with torch.no_grad():
                    # Keep obs normalization fixed within a rollout/update pair so
                    # PPO ratios compare log-probs under the same preprocessing.
                    agent.update_obs_rms(obs[1:].reshape(-1, envs.num_obs))
                    agent.update_obs_rms(next_obs)

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            current_lr = optimizer.param_groups[0]["lr"]
            sps = int(global_step / (time.time() - start_time))
            avg_grad_norm = grad_norm_sum / max(grad_norm_count, 1)
            rollout_count = max(rollout_diag["count"], 1)
            current_actor_std = torch.exp(
                torch.clamp(agent.actor_logstd, min=agent.log_std_min, max=agent.log_std_max)
            ).mean().item()

            # LR scheduler details
            lr_info = {}
            if args.use_lr_scheduler:
                # 计算调度器状态（基于global_step）
                is_warmup = global_step < args.warmup_steps
                warmup_progress = min(global_step / args.warmup_steps, 1.0) if args.warmup_steps > 0 else 1.0
                total_progress = global_step / args.total_timesteps

                lr_info = {
                    "lr_scheduler/is_warmup": float(is_warmup),
                    "lr_scheduler/warmup_progress": warmup_progress,
                    "lr_scheduler/total_progress": total_progress,
                    "lr_scheduler/warmup_steps": args.warmup_steps,
                    "lr_scheduler/current_step": global_step,
                    "lr_scheduler/total_steps": args.total_timesteps,
                }

                # 记录到TensorBoard
                for key, value in lr_info.items():
                    writer.add_scalar(key, value, global_step)

            writer.add_scalar("charts/learning_rate", current_lr, global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("charts/SPS", sps, global_step)
            writer.add_scalar("charts/entropy_coef", current_ent_coef, global_step)
            writer.add_scalar("network/effective_actor_std", current_actor_std, global_step)
            writer.add_scalar(
                "network/policy_mean_abs_mean",
                rollout_diag["policy_mean_abs_mean"] / rollout_count,
                global_step,
            )
            writer.add_scalar(
                "network/policy_mean_abs_max",
                rollout_diag["policy_mean_abs_max"],
                global_step,
            )
            writer.add_scalar(
                "network/pre_tanh_action_abs_mean",
                rollout_diag["pre_tanh_action_abs_mean"] / rollout_count,
                global_step,
            )
            writer.add_scalar(
                "network/pre_tanh_action_abs_max",
                rollout_diag["pre_tanh_action_abs_max"],
                global_step,
            )
            writer.add_scalar(
                "network/pre_tanh_action_std",
                rollout_diag["pre_tanh_action_std"] / rollout_count,
                global_step,
            )
            writer.add_scalar(
                "network/pre_tanh_action_oob_ratio",
                rollout_diag["pre_tanh_action_oob_ratio"] / rollout_count,
                global_step,
            )
            writer.add_scalar(
                "network/squashed_action_sat_ratio",
                rollout_diag["squashed_action_sat_ratio"] / rollout_count,
                global_step,
            )
            for dim in range(envs.num_actions):
                writer.add_scalar(
                    f"network/squashed_sat_dim_{dim}",
                    (rollout_diag["squashed_sat_per_dim"][dim] / rollout_count).item(),
                    global_step,
                )
            for dim, name in enumerate(output_action_command_names):
                writer.add_scalar(
                    f"outpu_action/command/mean_{name}",
                    (rollout_diag["output_action_command_mean"][dim] / rollout_count).item(),
                    global_step,
                )
                writer.add_scalar(
                    f"outpu_action/command/abs_mean_{name}",
                    (rollout_diag["output_action_command_abs_mean"][dim] / rollout_count).item(),
                    global_step,
                )
                writer.add_scalar(
                    f"outpu_action/command/abs_max_{name}",
                    rollout_diag["output_action_command_abs_max"][dim].item(),
                    global_step,
                )
            for dim, name in enumerate(OUTPUT_ACTION_EXEC_NAMES):
                writer.add_scalar(
                    f"outpu_action/exec/mean_{name}",
                    (rollout_diag["output_action_exec_mean"][dim] / rollout_count).item(),
                    global_step,
                )
                writer.add_scalar(
                    f"outpu_action/exec/abs_mean_{name}",
                    (rollout_diag["output_action_exec_abs_mean"][dim] / rollout_count).item(),
                    global_step,
                )
                writer.add_scalar(
                    f"outpu_action/exec/abs_max_{name}",
                    rollout_diag["output_action_exec_abs_max"][dim].item(),
                    global_step,
                )

            writer.add_scalar("diagnostics/reward_mean", reward_mean, global_step)
            writer.add_scalar("diagnostics/reward_std", reward_std, global_step)
            writer.add_scalar("diagnostics/adv_mean", adv_mean, global_step)
            writer.add_scalar("diagnostics/adv_std", adv_std, global_step)
            writer.add_scalar("diagnostics/adv_abs_mean", adv_abs_mean, global_step)
            writer.add_scalar("diagnostics/return_mean", return_mean, global_step)
            writer.add_scalar("diagnostics/return_std", return_std, global_step)
            writer.add_scalar("diagnostics/value_mean", value_mean, global_step)
            writer.add_scalar("diagnostics/value_std", value_std, global_step)
            writer.add_scalar("diagnostics/explained_variance", explained_variance, global_step)
            writer.add_scalar("diagnostics/grad_norm", avg_grad_norm, global_step)
            wandb_update_log = {
                "charts/learning_rate": float(current_lr),
                "losses/value_loss": float(v_loss.item()),
                "losses/policy_loss": float(pg_loss.item()),
                "losses/entropy": float(entropy_loss.item()),
                "losses/old_approx_kl": float(old_approx_kl.item()),
                "losses/approx_kl": float(approx_kl.item()),
                "losses/clipfrac": float(np.mean(clipfracs)),
                "charts/SPS": float(sps),
                "charts/entropy_coef": float(current_ent_coef),
                "network/effective_actor_std": float(current_actor_std),
                "network/policy_mean_abs_mean": float(rollout_diag["policy_mean_abs_mean"] / rollout_count),
                "network/policy_mean_abs_max": float(rollout_diag["policy_mean_abs_max"]),
                "network/pre_tanh_action_abs_mean": float(rollout_diag["pre_tanh_action_abs_mean"] / rollout_count),
                "network/pre_tanh_action_abs_max": float(rollout_diag["pre_tanh_action_abs_max"]),
                "network/pre_tanh_action_std": float(rollout_diag["pre_tanh_action_std"] / rollout_count),
                "network/pre_tanh_action_oob_ratio": float(rollout_diag["pre_tanh_action_oob_ratio"] / rollout_count),
                "network/squashed_action_sat_ratio": float(rollout_diag["squashed_action_sat_ratio"] / rollout_count),
                "diagnostics/reward_mean": float(reward_mean),
                "diagnostics/reward_std": float(reward_std),
                "diagnostics/adv_mean": float(adv_mean),
                "diagnostics/adv_std": float(adv_std),
                "diagnostics/adv_abs_mean": float(adv_abs_mean),
                "diagnostics/return_mean": float(return_mean),
                "diagnostics/return_std": float(return_std),
                "diagnostics/value_mean": float(value_mean),
                "diagnostics/value_std": float(value_std),
                "diagnostics/explained_variance": float(explained_variance),
                "diagnostics/grad_norm": float(avg_grad_norm),
            }
            wandb_update_log.update({key: float(value) for key, value in lr_info.items()})
            for dim in range(envs.num_actions):
                wandb_update_log[f"network/squashed_sat_dim_{dim}"] = float(
                    (rollout_diag["squashed_sat_per_dim"][dim] / rollout_count).item()
                )
            for dim, name in enumerate(output_action_command_names):
                wandb_update_log[f"outpu_action/command/mean_{name}"] = float(
                    (rollout_diag["output_action_command_mean"][dim] / rollout_count).item()
                )
                wandb_update_log[f"outpu_action/command/abs_mean_{name}"] = float(
                    (rollout_diag["output_action_command_abs_mean"][dim] / rollout_count).item()
                )
                wandb_update_log[f"outpu_action/command/abs_max_{name}"] = float(
                    rollout_diag["output_action_command_abs_max"][dim].item()
                )
            for dim, name in enumerate(OUTPUT_ACTION_EXEC_NAMES):
                wandb_update_log[f"outpu_action/exec/mean_{name}"] = float(
                    (rollout_diag["output_action_exec_mean"][dim] / rollout_count).item()
                )
                wandb_update_log[f"outpu_action/exec/abs_mean_{name}"] = float(
                    (rollout_diag["output_action_exec_abs_mean"][dim] / rollout_count).item()
                )
                wandb_update_log[f"outpu_action/exec/abs_max_{name}"] = float(
                    rollout_diag["output_action_exec_abs_max"][dim].item()
                )
            wandb_log(wandb_update_log, global_step, commit=True)
            writer.flush()

            # update summary
            lr_status = ""
            if args.use_lr_scheduler and lr_info:
                if lr_info["lr_scheduler/is_warmup"]:
                    lr_status = f" [Warmup {lr_info['lr_scheduler/warmup_progress']:.1%}]"
                else:
                    lr_status = f" [Cosine {lr_info['lr_scheduler/total_progress']:.1%}]"

            print(f"Update {update:4d} | SPS: {sps:5d} | LR: {current_lr:.2e}{lr_status} | "
                  f"V_loss: {v_loss.item():.4f} | P_loss: {pg_loss.item():.4f} | "
                  f"KL: {approx_kl.item():.4f}")

            improved = False
            if last_update_score is not None:
                improved = is_better_score(
                    last_update_score,
                    best_score,
                    args.early_stop_success_delta,
                    args.early_stop_length_delta,
                )
                if improved:
                    best_score = last_update_score
                    best_update = update
                    no_improve_updates = 0
                elif best_score is not None:
                    no_improve_updates += 1

            # save best
            if args.save_best and improved:
                    torch.save(agent.state_dict(), best_path)
                    score_text = format_score(best_score)
                    print(
                        f"New best model saved to {best_path} | metric=hybrid_success_then_return_fallback "
                        f"| score={score_text}"
                    )
                    writer.add_scalar("best/success_rate", best_score[0], global_step)
                    writer.add_scalar("best/avg_return", best_score[1], global_step)
                    writer.add_scalar("best/avg_min_relative_dist", best_score[2], global_step)
                    writer.add_scalar("best/avg_episode_length", best_score[3], global_step)
                    writer.add_scalar("best/update", update, global_step)

            latest_checkpoint = build_training_checkpoint(
                agent=agent,
                optimizer=optimizer,
                args=args,
                global_step=global_step,
                update=update,
                best_score=best_score,
                best_update=best_update,
                no_improve_updates=no_improve_updates,
                run_name=run_name,
                run_dir=run_dir,
                wandb_run_id=wandb_run_id,
            )
            if curriculum_controller is not None:
                latest_checkpoint["curriculum"] = curriculum_controller.get_state_dict()
            torch.save(latest_checkpoint, latest_path)

            if should_save_policy_pool_checkpoint(
                args, update, start_update, policy_pool_curriculum_transition
            ):
                policy_pool_checkpoint_path = os.path.join(
                    policy_pool_dir,
                    f"ppo_upd_{update:06d}_step_{global_step:010d}.pth",
                )
                policy_pool_reasons = policy_pool_save_reason(
                    args, update, start_update, policy_pool_curriculum_transition
                )
                policy_pool_metadata = build_policy_pool_metadata(
                    args=args,
                    run_name=run_name,
                    run_dir=run_dir,
                    checkpoint_path=policy_pool_checkpoint_path,
                    global_step=global_step,
                    update=update,
                    start_update=start_update,
                    score=policy_pool_score,
                    episode_metrics=policy_pool_episode_metrics,
                    update_metrics=wandb_update_log,
                    curriculum_controller=curriculum_controller,
                    curriculum_transition=policy_pool_curriculum_transition,
                    reasons=policy_pool_reasons,
                )
                policy_pool_checkpoint = dict(latest_checkpoint)
                policy_pool_checkpoint["policy_pool_metadata"] = policy_pool_metadata
                torch.save(policy_pool_checkpoint, policy_pool_checkpoint_path)
                append_jsonl(policy_pool_manifest_path, policy_pool_metadata)
                writer.add_scalar("policy_pool/saved_update", float(update), global_step)
                print(
                    f"Policy pool checkpoint saved: {policy_pool_checkpoint_path} "
                    f"| reasons={','.join(policy_pool_reasons)}"
                )

            if args.early_stop:
                # When curriculum is active, only apply early stop on the final stage
                if curriculum_controller is not None and not curriculum_controller.is_on_final_stage():
                    writer.add_scalar("early_stop/armed", 0.0, global_step)
                    writer.add_scalar("early_stop/no_improve_updates", 0, global_step)
                elif last_update_score is None:
                    print(
                        f"Early-stop check skipped at update {update}: "
                        f"finished_episodes < {args.early_stop_min_episodes}"
                    )
                else:
                    early_stop_armed = (
                        best_score is not None
                        and best_score[0] >= args.early_stop_start_success_rate
                    )
                    writer.add_scalar(
                        "early_stop/armed",
                        float(early_stop_armed),
                        global_step,
                    )
                    writer.add_scalar("early_stop/no_improve_updates", no_improve_updates, global_step)
                    if not early_stop_armed:
                        best_success_text = "N/A" if best_score is None else f"{best_score[0]:.3f}"
                        print(
                            f"Early-stop armed=FALSE at update {update}: "
                            f"best_success_rate={best_success_text} < {args.early_stop_start_success_rate:.3f}"
                        )
                    elif no_improve_updates >= args.early_stop_patience:
                        print(
                            f"Early stopping triggered at update {update}. "
                            f"Best update={best_update}, best_score={format_score(best_score)}, "
                            f"patience={args.early_stop_patience}"
                        )
                        break
        # 训练结束兜底：若从未达到保存条件且未生成best.pth，保存当前权重为best
        if args.save_best:
            try:
                if not os.path.exists(best_path):
                    torch.save(agent.state_dict(), best_path)
                    print(f"No best checkpoint found during training. Fallback saved to {best_path}.")
            except Exception as e:
                print(f"Failed to write fallback best checkpoint: {e}")

    else:
        # --- PLAY/EVAL: rollout one episode and plot trajectories ---
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Rectangle
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - required for 3D projection
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        import numpy as np

        base_env = envs.env

        # `--play` 复用 checkpoint 所在目录保存回测产物，不创建新的 run 目录。
        ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint)) if args.checkpoint else os.getcwd()
        play_output_stem = (
            f"play_trajectory_{args.target_asset_type}"
            if args.target_asset_type is not None
            else "play_trajectory"
        )

        def _extract_positions(env, env_idx):
            attacker_pos = env.robot_state[env_idx, 0:3].detach().cpu().numpy()
            target_pos = env.target_state[env_idx, 0:3].detach().cpu().numpy()
            return attacker_pos, target_pos

        def _extract_terminal_positions(infos, env_idx):
            terminal_valid = infos.get("terminal_valid", None)
            if terminal_valid is None or not bool(terminal_valid[env_idx].item()):
                return None

            attacker_pos = infos["terminal_actor1_pos"][env_idx].detach().cpu().numpy()
            target_pos = infos["terminal_actor2_pos"][env_idx].detach().cpu().numpy()
            target_cmd = infos["terminal_target_cmd"][env_idx].detach().cpu().numpy()
            return attacker_pos, target_pos, target_cmd

        def _get_room_bounds(env, attacker_xyz, target_xyz, target_cmd_xyz):
            room_cfg = getattr(env.task_config, "room", None)
            if room_cfg is not None and bool(getattr(room_cfg, "enabled", False)):
                room_size = np.asarray(getattr(room_cfg, "size", [0.0, 0.0, 0.0]), dtype=np.float32)
                return (0.0, room_size[0], 0.0, room_size[1], 0.0, room_size[2])

            all_points = np.vstack([attacker_xyz, target_xyz, target_cmd_xyz])
            mins = all_points.min(axis=0)
            maxs = all_points.max(axis=0)
            padding = 5.0
            return (
                mins[0] - padding,
                maxs[0] + padding,
                mins[1] - padding,
                maxs[1] + padding,
                mins[2] - padding,
                maxs[2] + padding,
            )

        def _draw_room_faces(ax, bounds):
            x0, x1, y0, y1, z0, z1 = bounds
            faces = [
                [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
                [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
                [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
                [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],
                [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],
                [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
            ]
            face_collection = Poly3DCollection(
                faces,
                facecolors=(0.75, 0.75, 0.75, 0.08),
                edgecolors=(0.45, 0.45, 0.45, 0.35),
                linewidths=0.6,
            )
            ax.add_collection3d(face_collection)

        def _draw_obstacles_3d(ax, obstacle_poses, radius, height):
            if not obstacle_poses or radius <= 0.0 or height <= 0.0:
                return
            theta = np.linspace(0.0, 2.0 * np.pi, 30)
            z_line = np.linspace(0.0, height, 2)
            theta_grid, z_grid = np.meshgrid(theta, z_line)
            for obstacle_pose in obstacle_poses:
                cx, cy, cz = obstacle_pose["position"]
                base_z = cz - height / 2.0
                x_grid = cx + radius * np.cos(theta_grid)
                y_grid = cy + radius * np.sin(theta_grid)
                ax.plot_surface(
                    x_grid,
                    y_grid,
                    base_z + z_grid,
                    color=obstacle_pose.get("color", [0.55, 0.55, 0.55]),
                    alpha=0.18,
                    linewidth=0.0,
                    shade=False,
                )

        def _plot_projection(ax, plane, bounds, obstacle_poses, radius, height, attacker_xyz, target_xyz, target_cmd_xyz):
            x0, x1, y0, y1, z0, z1 = bounds
            if plane == "top":
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="0.4", linewidth=1.5))
                for obstacle_pose in obstacle_poses:
                    cx, cy, _ = obstacle_pose["position"]
                    ax.add_patch(
                        Circle(
                            (cx, cy),
                            radius,
                            facecolor=obstacle_pose.get("color", [0.55, 0.55, 0.55]),
                            edgecolor="0.25",
                            alpha=0.25,
                        )
                    )
                ax.plot(attacker_xyz[:, 0], attacker_xyz[:, 1], color="tab:blue", linewidth=2.0, label="attacker actual")
                ax.plot(target_xyz[:, 0], target_xyz[:, 1], color="tab:orange", linewidth=1.8, label="target actual")
                ax.plot(target_cmd_xyz[:, 0], target_cmd_xyz[:, 1], color="tab:green", linestyle="--", linewidth=1.6, label="target desired")
                ax.set_xlabel("x [m]")
                ax.set_ylabel("y [m]")
                ax.set_title("Top View (XY)")
            elif plane == "side":
                ax.add_patch(Rectangle((x0, z0), x1 - x0, z1 - z0, fill=False, edgecolor="0.4", linewidth=1.5))
                for obstacle_pose in obstacle_poses:
                    cx, _, cz = obstacle_pose["position"]
                    ax.add_patch(
                        Rectangle(
                            (cx - radius, cz - height / 2.0),
                            2.0 * radius,
                            height,
                            facecolor=obstacle_pose.get("color", [0.55, 0.55, 0.55]),
                            edgecolor="0.25",
                            alpha=0.25,
                        )
                    )
                ax.plot(attacker_xyz[:, 0], attacker_xyz[:, 2], color="tab:blue", linewidth=2.0, label="attacker actual")
                ax.plot(target_xyz[:, 0], target_xyz[:, 2], color="tab:orange", linewidth=1.8, label="target actual")
                ax.plot(target_cmd_xyz[:, 0], target_cmd_xyz[:, 2], color="tab:green", linestyle="--", linewidth=1.6, label="target desired")
                ax.set_xlabel("x [m]")
                ax.set_ylabel("z [m]")
                ax.set_title("Side View (XZ)")
            else:
                ax.add_patch(Rectangle((y0, z0), y1 - y0, z1 - z0, fill=False, edgecolor="0.4", linewidth=1.5))
                for obstacle_pose in obstacle_poses:
                    _, cy, cz = obstacle_pose["position"]
                    ax.add_patch(
                        Rectangle(
                            (cy - radius, cz - height / 2.0),
                            2.0 * radius,
                            height,
                            facecolor=obstacle_pose.get("color", [0.55, 0.55, 0.55]),
                            edgecolor="0.25",
                            alpha=0.25,
                        )
                    )
                ax.plot(attacker_xyz[:, 1], attacker_xyz[:, 2], color="tab:blue", linewidth=2.0, label="attacker actual")
                ax.plot(target_xyz[:, 1], target_xyz[:, 2], color="tab:orange", linewidth=1.8, label="target actual")
                ax.plot(target_cmd_xyz[:, 1], target_cmd_xyz[:, 2], color="tab:green", linestyle="--", linewidth=1.6, label="target desired")
                ax.set_xlabel("y [m]")
                ax.set_ylabel("z [m]")
                ax.set_title("Front View (YZ)")

            ax.grid(True, alpha=0.25)
            ax.set_aspect("equal", adjustable="box")
            ax.legend(loc="best")

        def _set_axes_equal(ax, bounds):
            x0, x1, y0, y1, z0, z1 = bounds
            x_mid = 0.5 * (x0 + x1)
            y_mid = 0.5 * (y0 + y1)
            z_mid = 0.5 * (z0 + z1)
            max_range = max(x1 - x0, y1 - y0, z1 - z0, 1e-6)
            ax.set_xlim(x_mid - max_range / 2.0, x_mid + max_range / 2.0)
            ax.set_ylim(y_mid - max_range / 2.0, y_mid + max_range / 2.0)
            ax.set_zlim(z_mid - max_range / 2.0, z_mid + max_range / 2.0)

        def _tensor_row(tensor, row_idx, cols=None):
            if cols is None:
                return tensor[row_idx].detach().cpu().numpy().copy()
            return tensor[row_idx, cols].detach().cpu().numpy().copy()

        idx = 0  # log the first env if multiple
        episode_len_steps = int(getattr(base_env.task_config, "episode_len_steps", 2000))
        max_steps = min(int(args.play_steps), episode_len_steps)
        dt = float(getattr(base_env, "dt", 0.01))

        t_list = []
        attacker_xyz = []
        attacker_quat = []
        attacker_linvel = []
        attacker_body_linvel = []
        attacker_body_angvel = []
        target_xyz = []
        target_quat = []
        target_linvel = []
        target_angvel = []
        target_cmd_xyz = []
        target_cmd_yaw = []
        rel_d = []
        reward_log = []
        done_log = []
        raw_action_log = []
        policy_mean_log = []
        policy_std_log = []
        pre_tanh_action_log = []
        squashed_action_log = []
        actor1_command_log = []
        actor1_exec_log = []
        target_force_log = []
        target_torque_log = []

        done_reached = False
        for step in range(max_steps):
            with torch.no_grad():
                action, _, _, _, policy_aux = agent.get_action_and_value(
                    next_obs, return_aux=True
                )
            next_obs, rewards, next_done, info = envs.step(action)

            raw_action_log.append(_tensor_row(action, idx))
            policy_mean_log.append(_tensor_row(policy_aux["action_mean"], idx))
            policy_std_log.append(_tensor_row(policy_aux["action_std"], idx))
            pre_tanh_action_log.append(_tensor_row(policy_aux["pre_tanh_action"], idx))
            squashed_action_log.append(_tensor_row(policy_aux["squashed_action"], idx))
            actor1_command_log.append(_tensor_row(base_env.last_actor1_attitude_command, idx))
            actor1_exec_log.append(
                np.concatenate(
                    (
                        np.array(
                            [
                                base_env.last_actor1_output_thrust_normalized[idx].detach().cpu().item(),
                                base_env.last_actor1_force_z[idx].detach().cpu().item(),
                            ],
                            dtype=np.float32,
                        ),
                        _tensor_row(base_env.last_actor1_output_torques, idx),
                    )
                )
            )
            target_force_log.append(_tensor_row(base_env.target_force_tensor[:, 0, :], idx))
            target_torque_log.append(_tensor_row(base_env.target_torque_tensor[:, 0, :], idx))
            reward_log.append(float(rewards[idx].detach().cpu().item()))
            done_log.append(float(next_done[idx].detach().cpu().item()))

            attacker_quat.append(_tensor_row(base_env.robot_state, idx, slice(3, 7)))
            attacker_linvel.append(_tensor_row(base_env.robot_state, idx, slice(7, 10)))
            attacker_body_linvel.append(_tensor_row(base_env.obs_dict["robot_body_linvel"], idx))
            attacker_body_angvel.append(_tensor_row(base_env.obs_dict["robot_body_angvel"], idx))
            target_quat.append(_tensor_row(base_env.target_state, idx, slice(3, 7)))
            target_linvel.append(_tensor_row(base_env.target_state, idx, slice(7, 10)))
            target_angvel.append(_tensor_row(base_env.target_state, idx, slice(10, 13)))
            target_cmd_yaw.append(float(base_env.target_command[idx, 3].detach().cpu().item()))

            if next_done[idx].item() == 1:
                terminal_sample = _extract_terminal_positions(info, idx)
                if terminal_sample is not None:
                    attacker_pos, target_pos, target_cmd = terminal_sample
                    attacker_xyz.append(attacker_pos.copy())
                    target_xyz.append(target_pos.copy())
                    target_cmd_xyz.append(target_cmd.copy())
                    rel_d.append(float(np.linalg.norm(target_pos - attacker_pos)))
                    t_list.append((step + 1) * dt)
                done_reached = True
                break

            attacker_pos, target_pos = _extract_positions(base_env, idx)
            target_cmd = base_env.target_command[idx, 0:3].detach().cpu().numpy()

            attacker_xyz.append(attacker_pos.copy())
            target_xyz.append(target_pos.copy())
            target_cmd_xyz.append(target_cmd.copy())
            rel_d.append(float(np.linalg.norm(target_pos - attacker_pos)))
            t_list.append((step + 1) * dt)

        t = np.array(t_list)
        attacker_xyz = np.array(attacker_xyz)
        attacker_quat = np.array(attacker_quat)
        attacker_linvel = np.array(attacker_linvel)
        attacker_body_linvel = np.array(attacker_body_linvel)
        attacker_body_angvel = np.array(attacker_body_angvel)
        target_xyz = np.array(target_xyz)
        target_quat = np.array(target_quat)
        target_linvel = np.array(target_linvel)
        target_angvel = np.array(target_angvel)
        target_cmd_xyz = np.array(target_cmd_xyz)
        target_cmd_yaw = np.array(target_cmd_yaw)
        rel_d = np.array(rel_d)
        reward_log = np.array(reward_log)
        done_log = np.array(done_log)
        raw_action_log = np.array(raw_action_log)
        policy_mean_log = np.array(policy_mean_log)
        policy_std_log = np.array(policy_std_log)
        pre_tanh_action_log = np.array(pre_tanh_action_log)
        squashed_action_log = np.array(squashed_action_log)
        actor1_command_log = np.array(actor1_command_log)
        actor1_exec_log = np.array(actor1_exec_log)
        target_force_log = np.array(target_force_log)
        target_torque_log = np.array(target_torque_log)
        sample_count = len(t)
        attacker_quat = attacker_quat[:sample_count]
        attacker_linvel = attacker_linvel[:sample_count]
        attacker_body_linvel = attacker_body_linvel[:sample_count]
        attacker_body_angvel = attacker_body_angvel[:sample_count]
        target_quat = target_quat[:sample_count]
        target_linvel = target_linvel[:sample_count]
        target_angvel = target_angvel[:sample_count]
        target_cmd_yaw = target_cmd_yaw[:sample_count]
        reward_log = reward_log[:sample_count]
        done_log = done_log[:sample_count]
        raw_action_log = raw_action_log[:sample_count]
        policy_mean_log = policy_mean_log[:sample_count]
        policy_std_log = policy_std_log[:sample_count]
        pre_tanh_action_log = pre_tanh_action_log[:sample_count]
        squashed_action_log = squashed_action_log[:sample_count]
        actor1_command_log = actor1_command_log[:sample_count]
        actor1_exec_log = actor1_exec_log[:sample_count]
        target_force_log = target_force_log[:sample_count]
        target_torque_log = target_torque_log[:sample_count]

        if attacker_xyz.size == 0 or target_xyz.size == 0 or target_cmd_xyz.size == 0:
            raise RuntimeError("No rollout samples were collected during --play")

        final_d = float(rel_d[-1]) if len(rel_d) else float("nan")
        min_d = float(np.min(rel_d)) if len(rel_d) else float("nan")

        room_bounds = _get_room_bounds(base_env, attacker_xyz, target_xyz, target_cmd_xyz)
        obstacle_cfg = getattr(base_env.task_config, "static_obstacles", None)
        obstacle_positions = list(getattr(obstacle_cfg, "positions", [])) if obstacle_cfg is not None else []
        obstacle_colors = list(getattr(obstacle_cfg, "colors", [])) if obstacle_cfg is not None else []
        obstacle_poses = [
            {
                "position": obstacle_positions[i],
                "color": obstacle_colors[i] if i < len(obstacle_colors) else [0.55, 0.55, 0.55],
            }
            for i in range(len(obstacle_positions))
        ]
        obstacle_radius = float(getattr(base_env, "static_obstacle_radius", 0.0))
        obstacle_height = room_bounds[5] - room_bounds[4]

        fig = plt.figure(figsize=(18, 12))
        ax_scene = fig.add_subplot(2, 2, 1, projection="3d")
        _draw_room_faces(ax_scene, room_bounds)
        _draw_obstacles_3d(ax_scene, obstacle_poses, obstacle_radius, obstacle_height)
        ax_scene.plot(attacker_xyz[:, 0], attacker_xyz[:, 1], attacker_xyz[:, 2], color="tab:blue", linewidth=2.0, label="attacker actual")
        ax_scene.plot(target_xyz[:, 0], target_xyz[:, 1], target_xyz[:, 2], color="tab:orange", linewidth=1.8, label="target actual")
        ax_scene.plot(target_cmd_xyz[:, 0], target_cmd_xyz[:, 1], target_cmd_xyz[:, 2], color="tab:green", linestyle="--", linewidth=1.6, label="target desired")
        ax_scene.scatter(attacker_xyz[0, 0], attacker_xyz[0, 1], attacker_xyz[0, 2], c="tab:blue", marker="o", s=22)
        ax_scene.scatter(attacker_xyz[-1, 0], attacker_xyz[-1, 1], attacker_xyz[-1, 2], c="tab:blue", marker="x", s=40)
        ax_scene.scatter(target_xyz[0, 0], target_xyz[0, 1], target_xyz[0, 2], c="tab:orange", marker="o", s=22)
        ax_scene.scatter(target_xyz[-1, 0], target_xyz[-1, 1], target_xyz[-1, 2], c="tab:orange", marker="x", s=40)
        ax_scene.set_xlabel("x [m]")
        ax_scene.set_ylabel("y [m]")
        ax_scene.set_zlabel("z [m]")
        ax_scene.set_title("Perspective View")
        ax_scene.legend(loc="upper left")
        _set_axes_equal(ax_scene, room_bounds)

        _plot_projection(fig.add_subplot(2, 2, 2), "top", room_bounds, obstacle_poses, obstacle_radius, obstacle_height, attacker_xyz, target_xyz, target_cmd_xyz)
        _plot_projection(fig.add_subplot(2, 2, 3), "side", room_bounds, obstacle_poses, obstacle_radius, obstacle_height, attacker_xyz, target_xyz, target_cmd_xyz)
        _plot_projection(fig.add_subplot(2, 2, 4), "front", room_bounds, obstacle_poses, obstacle_radius, obstacle_height, attacker_xyz, target_xyz, target_cmd_xyz)

        st_done = "done" if done_reached else "timeout"
        fig.suptitle(f"Play PPO Guidance | steps={len(t)} ({st_done}), final d={final_d:.2f} m, min d={min_d:.2f} m")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        out_png = os.path.join(ckpt_dir, f"{play_output_stem}.png")
        fig.savefig(out_png, dpi=150)
        plt.close(fig)

        fig3d = plt.figure(figsize=(8, 7))
        ax3d = fig3d.add_subplot(111, projection="3d")
        _draw_room_faces(ax3d, room_bounds)
        _draw_obstacles_3d(ax3d, obstacle_poses, obstacle_radius, obstacle_height)
        ax3d.plot(attacker_xyz[:, 0], attacker_xyz[:, 1], attacker_xyz[:, 2], color="tab:blue", linewidth=2.0, label="attacker actual")
        ax3d.plot(target_xyz[:, 0], target_xyz[:, 1], target_xyz[:, 2], color="tab:orange", linewidth=1.8, label="target actual")
        ax3d.plot(target_cmd_xyz[:, 0], target_cmd_xyz[:, 1], target_cmd_xyz[:, 2], color="tab:green", linestyle="--", linewidth=1.6, label="target desired")
        ax3d.scatter(attacker_xyz[0, 0], attacker_xyz[0, 1], attacker_xyz[0, 2], c="tab:blue", marker="o", s=20)
        ax3d.scatter(attacker_xyz[-1, 0], attacker_xyz[-1, 1], attacker_xyz[-1, 2], c="tab:blue", marker="x", s=30)
        ax3d.scatter(target_xyz[0, 0], target_xyz[0, 1], target_xyz[0, 2], c="tab:orange", marker="o", s=20)
        ax3d.scatter(target_xyz[-1, 0], target_xyz[-1, 1], target_xyz[-1, 2], c="tab:orange", marker="x", s=30)
        ax3d.set_xlabel("x [m]")
        ax3d.set_ylabel("y [m]")
        ax3d.set_zlabel("z [m]")
        ax3d.set_title(f"3D Trajectory | steps={len(t)} ({st_done})")
        ax3d.legend()
        _set_axes_equal(ax3d, room_bounds)

        out_png3d = os.path.join(ckpt_dir, f"{play_output_stem}_3d.png")
        fig3d.savefig(out_png3d, dpi=150)
        plt.close(fig3d)

        np.savez(
            os.path.join(ckpt_dir, f"{play_output_stem}.npz"),
            target_asset_type=np.array(args.target_asset_type or "", dtype="U32"),
            t=t,
            a1_x=attacker_xyz[:, 0], a1_y=attacker_xyz[:, 1], a1_z=attacker_xyz[:, 2],
            a1_qx=attacker_quat[:, 0], a1_qy=attacker_quat[:, 1],
            a1_qz=attacker_quat[:, 2], a1_qw=attacker_quat[:, 3],
            a1_vx=attacker_linvel[:, 0], a1_vy=attacker_linvel[:, 1], a1_vz=attacker_linvel[:, 2],
            a1_body_vx=attacker_body_linvel[:, 0],
            a1_body_vy=attacker_body_linvel[:, 1],
            a1_body_vz=attacker_body_linvel[:, 2],
            a1_body_p=attacker_body_angvel[:, 0],
            a1_body_q=attacker_body_angvel[:, 1],
            a1_body_r=attacker_body_angvel[:, 2],
            a2_x=target_xyz[:, 0], a2_y=target_xyz[:, 1], a2_z=target_xyz[:, 2],
            a2_qx=target_quat[:, 0], a2_qy=target_quat[:, 1],
            a2_qz=target_quat[:, 2], a2_qw=target_quat[:, 3],
            a2_vx=target_linvel[:, 0], a2_vy=target_linvel[:, 1], a2_vz=target_linvel[:, 2],
            a2_wx=target_angvel[:, 0], a2_wy=target_angvel[:, 1], a2_wz=target_angvel[:, 2],
            target_cmd_x=target_cmd_xyz[:, 0], target_cmd_y=target_cmd_xyz[:, 1], target_cmd_z=target_cmd_xyz[:, 2],
            target_cmd_yaw=target_cmd_yaw,
            reward=reward_log,
            done=done_log,
            raw_action=raw_action_log,
            policy_mean=policy_mean_log,
            policy_std=policy_std_log,
            pre_tanh_action=pre_tanh_action_log,
            squashed_action=squashed_action_log,
            actor1_cmd_thrust=actor1_command_log[:, 0],
            actor1_cmd_p_rate=actor1_command_log[:, 1],
            actor1_cmd_q_rate=actor1_command_log[:, 2],
            actor1_cmd_r_rate=actor1_command_log[:, 3],
            actor1_exec_thrust_normalized=actor1_exec_log[:, 0],
            actor1_exec_force_z=actor1_exec_log[:, 1],
            actor1_exec_torque_x=actor1_exec_log[:, 2],
            actor1_exec_torque_y=actor1_exec_log[:, 3],
            actor1_exec_torque_z=actor1_exec_log[:, 4],
            target_force_x=target_force_log[:, 0],
            target_force_y=target_force_log[:, 1],
            target_force_z=target_force_log[:, 2],
            target_torque_x=target_torque_log[:, 0],
            target_torque_y=target_torque_log[:, 1],
            target_torque_z=target_torque_log[:, 2],
            rel_d=rel_d, final_d=final_d, min_d=min_d,
        )

        print(f"Saved plot to: {out_png}")
        print(f"Saved 3D plot to: {out_png3d}")
        print(f"Final distance: {final_d:.3f} m | Min distance: {min_d:.3f} m | steps: {len(t)}")


    cleanup_errors = []
    if writer is not None:
        try:
            writer.flush()
            writer.close()
        except Exception as e:
            cleanup_errors.append(f"writer.close failed: {e}")

    try:
        envs.close()
    except Exception as e:
        cleanup_errors.append(f"envs.close failed: {e}")

    if wandb_run is not None and wandb is not None and getattr(wandb, "run", None) is not None:
        try:
            wandb.finish(exit_code=0)
        except Exception as e:
            cleanup_errors.append(f"wandb.finish failed: {e}")

    if cleanup_errors:
        print("Cleanup completed with warnings:")
        for error in cleanup_errors:
            print(f"  {error}")
