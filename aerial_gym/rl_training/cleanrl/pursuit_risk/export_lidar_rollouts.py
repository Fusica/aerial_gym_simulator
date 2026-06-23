"""Export formal pursuit LiDAR risk-dataset shards from a PPO policy pool.

The exported training input is the deployable sensor stream, not debug images:

- LiDAR range image chunks: ``lidar_range_norm`` as float16.
- Deployable ego observation: body velocity, body angular velocity, rotation
  matrix, and previous action.
- Behavior/candidate action. PPO distribution statistics are diagnostic metadata
  only and must not be used as the risk head action reference.
- Multi-horizon observability-risk labels derived from privileged state.

Privileged target/robot state is stored only in the metadata shard for label
auditing and future relabeling. It must not be used as risk-model input.
"""

# Isaac Gym must be imported before torch.
import isaacgym  # noqa: F401

import argparse
import json
import os
import random
import shutil
import sys
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

from aerial_gym.registry.env_registry import env_config_registry
from aerial_gym.registry.robot_registry import robot_registry
from aerial_gym.registry.task_registry import task_registry
from aerial_gym.rl_training.cleanrl.ppo_guidance import (
    Agent,
    RecordEpisodeStatisticsTorch,
    configure_target_asset_type,
    extract_agent_state_dict,
)
from aerial_gym.rl_training.cleanrl.pursuit_risk.branch import (
    branch_anchor_selection_stats,
    branch_horizon_label_payload,
    branch_loss_label,
    build_branch_index_artifact,
    build_branch_metadata_payload,
    candidate_actions_from_policy,
    lidar_stack_from_history,
    write_branch_chunk,
)
from aerial_gym.rl_training.cleanrl.pursuit_risk.contract import (
    ACTION_DIM,
    DEFAULT_LIDAR_STACK_FRAMES,
    DEFAULT_RISK_HORIZONS,
    RISK_SCALAR_WEIGHTS,
    STATE_DIM,
    label_key,
)
from aerial_gym.task.pursuit_guidance_task.risk_geometry import (
    DetectionFrustum,
    detection_frustum_from_lidar_config,
    observability_loss_labels,
    relative_position_body,
    target_in_detection_frustum,
    visibility_recovery_stats,
)
from aerial_gym.utils.math import quat_to_rotation_matrix


DEFAULT_RUN_DIR = "runs/PE_20260520_110828"
DEFAULT_OUTPUT_DIR = "runs/risk_dataset/PE_20260520_110828_ladder_v1_500ep"
LADDER_V1_CHECKPOINT_UPDATES = (
    1,
    120,
    220,
    340,
    349,
    460,
    464,
    610,
    700,
    754,
    830,
    950,
    1070,
    1200,
    1300,
)
OUTPUT_SCHEMA_VERSION = 7
EMPTY_BBOX_XYXY = (-1, -1, -1, -1)
DEFAULT_BRANCH_MIN_SEVERITY_SEPARATION = 1.0e-3
DEFAULT_QA_MIN_HARD_MISMATCH_FRAMES = 3
DEFAULT_LABEL_MIN_VISIBLE_PIXELS = 3
NONTERMINAL_TRANSITION_SCHEMA = "nonterminal_s_t_lidar_t_action_t_reward_tplus1_done_false"
BRANCH_TRANSITION_SCHEMA = (
    "anchor_s_t_lidar_t_candidate_action_t_then_teacher_rollout_horizon"
)
REQUIRED_QA_BASELINE_KEYS = (
    "update",
    "source",
    "episodes",
    "frames",
    "semantic_frames",
    "target_visible_frame_rate",
    "detectable_frames",
    "visible_given_detectable_rate",
    "hard_mismatch_eligible_frames",
    "hard_mismatch_rate",
)


@dataclass(frozen=True)
class SelectedCheckpoint:
    env_id: int
    update: int
    checkpoint_path: str
    checkpoint_abspath: str
    train_success_rate: float
    train_reach_rate_current_threshold: Optional[float]
    train_avg_return: Optional[float]
    train_avg_min_relative_dist: Optional[float]
    train_avg_episode_length: Optional[float]
    train_finished_episodes: Optional[int]
    curriculum_current_threshold: Optional[float]
    curriculum_stage_idx: Optional[int]
    curriculum_visibility_reward_weight_scale: Optional[float]
    curriculum_transition: Optional[dict]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--task", default="pursuit_guidance_task")
    parser.add_argument("--robot-name", default="base_quad_root_link_control_with_lidar")
    parser.add_argument("--controller-name", default="thrust_bodyrate_control")
    parser.add_argument("--target-asset-type", default="target_x500", choices=("target_quad", "target_x500"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--total-episodes",
        type=int,
        default=500,
        help="Total episodes across all selected checkpoints.",
    )
    parser.add_argument(
        "--parallel-envs",
        type=int,
        default=2,
        help="Number of simultaneous LiDAR envs. Keep small for 501x2401 full-frame export.",
    )
    parser.add_argument("--max-steps", type=int, default=3600)
    parser.add_argument("--persist-steps", type=int, default=10)
    parser.add_argument("--risk-horizons", type=int, nargs="+", default=list(DEFAULT_RISK_HORIZONS))
    parser.add_argument(
        "--risk-lidar-stack-frames",
        type=int,
        default=DEFAULT_LIDAR_STACK_FRAMES,
        help="Number of LiDAR history frames required by the risk encoder input contract.",
    )
    parser.add_argument(
        "--label-min-visible-pixels",
        type=int,
        default=DEFAULT_LABEL_MIN_VISIBLE_PIXELS,
        help="Minimum target semantic pixels required for a frame to be label-visible.",
    )
    parser.add_argument(
        "--dataset-mode",
        default="behavior",
        choices=("behavior", "branch"),
        help="Export behavior-action full streams or action-branch anchor rollouts.",
    )
    parser.add_argument(
        "--lidar-chunk-steps",
        type=int,
        default=16,
        help="Number of full LiDAR frames buffered before writing one compressed chunk.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--checkpoint-preset",
        default="ladder_v1",
        choices=("ladder_v1", "custom"),
        help="Use ladder_v1 unless --checkpoint-updates should fully define the pool.",
    )
    parser.add_argument(
        "--checkpoint-updates",
        type=int,
        nargs="+",
        default=None,
        help="Policy-pool update indices. Used with --checkpoint-preset custom or as an override.",
    )
    parser.add_argument(
        "--lidar-config-class",
        default=None,
        help="Optional class from pursuit_forward_lidar_config.py to override the robot LiDAR config.",
    )
    lidar_seg_group = parser.add_mutually_exclusive_group()
    lidar_seg_group.add_argument(
        "--enable-lidar-segmentation",
        dest="enable_lidar_segmentation",
        action="store_true",
        help="Enable semantic output for QA metadata such as target_pixel_count.",
    )
    lidar_seg_group.add_argument(
        "--disable-lidar-segmentation",
        dest="enable_lidar_segmentation",
        action="store_false",
        help="Disable semantic QA output. Not recommended for formal datasets.",
    )
    parser.set_defaults(enable_lidar_segmentation=True)
    parser.add_argument("--qa-window-episodes", type=int, default=5)
    parser.add_argument("--qa-min-semantic-frames", type=int, default=20)
    parser.add_argument("--qa-min-detectable-frames", type=int, default=10)
    parser.add_argument("--qa-min-visible-given-detectable", type=float, default=0.0)
    parser.add_argument("--qa-max-hard-mismatch-rate", type=float, default=0.01)
    parser.add_argument(
        "--qa-min-hard-mismatch-frames",
        type=int,
        default=DEFAULT_QA_MIN_HARD_MISMATCH_FRAMES,
        help=(
            "Minimum absolute hard-mismatch frames required before the "
            "hard-mismatch rate can fail the QA gate."
        ),
    )
    parser.add_argument("--qa-hard-mismatch-max-range-m", type=float, default=30.0)
    parser.add_argument(
        "--qa-min-visible-pixels",
        type=int,
        default=DEFAULT_LABEL_MIN_VISIBLE_PIXELS,
        help="Minimum target semantic pixels required by hard-mismatch QA.",
    )
    parser.add_argument(
        "--qa-mode",
        default="pilot",
        choices=("pilot", "collect"),
        help="pilot can generate baselines; collect requires a complete QA baseline.",
    )
    parser.add_argument("--qa-baseline-path", default=None)
    parser.add_argument("--qa-baseline-output", default=None)
    parser.add_argument("--qa-baseline-relative-tolerance", type=float, default=0.75)
    parser.add_argument("--qa-disable-gate", action="store_true")
    parser.add_argument("--branch-update", type=int, default=None)
    parser.add_argument(
        "--branch-updates",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Branch-mode teacher checkpoint updates. Overrides --branch-update; "
            "use this for multi-teacher branch pilots."
        ),
    )
    parser.add_argument("--branch-episodes", type=int, default=1)
    parser.add_argument("--branch-anchor-stride", type=int, default=50)
    parser.add_argument("--branch-max-anchors", type=int, default=4)
    parser.add_argument("--branch-candidates", type=int, default=4)
    parser.add_argument("--branch-horizon", type=int, default=150)
    parser.add_argument("--branch-edge-margin-px", type=int, default=80)
    parser.add_argument("--branch-low-pixel-max", type=int, default=50)
    parser.add_argument("--branch-risk-anchor-min-score", type=float, default=0.25)
    parser.add_argument("--branch-min-candidate-action-l2", type=float, default=0.0)
    parser.add_argument(
        "--branch-min-severity-separation",
        type=float,
        default=DEFAULT_BRANCH_MIN_SEVERITY_SEPARATION,
        help="Minimum same-anchor severity range counted as action-conditioned separation.",
    )
    parser.add_argument(
        "--branch-min-separated-anchors",
        type=int,
        default=1,
        help="Minimum branch anchors that must show label separation before the run is accepted.",
    )
    parser.add_argument("--branch-min-visible-anchor-rate", type=float, default=0.5)
    parser.add_argument("--branch-min-detectable-anchor-rate", type=float, default=0.5)
    return parser.parse_args()


def assert_runtime():
    executable = os.path.realpath(sys.executable)
    if "/envs/aerialgym_v2/" not in executable:
        raise RuntimeError(
            "Run this exporter under conda env aerialgym_v2, for example: "
            "conda run -n aerialgym_v2 python aerial_gym/rl_training/cleanrl/"
            "python aerial_gym/rl_training/cleanrl/pursuit_risk/export_lidar_rollouts.py --dry-run"
        )


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def write_json(path: str, payload: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(payload), f, indent=2, sort_keys=True)


def append_jsonl(path: str, payload: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(json_safe(payload), ensure_ascii=False, sort_keys=True) + "\n")


def optional_float(value):
    return None if value is None else float(value)


def optional_int(value):
    return None if value is None else int(value)


def load_policy_pool_manifest(run_dir: str) -> Dict[int, dict]:
    manifest_path = os.path.join(run_dir, "policy_pool", "manifest.jsonl")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Policy-pool manifest not found: {manifest_path}")

    by_update = {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            by_update[int(row["update"])] = row
    return by_update


def checkpoint_updates_from_args(args) -> List[int]:
    if args.checkpoint_updates is not None:
        return [int(update) for update in args.checkpoint_updates]
    if args.checkpoint_preset == "ladder_v1":
        return list(LADDER_V1_CHECKPOINT_UPDATES)
    raise ValueError("--checkpoint-updates is required when --checkpoint-preset custom is used")


def select_checkpoints(run_dir: str, updates: Iterable[int]) -> List[SelectedCheckpoint]:
    manifest = load_policy_pool_manifest(run_dir)
    selected = []
    for env_id, update in enumerate(updates):
        update = int(update)
        if update not in manifest:
            raise KeyError(f"Update {update} is not present in policy_pool/manifest.jsonl")
        row = manifest[update]
        episode_metrics = row.get("episode_metrics") or {}
        curriculum = row.get("curriculum") or {}
        checkpoint_abspath = row.get("checkpoint_abspath")
        if not checkpoint_abspath:
            checkpoint_abspath = os.path.join(run_dir, row["checkpoint_path"])
        checkpoint_abspath = os.path.abspath(checkpoint_abspath)
        if not os.path.exists(checkpoint_abspath):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_abspath}")
        selected.append(
            SelectedCheckpoint(
                env_id=env_id,
                update=update,
                checkpoint_path=row["checkpoint_path"],
                checkpoint_abspath=checkpoint_abspath,
                train_success_rate=float(episode_metrics.get("success_rate", 0.0)),
                train_reach_rate_current_threshold=optional_float(
                    episode_metrics.get("reach_rate_current_threshold")
                ),
                train_avg_return=optional_float(episode_metrics.get("avg_return")),
                train_avg_min_relative_dist=optional_float(
                    episode_metrics.get("avg_min_relative_dist")
                ),
                train_avg_episode_length=optional_float(
                    episode_metrics.get("avg_episode_length")
                ),
                train_finished_episodes=optional_int(episode_metrics.get("finished_episodes")),
                curriculum_current_threshold=optional_float(curriculum.get("current_threshold")),
                curriculum_stage_idx=optional_int(curriculum.get("stage_idx")),
                curriculum_visibility_reward_weight_scale=optional_float(
                    curriculum.get("visibility_reward_weight_scale")
                ),
                curriculum_transition=row.get("curriculum_transition"),
            )
        )
    return selected


def distribute_episode_counts(total_episodes: int, num_sources: int) -> List[int]:
    if total_episodes <= 0:
        raise ValueError("--total-episodes must be positive")
    if num_sources <= 0:
        raise ValueError("at least one checkpoint is required")
    base = total_episodes // num_sources
    remainder = total_episodes % num_sources
    return [base + (1 if idx < remainder else 0) for idx in range(num_sources)]


def print_plan(selected: Sequence[SelectedCheckpoint], episode_counts: Sequence[int]):
    print("dataset export plan")
    print("env update episodes train_sr stage vis_weight checkpoint")
    for item, count in zip(selected, episode_counts):
        stage = "NA" if item.curriculum_stage_idx is None else str(item.curriculum_stage_idx)
        vis = (
            "NA"
            if item.curriculum_visibility_reward_weight_scale is None
            else f"{item.curriculum_visibility_reward_weight_scale:.2f}"
        )
        print(
            f"{item.env_id:02d} {item.update:04d} {int(count):04d} "
            f"{item.train_success_rate:.6f} {stage:>5} {vis:>8} {item.checkpoint_path}"
        )


def target_semantic_id_for_asset(target_asset_type: str) -> int:
    from aerial_gym.config.asset_config import pursuit_guidance_asset_config

    class_name = f"{target_asset_type}_asset_params"
    asset_config = getattr(pursuit_guidance_asset_config, class_name)
    return int(asset_config.semantic_id)


def configure_lidar_task(args, num_envs: int):
    import aerial_gym  # noqa: F401
    from aerial_gym.config.sensor_config.lidar_config import pursuit_forward_lidar_config

    task_config = task_registry.get_task_config(args.task)
    task_config.robot_name = args.robot_name
    task_config.controller_name = args.controller_name
    task_config.use_warp = True
    task_config.headless = True
    task_config.device = args.device
    task_config.num_envs = num_envs

    env_config = env_config_registry.get_env_config(task_config.env_name)
    env_config.env.num_envs = num_envs
    env_config.env.use_warp = True
    configure_target_asset_type(task_config, args.target_asset_type)

    robot_config = robot_registry.get_robot_config(args.robot_name)
    if args.lidar_config_class is not None:
        lidar_config = getattr(pursuit_forward_lidar_config, args.lidar_config_class)
        robot_config.sensor_config.lidar_config = lidar_config
    if args.enable_lidar_segmentation:
        robot_config.sensor_config.lidar_config.segmentation_camera = True
    else:
        robot_config.sensor_config.lidar_config.segmentation_camera = False


def build_env(args, num_envs: int, seed: int):
    configure_lidar_task(args, num_envs)
    original_argv = sys.argv[:]
    sys.argv = [
        original_argv[0],
        "--headless",
        "True",
        "--num_envs",
        str(num_envs),
        "--use_warp",
        "True",
        "--sim_device",
        args.device,
    ]
    try:
        env = task_registry.make_task(
            task_name=args.task,
            seed=seed,
            num_envs=num_envs,
            headless=True,
            use_warp=True,
            device=args.device,
        )
    finally:
        sys.argv = original_argv
    return RecordEpisodeStatisticsTorch(env, args.device)


def load_agents(selected: Sequence[SelectedCheckpoint], envs, device: str):
    agents = []
    for item in selected:
        checkpoint = torch.load(item.checkpoint_abspath, map_location="cpu")
        state_dict = extract_agent_state_dict(checkpoint)
        agent = Agent(envs).to(device)
        missing_keys, unexpected_keys = agent.load_state_dict(state_dict, strict=False)
        missing_non_obs = [key for key in missing_keys if not key.startswith("obs_rms.")]
        if missing_non_obs or unexpected_keys:
            raise RuntimeError(
                f"Checkpoint {item.checkpoint_abspath} is incompatible. "
                f"Missing non-obs keys: {missing_non_obs}, unexpected: {unexpected_keys}"
            )
        if any(key.startswith("obs_rms.") for key in missing_keys):
            raise RuntimeError(
                f"Checkpoint {item.checkpoint_abspath} is missing obs_rms stats; "
                "refusing to export misleading rollouts."
            )
        agent.eval()
        agents.append(agent)
    return agents


def to_cpu_np(tensor: torch.Tensor):
    return tensor.detach().cpu().numpy().copy()


def tensor_rows_to_numpy(rows: List[torch.Tensor], width: Optional[int] = None, dtype=np.float32):
    if not rows:
        shape = (0,) if width is None else (0, width)
        return np.empty(shape, dtype=dtype)
    return to_cpu_np(torch.stack(rows, dim=0)).astype(dtype, copy=False)


def nullable_float(value):
    return np.nan if value is None else float(value)


def extract_done_reason(info: dict, env_id: int) -> str:
    for key, name in (
        ("done_reason_success", "success"),
        ("done_reason_collision", "collision"),
        ("done_reason_far", "far"),
        ("done_reason_timeout", "timeout"),
    ):
        value = info.get(key)
        if value is not None and bool(value[env_id].item()):
            return name
    return "not_done"


def target_mask_stats(semantic_frame: Optional[np.ndarray], target_semantic_id: int) -> dict:
    if semantic_frame is None:
        return {
            "pixel_count": -1,
            "bbox_xyxy": EMPTY_BBOX_XYXY,
            "centroid_uv": (np.nan, np.nan),
            "area_ratio": np.nan,
        }

    target_mask = semantic_frame == int(target_semantic_id)
    pixel_count = int(np.count_nonzero(target_mask))
    if pixel_count == 0:
        return {
            "pixel_count": 0,
            "bbox_xyxy": EMPTY_BBOX_XYXY,
            "centroid_uv": (np.nan, np.nan),
            "area_ratio": 0.0,
        }

    rows, cols = np.nonzero(target_mask)
    return {
        "pixel_count": pixel_count,
        "bbox_xyxy": (
            int(cols.min()),
            int(rows.min()),
            int(cols.max()) + 1,
            int(rows.max()) + 1,
        ),
        "centroid_uv": (float(cols.mean()), float(rows.mean())),
        "area_ratio": float(pixel_count) / float(target_mask.size),
    }


def deployable_ego_obs(env, prev_action: torch.Tensor) -> torch.Tensor:
    robot_body_linvel = env.obs_dict["robot_body_linvel"]
    robot_body_angvel = env.obs_dict["robot_body_angvel"]
    robot_quat = env.obs_dict["robot_orientation"]
    rot = quat_to_rotation_matrix(robot_quat).reshape(env.num_envs, 9)
    return torch.cat((robot_body_linvel, robot_body_angvel, rot, prev_action), dim=1)


TASK_SNAPSHOT_ATTRS = (
    "actions",
    "prev_actions",
    "last_actor1_attitude_command",
    "trajectory_time",
    "prev_relative_dist",
    "visibility_loss_steps",
    "visibility_episode_invisible_steps",
    "visibility_episode_loss_segments",
    "visibility_episode_recovered_segments",
    "visibility_episode_max_loss_steps",
    "visibility_episode_over_horizon_steps",
    "visibility_episode_recovered_within_horizon_segments",
    "visibility_episode_loss_area",
    "target_command",
    "apf_velocity_state",
    "target_motion_type_ids",
    "simple_start_pos",
    "simple_direction",
    "simple_speed",
    "sine_start_pos",
    "sine_direction",
    "sine_lateral_axis",
    "sine_speed",
    "sine_amp_y",
    "sine_omega_y",
    "sine_amp_z",
    "sine_omega_z",
    "sine_phase_y",
    "sine_phase_z",
    "ellipse_focus1",
    "ellipse_focus2",
    "ellipse_semi_major_axis",
    "ellipse_angular_rate",
    "ellipse_height",
    "ellipse_initial_phase",
    "apf_start_pos",
    "apf_baseline_direction",
    "apf_baseline_speed",
)
CONTROLLER_SNAPSHOT_ATTRS = (
    "last_scaled_input",
    "rate_error_integral",
    "prev_rate_error",
    "prev_error_valid",
)
CONTROL_ALLOCATOR_SNAPSHOT_ATTRS = (
    "output_wrench",
)
MOTOR_MODEL_SNAPSHOT_ATTRS = (
    "current_motor_thrust",
    "motor_time_constants_increasing",
    "motor_time_constants_decreasing",
    "motor_rate",
    "motor_thrust_constant",
)


def capture_task_snapshot(task, env_id: int) -> dict:
    snapshot = {
        "robot_state": task.robot_state[env_id].detach().clone(),
        "env_asset_state_tensor": task.obs_dict["env_asset_state_tensor"][env_id].detach().clone(),
        "sim_steps": task.sim_env.sim_steps[env_id].detach().clone(),
        "env_lower_bound": task.sim_env.IGE_env.env_lower_bound[env_id].detach().clone(),
        "env_upper_bound": task.sim_env.IGE_env.env_upper_bound[env_id].detach().clone(),
        "task_attrs": {},
        "controller_attrs": {},
        "control_allocator_attrs": {},
        "motor_model_attrs": {},
    }
    for attr_name in TASK_SNAPSHOT_ATTRS:
        value = getattr(task, attr_name, None)
        if torch.is_tensor(value):
            snapshot["task_attrs"][attr_name] = value[env_id].detach().clone()
    controller = getattr(task.sim_env.robot_manager.robot, "controller", None)
    for attr_name in CONTROLLER_SNAPSHOT_ATTRS:
        value = getattr(controller, attr_name, None)
        if torch.is_tensor(value):
            snapshot["controller_attrs"][attr_name] = value[env_id].detach().clone()
    control_allocator = getattr(task.sim_env.robot_manager.robot, "control_allocator", None)
    for attr_name in CONTROL_ALLOCATOR_SNAPSHOT_ATTRS:
        value = getattr(control_allocator, attr_name, None)
        if torch.is_tensor(value):
            snapshot["control_allocator_attrs"][attr_name] = value[env_id].detach().clone()
    motor_model = getattr(control_allocator, "motor_model", None)
    for attr_name in MOTOR_MODEL_SNAPSHOT_ATTRS:
        value = getattr(motor_model, attr_name, None)
        if torch.is_tensor(value):
            snapshot["motor_model_attrs"][attr_name] = value[env_id].detach().clone()
    return snapshot


def _expand_snapshot_value(value: torch.Tensor, count: int) -> torch.Tensor:
    if value.ndim == 0:
        return value.reshape(1).expand(count)
    return value.unsqueeze(0).expand((count,) + tuple(value.shape))


def restore_task_snapshot(task, snapshot: dict, env_ids: torch.Tensor):
    env_ids = env_ids.to(device=task.device, dtype=torch.long)
    count = int(env_ids.numel())
    task.robot_state[env_ids] = _expand_snapshot_value(snapshot["robot_state"], count)
    task.obs_dict["env_asset_state_tensor"][env_ids] = _expand_snapshot_value(
        snapshot["env_asset_state_tensor"], count
    )
    task.sim_env.sim_steps[env_ids] = _expand_snapshot_value(snapshot["sim_steps"], count)
    task.sim_env.IGE_env.env_lower_bound[env_ids] = _expand_snapshot_value(
        snapshot["env_lower_bound"], count
    )
    task.sim_env.IGE_env.env_upper_bound[env_ids] = _expand_snapshot_value(
        snapshot["env_upper_bound"], count
    )
    for attr_name, value in snapshot["task_attrs"].items():
        attr = getattr(task, attr_name, None)
        if torch.is_tensor(attr):
            attr[env_ids] = _expand_snapshot_value(value, count)
    controller = getattr(task.sim_env.robot_manager.robot, "controller", None)
    for attr_name, value in snapshot.get("controller_attrs", {}).items():
        attr = getattr(controller, attr_name, None)
        if torch.is_tensor(attr):
            attr[env_ids] = _expand_snapshot_value(value, count)
    control_allocator = getattr(task.sim_env.robot_manager.robot, "control_allocator", None)
    for attr_name, value in snapshot.get("control_allocator_attrs", {}).items():
        attr = getattr(control_allocator, attr_name, None)
        if torch.is_tensor(attr):
            attr[env_ids] = _expand_snapshot_value(value, count)
    motor_model = getattr(control_allocator, "motor_model", None)
    for attr_name, value in snapshot.get("motor_model_attrs", {}).items():
        attr = getattr(motor_model, attr_name, None)
        if torch.is_tensor(attr):
            attr[env_ids] = _expand_snapshot_value(value, count)
    task.sim_env.IGE_env.write_to_sim()
    task.sim_env.robot_manager.robot.update_states()
    task.sim_env.render(render_components="sensors")
    task.process_obs_for_task()


def current_observations(task) -> torch.Tensor:
    task.process_obs_for_task()
    return task.task_obs["observations"].detach().clone()


def lidar_frame_np(task, env_id: int) -> np.ndarray:
    tensor = task.obs_dict.get("depth_range_pixels")
    if tensor is None:
        raise RuntimeError("depth_range_pixels tensor was not created")
    return tensor[env_id, 0].detach().cpu().numpy().astype(np.float32, copy=False)


def semantic_frame_np(task, env_id: int, args) -> Optional[np.ndarray]:
    tensor = task.obs_dict.get("segmentation_pixels")
    if tensor is None:
        if bool(args.enable_lidar_segmentation):
            raise RuntimeError("LiDAR segmentation QA is enabled but segmentation_pixels is missing.")
        return None
    return tensor[env_id, 0].detach().cpu().numpy().astype(np.int32, copy=False)


def collect_branch_dataset(args, selected: Sequence[SelectedCheckpoint]):
    if len(selected) < 1:
        raise ValueError("branch dataset mode requires at least one selected checkpoint")
    output_dir = args.output_dir
    index_path = os.path.join(output_dir, "index.jsonl")
    if os.path.exists(index_path):
        os.remove(index_path)
    branch_index_path = os.path.join(output_dir, "branch_index.jsonl")
    if os.path.exists(branch_index_path):
        os.remove(branch_index_path)

    num_candidates = int(args.branch_candidates)
    num_envs = num_candidates + 1
    envs = build_env(args, num_envs, seed=int(args.seed))
    task = envs.env
    lidar_cfg = task.sim_env.robot_manager.cfg.sensor_config.lidar_config
    frustum = detection_frustum_from_lidar_config(lidar_cfg)
    target_semantic_id = target_semantic_id_for_asset(args.target_asset_type)
    dataset_anchor_id = 0
    try:
        for selected_idx, item in enumerate(selected):
            print(
                "starting branch teacher "
                f"update={item.update} episodes={int(args.branch_episodes)} "
                f"teacher_index={selected_idx}"
            )
            agent = load_agents([item], envs, args.device)[0]
            dataset_anchor_id = collect_branch_dataset_for_item(
                args=args,
                item=item,
                agent=agent,
                envs=envs,
                lidar_cfg=lidar_cfg,
                frustum=frustum,
                target_semantic_id=target_semantic_id,
                dataset_anchor_id=dataset_anchor_id,
                index_path=index_path,
                branch_index_path=branch_index_path,
            )
    finally:
        try:
            envs.close()
        except Exception as exc:
            print(f"Cleanup failed: {exc}", file=sys.stderr)


def collect_branch_dataset_for_item(
    args,
    item: SelectedCheckpoint,
    agent: Agent,
    envs,
    lidar_cfg,
    frustum: DetectionFrustum,
    target_semantic_id: int,
    dataset_anchor_id: int,
    index_path: str,
    branch_index_path: str,
) -> int:
    output_dir = args.output_dir
    num_candidates = int(args.branch_candidates)
    num_envs = num_candidates + 1
    task = envs.env
    branch_env_ids = torch.arange(1, num_candidates + 1, device=args.device)
    label_horizons = [horizon for horizon in args.risk_horizons if horizon <= args.branch_horizon]
    anchor_id = 0
    for episode_idx in range(int(args.branch_episodes)):
        obs, _ = envs.reset()
        task.sim_env.render(render_components="sensors")
        source_lidar_history = []
        anchors_written = 0
        for step in range(int(args.max_steps)):
            obs_before = obs.detach().clone()
            source_lidar_history.append(lidar_frame_np(task, anchor_id).copy())
            max_history = int(args.risk_lidar_stack_frames)
            if len(source_lidar_history) > max_history:
                source_lidar_history = source_lidar_history[-max_history:]
            if step % int(args.branch_anchor_stride) == 0 and anchors_written < int(args.branch_max_anchors):
                snapshot = capture_task_snapshot(task, anchor_id)
                anchor_rel_body = relative_position_body(
                    task.robot_state[anchor_id : anchor_id + 1, 0:3],
                    task.robot_state[anchor_id : anchor_id + 1, 3:7],
                    task.target_state[anchor_id : anchor_id + 1, 0:3],
                )
                anchor_detectable = bool(
                    target_in_detection_frustum(anchor_rel_body, frustum)[0].item()
                )
                semantic_stats = target_mask_stats(
                    semantic_frame_np(task, anchor_id, args), target_semantic_id
                )
                anchor_visible = (
                    int(semantic_stats["pixel_count"]) >= int(args.label_min_visible_pixels)
                )
                selection_stats = branch_anchor_selection_stats(
                    semantic_stats=semantic_stats,
                    lidar_height=int(lidar_cfg.height),
                    lidar_width=int(lidar_cfg.width),
                    args=args,
                )
                if not (anchor_visible and selection_stats["score"] >= args.branch_risk_anchor_min_score):
                    skip_log_stride = max(int(args.branch_anchor_stride) * 10, 10)
                    if step % skip_log_stride == 0:
                        print(
                            "skip branch anchor "
                            f"update={item.update} episode={episode_idx} step={step} "
                            f"visible={anchor_visible} detectable={anchor_detectable} "
                            f"score={selection_stats['score']:.3f} "
                            f"reason={selection_stats['reason']}"
                        )
                    with torch.no_grad():
                        action, _, _, _, _aux = agent.get_action_and_value(
                            obs[anchor_id : anchor_id + 1], return_aux=True
                        )
                    actions = torch.zeros(
                        (num_envs, envs.num_actions),
                        dtype=torch.float32,
                        device=args.device,
                    )
                    actions[anchor_id] = action[0]
                    obs, _rewards, done_float, _info = envs.step(actions)
                    if bool(done_float[anchor_id].item()):
                        break
                    continue

                anchor_dir = os.path.join(
                    output_dir,
                    "branches",
                    f"upd_{item.update:06d}",
                    f"anchor_{dataset_anchor_id:06d}",
                )
                os.makedirs(anchor_dir, exist_ok=True)
                anchor_lidar_relpath = os.path.relpath(
                    os.path.join(anchor_dir, "anchor_lidar.npz"), output_dir
                )
                anchor_ego_obs_np = (
                    deployable_ego_obs(task, task.actions)[anchor_id]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                anchor_lidar_stack = lidar_stack_from_history(
                    source_lidar_history, int(args.risk_lidar_stack_frames)
                )
                anchor_lidar_bytes = write_branch_chunk(
                    os.path.join(anchor_dir, "anchor_lidar.npz"),
                    anchor_lidar_stack,
                    schema_version=OUTPUT_SCHEMA_VERSION,
                )
                candidate_actions = candidate_actions_from_policy(
                    agent,
                    obs_before[anchor_id : anchor_id + 1],
                    num_candidates,
                    args,
                )

                restore_task_snapshot(task, snapshot, branch_env_ids)
                branch_obs = current_observations(task)
                done_flags = torch.zeros(num_candidates, dtype=torch.bool, device=args.device)
                visible_history = [[] for _ in range(num_candidates)]
                first_actions = torch.zeros(
                    (num_envs, envs.num_actions), dtype=torch.float32, device=args.device
                )
                first_actions[1:] = candidate_actions
                branch_obs, _rewards, done_float, _info = envs.step(first_actions)
                done_flags |= done_float[1:].bool()
                for horizon_step in range(int(args.branch_horizon)):
                    for idx in range(num_candidates):
                        if bool(done_flags[idx].item()):
                            continue
                        branch_slot = idx + 1
                        stats = target_mask_stats(
                            semantic_frame_np(task, branch_slot, args),
                            target_semantic_id,
                        )
                        visible_history[idx].append(
                            int(stats["pixel_count"]) >= int(args.label_min_visible_pixels)
                        )
                    if horizon_step + 1 >= int(args.branch_horizon):
                        break
                    teacher_actions = torch.zeros(
                        (num_envs, envs.num_actions), dtype=torch.float32, device=args.device
                    )
                    with torch.no_grad():
                        for idx in range(num_candidates):
                            if bool(done_flags[idx].item()):
                                continue
                            branch_slot = idx + 1
                            teacher_action, _, _, _, _teacher_aux = agent.get_action_and_value(
                                branch_obs[branch_slot : branch_slot + 1], return_aux=True
                            )
                            teacher_actions[branch_slot] = teacher_action[0]
                    branch_obs, _rewards, done_float, _info = envs.step(teacher_actions)
                    done_flags |= done_float[1:].bool()

                labels = []
                severities = []
                for values in visible_history:
                    label, _offset, severity = branch_loss_label(values, int(args.persist_steps))
                    labels.append(float(label))
                    severities.append(float(severity))
                losses = np.asarray(labels, dtype=np.float32)
                severities_np = np.asarray(severities, dtype=np.float32)
                horizon_label_payload = branch_horizon_label_payload(
                    visible_history,
                    label_horizons,
                    int(args.persist_steps),
                )
                loss_range = float(np.max(losses) - np.min(losses))
                severity_range = float(np.max(severities_np) - np.min(severities_np))
                label_has_separation = bool(
                    loss_range > 0.0
                    or severity_range >= float(args.branch_min_severity_separation)
                )
                metadata_path = os.path.join(anchor_dir, "branch_metadata.npz")
                metadata_payload = build_branch_metadata_payload(
                    schema_version=OUTPUT_SCHEMA_VERSION,
                    anchor_lidar_relpath=anchor_lidar_relpath,
                    anchor_ego_obs_np=anchor_ego_obs_np,
                    candidate_actions=candidate_actions,
                    horizon_label_payload=horizon_label_payload,
                )
                np.savez_compressed(metadata_path, **metadata_payload)
                artifact = build_branch_index_artifact(
                    schema_version=OUTPUT_SCHEMA_VERSION,
                    transition_schema=BRANCH_TRANSITION_SCHEMA,
                    args=args,
                    item=item,
                    dataset_anchor_id=dataset_anchor_id,
                    episode_idx=episode_idx,
                    step=step,
                    num_candidates=num_candidates,
                    loss_range=loss_range,
                    severity_range=severity_range,
                    label_has_separation=label_has_separation,
                    anchor_lidar_bytes=anchor_lidar_bytes,
                    anchor_lidar_stack=anchor_lidar_stack,
                    metadata_relpath=os.path.relpath(metadata_path, output_dir),
                    semantic_stats=semantic_stats,
                    anchor_detectable=anchor_detectable,
                    anchor_visible=anchor_visible,
                    selection_stats=selection_stats,
                )
                append_jsonl(branch_index_path, artifact)
                append_jsonl(index_path, artifact)
                print(
                    "saved branch "
                    f"anchor={dataset_anchor_id:06d} update={item.update} "
                    f"step={step} loss_range={artifact['candidate_loss_range']:.3f} "
                    f"severity_range={artifact['candidate_severity_range']:.3f} "
                    f"separated={artifact['candidate_label_has_separation']}"
                )
                dataset_anchor_id += 1
                anchors_written += 1
                restore_task_snapshot(task, snapshot, torch.tensor([anchor_id], device=args.device))
                obs = current_observations(task)

            with torch.no_grad():
                action, _, _, _, _aux = agent.get_action_and_value(
                    obs[anchor_id : anchor_id + 1], return_aux=True
                )
            actions = torch.zeros((num_envs, envs.num_actions), dtype=torch.float32, device=args.device)
            actions[anchor_id] = action[0]
            obs, _rewards, done_float, _info = envs.step(actions)
            if bool(done_float[anchor_id].item()):
                break
    return int(dataset_anchor_id)


def compute_geometry_observability(
    robot_state: torch.Tensor,
    target_state: torch.Tensor,
    frustum: DetectionFrustum,
):
    rel_body = relative_position_body(robot_state[..., 0:3], robot_state[..., 3:7], target_state[..., 0:3])
    detectable = target_in_detection_frustum(rel_body, frustum)
    return rel_body, detectable


def future_invisible_fraction(visible: np.ndarray, horizon_steps: int) -> np.ndarray:
    invisible = (~visible.astype(np.bool_)).astype(np.float32)
    cumsum = np.concatenate(([0.0], np.cumsum(invisible, dtype=np.float64)))
    out = np.zeros_like(invisible, dtype=np.float32)
    time_steps = len(invisible)
    for t in range(time_steps):
        end = min(time_steps, t + int(horizon_steps))
        denom = max(end - t, 1)
        out[t] = float((cumsum[end] - cumsum[t]) / denom)
    return out


def future_recovery_labels(visible: np.ndarray, horizon_steps: int):
    visible = visible.astype(np.bool_)
    valid = ~visible
    recovered = np.zeros_like(visible, dtype=np.bool_)
    time_steps = len(visible)
    for t in range(time_steps):
        if not valid[t]:
            continue
        start = t + 1
        end = min(time_steps, t + int(horizon_steps))
        recovered[t] = bool(start < end and np.any(visible[start:end]))
    return recovered, valid


def label_valid_mask(time_steps: int, horizon_steps: int, persist_steps: int) -> np.ndarray:
    """Return samples with enough future context to supervise a horizon label."""
    latest_needed_exclusive = np.arange(time_steps, dtype=np.int64) + int(horizon_steps) + int(persist_steps) - 1
    return latest_needed_exclusive <= int(time_steps)


def semantic_visible_from_counts(target_counts: np.ndarray, min_visible_pixels: int) -> np.ndarray:
    """Return sensor-space target visibility from semantic target pixel counts."""
    if int(min_visible_pixels) <= 0:
        raise ValueError("min_visible_pixels must be positive")
    return np.asarray(target_counts, dtype=np.int32) >= int(min_visible_pixels)


def selected_to_dict(item: SelectedCheckpoint) -> dict:
    return asdict(item)


class EpisodeShardWriter:
    def __init__(
        self,
        output_dir: str,
        item: SelectedCheckpoint,
        dataset_episode_id: int,
        checkpoint_episode_idx: int,
        lidar_height: int,
        lidar_width: int,
        chunk_steps: int,
    ):
        self.item = item
        self.output_dir = output_dir
        self.dataset_episode_id = int(dataset_episode_id)
        self.checkpoint_episode_idx = int(checkpoint_episode_idx)
        self.lidar_height = int(lidar_height)
        self.lidar_width = int(lidar_width)
        self.chunk_steps = int(chunk_steps)
        self.episode_dir = os.path.join(
            output_dir,
            "episodes",
            f"upd_{item.update:06d}",
            f"episode_{dataset_episode_id:06d}",
        )
        os.makedirs(self.episode_dir, exist_ok=True)

        self.chunk_frames = []
        self.chunk_frame_steps = []
        self.chunk_relpaths = []
        self.chunk_num_frames = []
        self.chunk_bytes = []

        self.step = []
        self.robot_state = []
        self.target_state = []
        self.ego_obs = []
        self.prev_action = []
        self.action = []
        self.policy_mean = []
        self.policy_std = []
        self.pre_tanh_action = []
        self.reward = []
        self.relative_distance = []
        self.lidar_finite_fraction = []
        self.lidar_min = []
        self.lidar_max = []
        self.target_pixel_count = []
        self.target_bbox_xyxy = []
        self.target_centroid_uv = []
        self.target_area_ratio = []
        self.discarded_terminal_transition = False
        self.terminal_step = -1
        self.terminal_done_reason = "not_done"
        self.terminal_reward = np.nan
        self.terminal_relative_distance = np.nan

    def has_frames(self) -> bool:
        return bool(self.step)

    def mark_discarded_terminal_transition(
        self,
        step: int,
        reward: float,
        done_reason: str,
        relative_distance: float,
    ):
        self.discarded_terminal_transition = True
        self.terminal_step = int(step)
        self.terminal_reward = float(reward)
        self.terminal_done_reason = str(done_reason)
        self.terminal_relative_distance = float(relative_distance)

    def append_frame(
        self,
        step: int,
        robot_state: torch.Tensor,
        target_state: torch.Tensor,
        ego_obs: torch.Tensor,
        prev_action: torch.Tensor,
        action: torch.Tensor,
        policy_mean: torch.Tensor,
        policy_std: torch.Tensor,
        pre_tanh_action: torch.Tensor,
        reward: float,
        done: bool,
        done_reason: str,
        relative_distance: float,
        lidar_norm: np.ndarray,
        semantic_stats: dict,
    ):
        if done:
            raise ValueError("done=True transitions must be recorded as terminal metadata, not frames")
        if done_reason != "not_done":
            raise ValueError(f"non-terminal frame cannot carry done_reason={done_reason}")
        self.step.append(int(step))
        self.robot_state.append(robot_state.detach().clone())
        self.target_state.append(target_state.detach().clone())
        self.ego_obs.append(ego_obs.detach().clone())
        self.prev_action.append(prev_action.detach().clone())
        self.action.append(action.detach().clone())
        self.policy_mean.append(policy_mean.detach().clone())
        self.policy_std.append(policy_std.detach().clone())
        self.pre_tanh_action.append(pre_tanh_action.detach().clone())
        self.reward.append(float(reward))
        self.relative_distance.append(float(relative_distance))
        finite_mask = np.isfinite(lidar_norm)
        self.lidar_finite_fraction.append(float(np.mean(finite_mask)))
        self.lidar_min.append(float(np.nanmin(lidar_norm)))
        self.lidar_max.append(float(np.nanmax(lidar_norm)))
        self.target_pixel_count.append(int(semantic_stats["pixel_count"]))
        self.target_bbox_xyxy.append(semantic_stats["bbox_xyxy"])
        self.target_centroid_uv.append(semantic_stats["centroid_uv"])
        self.target_area_ratio.append(semantic_stats["area_ratio"])

        self.chunk_frames.append(lidar_norm.astype(np.float16, copy=True))
        self.chunk_frame_steps.append(int(step))
        if len(self.chunk_frames) >= self.chunk_steps:
            self.flush_lidar_chunk()

    def flush_lidar_chunk(self):
        if not self.chunk_frames:
            return
        chunk_idx = len(self.chunk_relpaths)
        chunk_name = f"lidar_chunk_{chunk_idx:04d}.npz"
        chunk_path = os.path.join(self.episode_dir, chunk_name)
        frames = np.stack(self.chunk_frames, axis=0)
        steps = np.asarray(self.chunk_frame_steps, dtype=np.int32)
        np.savez_compressed(
            chunk_path,
            schema_version=np.array(OUTPUT_SCHEMA_VERSION, dtype=np.int32),
            dataset_episode_id=np.array(self.dataset_episode_id, dtype=np.int32),
            update=np.array(self.item.update, dtype=np.int32),
            step=steps,
            lidar_range_norm=frames,
        )
        relpath = os.path.relpath(chunk_path, self.output_dir)
        self.chunk_relpaths.append(relpath)
        self.chunk_num_frames.append(int(frames.shape[0]))
        self.chunk_bytes.append(int(os.path.getsize(chunk_path)))
        self.chunk_frames.clear()
        self.chunk_frame_steps.clear()

    def close(
        self,
        args,
        frustum: DetectionFrustum,
        target_semantic_id: int,
    ) -> dict:
        self.flush_lidar_chunk()
        if not self.step:
            return None

        robot_np = tensor_rows_to_numpy(self.robot_state, width=13)
        target_np = tensor_rows_to_numpy(self.target_state, width=13)
        robot_state_t = torch.from_numpy(robot_np.astype(np.float32))
        target_state_t = torch.from_numpy(target_np.astype(np.float32))

        rel_body, detectable = compute_geometry_observability(
            robot_state=robot_state_t.unsqueeze(1),
            target_state=target_state_t.unsqueeze(1),
            frustum=frustum,
        )
        rel_body_np = to_cpu_np(rel_body[:, 0]).astype(np.float32)
        detectable_np = to_cpu_np(detectable[:, 0]).astype(np.bool_)
        target_counts = np.asarray(self.target_pixel_count, dtype=np.int32)
        semantic_available_np = target_counts >= 0
        target_visible_np = semantic_visible_from_counts(
            target_counts,
            int(args.label_min_visible_pixels),
        )
        visible_t = torch.from_numpy(target_visible_np.astype(np.bool_)).unsqueeze(1)
        label_payload = {}
        for horizon_steps in args.risk_horizons:
            loss_labels = observability_loss_labels(
                visible_t,
                horizon_steps=int(horizon_steps),
                persist_steps=args.persist_steps,
            )
            loss_np = to_cpu_np(loss_labels[:, 0]).astype(np.float32)
            severity_np = future_invisible_fraction(target_visible_np, int(horizon_steps))
            recovery_np, recovery_valid_np = future_recovery_labels(target_visible_np, int(horizon_steps))
            valid_np = label_valid_mask(len(target_visible_np), int(horizon_steps), int(args.persist_steps))
            loss_np[~valid_np] = 0.0
            severity_np[~valid_np] = 0.0
            recovery_valid_np &= valid_np
            label_payload[label_key("label_loss_prob", horizon_steps)] = loss_np
            label_payload[label_key("label_loss_severity", horizon_steps)] = severity_np
            label_payload[label_key("label_valid", horizon_steps)] = valid_np.astype(np.bool_)
            label_payload[label_key("label_recovery", horizon_steps)] = recovery_np.astype(np.float32)
            label_payload[label_key("label_recovery_valid", horizon_steps)] = recovery_valid_np.astype(np.bool_)

        visibility_stats = visibility_recovery_stats(
            target_visible_np,
            persist_steps=args.persist_steps,
            horizon_steps=max(int(value) for value in args.risk_horizons),
        )

        steps = np.asarray(self.step, dtype=np.int32)
        rel_d = np.asarray(self.relative_distance, dtype=np.float32)
        detectable_not_visible_np = semantic_available_np & detectable_np & (~target_visible_np)
        visible_not_detectable_np = semantic_available_np & target_visible_np & (~detectable_np)
        hard_mismatch_np = (
            semantic_available_np
            & detectable_np
            & (rel_d <= float(args.qa_hard_mismatch_max_range_m))
            & (target_counts < int(args.qa_min_visible_pixels))
        )
        hard_mismatch_eligible_np = (
            semantic_available_np
            & detectable_np
            & (rel_d <= float(args.qa_hard_mismatch_max_range_m))
        )
        metadata_name = "metadata.npz"
        metadata_path = os.path.join(self.episode_dir, metadata_name)
        np.savez_compressed(
            metadata_path,
            schema_version=np.array(OUTPUT_SCHEMA_VERSION, dtype=np.int32),
            dataset_episode_id=np.array(self.dataset_episode_id, dtype=np.int32),
            checkpoint_episode_idx=np.array(self.checkpoint_episode_idx, dtype=np.int32),
            run_dir=np.array(args.run_dir, dtype="U256"),
            checkpoint_path=np.array(self.item.checkpoint_path, dtype="U256"),
            checkpoint_abspath=np.array(self.item.checkpoint_abspath, dtype="U512"),
            update=np.array(self.item.update, dtype=np.int32),
            train_success_rate=np.array(self.item.train_success_rate, dtype=np.float32),
            target_asset_type=np.array(args.target_asset_type, dtype="U32"),
            target_semantic_id=np.array(target_semantic_id, dtype=np.int32),
            label_source=np.array("semantic_target_pixel_count", dtype="U64"),
            label_min_visible_pixels=np.array(args.label_min_visible_pixels, dtype=np.int32),
            transition_schema=np.array(NONTERMINAL_TRANSITION_SCHEMA, dtype="U96"),
            discarded_terminal_transition=np.array(
                self.discarded_terminal_transition, dtype=np.bool_
            ),
            terminal_step=np.array(self.terminal_step, dtype=np.int32),
            terminal_done_reason=np.array(self.terminal_done_reason, dtype="U16"),
            terminal_reward=np.array(self.terminal_reward, dtype=np.float32),
            terminal_relative_distance=np.array(self.terminal_relative_distance, dtype=np.float32),
            step=steps,
            ego_obs=tensor_rows_to_numpy(self.ego_obs, width=STATE_DIM),
            prev_action=tensor_rows_to_numpy(self.prev_action, width=ACTION_DIM),
            behavior_action=tensor_rows_to_numpy(self.action, width=ACTION_DIM),
            policy_mean=tensor_rows_to_numpy(self.policy_mean, width=ACTION_DIM),
            policy_std=tensor_rows_to_numpy(self.policy_std, width=ACTION_DIM),
            pre_tanh_action=tensor_rows_to_numpy(self.pre_tanh_action, width=ACTION_DIM),
            reward=np.asarray(self.reward, dtype=np.float32),
            relative_distance=rel_d,
            lidar_finite_fraction=np.asarray(self.lidar_finite_fraction, dtype=np.float32),
            lidar_min=np.asarray(self.lidar_min, dtype=np.float32),
            lidar_max=np.asarray(self.lidar_max, dtype=np.float32),
            label_robot_state=robot_np,
            label_target_state=target_np,
            label_relative_position_body=rel_body_np,
            label_detectable=detectable_np,
            target_pixel_count=target_counts,
            target_bbox_xyxy=np.asarray(self.target_bbox_xyxy, dtype=np.int32),
            target_centroid_uv=np.asarray(self.target_centroid_uv, dtype=np.float32),
            target_area_ratio=np.asarray(self.target_area_ratio, dtype=np.float32),
            target_visible=target_visible_np,
            qa_semantic_available=semantic_available_np,
            qa_detectable_not_visible=detectable_not_visible_np,
            qa_visible_not_detectable=visible_not_detectable_np,
            qa_hard_mismatch_eligible=hard_mismatch_eligible_np,
            qa_hard_mismatch=hard_mismatch_np,
            lidar_chunk_relative_paths=np.asarray(self.chunk_relpaths, dtype="U256"),
            lidar_chunk_num_frames=np.asarray(self.chunk_num_frames, dtype=np.int32),
            lidar_chunk_bytes=np.asarray(self.chunk_bytes, dtype=np.int64),
            lidar_height=np.array(self.lidar_height, dtype=np.int32),
            lidar_width=np.array(self.lidar_width, dtype=np.int32),
            lidar_dtype=np.array("float16", dtype="U16"),
            risk_horizons=np.asarray(args.risk_horizons, dtype=np.int32),
            **label_payload,
        )

        valid_target_counts = target_counts[target_counts >= 0]
        visible_target_counts = target_counts[target_visible_np]
        metadata_relpath = os.path.relpath(metadata_path, args.output_dir)
        total_lidar_bytes = int(sum(self.chunk_bytes))
        label_summary = {}
        for horizon_steps in args.risk_horizons:
            valid_key = label_key("label_valid", horizon_steps)
            loss_key = label_key("label_loss_prob", horizon_steps)
            valid_np = label_payload[valid_key].astype(np.bool_)
            loss_np = label_payload[loss_key]
            suffix = f"h{int(horizon_steps):03d}"
            label_summary[f"label_valid_{suffix}_rate"] = float(np.mean(valid_np))
            label_summary[f"label_loss_prob_{suffix}_rate"] = (
                float(np.mean(loss_np[valid_np])) if np.any(valid_np) else None
            )
        artifact = {
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "dataset_episode_id": self.dataset_episode_id,
            "checkpoint_episode_idx": self.checkpoint_episode_idx,
            "env_id": self.item.env_id,
            "update": self.item.update,
            "checkpoint_path": self.item.checkpoint_path,
            "num_steps": int(len(steps)),
            "transition_schema": NONTERMINAL_TRANSITION_SCHEMA,
            "discarded_terminal_transition": bool(self.discarded_terminal_transition),
            "terminal_step": int(self.terminal_step),
            "terminal_done_reason": self.terminal_done_reason,
            "terminal_reward": nullable_float(self.terminal_reward),
            "terminal_relative_distance": nullable_float(self.terminal_relative_distance),
            "final_relative_distance": float(rel_d[-1]),
            "last_observed_relative_distance": float(rel_d[-1]),
            "min_relative_distance": float(np.min(rel_d)),
            "detectable_rate": float(np.mean(detectable_np)),
            "label_source": "semantic_target_pixel_count",
            "label_min_visible_pixels": int(args.label_min_visible_pixels),
            **label_summary,
            **visibility_stats,
            "target_semantic_frames": int(len(valid_target_counts)),
            "target_visible_frames": int(len(visible_target_counts)),
            "target_visible_frame_rate": (
                float(len(visible_target_counts) / len(valid_target_counts))
                if len(valid_target_counts)
                else None
            ),
            "target_pixel_count_mean_visible_frames": (
                float(np.mean(visible_target_counts)) if len(visible_target_counts) else None
            ),
            "target_pixel_count_max_frames": (
                int(np.max(valid_target_counts)) if len(valid_target_counts) else None
            ),
            "qa_semantic_available_frames": int(np.count_nonzero(semantic_available_np)),
            "qa_detectable_frames": int(np.count_nonzero(detectable_np)),
            "qa_detectable_visible_frames": int(
                np.count_nonzero(semantic_available_np & detectable_np & target_visible_np)
            ),
            "qa_detectable_not_visible_frames": int(np.count_nonzero(detectable_not_visible_np)),
            "qa_visible_not_detectable_frames": int(np.count_nonzero(visible_not_detectable_np)),
            "qa_hard_mismatch_eligible_frames": int(
                np.count_nonzero(hard_mismatch_eligible_np)
            ),
            "qa_hard_mismatch_frames": int(np.count_nonzero(hard_mismatch_np)),
            "qa_visible_given_detectable_rate": (
                float(
                    np.count_nonzero(semantic_available_np & detectable_np & target_visible_np)
                    / np.count_nonzero(semantic_available_np & detectable_np)
                )
                if np.count_nonzero(semantic_available_np & detectable_np)
                else None
            ),
            "qa_detectable_not_visible_rate": (
                float(np.count_nonzero(detectable_not_visible_np) / np.count_nonzero(detectable_np))
                if np.count_nonzero(detectable_np)
                else None
            ),
            "qa_hard_mismatch_rate": (
                float(np.count_nonzero(hard_mismatch_np) / np.count_nonzero(hard_mismatch_eligible_np))
                if np.count_nonzero(hard_mismatch_eligible_np)
                else None
            ),
            "qa_hard_mismatch_frame_rate": (
                float(np.count_nonzero(hard_mismatch_np) / len(steps)) if len(steps) else None
            ),
            "lidar_finite_fraction_min": float(np.min(self.lidar_finite_fraction)),
            "lidar_finite_fraction_mean": float(np.mean(self.lidar_finite_fraction)),
            "lidar_norm_min": float(np.min(self.lidar_min)),
            "lidar_norm_max": float(np.max(self.lidar_max)),
            "lidar_chunks": int(len(self.chunk_relpaths)),
            "lidar_frames": int(sum(self.chunk_num_frames)),
            "lidar_bytes": total_lidar_bytes,
            "metadata_relative_path": metadata_relpath,
            "episode_relative_dir": os.path.relpath(self.episode_dir, args.output_dir),
        }
        return artifact


def _is_relative_to(path: str, base: str) -> bool:
    try:
        return os.path.commonpath([path, base]) == base
    except ValueError:
        return False


def validate_output_dir_safety(args):
    output_dir = os.path.realpath(args.output_dir)
    run_dir = os.path.realpath(args.run_dir)
    cwd = os.path.realpath(os.getcwd())
    forbidden = {
        run_dir,
        cwd,
        os.path.realpath(os.path.join(run_dir, "policy_pool")),
    }
    if output_dir in forbidden:
        raise ValueError(f"Refusing dangerous output directory: {args.output_dir}")
    if _is_relative_to(run_dir, output_dir) or _is_relative_to(output_dir, run_dir):
        raise ValueError(
            "Refusing output directory that overlaps the source run directory: "
            f"output_dir={args.output_dir}, run_dir={args.run_dir}"
        )


def prepare_output_dir(args):
    if args.dry_run:
        return
    validate_output_dir_safety(args)
    marker_path = os.path.join(args.output_dir, ".pursuit_lidar_risk_dataset")
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        if not args.overwrite:
            raise FileExistsError(
                f"Output dir is not empty: {args.output_dir}. Pass --overwrite to replace it."
            )
        if not os.path.exists(marker_path):
            raise FileExistsError(
                f"Refusing to overwrite unmarked directory: {args.output_dir}. "
                "Delete it manually or use a directory created by this exporter."
            )
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "episodes"), exist_ok=True)
    with open(marker_path, "w", encoding="utf-8") as f:
        f.write("pursuit_lidar_risk_dataset\n")


def build_dataset_manifest_payload(
    args,
    selected: Sequence[SelectedCheckpoint],
    episode_counts: Sequence[int],
    lidar_cfg,
    status: str,
    summary: Optional[dict] = None,
):
    selected_payload = []
    for item, count in zip(selected, episode_counts):
        row = selected_to_dict(item)
        row["planned_episodes"] = int(count)
        selected_payload.append(row)

    payload = {
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "dataset_kind": (
                "pursuit_lidar_branch_risk_dataset"
                if args.dataset_mode == "branch"
                else "pursuit_lidar_risk_dataset"
            ),
            "status": status,
            "dataset_mode": args.dataset_mode,
            "run_dir": args.run_dir,
            "output_dir": args.output_dir,
            "task": args.task,
            "robot_name": args.robot_name,
            "controller_name": args.controller_name,
            "target_asset_type": args.target_asset_type,
            "total_episodes": int(args.total_episodes),
            "parallel_envs": int(args.parallel_envs),
            "max_steps": int(args.max_steps),
            "seed": int(args.seed),
            "persist_steps": int(args.persist_steps),
            "risk_horizons": [int(value) for value in args.risk_horizons],
            "lidar_storage": {
                "mode": "full_stream_chunked_npz",
                "path_base": "output_dir",
                "chunk_steps": int(args.lidar_chunk_steps),
                "dtype": "float16",
                "array_name": "lidar_range_norm",
                "risk_lidar_stack_frames": int(args.risk_lidar_stack_frames),
                "branch_anchor_stack_frames": (
                    int(args.risk_lidar_stack_frames)
                    if args.dataset_mode == "branch"
                    else None
                ),
            },
            "risk_model_contract": {
                "s_red": (
                    f"ego_obs[{STATE_DIM}]=body_linvel(3)+body_angvel(3)+"
                    f"rotation_matrix(9)+prev_action({ACTION_DIM})"
                ),
                "z_lidar_source": f"K={args.risk_lidar_stack_frames} lidar_range_norm frames",
                "action_input": "u=[c,p,q,r], plus u-prev_action, abs(u), u^2 in the risk loader/model",
                "policy_mean_usage": "metadata_and_candidate_generation_only_not_u_ref",
                "ppo_v0_label_heads": [
                    "label_loss_prob_h050",
                    "label_loss_severity_h050",
                    "label_loss_prob_h150",
                    "label_loss_severity_h150",
                ],
                "auxiliary_label_heads": [
                    "label_recovery_h150",
                ],
                "risk_scalar_weights": dict(RISK_SCALAR_WEIGHTS),
            },
            "transition_schema": (
                BRANCH_TRANSITION_SCHEMA
                if args.dataset_mode == "branch"
                else NONTERMINAL_TRANSITION_SCHEMA
            ),
            "terminal_transition_policy": {
                "done_true_transitions": "discarded",
                "reason": "post-done sensor render may include reset state",
                "terminal_outcome": "stored only as scalar episode metadata",
            },
            "label_validity": {
                "source": "semantic_target_pixel_count",
                "min_visible_pixels": int(args.label_min_visible_pixels),
                "mask_prefix": "label_valid_h",
                "rule": "valid when t + horizon_steps + persist_steps - 1 <= num_steps",
            },
            "training_inputs": (
                [
                    "anchor_lidar_relative_path:lidar_range_norm",
                    "ego_obs",
                    "candidate_action",
                ]
                if args.dataset_mode == "branch"
                else [
                    "lidar_range_norm",
                    "ego_obs",
                    "prev_action",
                    "behavior_action",
                ]
            ),
            "diagnostic_metadata_fields": [
                "policy_mean",
                "policy_std",
                "pre_tanh_action",
            ],
            "label_only_fields": [
                "label_robot_state",
                "label_target_state",
                "label_relative_position_body",
                "label_detectable",
                "relative_distance",
                "target_visible",
                "target_pixel_count",
                "target_bbox_xyxy",
                "target_centroid_uv",
                "target_area_ratio",
                "anchor_detectable",
                "label_loss_prob_h*",
                "label_loss_severity_h*",
                "label_valid_h*",
                "label_recovery_h*",
                "label_recovery_valid_h*",
                "qa_*",
                "terminal_*",
                "discarded_terminal_transition",
            ],
            "leakage_excluded_from_training_inputs": [
                "label_*",
                "target_*",
                "anchor_detectable",
                "relative_distance",
                "reward",
                "pre_tanh_action",
                "terminal_*",
                "discarded_terminal_transition",
            ],
            "counterfactual_action_labels": bool(args.dataset_mode == "branch"),
            "same_state_action_branching": (
                "branch_mvp_same_anchor_candidate_actions"
                if args.dataset_mode == "branch"
                else "not_collected_in_dataset_v1"
            ),
            "branch_sampling": (
                {
                    "update": args.branch_update,
                    "updates": [int(item.update) for item in selected],
                    "episodes_per_update": int(args.branch_episodes),
                    "anchor_stride": int(args.branch_anchor_stride),
                    "max_anchors": int(args.branch_max_anchors),
                    "candidates": int(args.branch_candidates),
                    "horizon": int(args.branch_horizon),
                    "edge_margin_px": int(args.branch_edge_margin_px),
                    "low_pixel_max": int(args.branch_low_pixel_max),
                    "risk_anchor_min_score": float(args.branch_risk_anchor_min_score),
                    "min_candidate_action_l2": float(args.branch_min_candidate_action_l2),
                    "min_severity_separation": float(args.branch_min_severity_separation),
                    "min_separated_anchors": int(args.branch_min_separated_anchors),
                    "min_visible_anchor_rate": float(args.branch_min_visible_anchor_rate),
                    "min_detectable_anchor_rate": float(args.branch_min_detectable_anchor_rate),
                }
                if args.dataset_mode == "branch"
                else None
            ),
            "lidar_config": lidar_cfg.__name__,
            "lidar_segmentation_camera": bool(getattr(lidar_cfg, "segmentation_camera", False)),
            "enable_lidar_segmentation": bool(args.enable_lidar_segmentation),
            "target_semantic_id": target_semantic_id_for_asset(args.target_asset_type),
            "label_source": "semantic_target_pixel_count",
            "label_min_visible_pixels": int(args.label_min_visible_pixels),
            "qa_gate": {
                "enabled": not bool(args.qa_disable_gate),
                "window_episodes": int(args.qa_window_episodes),
                "min_semantic_frames": int(args.qa_min_semantic_frames),
                "min_detectable_frames": int(args.qa_min_detectable_frames),
                "min_visible_given_detectable": float(args.qa_min_visible_given_detectable),
                "max_hard_mismatch_rate": float(args.qa_max_hard_mismatch_rate),
                "min_hard_mismatch_frames": int(args.qa_min_hard_mismatch_frames),
                "hard_mismatch_max_range_m": float(args.qa_hard_mismatch_max_range_m),
                "min_visible_pixels": int(args.qa_min_visible_pixels),
                "baseline_path": args.qa_baseline_path,
            },
            "lidar_height": int(lidar_cfg.height),
            "lidar_width": int(lidar_cfg.width),
            "lidar_max_range_m": float(lidar_cfg.max_range),
            "lidar_min_range_m": float(getattr(lidar_cfg, "min_range", 0.0)),
            "lidar_horizontal_fov_deg": [
                float(lidar_cfg.horizontal_fov_deg_min),
                float(lidar_cfg.horizontal_fov_deg_max),
            ],
            "lidar_vertical_fov_deg": [
                float(lidar_cfg.vertical_fov_deg_min),
                float(lidar_cfg.vertical_fov_deg_max),
            ],
            "checkpoints": selected_payload,
    }
    if summary is not None:
        payload["actual_summary"] = summary
    return payload


def write_dataset_manifest(path: str, payload: dict):
    write_json(path, payload)


def _numeric_values(rows: Sequence[dict], key: str) -> List[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        values.append(float(value))
    return values


def summarize_branch_rows(rows: Sequence[dict]) -> dict:
    anchors = len(rows)
    updates = sorted({int(row["update"]) for row in rows}) if rows else []
    loss_ranges = _numeric_values(rows, "candidate_loss_range")
    severity_ranges = _numeric_values(rows, "candidate_severity_range")
    anchor_scores = _numeric_values(rows, "anchor_selection_score")
    separated_anchors = sum(1 for row in rows if bool(row.get("candidate_label_has_separation")))
    visible_anchors = sum(1 for row in rows if int(row.get("target_pixel_count") or 0) > 0)
    detectable_anchors = sum(1 for row in rows if bool(row.get("anchor_detectable")))

    def mean_or_none(values: Sequence[float]) -> Optional[float]:
        return float(np.mean(values)) if values else None

    def max_or_none(values: Sequence[float]) -> Optional[float]:
        return float(np.max(values)) if values else None

    return {
        "episodes": anchors,
        "anchors": anchors,
        "lidar_bytes": sum(int(row.get("lidar_bytes") or 0) for row in rows),
        "lidar_frames": sum(int(row.get("lidar_frames") or 0) for row in rows),
        "updates": updates,
        "branch_label_separated_anchors": int(separated_anchors),
        "branch_label_separation_rate": safe_rate(separated_anchors, anchors),
        "branch_binary_loss_separated_anchors": int(sum(value > 0.0 for value in loss_ranges)),
        "branch_severity_separated_anchors": int(
            sum(value >= DEFAULT_BRANCH_MIN_SEVERITY_SEPARATION for value in severity_ranges)
        ),
        "branch_visible_anchor_rate": safe_rate(visible_anchors, anchors),
        "branch_detectable_anchor_rate": safe_rate(detectable_anchors, anchors),
        "candidate_loss_range_mean": mean_or_none(loss_ranges),
        "candidate_loss_range_max": max_or_none(loss_ranges),
        "candidate_severity_range_mean": mean_or_none(severity_ranges),
        "candidate_severity_range_max": max_or_none(severity_ranges),
        "anchor_selection_score_mean": mean_or_none(anchor_scores),
    }


def summarize_index(index_path: str) -> dict:
    num_episodes = 0
    lidar_bytes = 0
    lidar_frames = 0
    updates = set()
    rows = []
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                rows.append(row)
                num_episodes += 1
                lidar_bytes += int(row.get("lidar_bytes") or 0)
                lidar_frames += int(row.get("lidar_frames") or 0)
                updates.add(int(row["update"]))
    if rows and rows[0].get("dataset_kind") == "pursuit_lidar_branch_risk_dataset":
        return summarize_branch_rows(rows)
    return {
        "episodes": num_episodes,
        "lidar_bytes": lidar_bytes,
        "lidar_frames": lidar_frames,
        "updates": sorted(updates),
    }


def validate_branch_summary(args, summary: dict):
    anchors = int(summary.get("anchors") or 0)
    separated = int(summary.get("branch_label_separated_anchors") or 0)
    min_separated = int(args.branch_min_separated_anchors)
    visible_rate = summary.get("branch_visible_anchor_rate")
    detectable_rate = summary.get("branch_detectable_anchor_rate")
    if anchors <= 0:
        raise RuntimeError(
            "Branch sampler wrote no anchors. Increase --max-steps, lower "
            "--branch-anchor-stride, or lower --branch-risk-anchor-min-score."
        )
    if (
        visible_rate is None
        or visible_rate < float(args.branch_min_visible_anchor_rate)
    ):
        raise RuntimeError(
            "Branch QA failed visible-anchor coverage: "
            f"visible_rate={visible_rate}, required={float(args.branch_min_visible_anchor_rate):.3f}"
        )
    if (
        detectable_rate is None
        or detectable_rate < float(args.branch_min_detectable_anchor_rate)
    ):
        raise RuntimeError(
            "Branch QA failed detectable-anchor coverage: "
            f"detectable_rate={detectable_rate}, required={float(args.branch_min_detectable_anchor_rate):.3f}"
        )
    if separated < min_separated:
        raise RuntimeError(
            "Branch labels are not sufficiently action-discriminative: "
            f"separated_anchors={separated}, required={min_separated}, anchors={anchors}. "
            "Increase --branch-max-anchors or sample harder/visible anchors before scaling collection."
        )


def load_qa_baseline(path: Optional[str]) -> dict:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return {int(row["update"]): row for row in payload.get("updates", [])}


def safe_rate(numerator: int, denominator: int) -> Optional[float]:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def aggregate_artifacts(rows: Sequence[dict], risk_horizons: Sequence[int]) -> dict:
    episodes = len(rows)
    frames = sum(int(row.get("num_steps") or 0) for row in rows)
    semantic_frames = sum(int(row.get("qa_semantic_available_frames") or 0) for row in rows)
    detectable_frames = sum(int(row.get("qa_detectable_frames") or 0) for row in rows)
    detectable_visible_frames = sum(
        int(row.get("qa_detectable_visible_frames") or 0) for row in rows
    )
    detectable_not_visible_frames = sum(
        int(row.get("qa_detectable_not_visible_frames") or 0) for row in rows
    )
    hard_mismatch_frames = sum(int(row.get("qa_hard_mismatch_frames") or 0) for row in rows)
    hard_mismatch_eligible_frames = sum(
        int(row.get("qa_hard_mismatch_eligible_frames") or 0) for row in rows
    )
    visible_frames = sum(int(row.get("target_visible_frames") or 0) for row in rows)
    target_semantic_frames = sum(int(row.get("target_semantic_frames") or 0) for row in rows)
    summary = {
        "episodes": episodes,
        "frames": frames,
        "semantic_frames": semantic_frames,
        "target_visible_frames": visible_frames,
        "target_visible_frame_rate": safe_rate(visible_frames, target_semantic_frames),
        "detectable_frames": detectable_frames,
        "detectable_visible_frames": detectable_visible_frames,
        "detectable_not_visible_frames": detectable_not_visible_frames,
        "visible_given_detectable_rate": safe_rate(
            detectable_visible_frames, detectable_frames
        ),
        "detectable_not_visible_rate": safe_rate(
            detectable_not_visible_frames, detectable_frames
        ),
        "hard_mismatch_eligible_frames": hard_mismatch_eligible_frames,
        "hard_mismatch_frames": hard_mismatch_frames,
        "hard_mismatch_rate": safe_rate(hard_mismatch_frames, hard_mismatch_eligible_frames),
        "hard_mismatch_frame_rate": safe_rate(hard_mismatch_frames, frames),
        "lidar_finite_fraction_min": min(
            float(row.get("lidar_finite_fraction_min", 1.0)) for row in rows
        ) if rows else None,
        "lidar_finite_fraction_mean": float(
            np.mean([float(row.get("lidar_finite_fraction_mean", 1.0)) for row in rows])
        ) if rows else None,
    }
    for horizon_steps in risk_horizons:
        suffix = f"h{int(horizon_steps):03d}"
        values = [
            row.get(f"label_loss_prob_{suffix}_rate")
            for row in rows
            if row.get(f"label_loss_prob_{suffix}_rate") is not None
        ]
        summary[f"label_loss_prob_{suffix}_rate"] = (
            float(np.mean(values)) if values else None
        )
    return summary


def validate_qa_window(
    args,
    update: int,
    summary: dict,
    baseline_row: Optional[dict],
    final: bool,
) -> tuple:
    failures = []
    warnings = []
    semantic_frames = int(summary.get("semantic_frames") or 0)
    detectable_frames = int(summary.get("detectable_frames") or 0)
    hard_eligible_frames = int(summary.get("hard_mismatch_eligible_frames") or 0)
    hard_mismatch_frames = int(summary.get("hard_mismatch_frames") or 0)
    hard_rate = summary.get("hard_mismatch_rate")
    visible_given_detectable = summary.get("visible_given_detectable_rate")
    full_window = int(summary.get("episodes") or 0) >= int(args.qa_window_episodes)

    if final and not full_window:
        warnings.append(
            f"final_partial_qa_window_episodes={int(summary.get('episodes') or 0)} below "
            f"{int(args.qa_window_episodes)}"
        )
    if bool(args.enable_lidar_segmentation) and semantic_frames <= 0:
        failures.append("semantic QA requested but no semantic frames were recorded")
    if semantic_frames >= int(args.qa_min_semantic_frames):
        if (
            hard_eligible_frames >= int(args.qa_min_detectable_frames)
            and hard_mismatch_frames >= int(args.qa_min_hard_mismatch_frames)
            and hard_rate is not None
            and hard_rate > float(args.qa_max_hard_mismatch_rate)
        ):
            failures.append(
                f"hard_mismatch_rate={hard_rate:.4f} exceeds "
                f"{float(args.qa_max_hard_mismatch_rate):.4f} "
                f"with frames={hard_mismatch_frames}"
            )
        if detectable_frames >= int(args.qa_min_detectable_frames):
            min_visible = float(args.qa_min_visible_given_detectable)
            if visible_given_detectable is not None and visible_given_detectable < min_visible:
                failures.append(
                    f"visible_given_detectable={visible_given_detectable:.4f} below "
                    f"{min_visible:.4f}"
                )
        else:
            warnings.append(
                f"insufficient_detectable_frames={detectable_frames} below "
                f"{int(args.qa_min_detectable_frames)}"
            )
    else:
        warnings.append(
            f"insufficient_semantic_frames={semantic_frames} below "
            f"{int(args.qa_min_semantic_frames)}"
        )

    if baseline_row is not None:
        tolerance = float(args.qa_baseline_relative_tolerance)
        for key in (
            "target_visible_frame_rate",
            "visible_given_detectable_rate",
            "hard_mismatch_rate",
        ):
            expected = baseline_row.get(key)
            actual = summary.get(key)
            if expected is None or actual is None:
                continue
            if key == "target_visible_frame_rate" and semantic_frames < int(args.qa_min_semantic_frames):
                warnings.append(f"skip baseline {key}: insufficient semantic frames")
                continue
            if key == "visible_given_detectable_rate" and detectable_frames < int(args.qa_min_detectable_frames):
                warnings.append(f"skip baseline {key}: insufficient detectable frames")
                continue
            if key == "hard_mismatch_rate" and hard_eligible_frames < int(args.qa_min_detectable_frames):
                warnings.append(f"skip baseline {key}: insufficient hard-mismatch eligible frames")
                continue
            if key == "hard_mismatch_rate" and hard_mismatch_frames < int(args.qa_min_hard_mismatch_frames):
                warnings.append(
                    f"skip baseline {key}: hard_mismatch_frames={hard_mismatch_frames} below "
                    f"{int(args.qa_min_hard_mismatch_frames)}"
                )
                continue
            if not full_window and final:
                warnings.append(f"skip baseline {key}: final partial QA window")
                continue
            floor = max(float(expected) * (1.0 - tolerance), 0.0)
            if key == "hard_mismatch_rate":
                ceiling = max(
                    float(expected) * (1.0 + tolerance),
                    float(args.qa_max_hard_mismatch_rate),
                )
                if actual > ceiling:
                    failures.append(
                        f"{key}={actual:.4f} above baseline ceiling {ceiling:.4f} "
                        f"for update={update}"
                    )
                continue
            if actual < floor:
                failures.append(
                    f"{key}={actual:.4f} below baseline floor {floor:.4f} "
                    f"for update={update}"
                )
    elif args.qa_mode == "collect":
        failures.append(f"QA baseline is missing update={update}")

    if failures:
        return "fail", failures, warnings
    if warnings:
        return "insufficient", failures, warnings
    return "pass", failures, warnings


class RollingQAGate:
    def __init__(self, args, output_dir: str):
        self.args = args
        self.qa_path = os.path.join(output_dir, "qa", "qa_windows.jsonl")
        self.baseline = load_qa_baseline(args.qa_baseline_path)
        self.windows: Dict[int, List[dict]] = {}

    def add(self, artifact: Optional[dict]):
        if artifact is None:
            return
        update = int(artifact["update"])
        window = self.windows.setdefault(update, [])
        window.append(artifact)
        if len(window) >= int(self.args.qa_window_episodes):
            self.flush(update, final=False)

    def flush(self, update: int, final: bool):
        rows = self.windows.get(update) or []
        if not rows:
            return
        summary = aggregate_artifacts(rows, self.args.risk_horizons)
        baseline_row = self.baseline.get(update)
        status, failures, warnings = validate_qa_window(
            self.args,
            update,
            summary,
            baseline_row,
            final=final,
        )
        payload = {
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "update": int(update),
            "final": bool(final),
            "status": status,
            "failures": failures,
            "warnings": warnings,
            **summary,
        }
        append_jsonl(self.qa_path, payload)
        self.windows[update] = []
        if failures and not bool(self.args.qa_disable_gate):
            raise RuntimeError(f"QA gate failed for update {update}: {failures}")

    def flush_all(self):
        for update in list(self.windows.keys()):
            self.flush(update, final=True)


def build_qa_baseline(index_path: str, output_path: str, args):
    rows_by_update: Dict[int, List[dict]] = {}
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows_by_update.setdefault(int(row["update"]), []).append(row)

    updates = []
    for update in sorted(rows_by_update):
        summary = aggregate_artifacts(rows_by_update[update], args.risk_horizons)
        updates.append(
            {
                "update": int(update),
                "source": "pilot",
                **summary,
            }
        )
    payload = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "baseline_kind": "ladder_qa_pilot",
        "relative_tolerance": float(args.qa_baseline_relative_tolerance),
        "updates": updates,
    }
    write_json(output_path, payload)


def validate_qa_baseline_coverage(path: str, selected: Sequence[SelectedCheckpoint]):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if int(payload.get("schema_version", -1)) != OUTPUT_SCHEMA_VERSION:
        raise RuntimeError(
            f"QA baseline schema mismatch: expected={OUTPUT_SCHEMA_VERSION}, "
            f"actual={payload.get('schema_version')}, path={path}"
        )
    if payload.get("baseline_kind") != "ladder_qa_pilot":
        raise RuntimeError(f"Unexpected QA baseline kind: {payload.get('baseline_kind')}")
    rows = payload.get("updates")
    if not isinstance(rows, list):
        raise RuntimeError(f"QA baseline is missing updates list: {path}")
    for row in rows:
        missing_keys = [key for key in REQUIRED_QA_BASELINE_KEYS if key not in row]
        if missing_keys:
            raise RuntimeError(
                f"QA baseline update={row.get('update')} missing keys={missing_keys}"
            )
    baseline = {int(row["update"]): row for row in rows}
    expected = {int(item.update) for item in selected}
    actual = set(baseline.keys())
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise RuntimeError(
            f"QA baseline coverage mismatch. missing={missing}, extra={extra}, path={path}"
        )


def collect_batch(
    args,
    envs,
    lidar_cfg,
    frustum: DetectionFrustum,
    target_semantic_id: int,
    selected_batch: Sequence[SelectedCheckpoint],
    episode_counts_batch: Sequence[int],
    batch_start_idx: int,
    next_episode_id: int,
    index_path: str,
    qa_gate: RollingQAGate,
) -> int:
    batch_size = int(envs.num_envs)
    batch_seed = int(args.seed) + int(batch_start_idx)
    random.seed(batch_seed)
    np.random.seed(batch_seed)
    torch.manual_seed(batch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(batch_seed)

    env = envs.env
    agents = load_agents(selected_batch, envs, args.device)

    max_episodes = max(int(count) for count in episode_counts_batch)
    for checkpoint_episode_idx in range(max_episodes):
        active_slots = [
            env_id
            for env_id, count in enumerate(episode_counts_batch)
            if checkpoint_episode_idx < int(count)
        ]
        if not active_slots:
            continue

        next_obs, _ = envs.reset()
        env.sim_env.render(render_components="sensors")
        alive = torch.zeros(batch_size, dtype=torch.bool, device=args.device)
        alive[active_slots] = True
        writers = {}
        for env_id in active_slots:
            writers[env_id] = EpisodeShardWriter(
                output_dir=args.output_dir,
                item=selected_batch[env_id],
                dataset_episode_id=next_episode_id,
                checkpoint_episode_idx=checkpoint_episode_idx,
                lidar_height=int(lidar_cfg.height),
                lidar_width=int(lidar_cfg.width),
                chunk_steps=int(args.lidar_chunk_steps),
            )
            next_episode_id += 1

        for step in range(int(args.max_steps)):
            obs_before = next_obs.detach().clone()
            robot_state_before = env.robot_state.detach().clone()
            target_state_before = env.target_state.detach().clone()
            prev_action_before = env.actions.detach().clone()
            ego_obs_before = deployable_ego_obs(env, prev_action_before).detach().clone()
            lidar_tensor_before = env.obs_dict.get("depth_range_pixels")
            if lidar_tensor_before is None:
                raise RuntimeError("depth_range_pixels tensor was not created")
            lidar_tensor_before = lidar_tensor_before.detach().clone()
            segmentation_tensor_before = env.obs_dict.get("segmentation_pixels")
            if segmentation_tensor_before is not None:
                segmentation_tensor_before = segmentation_tensor_before.detach().clone()

            actions = torch.zeros(
                (batch_size, envs.num_actions), dtype=torch.float32, device=args.device
            )
            policy_mean = torch.zeros_like(actions)
            policy_std = torch.zeros_like(actions)
            pre_tanh = torch.zeros_like(actions)

            with torch.no_grad():
                for env_id, agent in enumerate(agents):
                    if not bool(alive[env_id].item()):
                        continue
                    action, _, _, _, aux = agent.get_action_and_value(
                        obs_before[env_id : env_id + 1], return_aux=True
                    )
                    actions[env_id] = action[0]
                    policy_mean[env_id] = aux["action_mean"][0]
                    policy_std[env_id] = aux["action_std"][0]
                    pre_tanh[env_id] = aux["pre_tanh_action"][0]

            next_obs, rewards, done_float, info = envs.step(actions)

            for env_id in active_slots:
                if not bool(alive[env_id].item()):
                    continue
                done = bool(done_float[env_id].item())
                done_reason = extract_done_reason(info, env_id)
                frame_robot_state = robot_state_before[env_id]
                frame_target_state = target_state_before[env_id]
                rel_dist = torch.linalg.norm(frame_target_state[0:3] - frame_robot_state[0:3])
                if done:
                    terminal_rel_dist = float(rel_dist.detach().cpu().item())
                    info_rel_dist = info.get("relative_dist")
                    if info_rel_dist is not None:
                        terminal_rel_dist = float(info_rel_dist[env_id].detach().cpu().item())
                    writers[env_id].mark_discarded_terminal_transition(
                        step=step,
                        reward=float(rewards[env_id].detach().cpu().item()),
                        done_reason=done_reason,
                        relative_distance=terminal_rel_dist,
                    )
                    alive[env_id] = False
                    continue
                lidar_norm = (
                    lidar_tensor_before[env_id, 0]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )
                semantic_frame = None
                if segmentation_tensor_before is not None:
                    semantic_frame = (
                        segmentation_tensor_before[env_id, 0]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.int32, copy=False)
                    )
                elif bool(args.enable_lidar_segmentation):
                    raise RuntimeError(
                        "LiDAR segmentation QA is enabled but segmentation_pixels is missing."
                    )
                semantic_stats = target_mask_stats(semantic_frame, target_semantic_id)
                writers[env_id].append_frame(
                    step=step,
                    robot_state=frame_robot_state,
                    target_state=frame_target_state,
                    ego_obs=ego_obs_before[env_id],
                    prev_action=prev_action_before[env_id],
                    action=actions[env_id],
                    policy_mean=policy_mean[env_id],
                    policy_std=policy_std[env_id],
                    pre_tanh_action=pre_tanh[env_id],
                    reward=float(rewards[env_id].detach().cpu().item()),
                    done=done,
                    done_reason=done_reason,
                    relative_distance=float(rel_dist.detach().cpu().item()),
                    lidar_norm=lidar_norm,
                    semantic_stats=semantic_stats,
                )

            if not torch.any(alive):
                break

        for env_id in active_slots:
            artifact = writers[env_id].close(
                args=args,
                frustum=frustum,
                target_semantic_id=target_semantic_id,
            )
            if artifact is None:
                print(
                    "skipped empty terminal-only episode "
                    f"update={selected_batch[env_id].update} "
                    f"checkpoint_episode_idx={checkpoint_episode_idx}"
                )
                continue
            append_jsonl(index_path, artifact)
            qa_gate.add(artifact)
            print(
                "saved "
                f"episode={artifact['dataset_episode_id']:06d} "
                f"update={artifact['update']} "
                f"steps={artifact['num_steps']} "
                f"terminal={artifact['terminal_done_reason']} "
                f"lidar_mb={artifact['lidar_bytes'] / (1024.0 * 1024.0):.1f}"
            )

    return next_episode_id


def collect_dataset(args, selected: Sequence[SelectedCheckpoint], episode_counts: Sequence[int]):
    index_path = os.path.join(args.output_dir, "index.jsonl")
    if os.path.exists(index_path):
        os.remove(index_path)
    qa_path = os.path.join(args.output_dir, "qa", "qa_windows.jsonl")
    if os.path.exists(qa_path):
        os.remove(qa_path)

    next_episode_id = 0
    parallel_envs = max(1, int(args.parallel_envs))
    qa_gate = RollingQAGate(args, args.output_dir)
    envs = build_env(args, parallel_envs, seed=int(args.seed))
    env = envs.env
    lidar_cfg = env.sim_env.robot_manager.cfg.sensor_config.lidar_config
    frustum = detection_frustum_from_lidar_config(lidar_cfg)
    target_semantic_id = target_semantic_id_for_asset(args.target_asset_type)
    try:
        for batch_start in range(0, len(selected), parallel_envs):
            selected_batch = selected[batch_start : batch_start + parallel_envs]
            counts_batch = episode_counts[batch_start : batch_start + parallel_envs]
            print(
                "starting batch "
                f"{batch_start // parallel_envs + 1} "
                f"updates={[item.update for item in selected_batch]} "
                f"episodes={counts_batch}"
            )
            next_episode_id = collect_batch(
                args=args,
                envs=envs,
                lidar_cfg=lidar_cfg,
                frustum=frustum,
                target_semantic_id=target_semantic_id,
                selected_batch=selected_batch,
                episode_counts_batch=counts_batch,
                batch_start_idx=batch_start,
                next_episode_id=next_episode_id,
                index_path=index_path,
                qa_gate=qa_gate,
            )
        qa_gate.flush_all()
    finally:
        try:
            envs.close()
        except Exception as exc:
            print(f"Cleanup failed: {exc}", file=sys.stderr)


def main():
    args = parse_args()
    assert_runtime()
    
    args.risk_horizons = sorted({int(horizon) for horizon in args.risk_horizons})

    if args.dataset_mode == "branch":
        if args.branch_updates is not None:
            checkpoint_updates = [int(update) for update in args.branch_updates]
        elif args.checkpoint_updates is not None:
            checkpoint_updates = [int(update) for update in args.checkpoint_updates]
        elif args.branch_update is not None:
            checkpoint_updates = [int(args.branch_update)]
        
        args.total_episodes = int(args.branch_episodes) * len(checkpoint_updates)
    else:
        checkpoint_updates = checkpoint_updates_from_args(args)
    selected = select_checkpoints(args.run_dir, checkpoint_updates)
    episode_counts = (
        [int(args.branch_episodes) for _ in selected]
        if args.dataset_mode == "branch"
        else distribute_episode_counts(args.total_episodes, len(selected))
    )
    print_plan(selected, episode_counts)
    if args.dataset_mode == "branch":
        print(
            "Branch dataset mode stores same-anchor candidate actions with teacher-rollout risk labels."
        )
    else:
        print(
            "This dataset v1 stores behavior-action transitions only; "
            "same-state counterfactual action labels require branch dataset mode."
        )

    if args.dataset_mode == "behavior" and args.qa_mode == "collect":
        if args.qa_baseline_path is None:
            raise ValueError("--qa-mode collect requires --qa-baseline-path")
        validate_qa_baseline_coverage(args.qa_baseline_path, selected)

    prepare_output_dir(args)
    configure_lidar_task(args, num_envs=1)
    robot_config = robot_registry.get_robot_config(args.robot_name)
    lidar_cfg = robot_config.sensor_config.lidar_config
    pending_manifest_path = os.path.join(args.output_dir, "manifest.pending.json")
    final_manifest_path = os.path.join(args.output_dir, "manifest.json")
    write_dataset_manifest(
        pending_manifest_path,
        build_dataset_manifest_payload(
            args,
            selected,
            episode_counts,
            lidar_cfg,
            status="pending",
        ),
    )
    if args.dataset_mode == "branch":
        collect_branch_dataset(args, selected)
    else:
        collect_dataset(args, selected, episode_counts)
    index_path = os.path.join(args.output_dir, "index.jsonl")
    if args.qa_baseline_output is not None:
        baseline_output = args.qa_baseline_output
        if not os.path.isabs(baseline_output):
            baseline_output = os.path.join(args.output_dir, baseline_output)
        build_qa_baseline(index_path, baseline_output, args)
        validate_qa_baseline_coverage(baseline_output, selected)
    summary = summarize_index(index_path)
    if args.dataset_mode == "branch":
        validate_branch_summary(args, summary)
        print(
            "branch label separation summary "
            f"anchors={summary['anchors']} "
            f"separated={summary['branch_label_separated_anchors']} "
            f"rate={summary['branch_label_separation_rate']}"
        )
    write_dataset_manifest(
        pending_manifest_path,
        build_dataset_manifest_payload(
            args,
            selected,
            episode_counts,
            lidar_cfg,
            status="complete",
            summary=summary,
        ),
    )
    os.replace(pending_manifest_path, final_manifest_path)
    print(f"Saved formal LiDAR risk dataset to {args.output_dir}")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
