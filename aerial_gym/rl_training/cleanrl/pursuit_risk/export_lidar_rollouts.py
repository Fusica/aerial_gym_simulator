"""Export branch-only pursuit LiDAR risk-dataset shards from a PPO policy pool."""

# Isaac Gym must be imported before torch.
import isaacgym

import argparse
import glob
import json
import os
import random
import sys
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from aerial_gym.config.asset_config import pursuit_guidance_asset_config
from aerial_gym.config.sensor_config.lidar_config import pursuit_forward_lidar_config
from aerial_gym.registry.robot_registry import robot_registry
from aerial_gym.registry.task_registry import task_registry
from aerial_gym.rl_training.cleanrl.ppo_guidance import (
    Agent,
    configure_target_asset_type,
)
from aerial_gym.rl_training.cleanrl.pursuit_risk.branch import (
    branch_anchor_selection_stats,
    branch_horizon_label_payload,
    build_branch_metadata_payload,
    candidate_actions_from_policy,
    lidar_stack_from_history,
    write_branch_chunk,
)
from aerial_gym.rl_training.cleanrl.pursuit_risk.contract import (
    DEFAULT_LIDAR_STACK_FRAMES,
    DEFAULT_RISK_HORIZONS,
    label_key,
)
from aerial_gym.utils.math import quat_to_rotation_matrix


DEFAULT_RUN_DIR = "runs/PE_20260520_110828"
DEFAULT_OUTPUT_DIR = "runs/risk_dataset/D0_branch_full_v1_10ckpt_100kanchor_h150_k3_m16"
DEFAULT_BRANCH_UPDATES = (170, 220, 300, 350, 464, 610, 754, 900, 1110, 1300)
OUTPUT_SCHEMA_VERSION = 7
EMPTY_BBOX_XYXY = (-1, -1, -1, -1)
DEFAULT_LABEL_MIN_VISIBLE_PIXELS = 3
BRANCH_TRANSITION_SCHEMA = "anchor_s_t_lidar_t_candidate_action_t_then_teacher_rollout_horizon"


@dataclass(frozen=True)
class SelectedCheckpoint:
    update: int
    checkpoint_abspath: str


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted export from output-dir/branch_index.jsonl.",
    )
    parser.add_argument("--task", default="pursuit_guidance_task")
    parser.add_argument("--robot-name", default="base_quad_root_link_control_with_lidar")
    parser.add_argument("--controller-name", default="thrust_bodyrate_control")
    parser.add_argument("--target-asset-type", default="target_x500", choices=("target_quad", "target_x500"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=3600)
    parser.add_argument("--persist-steps", type=int, default=10)
    parser.add_argument("--risk-horizons", type=int, nargs="+", default=list(DEFAULT_RISK_HORIZONS))
    parser.add_argument("--risk-lidar-stack-frames", type=int, default=DEFAULT_LIDAR_STACK_FRAMES)
    parser.add_argument("--label-min-visible-pixels", type=int, default=DEFAULT_LABEL_MIN_VISIBLE_PIXELS)
    parser.add_argument("--branch-updates", type=int, nargs="+", default=list(DEFAULT_BRANCH_UPDATES))
    parser.add_argument("--branch-anchor-stride", type=int, default=25)
    parser.add_argument("--branch-max-anchors", type=int, default=10000)
    parser.add_argument("--branch-max-anchors-per-episode", type=int, default=10)
    parser.add_argument("--branch-candidates", type=int, default=16)
    parser.add_argument("--branch-horizon", type=int, default=150)
    parser.add_argument("--branch-edge-margin-px", type=int, default=80)
    parser.add_argument("--branch-low-pixel-max", type=int, default=50)
    parser.add_argument("--branch-risk-anchor-min-score", type=float, default=0.25)
    parser.add_argument("--branch-min-candidate-action-l2", type=float, default=0.2)
    return parser.parse_args()


def build_env(args, num_envs: int, seed: int):
    task_config = task_registry.get_task_config(args.task)
    task_config.robot_name = args.robot_name
    task_config.controller_name = args.controller_name
    task_config.use_warp = True
    task_config.headless = True
    task_config.device = args.device
    task_config.num_envs = num_envs

    configure_target_asset_type(task_config, args.target_asset_type)

    robot_config = robot_registry.get_robot_config(args.robot_name)
    robot_config.sensor_config.lidar_config = (
        pursuit_forward_lidar_config.PursuitForwardM3_120x25_UltraHighResLidarConfig
    )
    robot_config.sensor_config.lidar_config.segmentation_camera = True

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
        "--pipeline",
        "gpu" if args.device.startswith("cuda") else "cpu",
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
    env.num_obs = env.task_config.observation_space_dim
    env.num_actions = env.task_config.action_space_dim
    return env


def target_mask_stats(semantic_frame: np.ndarray, target_semantic_id: int) -> dict:
    target_mask = semantic_frame == target_semantic_id
    pixel_count = int(np.count_nonzero(target_mask))
    if pixel_count == 0:
        return {
            "pixel_count": 0,
            "bbox_xyxy": EMPTY_BBOX_XYXY,
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
    }


TASK_SNAPSHOT_ATTRS = (
    "actions",
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
CONTROLLER_SNAPSHOT_ATTRS = ("rate_error_integral", "prev_rate_error", "prev_error_valid")
MOTOR_MODEL_SNAPSHOT_ATTRS = (
    "current_motor_thrust",
    "motor_time_constants_increasing",
    "motor_time_constants_decreasing",
    "motor_thrust_constant",
)


def capture_task_snapshot(task, env_id: int) -> dict:
    robot = task.sim_env.robot_manager.robot
    controller = robot.controller
    motor_model = robot.control_allocator.motor_model
    snapshot = {
        "robot_state": task.robot_state[env_id].detach().clone(),
        "env_asset_state_tensor": task.obs_dict["env_asset_state_tensor"][env_id].detach().clone(),
        "sim_steps": task.sim_env.sim_steps[env_id].detach().clone(),
        "env_lower_bound": task.sim_env.IGE_env.env_lower_bound[env_id].detach().clone(),
        "env_upper_bound": task.sim_env.IGE_env.env_upper_bound[env_id].detach().clone(),
        "task_attrs": {
            attr_name: getattr(task, attr_name)[env_id].detach().clone()
            for attr_name in TASK_SNAPSHOT_ATTRS
        },
        "controller_attrs": {
            attr_name: getattr(controller, attr_name)[env_id].detach().clone()
            for attr_name in CONTROLLER_SNAPSHOT_ATTRS
        },
        "motor_model_attrs": {
            attr_name: getattr(motor_model, attr_name)[env_id].detach().clone()
            for attr_name in MOTOR_MODEL_SNAPSHOT_ATTRS
        },
    }
    return snapshot


def _expand_snapshot_value(value: torch.Tensor, count: int) -> torch.Tensor:
    return value.reshape((1,) + value.shape).expand((count,) + value.shape)


def restore_task_snapshot(task, snapshot: dict, env_ids: torch.Tensor):
    env_ids = env_ids.to(device=task.device, dtype=torch.long)
    count = env_ids.numel()
    task.robot_state[env_ids] = _expand_snapshot_value(snapshot["robot_state"], count)
    task.obs_dict["env_asset_state_tensor"][env_ids] = _expand_snapshot_value(snapshot["env_asset_state_tensor"], count)
    task.sim_env.sim_steps[env_ids] = _expand_snapshot_value(snapshot["sim_steps"], count)
    task.sim_env.IGE_env.env_lower_bound[env_ids] = _expand_snapshot_value(snapshot["env_lower_bound"], count)
    task.sim_env.IGE_env.env_upper_bound[env_ids] = _expand_snapshot_value(snapshot["env_upper_bound"], count)
    for attr_name, value in snapshot["task_attrs"].items():
        getattr(task, attr_name)[env_ids] = _expand_snapshot_value(value, count)
    controller = task.sim_env.robot_manager.robot.controller
    for attr_name, value in snapshot["controller_attrs"].items():
        getattr(controller, attr_name)[env_ids] = _expand_snapshot_value(value, count)
    motor_model = task.sim_env.robot_manager.robot.control_allocator.motor_model
    for attr_name, value in snapshot["motor_model_attrs"].items():
        getattr(motor_model, attr_name)[env_ids] = _expand_snapshot_value(value, count)
    task.sim_env.IGE_env.write_to_sim()
    task.sim_env.robot_manager.robot.update_states()
    task.sim_env.render(render_components="sensors")
    task.process_obs_for_task()


def make_empty_branch_summary() -> dict:
    return {
        "anchors": 0,
        "candidate_transitions": 0,
        "anchors_by_update": {},
        "target_pixel_sum": 0,
        "selection_score_sum": 0.0,
        "labels": {},
    }


def add_anchor_to_summary(
    summary: dict,
    *,
    update: int,
    num_candidates: int,
    target_pixel_count: int,
    selection_score: float,
    horizon_label_payload,
    label_horizons: Sequence[int],
):
    summary["anchors"] += 1
    summary["candidate_transitions"] += num_candidates
    summary["anchors_by_update"][update] = summary["anchors_by_update"].get(update, 0) + 1
    summary["target_pixel_sum"] += target_pixel_count
    summary["selection_score_sum"] += selection_score
    for horizon in label_horizons:
        valid = horizon_label_payload[label_key("label_valid", horizon)].astype(bool)
        valid_count = int(np.count_nonzero(valid))
        label_stats = summary["labels"].setdefault(
            horizon,
            {"valid_candidates": 0, "loss_sum": 0.0, "severity_sum": 0.0},
        )
        label_stats["valid_candidates"] += valid_count
        label_stats["loss_sum"] += float(
            horizon_label_payload[label_key("label_loss_prob", horizon)][valid].sum()
        )
        label_stats["severity_sum"] += float(
            horizon_label_payload[label_key("label_loss_severity", horizon)][valid].sum()
        )


def next_anchor_id_from_dirs(output_dir: str) -> int:
    next_anchor_id = 0
    for anchor_dir in glob.glob(os.path.join(output_dir, "branches", "upd_*", "anchor_*")):
        basename = os.path.basename(anchor_dir)
        if not basename.startswith("anchor_"):
            continue
        try:
            next_anchor_id = max(next_anchor_id, int(basename[len("anchor_") :]) + 1)
        except ValueError:
            continue
    return next_anchor_id


def load_branch_resume_state(args, selected: Sequence[SelectedCheckpoint], branch_index_path: str) -> dict:
    selected_updates = {int(item.update) for item in selected}
    label_horizons = [horizon for horizon in args.risk_horizons if horizon <= args.branch_horizon]
    state = {
        "summary": make_empty_branch_summary(),
        "counts_by_update": {},
        "next_episode_by_update": {},
        "next_dataset_anchor_id": 0,
    }
    if not os.path.exists(branch_index_path):
        next_anchor_id = next_anchor_id_from_dirs(args.output_dir)
        if next_anchor_id:
            raise RuntimeError(
                f"Cannot resume {args.output_dir}: branch_index.jsonl is missing but anchor dirs exist."
            )
        return state

    indexed_anchors = 0
    with open(branch_index_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            update = int(row["update"])
            if update not in selected_updates:
                raise RuntimeError(
                    f"Existing index contains update={update}, but "
                    f"--branch-updates={sorted(selected_updates)}."
                )
            if int(row.get("schema_version", -1)) != OUTPUT_SCHEMA_VERSION:
                raise RuntimeError(f"Resume schema mismatch in {branch_index_path}:{line_no}")
            if int(row["branch_candidates"]) != args.branch_candidates:
                raise RuntimeError(f"Resume branch_candidates mismatch in {branch_index_path}:{line_no}")
            if int(row["branch_horizon"]) != args.branch_horizon:
                raise RuntimeError(f"Resume branch_horizon mismatch in {branch_index_path}:{line_no}")

            metadata_path = os.path.join(args.output_dir, row["metadata_relative_path"])
            with np.load(metadata_path) as metadata:
                add_anchor_to_summary(
                    state["summary"],
                    update=update,
                    num_candidates=int(row["branch_candidates"]),
                    target_pixel_count=int(metadata["anchor_target_pixel_count"].item()),
                    selection_score=float(metadata["anchor_selection_score"].item()),
                    horizon_label_payload=metadata,
                    label_horizons=label_horizons,
                )

            indexed_anchors += 1
            state["counts_by_update"][update] = state["counts_by_update"].get(update, 0) + 1
            state["next_episode_by_update"][update] = max(
                state["next_episode_by_update"].get(update, 0),
                int(row["source_episode_idx"]) + 1,
            )
            state["next_dataset_anchor_id"] = max(
                state["next_dataset_anchor_id"],
                int(row["dataset_anchor_id"]) + 1,
            )

    state["next_dataset_anchor_id"] = max(
        state["next_dataset_anchor_id"],
        next_anchor_id_from_dirs(args.output_dir),
    )
    print(
        "resuming branch dataset "
        f"indexed_anchors={indexed_anchors} next_anchor={state['next_dataset_anchor_id']} "
        f"anchors_by_update={dict(sorted(state['counts_by_update'].items()))}"
    )
    return state


def collect_branch_dataset(args, selected: Sequence[SelectedCheckpoint], target_semantic_id: int):
    output_dir = args.output_dir
    branch_index_path = os.path.join(output_dir, "branch_index.jsonl")

    if args.resume:
        resume_state = load_branch_resume_state(args, selected, branch_index_path)
        summary = resume_state["summary"]
        existing_counts_by_update = resume_state["counts_by_update"]
        next_episode_by_update = resume_state["next_episode_by_update"]
        dataset_anchor_id = resume_state["next_dataset_anchor_id"]
    else:
        summary = make_empty_branch_summary()
        existing_counts_by_update = {}
        next_episode_by_update = {}
        dataset_anchor_id = 0
    num_candidates = args.branch_candidates
    num_envs = num_candidates + 1
    envs = build_env(args, num_envs, seed=args.seed)
    task = envs
    lidar_cfg = task.sim_env.robot_manager.cfg.sensor_config.lidar_config
    try:
        for selected_idx, item in enumerate(selected):
            existing_count = existing_counts_by_update.get(int(item.update), 0)
            if existing_count >= args.branch_max_anchors:
                print(
                    "skipping complete branch teacher "
                    f"update={item.update} existing_anchors={existing_count} "
                    f"max_anchors={args.branch_max_anchors} teacher_index={selected_idx}"
                )
                continue
            print(
                "starting branch teacher "
                f"update={item.update} max_anchors={args.branch_max_anchors} "
                f"max_anchors_per_episode={args.branch_max_anchors_per_episode} "
                f"existing_anchors={existing_count} "
                f"teacher_index={selected_idx}"
            )
            checkpoint = torch.load(item.checkpoint_abspath, map_location="cpu")
            agent = Agent(envs).to(args.device)
            agent.load_state_dict(checkpoint["agent"])
            agent.eval()
            dataset_anchor_id = collect_branch_dataset_for_item(
                args=args,
                item=item,
                agent=agent,
                envs=envs,
                lidar_cfg=lidar_cfg,
                target_semantic_id=target_semantic_id,
                dataset_anchor_id=dataset_anchor_id,
                branch_index_path=branch_index_path,
                summary=summary,
                initial_anchors_written=existing_count,
                start_episode_idx=next_episode_by_update.get(int(item.update), 0),
            )
    finally:
        envs.close()
    anchors = summary["anchors"]
    candidates = summary["candidate_transitions"]
    labels = {}
    for horizon, label_stats in sorted(summary["labels"].items(), key=lambda item: item[0]):
        valid = label_stats["valid_candidates"]
        labels[horizon] = {
            "valid_candidates": valid,
            "valid_rate": valid / candidates if candidates else 0.0,
            "loss_rate": label_stats["loss_sum"] / valid if valid else 0.0,
            "loss_severity_mean": label_stats["severity_sum"] / valid if valid else 0.0,
        }
    return {
        "anchors": anchors,
        "candidate_transitions": candidates,
        "anchors_by_update": dict(
            sorted(summary["anchors_by_update"].items(), key=lambda item: item[0])
        ),
        "quality": {
            "mean_target_pixels": summary["target_pixel_sum"] / anchors if anchors else 0.0,
            "mean_anchor_selection_score": summary["selection_score_sum"] / anchors if anchors else 0.0,
        },
        "labels": labels,
    }


def collect_branch_dataset_for_item(
    args,
    item: SelectedCheckpoint,
    agent: Agent,
    envs,
    lidar_cfg,
    target_semantic_id: int,
    dataset_anchor_id: int,
    branch_index_path: str,
    summary: dict,
    initial_anchors_written: int = 0,
    start_episode_idx: int = 0,
) -> int:
    output_dir = args.output_dir
    num_candidates = args.branch_candidates
    num_envs = num_candidates + 1
    task = envs
    branch_env_ids = torch.arange(1, num_candidates + 1, device=args.device)
    label_horizons = [horizon for horizon in args.risk_horizons if horizon <= args.branch_horizon]
    anchor_id = 0
    anchors_written = int(initial_anchors_written)
    episode_idx = int(start_episode_idx)
    while anchors_written < args.branch_max_anchors:
        task_obs, _, _, _, _ = envs.reset()
        obs = task_obs["observations"]
        task.sim_env.render(render_components="sensors")
        source_lidar_history = []
        source_lidar_step_history = []
        episode_anchors = 0
        for step in range(args.max_steps):
            obs_before = obs.detach().clone()
            source_lidar_step_history.append(
                task.sim_env.sim_steps[anchor_id].detach().cpu().numpy().copy()
            )
            source_lidar_history.append(
                task.obs_dict["depth_range_pixels"][anchor_id, 0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
                .copy()
            )
            if len(source_lidar_history) > args.risk_lidar_stack_frames:
                source_lidar_history = source_lidar_history[-args.risk_lidar_stack_frames:]
                source_lidar_step_history = source_lidar_step_history[-args.risk_lidar_stack_frames:]
            if step % args.branch_anchor_stride == 0 and episode_anchors < args.branch_max_anchors_per_episode:
                snapshot = capture_task_snapshot(task, anchor_id)
                semantic_stats = target_mask_stats(
                    task.obs_dict["segmentation_pixels"][anchor_id, 0]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.int32, copy=False),
                    target_semantic_id,
                )
                selection_stats = branch_anchor_selection_stats(
                    semantic_stats=semantic_stats,
                    lidar_height=lidar_cfg.height,
                    lidar_width=lidar_cfg.width,
                    args=args,
                )
                if selection_stats["score"] < args.branch_risk_anchor_min_score:
                    with torch.no_grad():
                        action, _, _, _, _ = agent.get_action_and_value(
                            obs[anchor_id : anchor_id + 1],
                            return_aux=True,
                        )
                    actions = torch.zeros((num_envs, envs.num_actions), dtype=torch.float32, device=args.device)
                    actions[anchor_id] = action[0]
                    task_obs, _, terminations, truncations, _ = envs.step(actions)
                    if bool((terminations | truncations)[anchor_id].item()):
                        break
                    obs = task_obs["observations"]
                    continue

                anchor_dir = os.path.join(
                    output_dir,
                    "branches",
                    f"upd_{item.update:06d}",
                    f"anchor_{dataset_anchor_id:06d}",
                )
                os.makedirs(anchor_dir, exist_ok=True)
                anchor_lidar_relpath = os.path.relpath(os.path.join(anchor_dir, "anchor_lidar.npz"), output_dir)
                robot_body_linvel = task.obs_dict["robot_body_linvel"]
                robot_body_angvel = task.obs_dict["robot_body_angvel"]
                robot_quat = task.obs_dict["robot_orientation"]
                rot = quat_to_rotation_matrix(robot_quat).reshape(task.num_envs, 9)
                anchor_ego_obs_np = (
                    torch.cat((robot_body_linvel, robot_body_angvel, rot, task.actions), dim=1)[anchor_id]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                anchor_lidar_stack = lidar_stack_from_history(source_lidar_history, args.risk_lidar_stack_frames)
                source_lidar_frame_steps = list(source_lidar_step_history[-args.risk_lidar_stack_frames:])
                while len(source_lidar_frame_steps) < args.risk_lidar_stack_frames:
                    source_lidar_frame_steps.insert(0, source_lidar_frame_steps[0])
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
                done_flags = torch.zeros(num_candidates, dtype=torch.bool, device=args.device)
                visible_history = [[] for _ in range(num_candidates)]
                branch_target_pixel_count_history = np.full(
                    (num_candidates, args.branch_horizon),
                    -1,
                    dtype=np.int32,
                )
                branch_valid_history = np.zeros((num_candidates, args.branch_horizon), dtype=np.bool_)
                branch_sim_step_history = np.full(
                    (num_candidates, args.branch_horizon),
                    -1,
                    dtype=np.int64,
                )
                first_actions = torch.zeros((num_envs, envs.num_actions), dtype=torch.float32, device=args.device)
                first_actions[1:] = candidate_actions
                task_obs, _, terminations, truncations, _ = envs.step(first_actions)
                branch_obs = task_obs["observations"]
                done_flags |= (terminations | truncations)[1:]
                for horizon_step in range(args.branch_horizon):
                    semantic_tensor = task.obs_dict["segmentation_pixels"]
                    for idx in range(num_candidates):
                        if bool(done_flags[idx].item()):
                            continue
                        branch_slot = idx + 1
                        stats = target_mask_stats(
                            semantic_tensor[branch_slot, 0].detach().cpu().numpy().astype(np.int32, copy=False),
                            target_semantic_id,
                        )
                        branch_target_pixel_count_history[idx, horizon_step] = stats["pixel_count"]
                        branch_valid_history[idx, horizon_step] = True
                        branch_sim_step_history[idx, horizon_step] = (
                            task.sim_env.sim_steps[branch_slot].detach().cpu().numpy().copy()
                        )
                        visible_history[idx].append(stats["pixel_count"] >= args.label_min_visible_pixels)
                    if torch.all(done_flags):
                        break
                    if horizon_step + 1 >= args.branch_horizon:
                        break
                    teacher_actions = torch.zeros((num_envs, envs.num_actions), dtype=torch.float32, device=args.device)
                    with torch.no_grad():
                        branch_slots = torch.nonzero(~done_flags, as_tuple=False).flatten() + 1
                        if branch_slots.numel():
                            teacher_action, _, _, _, _ = agent.get_action_and_value(
                                branch_obs[branch_slots],
                                return_aux=True,
                            )
                            teacher_actions[branch_slots] = teacher_action
                    task_obs, _, terminations, truncations, _ = envs.step(teacher_actions)
                    branch_obs = task_obs["observations"]
                    done_flags |= (terminations | truncations)[1:]

                horizon_label_payload = branch_horizon_label_payload(
                    visible_history,
                    label_horizons,
                    args.persist_steps,
                )
                add_anchor_to_summary(
                    summary,
                    update=int(item.update),
                    num_candidates=num_candidates,
                    target_pixel_count=int(semantic_stats["pixel_count"]),
                    selection_score=float(selection_stats["score"]),
                    horizon_label_payload=horizon_label_payload,
                    label_horizons=label_horizons,
                )
                metadata_path = os.path.join(anchor_dir, "branch_metadata.npz")
                np.savez_compressed(
                    metadata_path,
                    **build_branch_metadata_payload(
                        schema_version=OUTPUT_SCHEMA_VERSION,
                        anchor_lidar_relpath=anchor_lidar_relpath,
                        anchor_ego_obs_np=anchor_ego_obs_np,
                        candidate_actions=candidate_actions,
                        horizon_label_payload=horizon_label_payload,
                    ),
                    anchor_sim_step=np.asarray(source_lidar_frame_steps[-1], dtype=np.int64),
                    source_lidar_frame_steps=np.asarray(source_lidar_frame_steps, dtype=np.int64),
                    anchor_target_pixel_count=np.asarray(semantic_stats["pixel_count"], dtype=np.int32),
                    anchor_target_bbox_xyxy=np.asarray(semantic_stats["bbox_xyxy"], dtype=np.int32),
                    anchor_selection_score=np.asarray(selection_stats["score"], dtype=np.float32),
                    anchor_edge_score=np.asarray(selection_stats["edge_score"], dtype=np.float32),
                    anchor_low_pixel_score=np.asarray(selection_stats["low_pixel_score"], dtype=np.float32),
                    anchor_edge_distance_px=np.asarray(selection_stats["edge_distance_px"], dtype=np.int32),
                    anchor_robot_body_linvel=robot_body_linvel[anchor_id].detach().cpu().numpy().copy(),
                    anchor_robot_body_angvel=robot_body_angvel[anchor_id].detach().cpu().numpy().copy(),
                    anchor_robot_quat=robot_quat[anchor_id].detach().cpu().numpy().copy(),
                    anchor_prev_action=task.actions[anchor_id].detach().cpu().numpy().copy(),
                    branch_target_pixel_count_history=branch_target_pixel_count_history,
                    branch_valid_history=branch_valid_history,
                    branch_sim_step_history=branch_sim_step_history,
                )
                artifact = {
                    "schema_version": OUTPUT_SCHEMA_VERSION,
                    "dataset_kind": "pursuit_lidar_branch_risk_dataset",
                    "transition_schema": BRANCH_TRANSITION_SCHEMA,
                    "dataset_anchor_id": dataset_anchor_id,
                    "update": item.update,
                    "source_episode_idx": episode_idx,
                    "source_step": step,
                    "branch_candidates": num_candidates,
                    "branch_horizon": args.branch_horizon,
                    "metadata_relative_path": os.path.relpath(metadata_path, output_dir),
                }
                with open(branch_index_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(artifact, ensure_ascii=False, sort_keys=True) + "\n")
                print(
                    "saved branch "
                    f"anchor={dataset_anchor_id:06d} update={item.update} step={step} "
                    f"lidar_bytes={anchor_lidar_bytes}"
                )
                dataset_anchor_id += 1
                anchors_written += 1
                episode_anchors += 1
                restore_task_snapshot(task, snapshot, torch.tensor([anchor_id], device=args.device))
                obs = task.task_obs["observations"].detach().clone()
                if anchors_written >= args.branch_max_anchors or episode_anchors >= args.branch_max_anchors_per_episode:
                    break

            with torch.no_grad():
                action, _, _, _, _ = agent.get_action_and_value(
                    obs[anchor_id : anchor_id + 1],
                    return_aux=True,
                )
            actions = torch.zeros((num_envs, envs.num_actions), dtype=torch.float32, device=args.device)
            actions[anchor_id] = action[0]
            task_obs, _, terminations, truncations, _ = envs.step(actions)
            if bool((terminations | truncations)[anchor_id].item()):
                break
            obs = task_obs["observations"]
        episode_idx += 1
    return dataset_anchor_id


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    selected = []
    for update in args.branch_updates:
        pattern = os.path.join(args.run_dir, "policy_pool", f"ppo_upd_{update:06d}_step_*.pth")
        matches = sorted(glob.glob(pattern))
        checkpoint_abspath = os.path.abspath(matches[0])
        selected.append(
            SelectedCheckpoint(
                update=update,
                checkpoint_abspath=checkpoint_abspath,
            )
        )

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and not args.resume:
        raise FileExistsError(f"Output dir is not empty: {args.output_dir}")
    os.makedirs(os.path.join(args.output_dir, "branches"), exist_ok=True)

    target_semantic_id = getattr(
        pursuit_guidance_asset_config,
        f"{args.target_asset_type}_asset_params",
    ).semantic_id
    collection_summary = collect_branch_dataset(args, selected, target_semantic_id)
    manifest = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "dataset_kind": "pursuit_lidar_branch_risk_dataset",
        "index": "branch_index.jsonl",
        **collection_summary,
    }
    with open(os.path.join(args.output_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"Saved branch LiDAR risk dataset to {args.output_dir}")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
