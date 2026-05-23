"""Export LiDAR rollout smoke data from a balanced pursuit policy pool.

This script keeps the PPO policy input unchanged (32D privileged observation)
and records LiDAR/range-image observations only as risk-model data artifacts.
It is intentionally separate from ppo_guidance.py so rollout export does not
change training or playback behavior.
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
from typing import Dict, Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from aerial_gym.registry.env_registry import env_config_registry
from aerial_gym.registry.robot_registry import robot_registry
from aerial_gym.registry.task_registry import task_registry
from aerial_gym.rl_training.cleanrl.ppo_guidance import (
    Agent,
    RecordEpisodeStatisticsTorch,
    configure_target_asset_type,
    extract_agent_state_dict,
)
from aerial_gym.task.pursuit_guidance_task.risk_geometry import (
    DetectionFrustum,
    detection_frustum_from_lidar_config,
    first_observability_loss_offset,
    observability_loss_labels,
    relative_position_body,
    target_in_detection_frustum,
    visibility_recovery_stats,
)


DEFAULT_RUN_DIR = "runs/PE_20260513_175323"
DEFAULT_OUTPUT_DIR = "runs/lidar_smoke/PE_20260513_175323_x500_13ckpt_smoke"
DEFAULT_CHECKPOINT_UPDATES = (1, 140, 150, 160, 170, 218, 220, 436, 440, 450, 460, 470, 640)
OUTPUT_SCHEMA_VERSION = 4
EMPTY_BBOX_XYXY = (-1, -1, -1, -1)


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
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=3600)
    parser.add_argument("--horizon-steps", type=int, default=150)
    parser.add_argument("--persist-steps", type=int, default=10)
    parser.add_argument("--lidar-save-interval-steps", type=int, default=30)
    parser.add_argument("--far-threshold-m", type=float, default=0.05)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--checkpoint-updates",
        type=int,
        nargs="+",
        default=list(DEFAULT_CHECKPOINT_UPDATES),
        help="Policy-pool update indices to export, in env_id order.",
    )
    parser.add_argument(
        "--lidar-config-class",
        default=None,
        help="Optional class from pursuit_forward_lidar_config.py to override the robot LiDAR config.",
    )
    parser.add_argument(
        "--enable-lidar-segmentation",
        action="store_true",
        help="Enable LiDAR semantic output for export-time target pixel statistics.",
    )
    return parser.parse_args()


def assert_runtime():
    executable = os.path.realpath(sys.executable)
    if "/envs/aerialgym_v2/" not in executable:
        raise RuntimeError(
            "Run this exporter under conda env aerialgym_v2, for example: "
            "conda run -n aerialgym_v2 python aerial_gym/rl_training/cleanrl/"
            "export_pursuit_lidar_rollouts.py --dry-run"
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
                curriculum_transition=row.get("curriculum_transition"),
            )
        )
    return selected


def optional_float(value):
    return None if value is None else float(value)


def optional_int(value):
    return None if value is None else int(value)


def print_selection(selected: List[SelectedCheckpoint]):
    print("env update train_sr threshold checkpoint")
    for item in selected:
        threshold = "NA" if item.curriculum_current_threshold is None else f"{item.curriculum_current_threshold:.1f}"
        print(
            f"{item.env_id:02d} {item.update:04d} "
            f"{item.train_success_rate:.6f} {threshold:>4} {item.checkpoint_path}"
        )


def selected_to_dict(item: SelectedCheckpoint) -> dict:
    return asdict(item)


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

    if args.lidar_config_class is not None:
        lidar_config = getattr(pursuit_forward_lidar_config, args.lidar_config_class)
        robot_config = robot_registry.get_robot_config(args.robot_name)
        robot_config.sensor_config.lidar_config = lidar_config
    if args.enable_lidar_segmentation:
        robot_config = robot_registry.get_robot_config(args.robot_name)
        robot_config.sensor_config.lidar_config.segmentation_camera = True


def build_env(args, num_envs: int):
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
            seed=args.seed,
            num_envs=num_envs,
            headless=True,
            use_warp=True,
            device=args.device,
        )
    finally:
        sys.argv = original_argv
    return RecordEpisodeStatisticsTorch(env, args.device)


def load_agents(selected: List[SelectedCheckpoint], envs, device: str):
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


def target_distance_series(robot_state: torch.Tensor, target_state: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(target_state[..., 0:3] - robot_state[..., 0:3], dim=-1)


def compute_risk_labels(
    robot_state: torch.Tensor,
    target_state: torch.Tensor,
    frustum: DetectionFrustum,
    horizon_steps: int,
    persist_steps: int,
):
    rel_body = relative_position_body(robot_state[..., 0:3], robot_state[..., 3:7], target_state[..., 0:3])
    detectable = target_in_detection_frustum(rel_body, frustum)
    labels = observability_loss_labels(detectable, horizon_steps=horizon_steps, persist_steps=persist_steps)
    offsets = first_observability_loss_offset(
        detectable, horizon_steps=horizon_steps, persist_steps=persist_steps
    )
    return rel_body, detectable, labels, offsets


def to_cpu_np(tensor: torch.Tensor):
    return tensor.detach().cpu().numpy().copy()


def tensor_rows_to_numpy(rows: List[torch.Tensor]):
    if not rows:
        return np.empty((0,), dtype=np.float32)
    return to_cpu_np(torch.stack(rows, dim=0))


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


def should_save_lidar_frame(step: int, done: bool, max_steps: int, save_interval: int) -> bool:
    if step == 0 or done:
        return True
    if step == max_steps - 1:
        return True
    return save_interval > 0 and step % save_interval == 0


def lidar_range_and_mask(lidar_norm: np.ndarray, lidar_max_range: float, far_threshold_m: float):
    range_m = lidar_norm * lidar_max_range
    return_mask = (lidar_norm >= 0.0) & (range_m < (lidar_max_range - far_threshold_m))
    return range_m.astype(np.float32), return_mask


def target_mask_stats(semantic_frame: np.ndarray, target_semantic_id: int) -> dict:
    target_mask = semantic_frame == int(target_semantic_id)
    pixel_count = int(np.count_nonzero(target_mask))
    if pixel_count == 0:
        return {
            "mask": target_mask,
            "pixel_count": 0,
            "bbox_xyxy": EMPTY_BBOX_XYXY,
            "centroid_uv": (np.nan, np.nan),
            "area_ratio": 0.0,
        }

    rows, cols = np.nonzero(target_mask)
    return {
        "mask": target_mask,
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


def append_target_stats(buffer: dict, prefix: str, stats: Optional[dict]):
    if stats is None:
        buffer[f"{prefix}_target_pixel_count"].append(-1)
        buffer[f"{prefix}_target_bbox_xyxy"].append(EMPTY_BBOX_XYXY)
        buffer[f"{prefix}_target_centroid_uv"].append((np.nan, np.nan))
        buffer[f"{prefix}_target_area_ratio"].append(np.nan)
        return

    buffer[f"{prefix}_target_pixel_count"].append(stats["pixel_count"])
    buffer[f"{prefix}_target_bbox_xyxy"].append(stats["bbox_xyxy"])
    buffer[f"{prefix}_target_centroid_uv"].append(stats["centroid_uv"])
    buffer[f"{prefix}_target_area_ratio"].append(stats["area_ratio"])


def target_stats_arrays(buffer: dict, prefix: str):
    counts = np.asarray(buffer[f"{prefix}_target_pixel_count"], dtype=np.int32)
    bboxes = (
        np.asarray(buffer[f"{prefix}_target_bbox_xyxy"], dtype=np.int32)
        if buffer[f"{prefix}_target_bbox_xyxy"]
        else np.empty((0, 4), dtype=np.int32)
    )
    centroids = (
        np.asarray(buffer[f"{prefix}_target_centroid_uv"], dtype=np.float32)
        if buffer[f"{prefix}_target_centroid_uv"]
        else np.empty((0, 2), dtype=np.float32)
    )
    area_ratios = np.asarray(buffer[f"{prefix}_target_area_ratio"], dtype=np.float32)
    return counts, bboxes, centroids, area_ratios


def save_lidar_pngs(
    output_dir: str,
    basename: str,
    lidar_norm: np.ndarray,
    target_range_m: float,
    lidar_max_range: float,
    far_threshold_m: float,
    target_mask: Optional[np.ndarray] = None,
):
    os.makedirs(output_dir, exist_ok=True)
    range_m, return_mask = lidar_range_and_mask(lidar_norm, lidar_max_range, far_threshold_m)
    range_raw = (255.0 * np.clip(lidar_norm, 0.0, 1.0)).astype(np.uint8)
    near_bright = (255.0 * np.clip(1.0 - lidar_norm, 0.0, 1.0)).astype(np.uint8)
    mask_img = np.zeros((*return_mask.shape, 3), dtype=np.uint8)
    mask_img[return_mask] = (255, 40, 40)

    contrast = np.zeros_like(range_m, dtype=np.float32)
    window = max(2.0, float(target_range_m) * 0.02)
    valid = return_mask & (np.abs(range_m - float(target_range_m)) <= window)
    if valid.any():
        lo = max(0.0, float(target_range_m) - window)
        hi = float(target_range_m) + window
        contrast = 1.0 - np.clip((range_m - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        contrast[~valid] = 0.0

    Image.fromarray(range_raw).save(os.path.join(output_dir, f"{basename}_range_raw.png"))
    Image.fromarray(near_bright).save(os.path.join(output_dir, f"{basename}_near_bright.png"))
    Image.fromarray(mask_img).save(os.path.join(output_dir, f"{basename}_return_mask.png"))
    Image.fromarray((255.0 * contrast).astype(np.uint8)).save(
        os.path.join(output_dir, f"{basename}_target_contrast.png")
    )
    if target_mask is not None:
        Image.fromarray((target_mask.astype(np.uint8) * 255)).save(
            os.path.join(output_dir, f"{basename}_target_semantic_mask.png")
        )


def save_trajectory_png(path: str, robot_xyz: np.ndarray, target_xyz: np.ndarray, rel_d: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig = plt.figure(figsize=(12, 9))
    ax_xy = fig.add_subplot(2, 2, 1)
    ax_xz = fig.add_subplot(2, 2, 2)
    ax_yz = fig.add_subplot(2, 2, 3)
    ax_d = fig.add_subplot(2, 2, 4)

    ax_xy.plot(robot_xyz[:, 0], robot_xyz[:, 1], label="robot", color="tab:blue")
    ax_xy.plot(target_xyz[:, 0], target_xyz[:, 1], label="target", color="tab:orange")
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    ax_xy.set_title("XY")
    ax_xy.axis("equal")
    ax_xy.grid(True, alpha=0.25)
    ax_xy.legend()

    ax_xz.plot(robot_xyz[:, 0], robot_xyz[:, 2], label="robot", color="tab:blue")
    ax_xz.plot(target_xyz[:, 0], target_xyz[:, 2], label="target", color="tab:orange")
    ax_xz.set_xlabel("x [m]")
    ax_xz.set_ylabel("z [m]")
    ax_xz.set_title("XZ")
    ax_xz.grid(True, alpha=0.25)

    ax_yz.plot(robot_xyz[:, 1], robot_xyz[:, 2], label="robot", color="tab:blue")
    ax_yz.plot(target_xyz[:, 1], target_xyz[:, 2], label="target", color="tab:orange")
    ax_yz.set_xlabel("y [m]")
    ax_yz.set_ylabel("z [m]")
    ax_yz.set_title("YZ")
    ax_yz.grid(True, alpha=0.25)

    ax_d.plot(np.arange(len(rel_d)), rel_d, color="tab:green")
    ax_d.set_xlabel("step")
    ax_d.set_ylabel("relative distance [m]")
    ax_d.set_title(f"Distance, min={float(np.min(rel_d)):.2f}m final={float(rel_d[-1]):.2f}m")
    ax_d.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def collect_rollouts(args, selected: List[SelectedCheckpoint]):
    num_envs = len(selected)
    os.makedirs(args.output_dir, exist_ok=True)
    episodes_dir = os.path.join(args.output_dir, "episodes")
    png_root = os.path.join(args.output_dir, "png")
    for managed_dir in (episodes_dir, png_root):
        if os.path.exists(managed_dir):
            shutil.rmtree(managed_dir)
    os.makedirs(episodes_dir, exist_ok=True)
    os.makedirs(png_root, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    envs = build_env(args, num_envs)
    env = envs.env
    agents = load_agents(selected, envs, args.device)
    lidar_cfg = env.sim_env.robot_manager.cfg.sensor_config.lidar_config
    lidar_max_range = float(lidar_cfg.max_range)
    frustum = detection_frustum_from_lidar_config(lidar_cfg)
    target_semantic_id = target_semantic_id_for_asset(args.target_asset_type)

    selected_payload = [selected_to_dict(item) for item in selected]
    write_json(os.path.join(args.output_dir, "selected_checkpoints.json"), {"checkpoints": selected_payload})
    write_json(
        os.path.join(args.output_dir, "manifest.json"),
        {
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "run_dir": args.run_dir,
            "output_dir": args.output_dir,
            "task": args.task,
            "robot_name": args.robot_name,
            "controller_name": args.controller_name,
            "target_asset_type": args.target_asset_type,
            "num_envs": num_envs,
            "num_episodes": args.num_episodes,
            "max_steps": args.max_steps,
            "seed": args.seed,
            "horizon_steps": args.horizon_steps,
            "persist_steps": args.persist_steps,
            "lidar_save_mode": "keyframes",
            "lidar_save_interval_steps": args.lidar_save_interval_steps,
            "lidar_keyframe_dtype": "float16",
            "lidar_config": lidar_cfg.__name__,
            "lidar_segmentation_camera": bool(getattr(lidar_cfg, "segmentation_camera", False)),
            "enable_lidar_segmentation": bool(args.enable_lidar_segmentation),
            "target_semantic_id": target_semantic_id,
            "lidar_height": int(lidar_cfg.height),
            "lidar_width": int(lidar_cfg.width),
            "lidar_max_range_m": lidar_max_range,
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
        },
    )

    index_path = os.path.join(args.output_dir, "index.jsonl")
    if os.path.exists(index_path):
        os.remove(index_path)

    try:
        for episode_idx in range(args.num_episodes):
            next_obs, _ = envs.reset()
            env.sim_env.render(render_components="sensors")
            alive = torch.ones(num_envs, dtype=torch.bool, device=args.device)
            buffers = [make_episode_buffer() for _ in selected]

            for step in range(int(args.max_steps)):
                obs_before = next_obs.detach().clone()
                robot_state_before = env.robot_state.detach().clone()
                target_state_before = env.target_state.detach().clone()
                lidar_tensor_before = env.obs_dict.get("depth_range_pixels")
                if lidar_tensor_before is None:
                    raise RuntimeError("depth_range_pixels tensor was not created")
                lidar_norm_before = lidar_tensor_before[:, 0].detach().cpu().numpy().astype(np.float32)
                segmentation_tensor_before = env.obs_dict.get("segmentation_pixels")
                lidar_semantic_before = None
                if segmentation_tensor_before is not None:
                    lidar_semantic_before = (
                        segmentation_tensor_before[:, 0].detach().cpu().numpy().astype(np.int32)
                    )

                actions = torch.zeros(
                    (num_envs, envs.num_actions), dtype=torch.float32, device=args.device
                )
                policy_mean = torch.zeros_like(actions)
                policy_std = torch.zeros_like(actions)
                pre_tanh = torch.zeros_like(actions)

                with torch.no_grad():
                    for env_id, agent in enumerate(agents):
                        action, _, _, _, aux = agent.get_action_and_value(
                            next_obs[env_id : env_id + 1], return_aux=True
                        )
                        actions[env_id] = action[0]
                        policy_mean[env_id] = aux["action_mean"][0]
                        policy_std[env_id] = aux["action_std"][0]
                        pre_tanh[env_id] = aux["pre_tanh_action"][0]

                next_obs, rewards, done_float, info = envs.step(actions)

                for env_id, item in enumerate(selected):
                    if not bool(alive[env_id].item()):
                        continue
                    done = bool(done_float[env_id].item())
                    frame_robot_state = robot_state_before[env_id].detach().clone()
                    frame_target_state = target_state_before[env_id].detach().clone()
                    frame_lidar_norm = lidar_norm_before[env_id]
                    frame_lidar_semantic = (
                        None if lidar_semantic_before is None else lidar_semantic_before[env_id]
                    )
                    buffer = buffers[env_id]
                    buffer["robot_state"].append(frame_robot_state)
                    buffer["target_state"].append(frame_target_state)
                    buffer["obs"].append(obs_before[env_id].detach().clone())
                    buffer["action"].append(actions[env_id].detach().clone())
                    buffer["policy_mean"].append(policy_mean[env_id].detach().clone())
                    buffer["policy_std"].append(policy_std[env_id].detach().clone())
                    buffer["pre_tanh_action"].append(pre_tanh[env_id].detach().clone())
                    buffer["reward"].append(float(rewards[env_id].detach().cpu().item()))
                    buffer["done"].append(float(done))
                    buffer["done_reason"].append(extract_done_reason(info, env_id))
                    frame_rel_dist = torch.linalg.norm(
                        frame_target_state[0:3] - frame_robot_state[0:3]
                    )
                    buffer["relative_distance"].append(float(frame_rel_dist.detach().cpu().item()))
                    buffer["step"].append(step)

                    semantic_stats = None
                    if frame_lidar_semantic is not None:
                        semantic_stats = target_mask_stats(frame_lidar_semantic, target_semantic_id)
                    append_target_stats(buffer, "step", semantic_stats)

                    if should_save_lidar_frame(
                        step, done, int(args.max_steps), int(args.lidar_save_interval_steps)
                    ):
                        basename = f"episode_{episode_idx:03d}_step_{step:04d}"
                        png_dir = os.path.join(png_root, f"env_{env_id:02d}_upd_{item.update:06d}")
                        save_lidar_pngs(
                            png_dir,
                            basename,
                            frame_lidar_norm,
                            buffer["relative_distance"][-1],
                            lidar_max_range,
                            args.far_threshold_m,
                            target_mask=None if semantic_stats is None else semantic_stats["mask"],
                        )
                        buffer["lidar_keyframe_step"].append(step)
                        buffer["lidar_keyframe"].append(frame_lidar_norm.astype(np.float16))
                        buffer["lidar_keyframe_png_prefix"].append(
                            os.path.relpath(os.path.join(png_dir, basename), args.output_dir)
                        )
                        buffer["lidar_keyframe_index"].append(
                            len(buffer["lidar_keyframe_step"]) - 1
                        )
                        append_target_stats(buffer, "lidar_keyframe", semantic_stats)

                    if done:
                        alive[env_id] = False

                if not torch.any(alive):
                    break

            for env_id, item in enumerate(selected):
                artifact = write_episode_artifacts(
                    args=args,
                    item=item,
                    episode_idx=episode_idx,
                    buffer=buffers[env_id],
                    frustum=frustum,
                    episodes_dir=episodes_dir,
                    png_root=png_root,
                )
                append_jsonl(index_path, artifact)
                print(
                    f"saved env={env_id:02d} update={item.update} "
                    f"steps={artifact['num_steps']} done={artifact['done_reason']} "
                    f"npz={artifact['npz_path']}"
                )
    finally:
        try:
            envs.close()
        except Exception as exc:
            print(f"Cleanup failed: {exc}", file=sys.stderr)


def make_episode_buffer():
    return {
        "step": [],
        "robot_state": [],
        "target_state": [],
        "obs": [],
        "action": [],
        "policy_mean": [],
        "policy_std": [],
        "pre_tanh_action": [],
        "reward": [],
        "done": [],
        "done_reason": [],
        "relative_distance": [],
        "step_target_pixel_count": [],
        "step_target_bbox_xyxy": [],
        "step_target_centroid_uv": [],
        "step_target_area_ratio": [],
        "lidar_keyframe_step": [],
        "lidar_keyframe": [],
        "lidar_keyframe_png_prefix": [],
        "lidar_keyframe_index": [],
        "lidar_keyframe_target_pixel_count": [],
        "lidar_keyframe_target_bbox_xyxy": [],
        "lidar_keyframe_target_centroid_uv": [],
        "lidar_keyframe_target_area_ratio": [],
    }


def write_episode_artifacts(
    args,
    item: SelectedCheckpoint,
    episode_idx: int,
    buffer: dict,
    frustum: DetectionFrustum,
    episodes_dir: str,
    png_root: str,
) -> dict:
    robot_state = torch.stack(buffer["robot_state"], dim=0)
    target_state = torch.stack(buffer["target_state"], dim=0)
    rel_body, detectable, labels, offsets = compute_risk_labels(
        robot_state=robot_state.unsqueeze(1),
        target_state=target_state.unsqueeze(1),
        frustum=frustum,
        horizon_steps=args.horizon_steps,
        persist_steps=args.persist_steps,
    )
    rel_body_np = to_cpu_np(rel_body[:, 0])
    detectable_np = to_cpu_np(detectable[:, 0]).astype(np.bool_)
    labels_np = to_cpu_np(labels[:, 0]).astype(np.bool_)
    offsets_np = to_cpu_np(offsets[:, 0]).astype(np.int64)

    npz_name = f"env_{item.env_id:02d}_upd_{item.update:06d}_episode_{episode_idx:03d}.npz"
    npz_path = os.path.join(episodes_dir, npz_name)
    lidar_keyframes = (
        np.stack(buffer["lidar_keyframe"], axis=0)
        if buffer["lidar_keyframe"]
        else np.empty((0,), dtype=np.float32)
    )
    lidar_png_prefixes = np.asarray(buffer["lidar_keyframe_png_prefix"], dtype="U256")
    target_pixel_counts, target_bboxes, target_centroids, target_area_ratios = (
        target_stats_arrays(buffer, "step")
    )
    (
        keyframe_target_pixel_counts,
        keyframe_target_bboxes,
        keyframe_target_centroids,
        keyframe_target_area_ratios,
    ) = target_stats_arrays(buffer, "lidar_keyframe")

    steps = np.asarray(buffer["step"], dtype=np.int32)
    rel_d = np.asarray(buffer["relative_distance"], dtype=np.float32)
    done = np.asarray(buffer["done"], dtype=np.float32)
    done_reason = buffer["done_reason"][-1] if buffer["done_reason"] else "not_done"
    visibility_stats = visibility_recovery_stats(
        detectable_np,
        persist_steps=args.persist_steps,
        horizon_steps=args.horizon_steps,
    )

    robot_np = tensor_rows_to_numpy(buffer["robot_state"])
    target_np = tensor_rows_to_numpy(buffer["target_state"])
    save_trajectory_png(
        os.path.join(
            png_root,
            f"env_{item.env_id:02d}_upd_{item.update:06d}",
            f"episode_{episode_idx:03d}_trajectory.png",
        ),
        robot_np[:, 0:3],
        target_np[:, 0:3],
        rel_d,
    )

    np.savez_compressed(
        npz_path,
        schema_version=np.array(OUTPUT_SCHEMA_VERSION, dtype=np.int32),
        run_dir=np.array(args.run_dir, dtype="U256"),
        checkpoint_path=np.array(item.checkpoint_path, dtype="U256"),
        checkpoint_abspath=np.array(item.checkpoint_abspath, dtype="U512"),
        env_id=np.array(item.env_id, dtype=np.int32),
        update=np.array(item.update, dtype=np.int32),
        train_success_rate=np.array(item.train_success_rate, dtype=np.float32),
        target_asset_type=np.array(args.target_asset_type, dtype="U32"),
        step=steps,
        robot_state=robot_np,
        target_state=target_np,
        obs=tensor_rows_to_numpy(buffer["obs"]),
        action=tensor_rows_to_numpy(buffer["action"]),
        policy_mean=tensor_rows_to_numpy(buffer["policy_mean"]),
        policy_std=tensor_rows_to_numpy(buffer["policy_std"]),
        pre_tanh_action=tensor_rows_to_numpy(buffer["pre_tanh_action"]),
        reward=np.asarray(buffer["reward"], dtype=np.float32),
        done=done,
        done_reason=np.asarray(buffer["done_reason"], dtype="U16"),
        relative_distance=rel_d,
        relative_position_body=rel_body_np,
        detectable=detectable_np,
        observability_loss_label=labels_np,
        first_loss_offset=offsets_np,
        final_detectable=np.array(
            nullable_float(visibility_stats["final_detectable"]),
            dtype=np.float32,
        ),
        max_consecutive_invisible_steps=np.array(
            nullable_float(visibility_stats["max_consecutive_invisible_steps"]),
            dtype=np.float32,
        ),
        num_invisible_segments=np.array(
            nullable_float(visibility_stats["num_invisible_segments"]),
            dtype=np.float32,
        ),
        mean_invisible_segment_steps=np.array(
            nullable_float(visibility_stats["mean_invisible_segment_steps"]),
            dtype=np.float32,
        ),
        recovered_invisible_segments=np.array(
            nullable_float(visibility_stats["recovered_invisible_segments"]),
            dtype=np.float32,
        ),
        recovery_rate_within_persist_steps=np.array(
            nullable_float(visibility_stats["recovery_rate_within_persist_steps"]),
            dtype=np.float32,
        ),
        recovery_rate_within_horizon_steps=np.array(
            nullable_float(visibility_stats["recovery_rate_within_horizon_steps"]),
            dtype=np.float32,
        ),
        mean_recovery_steps=np.array(
            nullable_float(visibility_stats["mean_recovery_steps"]),
            dtype=np.float32,
        ),
        max_recovery_steps=np.array(
            nullable_float(visibility_stats["max_recovery_steps"]),
            dtype=np.float32,
        ),
        long_invisible_segments=np.array(
            nullable_float(visibility_stats["long_invisible_segments"]),
            dtype=np.float32,
        ),
        over_horizon_invisible_segments=np.array(
            nullable_float(visibility_stats["over_horizon_invisible_segments"]),
            dtype=np.float32,
        ),
        detectable_rate_first_half=np.array(
            nullable_float(visibility_stats["detectable_rate_first_half"]),
            dtype=np.float32,
        ),
        detectable_rate_second_half=np.array(
            nullable_float(visibility_stats["detectable_rate_second_half"]),
            dtype=np.float32,
        ),
        lidar_keyframe_step=np.asarray(buffer["lidar_keyframe_step"], dtype=np.int32),
        lidar_keyframe_index=np.asarray(buffer["lidar_keyframe_index"], dtype=np.int32),
        lidar_keyframe_range_norm=lidar_keyframes,
        lidar_keyframe_png_prefix=lidar_png_prefixes,
        target_semantic_id=np.array(target_semantic_id_for_asset(args.target_asset_type), dtype=np.int32),
        target_pixel_count=target_pixel_counts,
        target_bbox_xyxy=target_bboxes,
        target_centroid_uv=target_centroids,
        target_area_ratio=target_area_ratios,
        target_visible=target_pixel_counts > 0,
        lidar_keyframe_target_pixel_count=keyframe_target_pixel_counts,
        lidar_keyframe_target_bbox_xyxy=keyframe_target_bboxes,
        lidar_keyframe_target_centroid_uv=keyframe_target_centroids,
        lidar_keyframe_target_area_ratio=keyframe_target_area_ratios,
    )

    png_dir = os.path.join(png_root, f"env_{item.env_id:02d}_upd_{item.update:06d}")
    valid_target_counts = keyframe_target_pixel_counts[keyframe_target_pixel_counts >= 0]
    visible_target_counts = keyframe_target_pixel_counts[keyframe_target_pixel_counts > 0]
    valid_target_frame_counts = target_pixel_counts[target_pixel_counts >= 0]
    visible_target_frame_counts = target_pixel_counts[target_pixel_counts > 0]
    artifact = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "env_id": item.env_id,
        "episode_idx": episode_idx,
        "update": item.update,
        "checkpoint_path": item.checkpoint_path,
        "train_success_rate": item.train_success_rate,
        "num_steps": int(len(steps)),
        "done": bool(done[-1] > 0.5) if len(done) else False,
        "done_reason": done_reason,
        "final_relative_distance": float(rel_d[-1]) if len(rel_d) else None,
        "min_relative_distance": float(np.min(rel_d)) if len(rel_d) else None,
        "detectable_rate": float(np.mean(detectable_np)) if len(detectable_np) else None,
        "observability_loss_label_rate": float(np.mean(labels_np)) if len(labels_np) else None,
        **visibility_stats,
        "lidar_keyframes": int(len(buffer["lidar_keyframe_step"])),
        "target_semantic_keyframes": int(len(valid_target_counts)),
        "target_visible_keyframes": int(len(visible_target_counts)),
        "target_visible_keyframe_rate": (
            float(len(visible_target_counts) / len(valid_target_counts))
            if len(valid_target_counts)
            else None
        ),
        "target_pixel_count_mean_all_keyframes": (
            float(np.mean(valid_target_counts)) if len(valid_target_counts) else None
        ),
        "target_pixel_count_mean_visible_keyframes": (
            float(np.mean(visible_target_counts)) if len(visible_target_counts) else None
        ),
        "target_pixel_count_max": (
            int(np.max(valid_target_counts)) if len(valid_target_counts) else None
        ),
        "target_semantic_frames": int(len(valid_target_frame_counts)),
        "target_visible_frames": int(len(visible_target_frame_counts)),
        "target_visible_frame_rate": (
            float(len(visible_target_frame_counts) / len(valid_target_frame_counts))
            if len(valid_target_frame_counts)
            else None
        ),
        "target_pixel_count_mean_all_frames": (
            float(np.mean(valid_target_frame_counts)) if len(valid_target_frame_counts) else None
        ),
        "target_pixel_count_mean_visible_frames": (
            float(np.mean(visible_target_frame_counts)) if len(visible_target_frame_counts) else None
        ),
        "target_pixel_count_max_frames": (
            int(np.max(valid_target_frame_counts)) if len(valid_target_frame_counts) else None
        ),
        "npz_path": os.path.abspath(npz_path),
        "npz_output_relative_path": os.path.relpath(npz_path, args.output_dir),
        "png_dir": os.path.abspath(png_dir),
        "png_output_relative_dir": os.path.relpath(png_dir, args.output_dir),
    }
    return artifact


def main():
    args = parse_args()
    assert_runtime()
    if args.num_episodes <= 0:
        raise ValueError("--num-episodes must be positive")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.lidar_save_interval_steps < 0:
        raise ValueError("--lidar-save-interval-steps must be >= 0")

    selected = select_checkpoints(args.run_dir, args.checkpoint_updates)
    print_selection(selected)
    if args.dry_run:
        return 0

    collect_rollouts(args, selected)
    print(f"Saved LiDAR rollout smoke data to {args.output_dir}")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
