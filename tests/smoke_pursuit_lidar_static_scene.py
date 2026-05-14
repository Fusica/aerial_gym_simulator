"""Static pursuit LiDAR smoke test.

This script builds a minimal pursuit scene with Warp LiDAR enabled, keeps only
one selected target asset, places the target at controlled forward distances,
and saves range images plus return masks for manual inspection.
"""

# isaacgym must be imported before torch
import isaacgym  # noqa: F401

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image


@dataclass(frozen=True)
class SmokeCase:
    name: str
    distance_m: float


CASES = (
    SmokeCase("front_001m", 1.0),
    SmokeCase("front_005m", 5.0),
    SmokeCase("front_010m", 10.0),
    SmokeCase("front_025m", 25.0),
    SmokeCase("front_050m", 50.0),
    SmokeCase("front_100m", 100.0),
    SmokeCase("front_150m", 150.0),
    SmokeCase("front_200m", 200.0),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="/tmp/pursuit_lidar_static_smoke")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--task", default="pursuit_guidance_task")
    parser.add_argument("--robot-name", default="base_quad_root_link_control_with_lidar")
    parser.add_argument("--controller-name", default="thrust_bodyrate_control")
    parser.add_argument(
        "--target-asset-type",
        default="target_quad",
        choices=("target_quad", "target_x500"),
    )
    parser.add_argument(
        "--lidar-config-class",
        default=None,
        help="Optional class name from pursuit_forward_lidar_config.py to override the robot LiDAR config.",
    )
    parser.add_argument("--far-threshold-m", type=float, default=0.05)
    parser.add_argument(
        "--align-to-nearest-ray",
        action="store_true",
        help="Place each target center on the nearest central LiDAR ray.",
    )
    return parser.parse_args()


def assert_runtime():
    executable = os.path.realpath(sys.executable)
    if "/envs/aerialgym_v2/" not in executable:
        raise RuntimeError(
            "Run this smoke under conda env aerialgym_v2, for example: "
            "conda run -n aerialgym_v2 python tests/smoke_pursuit_lidar_static_scene.py"
        )


def keep_only_target_asset(env_config, target_asset_type):
    if target_asset_type not in env_config.env_config.asset_type_to_dict_map:
        raise ValueError(f"Unknown target asset type: {target_asset_type}")
    for asset_type in list(env_config.env_config.include_asset_type.keys()):
        env_config.env_config.include_asset_type[asset_type] = asset_type == target_asset_type


def build_task(args):
    import aerial_gym  # noqa: F401
    from aerial_gym.registry.env_registry import env_config_registry
    from aerial_gym.registry.robot_registry import robot_registry
    from aerial_gym.registry.task_registry import task_registry
    from aerial_gym.config.sensor_config.lidar_config import pursuit_forward_lidar_config

    task_config = task_registry.get_task_config(args.task)
    task_config.robot_name = args.robot_name
    task_config.controller_name = args.controller_name
    task_config.use_warp = True
    task_config.headless = True
    task_config.device = args.device
    task_config.num_envs = len(CASES)
    task_config.room.enabled = False
    task_config.static_obstacles.enabled = False

    env_config = env_config_registry.get_env_config(task_config.env_name)
    env_config.env.num_envs = len(CASES)
    env_config.env.use_warp = True
    keep_only_target_asset(env_config, args.target_asset_type)

    if args.lidar_config_class is not None:
        lidar_config = getattr(pursuit_forward_lidar_config, args.lidar_config_class)
        robot_config = robot_registry.get_robot_config(args.robot_name)
        robot_config.sensor_config.lidar_config = lidar_config

    original_argv = sys.argv[:]
    sys.argv = [
        original_argv[0],
        "--headless",
        "True",
        "--num_envs",
        str(len(CASES)),
        "--use_warp",
        "True",
        "--sim_device",
        args.device,
    ]
    try:
        return task_registry.make_task(
            task_name=args.task,
            seed=args.seed,
            num_envs=len(CASES),
            headless=True,
            use_warp=True,
            device=args.device,
        )
    finally:
        sys.argv = original_argv


def central_ray_direction(lidar_cfg):
    col = (int(lidar_cfg.width) - 1) // 2
    row = (int(lidar_cfg.height) - 1) // 2
    azimuth_deg = float(lidar_cfg.horizontal_fov_deg_max) - (
        float(lidar_cfg.horizontal_fov_deg_max) - float(lidar_cfg.horizontal_fov_deg_min)
    ) * (col / (int(lidar_cfg.width) - 1))
    elevation_deg = float(lidar_cfg.vertical_fov_deg_max) - (
        float(lidar_cfg.vertical_fov_deg_max) - float(lidar_cfg.vertical_fov_deg_min)
    ) * (row / (int(lidar_cfg.height) - 1))
    azimuth = np.deg2rad(azimuth_deg)
    elevation = np.deg2rad(elevation_deg)
    direction = np.array(
        [
            np.cos(azimuth) * np.cos(elevation),
            np.sin(azimuth) * np.cos(elevation),
            np.sin(elevation),
        ],
        dtype=np.float32,
    )
    direction /= np.linalg.norm(direction)
    return direction, {"row": int(row), "col": int(col), "azimuth_deg": azimuth_deg, "elevation_deg": elevation_deg}


def set_static_scene(env, device, align_to_nearest_ray):
    center = torch.tensor([500.0, 500.0, 500.0], dtype=torch.float32, device=device)
    identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32, device=device)
    lidar_cfg = env.sim_env.robot_manager.cfg.sensor_config.lidar_config
    ray_dir_np, ray_meta = central_ray_direction(lidar_cfg)
    ray_dir = torch.tensor(ray_dir_np, dtype=torch.float32, device=device)

    env.robot_state[:, 0:3] = center
    env.robot_state[:, 3:7] = identity_quat
    env.robot_state[:, 7:13] = 0.0

    for env_id, case in enumerate(CASES):
        if align_to_nearest_ray:
            offset = ray_dir * case.distance_m
        else:
            offset = torch.tensor([case.distance_m, 0.0, 0.0], dtype=torch.float32, device=device)
        env.target_state[env_id, 0:3] = center + offset
        env.target_state[env_id, 3:7] = identity_quat
        env.target_state[env_id, 7:13] = 0.0

    env.target_command[:, 0:3] = env.target_state[:, 0:3]
    env.target_command[:, 3] = 0.0
    env.sim_env.IGE_env.write_to_sim()
    env.sim_env.robot_manager.robot.update_states()
    env.sim_env.warp_env.reset_idx(torch.arange(env.num_envs, device=device))
    return ray_meta


def save_pngs(output_dir, case, range_norm, range_m, return_mask):
    basename = case.name
    range_raw = (255.0 * np.clip(range_norm, 0.0, 1.0)).astype(np.uint8)
    near_bright = (255.0 * np.clip(1.0 - range_norm, 0.0, 1.0)).astype(np.uint8)
    mask_img = np.zeros((*return_mask.shape, 3), dtype=np.uint8)
    mask_img[return_mask] = (255, 40, 40)

    contrast = np.zeros_like(range_m, dtype=np.float32)
    window = max(2.0, case.distance_m * 0.02)
    valid = return_mask & (np.abs(range_m - case.distance_m) <= window)
    if valid.any():
        lo = max(0.0, case.distance_m - window)
        hi = case.distance_m + window
        contrast = 1.0 - np.clip((range_m - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        contrast[~valid] = 0.0
    contrast_img = (255.0 * contrast).astype(np.uint8)

    Image.fromarray(range_raw).save(os.path.join(output_dir, f"{basename}_range_raw.png"))
    Image.fromarray(near_bright).save(os.path.join(output_dir, f"{basename}_near_bright.png"))
    Image.fromarray(mask_img).save(os.path.join(output_dir, f"{basename}_return_mask.png"))
    Image.fromarray(contrast_img).save(os.path.join(output_dir, f"{basename}_target_contrast.png"))


def summarize_case(case, range_norm, lidar_max_range, far_threshold_m):
    range_m = range_norm * lidar_max_range
    return_mask = (range_norm >= 0.0) & (range_m < (lidar_max_range - far_threshold_m))
    returns = range_m[return_mask]
    summary = {
        "case": case.name,
        "distance_m": float(case.distance_m),
        "return_pixel_count": int(return_mask.sum()),
        "return_ratio": float(return_mask.mean()),
        "range_min_m": float(range_m.min()),
        "range_max_m": float(range_m.max()),
    }
    if returns.size > 0:
        summary.update(
            {
                "return_range_min_m": float(returns.min()),
                "return_range_median_m": float(np.median(returns)),
                "return_range_max_m": float(returns.max()),
            }
        )
    else:
        summary.update(
            {
                "return_range_min_m": None,
                "return_range_median_m": None,
                "return_range_max_m": None,
            }
        )
    return summary, range_m, return_mask


def main():
    args = parse_args()
    assert_runtime()
    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    env = build_task(args)
    try:
        env.reset()
        ray_meta = set_static_scene(env, args.device, args.align_to_nearest_ray)
        env.sim_env.render(render_components="sensors")
        env.sim_env.render(render_components="sensors")

        lidar_tensor = env.obs_dict.get("depth_range_pixels")
        if lidar_tensor is None:
            raise RuntimeError("depth_range_pixels tensor was not created")

        lidar_cfg = env.sim_env.robot_manager.cfg.sensor_config.lidar_config
        lidar_max_range = float(lidar_cfg.max_range)
        range_norm = lidar_tensor[:, 0].detach().cpu().numpy().astype(np.float32)

        summaries = []
        for env_id, case in enumerate(CASES):
            summary, range_m, return_mask = summarize_case(
                case, range_norm[env_id], lidar_max_range, args.far_threshold_m
            )
            save_pngs(args.output_dir, case, range_norm[env_id], range_m, return_mask)
            summaries.append(summary)

        summary_path = os.path.join(args.output_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "schema_version": 1,
                    "robot_name": args.robot_name,
                    "controller_name": args.controller_name,
                    "target_asset_type": args.target_asset_type,
                    "lidar_config": lidar_cfg.__name__,
                    "lidar_config_override": args.lidar_config_class,
                    "lidar_height": int(lidar_cfg.height),
                    "lidar_width": int(lidar_cfg.width),
                    "lidar_max_range_m": lidar_max_range,
                    "lidar_vertical_fov_deg": [
                        float(lidar_cfg.vertical_fov_deg_min),
                        float(lidar_cfg.vertical_fov_deg_max),
                    ],
                    "lidar_horizontal_fov_deg": [
                        float(lidar_cfg.horizontal_fov_deg_min),
                        float(lidar_cfg.horizontal_fov_deg_max),
                    ],
                    "segmentation_camera": bool(lidar_cfg.segmentation_camera),
                    "far_threshold_m": float(args.far_threshold_m),
                    "align_to_nearest_ray": bool(args.align_to_nearest_ray),
                    "central_ray": ray_meta,
                    "cases": summaries,
                },
                f,
                indent=2,
                sort_keys=True,
            )

        print(f"Saved static LiDAR smoke artifacts to {args.output_dir}")
        print(f"Summary: {summary_path}")
        for item in summaries:
            print(
                f"{item['case']}: distance={item['distance_m']:.1f}m "
                f"returns={item['return_pixel_count']} "
                f"median={item['return_range_median_m']}"
            )
        return 0
    finally:
        try:
            env.close()
        except Exception as exc:
            print(f"Cleanup failed: {exc}", file=sys.stderr)


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
