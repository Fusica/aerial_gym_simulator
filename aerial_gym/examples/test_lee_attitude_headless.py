#!/usr/bin/env python3
"""
Headless validation test for Lee Attitude Controller with root-link force application.

Validates step response for roll angle, pitch angle, and yaw rate commands.
The Lee attitude controller accepts [thrust, roll_rad, pitch_rad, yaw_rate_rad_s]
and uses geometric attitude control on SO(3) to compute body torques.

Force application: root_link (base_quad_root_link_control), not direct.
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime

import isaacgym  # noqa: F401
import numpy as np
import torch

import aerial_gym  # noqa: F401
from aerial_gym.registry.task_registry import task_registry

# Lee attitude controller commands: [thrust, roll_rad, pitch_rad, yaw_rate_rad_s]
# thrust  : normalized offset, maps to (action[0] + 1.0) * m * g  →  0 means hover
# roll    : desired roll angle in radians
# pitch   : desired pitch angle in radians
# yaw_rate: desired yaw rate in rad/s
ZERO_COMMAND = (0.0, 0.0, 0.0, 0.0)

AXIS_SPECS = {
    "roll": {
        "command": (0.0, 0.25, 0.0, 0.0),
        "cmd_index": 1,
        "euler_index": 0,
        "is_angle": True,
    },
    "pitch": {
        "command": (0.0, 0.0, 0.25, 0.0),
        "cmd_index": 2,
        "euler_index": 1,
        "is_angle": True,
    },
    "yaw": {
        "command": (0.0, 0.0, 0.0, 0.5),
        "cmd_index": 3,
        "rate_index": 2,
        "is_angle": False,
    },
}

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, "runs", "rpy_test")


def parse_args():
    parser = argparse.ArgumentParser(
        "Headless validation test for Lee Attitude Controller (root-link force application)."
    )
    parser.add_argument("--task", type=str, default="pursuit_guidance_task")
    parser.add_argument("--axis", type=str, default="all",
                        choices=["all", "roll", "pitch", "yaw"])
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps-per-phase", type=int, default=300)
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)

    # Angle step thresholds
    parser.add_argument("--max-angle-steady-error-rad", type=float, default=0.06)
    parser.add_argument("--max-angle-overshoot-ratio", type=float, default=0.40)
    parser.add_argument("--max-angle-settling-time-s", type=float, default=2.0)
    parser.add_argument("--angle-settling-band-rad", type=float, default=0.03)
    parser.add_argument("--max-cross-axis-angle-rad", type=float, default=0.08)

    # Rate step thresholds (yaw)
    parser.add_argument("--max-rate-steady-error-rad-s", type=float, default=0.12)
    parser.add_argument("--max-rate-overshoot-ratio", type=float, default=0.35)
    parser.add_argument("--max-rate-settling-time-s", type=float, default=1.5)

    # Shared thresholds
    parser.add_argument("--max-body-rate", type=float, default=6.0)
    parser.add_argument("--max-tilt-deg", type=float, default=60.0)
    parser.add_argument("--max-hover-angle-rad", type=float, default=0.10)
    parser.add_argument("--max-hover-rate", type=float, default=0.5)
    parser.add_argument("--min-thrust-weight", type=float, default=0.65)
    parser.add_argument("--max-thrust-weight", type=float, default=1.40)

    parser.add_argument("--no-plot", action="store_true", default=False)
    return parser.parse_args()


def selected_axes(axis_arg):
    if axis_arg == "all":
        return ["roll", "pitch", "yaw"]
    return [axis_arg]


def build_axis_phases(axis_name):
    spec = AXIS_SPECS[axis_name]
    phase_name = f"{axis_name}_step"
    return [
        ("hover", ZERO_COMMAND),
        (phase_name, spec["command"]),
        ("recovery", ZERO_COMMAND),
    ]


def make_headless_task(args):
    original_argv = sys.argv[:]
    sys.argv = [
        original_argv[0],
        "--headless", "True",
        "--num_envs", str(args.num_envs),
        "--use_warp", "False",
        "--sim_device", args.device,
    ]
    try:
        task_config = task_registry.get_task_config(args.task)
        task_config.controller_name = "lee_attitude_control"
        return task_registry.make_task(
            task_name=args.task,
            seed=args.seed,
            num_envs=args.num_envs,
            headless=True,
            use_warp=False,
            device=args.device,
        )
    finally:
        sys.argv = original_argv


def quat_to_euler_xyz(quat_xyzw):
    """quat in [x, y, z, w] order → euler [roll, pitch, yaw] in radians."""
    x, y, z, w = quat_xyzw[:, 0], quat_xyzw[:, 1], quat_xyzw[:, 2], quat_xyzw[:, 3]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    sinp = torch.clamp(sinp, -1.0, 1.0)
    pitch = torch.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return torch.stack([roll, pitch, yaw], dim=1)


def quat_to_tilt_rad(quat_xyzw):
    x, y = quat_xyzw[:, 0], quat_xyzw[:, 1]
    w = quat_xyzw[:, 3]
    body_z_world_z = 1.0 - 2.0 * (x * x + y * y)
    body_z_world_z = torch.clamp(body_z_world_z, -1.0, 1.0)
    return torch.acos(body_z_world_z)


def finite_tensor(*tensors):
    return all(torch.isfinite(t).all().item() for t in tensors)


def settling_time(time_s, actual, target, band):
    error = np.abs(actual - target)
    within = error <= band
    for idx in range(within.size):
        if within[idx] and np.all(within[idx:]):
            return float(time_s[idx] - time_s[0])
    return None


def summarize_angle_response(rows, args, axis_name):
    """Summarize step response for roll/pitch angle commands."""
    checks = []
    spec = AXIS_SPECS[axis_name]
    euler_idx = spec["euler_index"]
    cmd_idx = spec["cmd_index"]
    command_angle = spec["command"][cmd_idx]

    settling_band = args.angle_settling_band_rad
    steady_error_limit = args.max_angle_steady_error_rad

    axis_rows = [r for r in rows if r["test_axis"] == axis_name]

    # Per-phase checks
    for phase, command in build_axis_phases(axis_name):
        phase_rows = [r for r in axis_rows if r["phase"] == phase]
        if not phase_rows:
            checks.append({"name": f"{axis_name}_{phase}_has_samples", "passed": False})
            continue

        rate_norms = [r["body_rate_norm"] for r in phase_rows]
        tilt_degs = [r["tilt_deg"] for r in phase_rows]
        roll_angles = [r["euler_roll_rad"] for r in phase_rows]
        pitch_angles = [r["euler_pitch_rad"] for r in phase_rows]
        force_ratios = [r["force_z"] / max(r["weight"], 1e-6) for r in phase_rows]

        max_rate = max(rate_norms)
        max_tilt = max(tilt_degs)
        mean_force_ratio = np.mean(force_ratios)

        checks.append({
            "name": f"{axis_name}_{phase}_rate_bounded",
            "passed": bool(max_rate < args.max_body_rate),
            "value": float(max_rate),
            "threshold": args.max_body_rate,
        })
        checks.append({
            "name": f"{axis_name}_{phase}_tilt_bounded",
            "passed": bool(max_tilt < args.max_tilt_deg),
            "value": float(max_tilt),
            "threshold": args.max_tilt_deg,
        })

        if phase in ("hover", "recovery"):
            mean_abs_roll = np.mean(np.abs(roll_angles))
            mean_abs_pitch = np.mean(np.abs(pitch_angles))
            checks.append({
                "name": f"{axis_name}_{phase}_euler_near_zero",
                "passed": bool(mean_abs_roll < args.max_hover_angle_rad
                               and mean_abs_pitch < args.max_hover_angle_rad),
                "value": [float(mean_abs_roll), float(mean_abs_pitch)],
                "threshold": args.max_hover_angle_rad,
            })
            checks.append({
                "name": f"{axis_name}_{phase}_rate_near_zero",
                "passed": bool(np.mean(rate_norms) < args.max_hover_rate),
                "value": float(np.mean(rate_norms)),
                "threshold": args.max_hover_rate,
            })

        checks.append({
            "name": f"{axis_name}_{phase}_thrust_bounded",
            "passed": bool(args.min_thrust_weight <= mean_force_ratio <= args.max_thrust_weight),
            "value": float(mean_force_ratio),
            "threshold": [args.min_thrust_weight, args.max_thrust_weight],
        })

    # Angle step response metrics
    step_phase = f"{axis_name}_step"
    step_rows = [r for r in axis_rows if r["phase"] == step_phase]
    if step_rows:
        time_s = np.asarray([r["time_s"] for r in step_rows], dtype=np.float64)
        euler_angles = np.asarray([[r["euler_roll_rad"], r["euler_pitch_rad"]]
                                    for r in step_rows], dtype=np.float64)
        actual_angle = euler_angles[:, euler_idx]
        cross_index = 1 - euler_idx  # 1 if roll test, 0 if pitch test
        cross_angle = euler_angles[:, cross_index]

        tail_start = int(0.7 * len(step_rows))
        tail_mean = float(np.mean(actual_angle[tail_start:]))
        steady_error = abs(tail_mean - command_angle)
        max_angle = float(np.max(actual_angle))
        overshoot = max(0.0, max_angle - command_angle) / max(abs(command_angle), 1e-6)
        cross_axis_peak = float(np.max(np.abs(cross_angle)))
        settle_time = settling_time(time_s, actual_angle, command_angle, settling_band)

        checks.extend([
            {
                "name": f"{axis_name}_step_angle_tracks",
                "passed": bool(steady_error <= steady_error_limit),
                "value": float(steady_error),
                "threshold": float(steady_error_limit),
            },
            {
                "name": f"{axis_name}_step_overshoot",
                "passed": bool(overshoot <= args.max_angle_overshoot_ratio),
                "value": float(overshoot),
                "threshold": args.max_angle_overshoot_ratio,
            },
            {
                "name": f"{axis_name}_step_settling_time",
                "passed": bool(settle_time is not None
                               and settle_time <= args.max_angle_settling_time_s),
                "value": float(settle_time) if settle_time is not None else None,
                "threshold": args.max_angle_settling_time_s,
            },
            {
                "name": f"{axis_name}_step_cross_axis_angle",
                "passed": bool(cross_axis_peak <= args.max_cross_axis_angle_rad),
                "value": float(cross_axis_peak),
                "threshold": args.max_cross_axis_angle_rad,
            },
        ])
    else:
        checks.append({"name": f"{axis_name}_{step_phase}_has_samples", "passed": False})

    return {"checks": checks, "passed": all(c["passed"] for c in checks)}


def summarize_rate_response(rows, args, axis_name):
    """Summarize step response for yaw rate command."""
    checks = []
    spec = AXIS_SPECS[axis_name]
    rate_idx = spec["rate_index"]
    cmd_idx = spec["cmd_index"]
    command_rate = spec["command"][cmd_idx]

    steady_error_limit = max(args.max_rate_steady_error_rad_s,
                             abs(command_rate) * 0.24)
    settling_band = max(0.06, abs(command_rate) * 0.12)

    axis_rows = [r for r in rows if r["test_axis"] == axis_name]

    for phase, command in build_axis_phases(axis_name):
        phase_rows = [r for r in axis_rows if r["phase"] == phase]
        if not phase_rows:
            checks.append({"name": f"{axis_name}_{phase}_has_samples", "passed": False})
            continue

        rate_norms = [r["body_rate_norm"] for r in phase_rows]
        tilt_degs = [r["tilt_deg"] for r in phase_rows]
        roll_angles = [r["euler_roll_rad"] for r in phase_rows]
        pitch_angles = [r["euler_pitch_rad"] for r in phase_rows]
        force_ratios = [r["force_z"] / max(r["weight"], 1e-6) for r in phase_rows]

        checks.append({
            "name": f"{axis_name}_{phase}_rate_bounded",
            "passed": bool(max(rate_norms) < args.max_body_rate),
            "value": float(max(rate_norms)),
            "threshold": args.max_body_rate,
        })
        checks.append({
            "name": f"{axis_name}_{phase}_tilt_bounded",
            "passed": bool(max(tilt_degs) < args.max_tilt_deg),
            "value": float(max(tilt_degs)),
            "threshold": args.max_tilt_deg,
        })
        checks.append({
            "name": f"{axis_name}_{phase}_thrust_bounded",
            "passed": bool(args.min_thrust_weight <= np.mean(force_ratios)
                           <= args.max_thrust_weight),
            "value": float(np.mean(force_ratios)),
            "threshold": [args.min_thrust_weight, args.max_thrust_weight],
        })

        if phase in ("hover", "recovery"):
            mean_abs_roll = np.mean(np.abs(roll_angles))
            mean_abs_pitch = np.mean(np.abs(pitch_angles))
            checks.append({
                "name": f"{axis_name}_{phase}_euler_near_zero",
                "passed": bool(mean_abs_roll < args.max_hover_angle_rad
                               and mean_abs_pitch < args.max_hover_angle_rad),
                "value": [float(mean_abs_roll), float(mean_abs_pitch)],
                "threshold": args.max_hover_angle_rad,
            })
            checks.append({
                "name": f"{axis_name}_{phase}_rate_near_zero",
                "passed": bool(np.mean(rate_norms) < args.max_hover_rate),
                "value": float(np.mean(rate_norms)),
                "threshold": args.max_hover_rate,
            })

    step_phase = f"{axis_name}_step"
    step_rows = [r for r in axis_rows if r["phase"] == step_phase]
    if step_rows:
        time_s = np.asarray([r["time_s"] for r in step_rows], dtype=np.float64)
        rates = np.stack([
            [r["body_rate_x"], r["body_rate_y"], r["body_rate_z"]] for r in step_rows
        ], axis=0)
        actual_rate = rates[:, rate_idx]
        cross_indices = [i for i in range(3) if i != rate_idx]
        roll_angles = np.asarray([r["euler_roll_rad"] for r in step_rows])
        pitch_angles = np.asarray([r["euler_pitch_rad"] for r in step_rows])

        tail_start = int(0.7 * len(step_rows))
        tail_mean_rate = float(np.mean(actual_rate[tail_start:]))
        steady_error = abs(tail_mean_rate - command_rate)
        max_rate = float(np.max(actual_rate))
        overshoot = max(0.0, max_rate - command_rate) / max(abs(command_rate), 1e-6)
        cross_axis_rate_peak = float(np.max(np.abs(rates[:, cross_indices])))
        settle_time = settling_time(time_s, actual_rate, command_rate, settling_band)
        cross_roll = float(np.max(np.abs(roll_angles)))
        cross_pitch = float(np.max(np.abs(pitch_angles)))

        checks.extend([
            {
                "name": f"{axis_name}_step_rate_tracks",
                "passed": bool(steady_error <= steady_error_limit),
                "value": float(steady_error),
                "threshold": float(steady_error_limit),
            },
            {
                "name": f"{axis_name}_step_rate_overshoot",
                "passed": bool(overshoot <= args.max_rate_overshoot_ratio),
                "value": float(overshoot),
                "threshold": args.max_rate_overshoot_ratio,
            },
            {
                "name": f"{axis_name}_step_rate_settling_time",
                "passed": bool(settle_time is not None
                               and settle_time <= args.max_rate_settling_time_s),
                "value": float(settle_time) if settle_time is not None else None,
                "threshold": args.max_rate_settling_time_s,
            },
            {
                "name": f"{axis_name}_step_cross_axis_rate",
                "passed": bool(cross_axis_rate_peak <= 0.25),
                "value": float(cross_axis_rate_peak),
                "threshold": 0.25,
            },
            {
                "name": f"{axis_name}_step_cross_axis_angle",
                "passed": bool(cross_roll < args.max_cross_axis_angle_rad
                               and cross_pitch < args.max_cross_axis_angle_rad),
                "value": [float(cross_roll), float(cross_pitch)],
                "threshold": args.max_cross_axis_angle_rad,
            },
        ])
    else:
        checks.append({"name": f"{axis_name}_{step_phase}_has_samples", "passed": False})

    return {"checks": checks, "passed": all(c["passed"] for c in checks)}


def summarize_trace(rows, args, axes):
    axis_summaries = {}
    all_checks = []
    for axis_name in axes:
        spec = AXIS_SPECS[axis_name]
        if spec["is_angle"]:
            summary = summarize_angle_response(rows, args, axis_name)
        else:
            summary = summarize_rate_response(rows, args, axis_name)
        axis_summaries[axis_name] = summary
        all_checks.extend(summary["checks"])

    return {
        "axes": axis_summaries,
        "checks": all_checks,
        "passed": all(c["passed"] for c in all_checks),
        "criteria": {
            "max_angle_steady_error_rad": args.max_angle_steady_error_rad,
            "max_angle_overshoot_ratio": args.max_angle_overshoot_ratio,
            "max_angle_settling_time_s": args.max_angle_settling_time_s,
            "angle_settling_band_rad": args.angle_settling_band_rad,
            "max_cross_axis_angle_rad": args.max_cross_axis_angle_rad,
            "max_rate_steady_error_rad_s": args.max_rate_steady_error_rad_s,
            "max_rate_overshoot_ratio": args.max_rate_overshoot_ratio,
            "max_rate_settling_time_s": args.max_rate_settling_time_s,
            "max_body_rate": args.max_body_rate,
            "max_tilt_deg": args.max_tilt_deg,
            "max_hover_angle_rad": args.max_hover_angle_rad,
            "max_hover_rate": args.max_hover_rate,
            "min_thrust_weight": args.min_thrust_weight,
            "max_thrust_weight": args.max_thrust_weight,
        },
    }


def _rows_to_arrays(rows):
    return {
        "time_s": np.asarray([r["time_s"] for r in rows], dtype=np.float64),
        "test_axis": np.asarray([r["test_axis"] for r in rows]),
        "phase": np.asarray([r["phase"] for r in rows]),
        "euler_roll": np.asarray([r["euler_roll_rad"] for r in rows], dtype=np.float64),
        "euler_pitch": np.asarray([r["euler_pitch_rad"] for r in rows], dtype=np.float64),
        "euler_yaw": np.asarray([r["euler_yaw_rad"] for r in rows], dtype=np.float64),
        "cmd_thrust": np.asarray([r["cmd_thrust"] for r in rows], dtype=np.float64),
        "cmd_roll": np.asarray([r["cmd_roll_rad"] for r in rows], dtype=np.float64),
        "cmd_pitch": np.asarray([r["cmd_pitch_rad"] for r in rows], dtype=np.float64),
        "cmd_yaw_rate": np.asarray([r["cmd_yaw_rate_rad_s"] for r in rows], dtype=np.float64),
        "body_rate_x": np.asarray([r["body_rate_x"] for r in rows], dtype=np.float64),
        "body_rate_y": np.asarray([r["body_rate_y"] for r in rows], dtype=np.float64),
        "body_rate_z": np.asarray([r["body_rate_z"] for r in rows], dtype=np.float64),
        "torque_x": np.asarray([r["torque_x"] for r in rows], dtype=np.float64),
        "torque_y": np.asarray([r["torque_y"] for r in rows], dtype=np.float64),
        "torque_z": np.asarray([r["torque_z"] for r in rows], dtype=np.float64),
        "force_z": np.asarray([r["force_z"] for r in rows], dtype=np.float64),
    }


def plot_step_responses(rows, run_dir, axes_to_plot):
    cache_dir = os.path.join(os.path.dirname(run_dir), "matplotlib_cache")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", cache_dir)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arrays = _rows_to_arrays(rows)
    plot_paths = []

    for axis_name in axes_to_plot:
        mask = arrays["test_axis"] == axis_name
        if not mask.any():
            continue

        time_s = arrays["time_s"][mask]
        time_s = time_s - time_s[0]
        spec = AXIS_SPECS[axis_name]

        fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True)
        fig.suptitle(f"Lee Attitude Controller step response: {axis_name}", fontsize=14)

        # Row 0: Euler angles (roll, pitch) + command
        if spec["is_angle"]:
            target_val = spec["command"][spec["cmd_index"]]
            cmd_array = arrays[f"cmd_{axis_name}"][mask]
            actual_array = arrays[f"euler_{axis_name}"][mask]
            axes[0].plot(time_s, cmd_array, "--", label=f"cmd_{axis_name}", linewidth=1.5,
                         color="C0")
            axes[0].plot(time_s, actual_array, "-", label=f"actual_{axis_name}", linewidth=1.5,
                         color="C0", alpha=0.7)
        else:
            # Yaw rate: show actual yaw rate vs command
            cmd_array = arrays["cmd_yaw_rate"][mask]
            actual_array = arrays["body_rate_z"][mask]
            axes[0].plot(time_s, cmd_array, "--", label="cmd_yaw_rate", linewidth=1.5,
                         color="C0")
            axes[0].plot(time_s, actual_array, "-", label="actual_yaw_rate", linewidth=1.5,
                         color="C0", alpha=0.7)

        # Also plot roll and pitch Euler angles
        axes[0].plot(time_s, arrays["euler_roll"][mask], "-", label="euler_roll",
                     linewidth=1.2, alpha=0.6)
        axes[0].plot(time_s, arrays["euler_pitch"][mask], "-", label="euler_pitch",
                     linewidth=1.2, alpha=0.6)
        axes[0].set_ylabel("angle [rad]")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(ncol=3, loc="upper right", fontsize="small")

        # Row 1: Body rates
        axes[1].plot(time_s, arrays["body_rate_x"][mask], label="p (roll rate)",
                     linewidth=1.5)
        axes[1].plot(time_s, arrays["body_rate_y"][mask], label="q (pitch rate)",
                     linewidth=1.5)
        axes[1].plot(time_s, arrays["body_rate_z"][mask], label="r (yaw rate)",
                     linewidth=1.5)
        axes[1].set_ylabel("body rate [rad/s]")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(ncol=3, loc="upper right", fontsize="small")

        # Row 2: Torques
        axes[2].plot(time_s, arrays["torque_x"][mask], label="tau_x", linewidth=1.5)
        axes[2].plot(time_s, arrays["torque_y"][mask], label="tau_y", linewidth=1.5)
        axes[2].plot(time_s, arrays["torque_z"][mask], label="tau_z", linewidth=1.5)
        axes[2].set_ylabel("torque [Nm]")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(ncol=3, loc="upper right", fontsize="small")

        # Row 3: Thrust / weight
        weight = np.asarray([r["weight"] for r in rows], dtype=np.float64)[mask]
        thrust_weight = arrays["force_z"][mask] / np.maximum(weight, 1e-6)
        axes[3].plot(time_s, thrust_weight, "-", label="force_z / weight", linewidth=1.5)
        axes[3].axhline(1.0, color="k", linewidth=0.8, alpha=0.3)
        axes[3].set_ylabel("thrust / weight")
        axes[3].set_xlabel("time [s]")
        axes[3].grid(True, alpha=0.3)
        axes[3].legend(loc="upper right")

        phase = arrays["phase"][mask]
        transition_idx = np.where(phase[1:] != phase[:-1])[0] + 1
        for ax in axes:
            for idx in transition_idx:
                ax.axvline(time_s[idx], color="0.55", linewidth=0.8, alpha=0.45)

        fig.tight_layout()
        path = os.path.join(run_dir, f"lee_attitude_response_{axis_name}.png")
        fig.savefig(path, dpi=180)
        plt.close(fig)
        plot_paths.append(path)

    # Combined figure
    fig, axes = plt.subplots(1, len(axes_to_plot), figsize=(6 * len(axes_to_plot), 5),
                             squeeze=False)
    axes = axes[0]
    fig.suptitle("Lee Attitude Controller — command vs actual", fontsize=14)
    for i, axis_name in enumerate(axes_to_plot):
        mask = arrays["test_axis"] == axis_name
        time_s = arrays["time_s"][mask] - arrays["time_s"][mask][0]
        spec = AXIS_SPECS[axis_name]
        if spec["is_angle"]:
            cmd = arrays[f"cmd_{axis_name}"][mask]
            actual = arrays[f"euler_{axis_name}"][mask]
        else:
            cmd = arrays["cmd_yaw_rate"][mask]
            actual = arrays["body_rate_z"][mask]
        axes[i].plot(time_s, cmd, "--", label="cmd", linewidth=1.5)
        axes[i].plot(time_s, actual, "-", label="actual", linewidth=1.5)
        axes[i].set_title(f"{axis_name}")
        axes[i].set_xlabel("time [s]")
        axes[i].set_ylabel("rad" if spec["is_angle"] else "rad/s")
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
    fig.tight_layout()
    combined_path = os.path.join(run_dir, "lee_attitude_combined.png")
    fig.savefig(combined_path, dpi=180)
    plt.close(fig)
    plot_paths.append(combined_path)

    return plot_paths


def main():
    args = parse_args()
    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    run_dir = os.path.join(output_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    env = make_headless_task(args)
    rows = []
    try:
        obs, rewards, terminations, truncations, infos = env.reset()
        robot = env.sim_env.robot_manager.robot
        axes_to_run = selected_axes(args.axis)

        print(f"robot_class={type(robot).__name__}")
        print(f"force_application_level={getattr(robot, 'force_application_level', 'unknown')}")
        print(f"output_mode={getattr(robot, 'output_mode', 'unknown')}")
        print(f"controller={type(robot.controller).__name__}")
        print(f"K_rot={robot.controller.K_rot_tensor_current[0].detach().cpu().tolist()}")
        print(f"K_angvel={robot.controller.K_angvel_tensor_current[0].detach().cpu().tolist()}")
        print(f"axes={axes_to_run}")

        global_step = 0
        finite_ok = True
        dt = float(getattr(env, "dt", 0.01))
        sample_every = max(args.sample_every, 1)

        for axis_name in axes_to_run:
            obs, rewards, terminations, truncations, infos = env.reset()
            axis_step = 0
            for phase, command in build_axis_phases(axis_name):
                action = torch.tensor(command, dtype=torch.float32, device=args.device).repeat(
                    args.num_envs, 1
                )
                for local_step in range(args.steps_per_phase):
                    obs, rewards, terminations, truncations, infos = env.step(action)
                    global_step += 1
                    axis_step += 1

                    quat = env.robot_state[:, 3:7].detach()
                    euler = quat_to_euler_xyz(quat)
                    body_rates = env.obs_dict["robot_body_angvel"].detach()
                    force_z = env.last_actor1_force_z.detach()
                    torques = env.last_actor1_output_torques.detach()
                    gravity_norm = torch.norm(env.obs_dict["gravity"], dim=1).detach()
                    weight = env.obs_dict["robot_mass"].detach() * gravity_norm
                    tilt = quat_to_tilt_rad(quat)

                    finite_ok = finite_ok and finite_tensor(
                        obs["observations"], rewards, quat, body_rates, force_z, torques
                    )

                    if local_step % sample_every == 0 or local_step == args.steps_per_phase - 1:
                        mean_euler = euler.mean(dim=0)
                        mean_rates = body_rates.mean(dim=0)
                        mean_torques = torques.mean(dim=0)
                        rows.append({
                            "global_step": global_step,
                            "axis_step": axis_step,
                            "time_s": axis_step * dt,
                            "test_axis": axis_name,
                            "phase": phase,
                            "cmd_thrust": float(command[0]),
                            "cmd_roll_rad": float(command[1]),
                            "cmd_pitch_rad": float(command[2]),
                            "cmd_yaw_rate_rad_s": float(command[3]),
                            "euler_roll_rad": float(mean_euler[0].item()),
                            "euler_pitch_rad": float(mean_euler[1].item()),
                            "euler_yaw_rad": float(mean_euler[2].item()),
                            "body_rate_x": float(mean_rates[0].item()),
                            "body_rate_y": float(mean_rates[1].item()),
                            "body_rate_z": float(mean_rates[2].item()),
                            "body_rate_norm": float(torch.norm(body_rates, dim=1).mean().item()),
                            "tilt_deg": float(torch.rad2deg(tilt).mean().item()),
                            "force_z": float(force_z.mean().item()),
                            "weight": float(weight.mean().item()),
                            "torque_x": float(mean_torques[0].item()),
                            "torque_y": float(mean_torques[1].item()),
                            "torque_z": float(mean_torques[2].item()),
                            "reward": float(rewards.mean().item()),
                            "done_frac": float((terminations | truncations).float().mean().item()),
                        })

        summary = summarize_trace(rows, args, axes_to_run)
        summary["finite_ok"] = bool(finite_ok)
        summary["passed"] = bool(summary["passed"] and finite_ok)
        summary["axis_arg"] = args.axis
        summary["axes_run"] = axes_to_run
        summary["num_envs"] = args.num_envs
        summary["steps_per_phase"] = args.steps_per_phase
        summary["task"] = args.task
        summary["robot_class"] = type(robot).__name__
        summary["force_application_level"] = getattr(robot, "force_application_level", "unknown")
        summary["output_mode"] = getattr(robot, "output_mode", "unknown")
        summary["controller"] = type(robot.controller).__name__
        summary["K_rot"] = robot.controller.K_rot_tensor_current[0].detach().cpu().tolist()
        summary["K_angvel"] = robot.controller.K_angvel_tensor_current[0].detach().cpu().tolist()
        summary["run_dir"] = run_dir

        if not args.no_plot:
            summary["plot_paths"] = plot_step_responses(rows, run_dir, axes_to_run)

        # Write per-axis CSV
        for axis_name in axes_to_run:
            axis_rows = [r for r in rows if r["test_axis"] == axis_name]
            csv_path = os.path.join(run_dir, f"lee_attitude_{axis_name}.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(axis_rows[0].keys()))
                writer.writeheader()
                writer.writerows(axis_rows)
            print(f"csv_{axis_name}={csv_path}")

        # Write combined trace
        trace_path = os.path.join(run_dir, "trace.csv")
        with open(trace_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"trace={trace_path}")

        # Write summary
        summary_path = os.path.join(run_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"summary={summary_path}")

        print(f"\npassed={summary['passed']}")
        print(f"{'='*60}")
        for check in summary["checks"]:
            status = "PASS" if check["passed"] else "FAIL"
            print(f"  {status}  {check['name']}: {check.get('value', '')}")
        print(f"{'='*60}")

        if not summary["passed"]:
            raise SystemExit(2)
    finally:
        env.close()


if __name__ == "__main__":
    main()
