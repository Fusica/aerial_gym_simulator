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


ZERO_COMMAND = (0.0, 0.0, 0.0, 0.0)
AXIS_SPECS = {
    "roll": {
        "phase": "roll_rate",
        "command": (0.0, 0.18, 0.0, 0.0),
        "rate_index": 0,
        "cmd_index": 1,
    },
    "pitch": {
        "phase": "pitch_rate",
        "command": (0.0, 0.0, 0.18, 0.0),
        "rate_index": 1,
        "cmd_index": 2,
    },
    "yaw": {
        "phase": "yaw_rate",
        "command": (0.0, 0.0, 0.0, 0.25),
        "rate_index": 2,
        "cmd_index": 3,
    },
}

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, "runs", "thrust_bodyrate_headless")


def parse_args():
    parser = argparse.ArgumentParser(
        "Headless smoke test for thrust + body-rate controller in the v2 simulator."
    )
    parser.add_argument("--task", type=str, default="pursuit_guidance_task")
    parser.add_argument("--axis", type=str, default="all", choices=["all", "roll", "pitch", "yaw"])
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps-per-phase", type=int, default=160)
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-body-rate", type=float, default=8.0)
    parser.add_argument("--max-tilt-deg", type=float, default=80.0)
    parser.add_argument("--min-rate-response", type=float, default=0.15)
    parser.add_argument("--max-hover-rate", type=float, default=0.35)
    parser.add_argument("--max-steady-state-error-rad-s", type=float, default=0.08)
    parser.add_argument("--max-steady-state-error-ratio", type=float, default=0.10)
    parser.add_argument("--max-overshoot-ratio", type=float, default=0.25)
    parser.add_argument("--max-settling-time-s", type=float, default=0.35)
    parser.add_argument("--settling-error-ratio", type=float, default=0.10)
    parser.add_argument("--max-cross-axis-rate", type=float, default=0.12)
    parser.add_argument("--min-thrust-weight", type=float, default=0.70)
    parser.add_argument("--max-thrust-weight", type=float, default=1.35)
    parser.add_argument("--no-plot", action="store_true", default=False)
    return parser.parse_args()


def selected_axes(axis_arg):
    if axis_arg == "all":
        return ["roll", "pitch", "yaw"]
    return [axis_arg]


def build_axis_phases(axis_name):
    spec = AXIS_SPECS[axis_name]
    return (
        ("hover", ZERO_COMMAND),
        (spec["phase"], spec["command"]),
        ("recovery", ZERO_COMMAND),
    )


def make_headless_task(args):
    original_argv = sys.argv[:]
    sys.argv = [
        original_argv[0],
        "--headless",
        "True",
        "--num_envs",
        str(args.num_envs),
        "--use_warp",
        "False",
        "--sim_device",
        args.device,
    ]
    try:
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


def quat_to_tilt_rad(quat_xyzw):
    x = quat_xyzw[:, 0]
    y = quat_xyzw[:, 1]
    z = quat_xyzw[:, 2]
    w = quat_xyzw[:, 3]
    body_z_world_z = 1.0 - 2.0 * (x * x + y * y)
    body_z_world_z = torch.clamp(body_z_world_z, -1.0, 1.0)
    return torch.acos(body_z_world_z)


def finite_tensor(*tensors):
    return all(torch.isfinite(tensor).all().item() for tensor in tensors)


def phase_tail_mask(phase_names, phase):
    idx = np.where(np.asarray(phase_names) == phase)[0]
    if idx.size == 0:
        return idx
    tail_start = idx[int(0.75 * idx.size)]
    return idx[idx >= tail_start]


def rows_for_axis_phase(rows, axis_name, phase):
    return [row for row in rows if row["test_axis"] == axis_name and row["phase"] == phase]


def rate_vector(row):
    return np.asarray(
        [row["body_rate_x"], row["body_rate_y"], row["body_rate_z"]], dtype=np.float64
    )


def settling_time(time_s, actual_rate, target_rate, band):
    error = np.abs(actual_rate - target_rate)
    within = error <= band
    for idx in range(within.size):
        if within[idx] and np.all(within[idx:]):
            return float(time_s[idx] - time_s[0])
    return None


def summarize_axis_response(rows, args, axis_name, scale_input):
    checks = []
    phases = {}
    spec = AXIS_SPECS[axis_name]
    step_phase = spec["phase"]
    rate_index = spec["rate_index"]
    cmd_index = spec["cmd_index"]
    command_rate = spec["command"][cmd_index] * scale_input[cmd_index]
    steady_error_limit = max(
        args.max_steady_state_error_rad_s,
        abs(command_rate) * args.max_steady_state_error_ratio,
    )
    settling_band = max(args.max_steady_state_error_rad_s, abs(command_rate) * args.settling_error_ratio)

    axis_rows = [row for row in rows if row["test_axis"] == axis_name]
    for phase, command in build_axis_phases(axis_name):
        phase_rows_all = rows_for_axis_phase(rows, axis_name, phase)
        if not phase_rows_all:
            checks.append({"name": f"{axis_name}_{phase}_has_samples", "passed": False})
            continue

        phase_names = [row["phase"] for row in axis_rows]
        tail_idx = phase_tail_mask(phase_names, phase)
        if tail_idx.size == 0:
            checks.append({"name": f"{axis_name}_{phase}_has_tail_samples", "passed": False})
            continue
        phase_rows = [axis_rows[i] for i in tail_idx]
        rate_samples = np.stack([rate_vector(row) for row in phase_rows], axis=0)
        mean_rates = np.mean(rate_samples, axis=0)
        max_rate = max(row["body_rate_norm"] for row in phase_rows)
        max_tilt = max(row["tilt_deg"] for row in phase_rows)
        force_ratio = np.mean([row["force_z"] / max(row["weight"], 1e-6) for row in phase_rows])
        phases[phase] = {
            "command": command,
            "mean_body_rate": mean_rates.tolist(),
            "max_body_rate_norm": float(max_rate),
            "max_tilt_deg": float(max_tilt),
            "mean_force_to_weight": float(force_ratio),
        }

        checks.append(
            {
                "name": f"{axis_name}_{phase}_rate_bounded",
                "passed": bool(max_rate < args.max_body_rate),
                "value": float(max_rate),
                "threshold": args.max_body_rate,
            }
        )
        checks.append(
            {
                "name": f"{axis_name}_{phase}_tilt_bounded",
                "passed": bool(max_tilt < args.max_tilt_deg),
                "value": float(max_tilt),
                "threshold": args.max_tilt_deg,
            }
        )

        if phase in ("hover", "recovery"):
            checks.append(
                {
                    "name": f"{axis_name}_{phase}_near_zero_body_rate",
                    "passed": bool(np.linalg.norm(mean_rates) < args.max_hover_rate),
                    "value": float(np.linalg.norm(mean_rates)),
                    "threshold": args.max_hover_rate,
                }
            )

    step_rows = rows_for_axis_phase(rows, axis_name, step_phase)
    if step_rows:
        time_s = np.asarray([row["time_s"] for row in step_rows], dtype=np.float64)
        rates = np.stack([rate_vector(row) for row in step_rows], axis=0)
        actual_axis_rate = rates[:, rate_index]
        other_indices = [idx for idx in range(3) if idx != rate_index]
        settle_ignore_steps = max(1, int(0.15 / max(time_s[1] - time_s[0], 1e-6))) if len(time_s) > 1 else 1
        tail_start = int(0.75 * len(step_rows))
        post_transient_start = min(settle_ignore_steps, len(step_rows) - 1)

        tail_mean_rate = float(np.mean(actual_axis_rate[tail_start:]))
        steady_error = abs(tail_mean_rate - command_rate)
        max_axis_rate = float(np.max(actual_axis_rate))
        overshoot_ratio = max(0.0, max_axis_rate - command_rate) / max(abs(command_rate), 1e-6)
        cross_axis_peak = float(np.max(np.abs(rates[post_transient_start:, other_indices])))
        force_weight = np.asarray(
            [row["force_z"] / max(row["weight"], 1e-6) for row in step_rows], dtype=np.float64
        )
        thrust_min = float(np.min(force_weight))
        thrust_max = float(np.max(force_weight))
        settle_time = settling_time(time_s, actual_axis_rate, command_rate, settling_band)

        checks.extend(
            [
                {
                    "name": f"{axis_name}_step_response_sign",
                    "passed": bool(tail_mean_rate > args.min_rate_response),
                    "value": tail_mean_rate,
                    "threshold": args.min_rate_response,
                },
                {
                    "name": f"{axis_name}_steady_state_error",
                    "passed": bool(steady_error <= steady_error_limit),
                    "value": float(steady_error),
                    "threshold": float(steady_error_limit),
                },
                {
                    "name": f"{axis_name}_overshoot_ratio",
                    "passed": bool(overshoot_ratio <= args.max_overshoot_ratio),
                    "value": float(overshoot_ratio),
                    "threshold": args.max_overshoot_ratio,
                },
                {
                    "name": f"{axis_name}_settling_time_s",
                    "passed": bool(
                        settle_time is not None and settle_time <= args.max_settling_time_s
                    ),
                    "value": float(settle_time) if settle_time is not None else None,
                    "threshold": args.max_settling_time_s,
                },
                {
                    "name": f"{axis_name}_cross_axis_rate_after_0p15s",
                    "passed": bool(cross_axis_peak <= args.max_cross_axis_rate),
                    "value": cross_axis_peak,
                    "threshold": args.max_cross_axis_rate,
                },
                {
                    "name": f"{axis_name}_thrust_weight_min",
                    "passed": bool(thrust_min >= args.min_thrust_weight),
                    "value": thrust_min,
                    "threshold": args.min_thrust_weight,
                },
                {
                    "name": f"{axis_name}_thrust_weight_max",
                    "passed": bool(thrust_max <= args.max_thrust_weight),
                    "value": thrust_max,
                    "threshold": args.max_thrust_weight,
                },
            ]
        )
        phases[step_phase]["step_metrics"] = {
            "command_rate_rad_s": float(command_rate),
            "tail_mean_rate_rad_s": tail_mean_rate,
            "steady_state_error_rad_s": float(steady_error),
            "steady_state_error_limit_rad_s": float(steady_error_limit),
            "overshoot_ratio": float(overshoot_ratio),
            "settling_time_s": float(settle_time) if settle_time is not None else None,
            "settling_band_rad_s": float(settling_band),
            "cross_axis_peak_after_0p15s_rad_s": cross_axis_peak,
            "thrust_weight_min": thrust_min,
            "thrust_weight_max": thrust_max,
        }
    else:
        checks.append({"name": f"{axis_name}_{step_phase}_has_samples", "passed": False})

    return {"phases": phases, "checks": checks, "passed": all(check["passed"] for check in checks)}


def summarize_trace(rows, args, axes, scale_input):
    axis_summaries = {}
    all_checks = []
    for axis_name in axes:
        axis_summary = summarize_axis_response(rows, args, axis_name, scale_input)
        axis_summaries[axis_name] = axis_summary
        all_checks.extend(axis_summary["checks"])

    return {
        "axes": axis_summaries,
        "checks": all_checks,
        "passed": all(check["passed"] for check in all_checks),
        "criteria": {
            "max_steady_state_error_rad_s": args.max_steady_state_error_rad_s,
            "max_steady_state_error_ratio": args.max_steady_state_error_ratio,
            "max_overshoot_ratio": args.max_overshoot_ratio,
            "max_settling_time_s": args.max_settling_time_s,
            "settling_error_ratio": args.settling_error_ratio,
            "max_cross_axis_rate_after_0p15s": args.max_cross_axis_rate,
            "min_thrust_weight": args.min_thrust_weight,
            "max_thrust_weight": args.max_thrust_weight,
            "max_hover_rate_norm": args.max_hover_rate,
        },
    }


def _rows_to_arrays(rows, scale_input):
    arrays = {
        "time_s": np.asarray([row["time_s"] for row in rows], dtype=np.float64),
        "test_axis": np.asarray([row["test_axis"] for row in rows]),
        "phase": np.asarray([row["phase"] for row in rows]),
        "cmd_p": np.asarray([row["cmd_p_rate"] for row in rows], dtype=np.float64)
        * scale_input[1],
        "cmd_q": np.asarray([row["cmd_q_rate"] for row in rows], dtype=np.float64)
        * scale_input[2],
        "cmd_r": np.asarray([row["cmd_r_rate"] for row in rows], dtype=np.float64)
        * scale_input[3],
        "actual_p": np.asarray([row["body_rate_x"] for row in rows], dtype=np.float64),
        "actual_q": np.asarray([row["body_rate_y"] for row in rows], dtype=np.float64),
        "actual_r": np.asarray([row["body_rate_z"] for row in rows], dtype=np.float64),
        "tau_x": np.asarray([row["torque_x"] for row in rows], dtype=np.float64),
        "tau_y": np.asarray([row["torque_y"] for row in rows], dtype=np.float64),
        "tau_z": np.asarray([row["torque_z"] for row in rows], dtype=np.float64),
        "cmd_thrust_norm": np.asarray([row["cmd_thrust"] for row in rows], dtype=np.float64)
        * scale_input[0]
        + 1.0,
        "actual_thrust_norm": np.asarray(
            [row["force_z"] / max(row["weight"], 1e-6) for row in rows], dtype=np.float64
        ),
    }
    return arrays


def plot_step_responses(rows, run_dir, scale_input, axes_to_plot):
    cache_dir = os.path.join(os.path.dirname(run_dir), "matplotlib_cache")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", cache_dir)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arrays = _rows_to_arrays(rows, scale_input)
    plot_paths = []

    for axis_name in axes_to_plot:
        mask = arrays["test_axis"] == axis_name
        if not mask.any():
            continue

        time_s = arrays["time_s"][mask]
        time_s = time_s - time_s[0]

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        fig.suptitle(f"Body-rate step response: {axis_name}", fontsize=14)

        axes[0].plot(time_s, arrays["cmd_p"][mask], "--", label="cmd_p", linewidth=1.5)
        axes[0].plot(time_s, arrays["actual_p"][mask], "-", label="actual_p", linewidth=1.5)
        axes[0].plot(time_s, arrays["cmd_q"][mask], "--", label="cmd_q", linewidth=1.5)
        axes[0].plot(time_s, arrays["actual_q"][mask], "-", label="actual_q", linewidth=1.5)
        axes[0].plot(time_s, arrays["cmd_r"][mask], "--", label="cmd_r", linewidth=1.5)
        axes[0].plot(time_s, arrays["actual_r"][mask], "-", label="actual_r", linewidth=1.5)
        axes[0].set_ylabel("body rate [rad/s]")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(ncol=3, loc="upper right")

        axes[1].plot(time_s, arrays["tau_x"][mask], label="tau_x", linewidth=1.5)
        axes[1].plot(time_s, arrays["tau_y"][mask], label="tau_y", linewidth=1.5)
        axes[1].plot(time_s, arrays["tau_z"][mask], label="tau_z", linewidth=1.5)
        axes[1].set_ylabel("torque")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(ncol=3, loc="upper right")

        axes[2].plot(
            time_s,
            arrays["cmd_thrust_norm"][mask],
            "--",
            label="cmd_thrust_norm",
            linewidth=1.5,
        )
        axes[2].plot(
            time_s,
            arrays["actual_thrust_norm"][mask],
            "-",
            label="actual_force_weight",
            linewidth=1.5,
        )
        axes[2].set_ylabel("thrust / weight")
        axes[2].set_xlabel("time [s]")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc="upper right")

        phase = arrays["phase"][mask]
        transition_idx = np.where(phase[1:] != phase[:-1])[0] + 1
        for ax in axes:
            for idx in transition_idx:
                ax.axvline(time_s[idx], color="0.55", linewidth=0.8, alpha=0.45)

        fig.tight_layout()
        path = os.path.join(run_dir, f"body_rate_response_{axis_name}.png")
        fig.savefig(path, dpi=180)
        plt.close(fig)
        plot_paths.append(path)

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
        print(f"kOmega={robot.controller.kOmega.detach().cpu().tolist()}")
        print(f"kOmegaI={robot.controller.kOmegaI.detach().cpu().tolist()}")
        print(f"kOmegaD={robot.controller.kOmegaD.detach().cpu().tolist()}")
        scale_input = robot.controller.scale_input.detach().cpu().numpy().astype(np.float64)
        print(f"scale_input={scale_input.tolist()}")
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

                    body_rates = env.obs_dict["robot_body_angvel"].detach()
                    quat = env.robot_state[:, 3:7].detach()
                    force_z = env.last_actor1_force_z.detach()
                    torques = env.last_actor1_output_torques.detach()
                    gravity_norm = torch.norm(env.obs_dict["gravity"], dim=1).detach()
                    weight = env.obs_dict["robot_mass"].detach() * gravity_norm
                    tilt = quat_to_tilt_rad(quat)

                    finite_ok = finite_ok and finite_tensor(
                        obs["observations"], rewards, body_rates, quat, force_z, torques
                    )

                    if local_step % sample_every == 0 or local_step == args.steps_per_phase - 1:
                        mean_rates = body_rates.mean(dim=0)
                        mean_torques = torques.mean(dim=0)
                        rows.append(
                            {
                                "global_step": global_step,
                                "axis_step": axis_step,
                                "time_s": axis_step * dt,
                                "test_axis": axis_name,
                                "phase": phase,
                                "cmd_thrust": command[0],
                                "cmd_p_rate": command[1],
                                "cmd_q_rate": command[2],
                                "cmd_r_rate": command[3],
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
                            }
                        )

        summary = summarize_trace(rows, args, axes_to_run, scale_input)
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
        summary["scale_input"] = scale_input.tolist()
        summary["run_dir"] = run_dir

        if not args.no_plot:
            summary["plot_paths"] = plot_step_responses(rows, run_dir, scale_input, axes_to_run)

        trace_path = os.path.join(run_dir, "trace.csv")
        with open(trace_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        summary_path = os.path.join(run_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"trace={trace_path}")
        print(f"summary={summary_path}")
        print(f"passed={summary['passed']}")
        for check in summary["checks"]:
            status = "PASS" if check["passed"] else "FAIL"
            print(f"{status} {check['name']}: {check.get('value', '')}")

        if not summary["passed"]:
            raise SystemExit(2)
    finally:
        env.close()


if __name__ == "__main__":
    main()
