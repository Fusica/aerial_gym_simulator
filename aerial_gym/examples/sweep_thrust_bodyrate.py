#!/usr/bin/env python3
"""
Sweep runner for thrust bodyrate controller parameter configurations.

Tests 12 parameter combinations against RL-tightened pass criteria,
aggregates results, and recommends the best configuration for RL training.

Run:  conda activate aerialgym_v2 && python aerial_gym/examples/sweep_thrust_bodyrate.py
"""

import json
import os
import subprocess
import sys
import time
from collections import OrderedDict
from datetime import datetime

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TEST_SCRIPT = os.path.join(REPO_ROOT, "aerial_gym", "examples", "test_thrust_bodyrate_headless.py")
OUTPUT_BASE = os.path.join(REPO_ROOT, "runs", "thrust_bodyrate_sweep")

# ── Parameter combinations ──────────────────────────────────────────
COMBOS = [
    # Group A — Yaw scale sweep
    OrderedDict([
        ("name", "A1_baseline"),
        ("group", "A_yaw_scale"),
        ("override_scale_input", [1.0, 2.0, 2.0, 0.2]),
        ("override_torquelimit", [2.4, 2.4, 0.18]),
        ("override_komega", [1.0, 1.0, 0.8]),
    ]),
    OrderedDict([
        ("name", "A2_yaw_scale_0.5"),
        ("group", "A_yaw_scale"),
        ("override_scale_input", [1.0, 2.0, 2.0, 0.5]),
        ("override_torquelimit", [2.4, 2.4, 0.18]),
        ("override_komega", [1.0, 1.0, 0.8]),
    ]),
    OrderedDict([
        ("name", "A3_yaw_scale_1.0"),
        ("group", "A_yaw_scale"),
        ("override_scale_input", [1.0, 2.0, 2.0, 1.0]),
        ("override_torquelimit", [2.4, 2.4, 0.18]),
        ("override_komega", [1.0, 1.0, 0.8]),
    ]),
    OrderedDict([
        ("name", "A4_yaw_scale_2.0"),
        ("group", "A_yaw_scale"),
        ("override_scale_input", [1.0, 2.0, 2.0, 2.0]),
        ("override_torquelimit", [2.4, 2.4, 0.18]),
        ("override_komega", [1.0, 1.0, 0.8]),
    ]),
    # Group B — Yaw torque sweep
    OrderedDict([
        ("name", "B1_yaw_torque_0.20"),
        ("group", "B_yaw_torque"),
        ("override_scale_input", [1.0, 2.0, 2.0, 1.0]),
        ("override_torquelimit", [2.4, 2.4, 0.20]),
        ("override_komega", [1.0, 1.0, 0.8]),
    ]),
    OrderedDict([
        ("name", "B2_yaw_torque_0.30"),
        ("group", "B_yaw_torque"),
        ("override_scale_input", [1.0, 2.0, 2.0, 1.0]),
        ("override_torquelimit", [2.4, 2.4, 0.30]),
        ("override_komega", [1.0, 1.0, 0.8]),
    ]),
    # Group C — Roll/pitch aggressiveness
    OrderedDict([
        ("name", "C1_rp_scale_3.0"),
        ("group", "C_rp_scale"),
        ("override_scale_input", [1.0, 3.0, 3.0, 1.0]),
        ("override_torquelimit", [2.4, 2.4, 0.20]),
        ("override_komega", [1.0, 1.0, 0.8]),
    ]),
    OrderedDict([
        ("name", "C2_rp_scale_4.0"),
        ("group", "C_rp_scale"),
        ("override_scale_input", [1.0, 4.0, 4.0, 1.0]),
        ("override_torquelimit", [2.4, 2.4, 0.20]),
        ("override_komega", [1.0, 1.0, 0.8]),
    ]),
    # Group D — Gain tuning
    OrderedDict([
        ("name", "D1_kOmega_high_s2"),
        ("group", "D_gain_tuning"),
        ("override_scale_input", [1.0, 2.0, 2.0, 1.0]),
        ("override_torquelimit", [2.4, 2.4, 0.20]),
        ("override_komega", [1.5, 1.5, 1.2]),
    ]),
    OrderedDict([
        ("name", "D2_kOmega_high_s3"),
        ("group", "D_gain_tuning"),
        ("override_scale_input", [1.0, 3.0, 3.0, 1.0]),
        ("override_torquelimit", [2.4, 2.4, 0.20]),
        ("override_komega", [1.5, 1.5, 1.2]),
    ]),
    # Group E — Combined candidates
    OrderedDict([
        ("name", "E1_balanced_aggressive"),
        ("group", "E_combined"),
        ("override_scale_input", [1.0, 3.0, 3.0, 1.5]),
        ("override_torquelimit", [2.4, 2.4, 0.20]),
        ("override_komega", [1.2, 1.2, 1.0]),
    ]),
    OrderedDict([
        ("name", "E2_full_aggressive"),
        ("group", "E_combined"),
        ("override_scale_input", [1.0, 4.0, 4.0, 2.0]),
        ("override_torquelimit", [2.6, 2.6, 0.20]),
        ("override_komega", [1.5, 1.5, 1.2]),
    ]),
]

# ── RL-tightened pass criteria (CLI flags) ────────────────────────
TIGHT_CRITERIA = [
    "--max-steady-state-error-rad-s", "0.03",
    "--max-steady-state-error-ratio", "0.05",
    "--max-overshoot-ratio", "0.15",
    "--max-settling-time-s", "0.20",
    "--settling-error-ratio", "0.05",
    "--max-cross-axis-rate", "0.06",
    "--max-hover-rate", "0.10",
    "--min-thrust-weight", "0.70",
    "--max-thrust-weight", "1.20",
    "--min-rate-response", "0.02",
]


def run_combo(combo, sweep_run_dir):
    """Run a single combo via subprocess. Returns (combo, summary_dict, success_bool)."""
    combo_dir = os.path.join(sweep_run_dir, combo["name"])
    os.makedirs(combo_dir, exist_ok=True)

    cmd = [
        sys.executable, TEST_SCRIPT,
        "--task", "pursuit_guidance_task",
        "--axis", "all",
        "--num-envs", "4",
        "--steps-per-phase", "160",
        "--output-dir", combo_dir,
        "--no-plot",
        "--override-scale-input",
        str(combo["override_scale_input"][0]),
        str(combo["override_scale_input"][1]),
        str(combo["override_scale_input"][2]),
        str(combo["override_scale_input"][3]),
        "--override-torquelimit",
        str(combo["override_torquelimit"][0]),
        str(combo["override_torquelimit"][1]),
        str(combo["override_torquelimit"][2]),
        "--override-komega",
        str(combo["override_komega"][0]),
        str(combo["override_komega"][1]),
        str(combo["override_komega"][2]),
    ] + TIGHT_CRITERIA

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    elapsed = time.time() - t0

    passed = result.returncode == 0
    summary = None

    # Find summary.json (timestamped subdirectory)
    for fname in sorted(os.listdir(combo_dir)):
        subdir = os.path.join(combo_dir, fname)
        summary_path = os.path.join(subdir, "summary.json")
        if os.path.isfile(summary_path):
            with open(summary_path) as f:
                summary = json.load(f)
            break

    if summary is None:
        summary = {
            "error": "no summary.json found",
            "stdout": result.stdout[-2000:] if result.stdout else "",
            "stderr": result.stderr[-2000:] if result.stderr else "",
        }

    summary["_elapsed_s"] = round(elapsed, 1)
    summary["_returncode"] = result.returncode
    return combo, summary, passed


def extract_axis_metrics(summary):
    """Extract per-axis step metrics from summary axes dict."""
    metrics = {}
    for axis_name in ["roll", "pitch", "yaw"]:
        axis = summary.get("axes", {}).get(axis_name, {})
        phases = axis.get("phases", {})
        step_phase_name = f"{axis_name}_rate"
        step_phase = phases.get(step_phase_name, {})
        step_metrics = step_phase.get("step_metrics", {})

        metrics[f"{axis_name}_steady_error"] = step_metrics.get("steady_state_error_rad_s", None)
        metrics[f"{axis_name}_overshoot"] = step_metrics.get("overshoot_ratio", None)
        metrics[f"{axis_name}_settling_time"] = step_metrics.get("settling_time_s", None)
        metrics[f"{axis_name}_cross_axis"] = step_metrics.get("cross_axis_peak_after_0p15s_rad_s", None)
        metrics[f"{axis_name}_thrust_min"] = step_metrics.get("thrust_weight_min", None)
        metrics[f"{axis_name}_thrust_max"] = step_metrics.get("thrust_weight_max", None)
        metrics[f"{axis_name}_cmd_rate"] = step_metrics.get("command_rate_rad_s", None)

        # Yaw ratio check: steady error / commanded rate
        if axis_name == "yaw" and metrics["yaw_cmd_rate"] is not None:
            se = metrics["yaw_steady_error"]
            cr = metrics["yaw_cmd_rate"]
            metrics["yaw_steady_error_ratio"] = se / cr if cr and cr > 1e-6 else None
    return metrics


def score_combo(summary, combo):
    """Return weighted score and detail dict. Base 0-32 + bonus 0-7 = max 39."""
    if "axes" not in summary:
        return 0, {"error": summary.get("error", "no axes data")}

    m = extract_axis_metrics(summary)
    checks = {c["name"]: c["passed"] for c in summary.get("checks", [])}
    score = 0
    details = {}

    # ── Base scoring (0-32) ──
    # Critical (3x3=9): steady-state error ≤ 0.03 all axes
    se_ok = all(
        m.get(f"{ax}_steady_error") is not None and m[f"{ax}_steady_error"] <= 0.03
        for ax in ["roll", "pitch", "yaw"]
    )
    details["steady_error_ok"] = se_ok
    details["steady_errors"] = {ax: m.get(f"{ax}_steady_error") for ax in ["roll", "pitch", "yaw"]}
    if se_ok:
        score += 9

    # Critical (3x3=9): settling time ≤ 0.20 all axes
    settle_ok = all(
        m.get(f"{ax}_settling_time") is not None and m[f"{ax}_settling_time"] <= 0.20
        for ax in ["roll", "pitch", "yaw"]
    )
    details["settling_ok"] = settle_ok
    details["settling_times"] = {ax: m.get(f"{ax}_settling_time") for ax in ["roll", "pitch", "yaw"]}
    if settle_ok:
        score += 9

    # High (2x3=6): overshoot ≤ 0.15 all axes
    overshoot_ok = all(
        m.get(f"{ax}_overshoot") is not None and m[f"{ax}_overshoot"] <= 0.15
        for ax in ["roll", "pitch", "yaw"]
    )
    details["overshoot_ok"] = overshoot_ok
    if overshoot_ok:
        score += 6

    # High (2x3=6): cross-axis ≤ 0.06 all axes
    cross_ok = all(
        m.get(f"{ax}_cross_axis") is not None and m[f"{ax}_cross_axis"] <= 0.06
        for ax in ["roll", "pitch", "yaw"]
    )
    details["cross_axis_ok"] = cross_ok
    if cross_ok:
        score += 6

    # Medium (1): thrust/weight in [0.70, 1.20] all axes
    thrust_ok = all(
        m.get(f"{ax}_thrust_min") is not None
        and m.get(f"{ax}_thrust_max") is not None
        and m[f"{ax}_thrust_min"] >= 0.70
        and m[f"{ax}_thrust_max"] <= 1.20
        for ax in ["roll", "pitch", "yaw"]
    )
    details["thrust_ok"] = thrust_ok
    if thrust_ok:
        score += 1

    # Medium (1): hover rate ≤ 0.10
    hover_ok = all(
        checks.get(f"{ax}_hover_near_zero_body_rate", False)
        for ax in ["roll", "pitch", "yaw"]
    )
    details["hover_ok"] = hover_ok
    if hover_ok:
        score += 1

    # ── Bonus scoring (0-7): differentiate among passing configs ──
    bonus = 0
    bonus_detail = {}

    # Yaw settling speed bonus (faster = better for RL sample efficiency)
    yaw_settle = m.get("yaw_settling_time")
    if yaw_settle is not None and settle_ok:
        if yaw_settle <= 0.05:
            bonus += 3
            bonus_detail["yaw_settle_bonus"] = "3 (≤0.05s)"
        elif yaw_settle <= 0.10:
            bonus += 2
            bonus_detail["yaw_settle_bonus"] = "2 (≤0.10s)"
        elif yaw_settle <= 0.15:
            bonus += 1
            bonus_detail["yaw_settle_bonus"] = "1 (≤0.15s)"

    # Yaw authority bonus (based on full scale_input[3], the RL action range)
    # RL actions [-1,1] × scale = max yaw rate available to the policy
    yaw_full_authority = combo.get("override_scale_input", [0, 0, 0, 0])[3]
    if settle_ok:
        if yaw_full_authority >= 1.5:
            bonus += 3
            bonus_detail["yaw_authority_bonus"] = f"3 (scale={yaw_full_authority}, max ±{yaw_full_authority} rad/s)"
        elif yaw_full_authority >= 1.0:
            bonus += 2
            bonus_detail["yaw_authority_bonus"] = f"2 (scale={yaw_full_authority}, max ±{yaw_full_authority} rad/s)"
        elif yaw_full_authority >= 0.5:
            bonus += 1
            bonus_detail["yaw_authority_bonus"] = f"1 (scale={yaw_full_authority}, max ±{yaw_full_authority} rad/s)"

    # Roll/pitch settling speed bonus
    rp_settle_avg = (
        (m.get("roll_settling_time") or 0) + (m.get("pitch_settling_time") or 0)
    ) / 2.0
    if settle_ok and rp_settle_avg <= 0.10:
        bonus += 1
        bonus_detail["rp_settle_bonus"] = "1 (avg ≤0.10s)"

    score += bonus
    details["bonus"] = bonus
    details["bonus_detail"] = bonus_detail
    details["score"] = score
    details["max_score"] = 39
    details["base_score"] = 32
    return score, details


def print_results_table(results):
    """Print formatted comparison table."""
    header = (
        f"{'#':<3} {'Combo':<24} {'Score':<6} {'PASS':<6}"
        f"{'Roll_SE':<10} {'Pitch_SE':<10} {'Yaw_SE':<10}"
        f"{'Roll_settle':<13} {'Yaw_settle':<13} {'Key Issue':<30}"
    )
    sep = "─" * len(header)

    print("\n" + "=" * len(header))
    print(" Thrust Bodyrate Controller Sweep — RL-Tightened Criteria")
    print("=" * len(header))
    print(f"  Pass: steady_err≤0.03, overshoot≤0.15, settle≤0.20s, cross≤0.06, thrust∈[0.70,1.20]")
    print(sep)
    print(header)
    print(sep)

    for i, r in enumerate(results, 1):
        m = r["metrics"]
        score = r["score"]
        passed_str = "PASS" if r["passed"] else "FAIL"
        key_issues = r.get("key_issues", "")

        def fmt(v):
            if v is None:
                return "N/A"
            return f"{v:.4f}"

        print(
            f"{i:<3} {r['name']:<24} {score:<6} {passed_str:<6}"
            f"{fmt(m.get('roll_steady_error')):<10} {fmt(m.get('pitch_steady_error')):<10} {fmt(m.get('yaw_steady_error')):<10}"
            f"{fmt(m.get('roll_settling_time')):<13} {fmt(m.get('yaw_settling_time')):<13} {key_issues:<30}"
        )

    print(sep)

    # Highlight best
    best = max(results, key=lambda r: (r["score"], r["metrics"].get("yaw_cmd_rate", 0) or 0))
    print(f"\n  >>> RECOMMENDED: {best['name']} (score={best['score']}/{best['max_score']})")
    print(f"      scale_input={best['scale_input']}")
    print(f"      torqueLimit={best['torque_limit']}")
    print(f"      kOmega={best['komega']}")
    print()


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_run_dir = os.path.join(OUTPUT_BASE, ts)
    os.makedirs(sweep_run_dir, exist_ok=True)
    print(f"Sweep output: {sweep_run_dir}")
    print(f"Combos: {len(COMBOS)}\n")

    results = []
    for i, combo in enumerate(COMBOS, 1):
        print(f"[{i}/{len(COMBOS)}] {combo['name']} ... ", end="", flush=True)
        try:
            _combo, summary, passed = run_combo(combo, sweep_run_dir)
        except subprocess.TimeoutExpired:
            print("TIMEOUT")
            summary = {"error": "timeout", "_elapsed_s": 300}
            passed = False
        except Exception as e:
            print(f"ERROR: {e}")
            summary = {"error": str(e)}
            passed = False

        score, details = score_combo(summary, combo)
        metrics = extract_axis_metrics(summary)

        # Determine key issues
        key_issues = []
        if not details.get("steady_error_ok"):
            bad = [ax for ax in ["roll", "pitch", "yaw"]
                   if metrics.get(f"{ax}_steady_error") is not None
                   and metrics[f"{ax}_steady_error"] > 0.03]
            if bad:
                key_issues.append(f"SS_err:{','.join(bad)}")
        if not details.get("settling_ok"):
            bad_s = [ax for ax in ["roll", "pitch", "yaw"]
                     if metrics.get(f"{ax}_settling_time") is not None
                     and metrics[f"{ax}_settling_time"] > 0.20]
            if bad_s:
                key_issues.append(f"settle:{','.join(bad_s)}")
        if not details.get("overshoot_ok"):
            key_issues.append("overshoot")
        if not details.get("cross_axis_ok"):
            key_issues.append("cross_axis")
        if not details.get("hover_ok"):
            key_issues.append("hover_rate")

        result_entry = {
            "name": combo["name"],
            "group": combo["group"],
            "scale_input": combo["override_scale_input"],
            "torque_limit": combo["override_torquelimit"],
            "komega": combo["override_komega"],
            "passed": passed,
            "score": score,
            "max_score": details.get("max_score", 32),
            "metrics": metrics,
            "details": details,
            "key_issues": "; ".join(key_issues) if key_issues else "",
            "summary": summary,
        }
        results.append(result_entry)

        status = "PASS" if passed else "FAIL"
        print(f"{status}  score={score}/{details.get('max_score',26)}  {result_entry['key_issues']}")

    # Save aggregated results
    print_results_table(results)

    # Write comparison CSV
    csv_path = os.path.join(sweep_run_dir, "sweep_comparison.csv")
    fieldnames = [
        "name", "group", "scale_input", "torque_limit", "komega",
        "passed", "score", "max_score",
        "roll_steady_error", "pitch_steady_error", "yaw_steady_error",
        "roll_overshoot", "pitch_overshoot", "yaw_overshoot",
        "roll_settling_time", "pitch_settling_time", "yaw_settling_time",
        "roll_cross_axis", "pitch_cross_axis", "yaw_cross_axis",
        "roll_thrust_min", "pitch_thrust_min", "yaw_thrust_min",
        "roll_thrust_max", "pitch_thrust_max", "yaw_thrust_max",
        "yaw_steady_error_ratio", "yaw_cmd_rate",
        "steady_error_ok", "settling_ok", "overshoot_ok", "cross_axis_ok", "hover_ok",
        "key_issues",
    ]
    import csv
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row = {**r, **r["metrics"], **r["details"]}
            # Convert list values to string
            for k in ["scale_input", "torque_limit", "komega"]:
                row[k] = str(row.get(k, ""))
            writer.writerow(row)
    print(f"CSV: {csv_path}")

    # Write summary JSON
    json_path = os.path.join(sweep_run_dir, "sweep_summary.json")
    best = max(results, key=lambda r: (r["score"], r["metrics"].get("yaw_cmd_rate", 0) or 0))
    # Clean non-serializable items
    clean_results = []
    for r in results:
        cr = {k: v for k, v in r.items() if k != "summary"}
        clean_results.append(cr)
    sweep_summary = {
        "sweep_run_dir": sweep_run_dir,
        "criteria": {
            "max_steady_state_error_rad_s": 0.03,
            "max_steady_state_error_ratio": 0.05,
            "max_overshoot_ratio": 0.15,
            "max_settling_time_s": 0.20,
            "settling_error_ratio": 0.05,
            "max_cross_axis_rate": 0.06,
            "max_hover_rate": 0.10,
            "min_thrust_weight": 0.70,
            "max_thrust_weight": 1.20,
        },
        "recommendation": {
            "name": best["name"],
            "scale_input": best["scale_input"],
            "torque_limit": best["torque_limit"],
            "komega": best["komega"],
            "score": best["score"],
            "max_score": best["max_score"],
        },
        "results": clean_results,
    }
    with open(json_path, "w") as f:
        json.dump(sweep_summary, f, indent=2)
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
