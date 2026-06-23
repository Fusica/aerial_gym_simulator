"""Evaluate a trained pursuit risk model on a cached behavior split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import PursuitRiskBehaviorDataset, move_batch_to_device
from .evaluate_branch import binary_report, regression_metrics
from .model import RiskNet, risk_scalar_from_outputs, risk_scalar_from_targets


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--behavior-cache", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--calibration-bins", type=int, default=15)
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    checkpoint_path = Path(args.checkpoint) if args.checkpoint is not None else run_dir / "checkpoint_best.pt"
    requested_cuda = str(args.device).startswith("cuda")
    cuda_available = torch.cuda.is_available()
    device = torch.device(args.device if cuda_available or not requested_cuda else "cpu")

    dataset = PursuitRiskBehaviorDataset(args.behavior_cache, split=args.split)
    loader_kwargs = {
        "batch_size": int(args.batch_size),
        "shuffle": False,
        "num_workers": int(args.num_workers),
        "pin_memory": device.type == "cuda",
        "drop_last": False,
    }
    if int(args.num_workers) > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = int(args.prefetch_factor)
    loader = DataLoader(dataset, **loader_kwargs)

    model = RiskNet(
        hidden_dim=int(config["hidden_dim"]),
        resize_hw=tuple(config["lidar_resize"]),
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    collected = {
        key: []
        for key in (
            "p50",
            "p150",
            "target50",
            "target150",
            "severity50",
            "severity150",
            "pred_severity50",
            "pred_severity150",
            "recovery150",
            "pred_recovery150",
            "recovery_valid150",
            "risk_scalar",
            "pred_risk_scalar",
        )
    }
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            targets = batch["targets"]
            outputs = model(batch["lidar"], batch["s_red"], batch["action"])
            collected["p50"].append(torch.sigmoid(outputs["p_loss_50_logit"]).detach().cpu().numpy())
            collected["p150"].append(torch.sigmoid(outputs["p_loss_150_logit"]).detach().cpu().numpy())
            collected["target50"].append(targets["loss_prob_50"].detach().cpu().numpy())
            collected["target150"].append(targets["loss_prob_150"].detach().cpu().numpy())
            collected["severity50"].append(targets["loss_severity_50"].detach().cpu().numpy())
            collected["severity150"].append(targets["loss_severity_150"].detach().cpu().numpy())
            collected["pred_severity50"].append(outputs["severity_50"].detach().cpu().numpy())
            collected["pred_severity150"].append(outputs["severity_150"].detach().cpu().numpy())
            collected["recovery150"].append(targets["recovery_150"].detach().cpu().numpy())
            collected["pred_recovery150"].append(torch.sigmoid(outputs["p_recover_150_logit"]).detach().cpu().numpy())
            collected["recovery_valid150"].append(targets["recovery_valid_150"].detach().cpu().numpy().astype(bool))
            cpu_targets = {key: value.detach().cpu() for key, value in targets.items()}
            cpu_outputs = {key: value.detach().cpu() for key, value in outputs.items()}
            collected["risk_scalar"].append(risk_scalar_from_targets(cpu_targets).numpy())
            collected["pred_risk_scalar"].append(risk_scalar_from_outputs(cpu_outputs).numpy())

    collected = {key: np.concatenate(value) for key, value in collected.items()}
    recovery_mask = collected["recovery_valid150"].astype(bool)
    report = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "num_samples": int(len(dataset)),
        "h50": binary_report(collected["target50"], collected["p50"], int(args.calibration_bins)),
        "h150": binary_report(collected["target150"], collected["p150"], int(args.calibration_bins)),
        "severity_50": regression_metrics(collected["severity50"], collected["pred_severity50"]),
        "severity_150": regression_metrics(collected["severity150"], collected["pred_severity150"]),
        "risk_scalar": regression_metrics(collected["risk_scalar"], collected["pred_risk_scalar"]),
    }
    report["mAP_h50_h150"] = float(np.mean([report["h50"]["ap"], report["h150"]["ap"]]))
    report["mean_auroc_h50_h150"] = float(np.mean([report["h50"]["auroc"], report["h150"]["auroc"]]))
    if recovery_mask.any():
        report["recovery_150"] = regression_metrics(
            collected["recovery150"][recovery_mask],
            collected["pred_recovery150"][recovery_mask],
        )
    else:
        report["recovery_150"] = None

    output_path = Path(args.output) if args.output is not None else run_dir / f"{args.split}_extended_metrics.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
