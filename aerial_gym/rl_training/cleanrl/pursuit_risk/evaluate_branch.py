"""Evaluate action-conditioned branch ranking quality for a pursuit risk model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .dataset import (
    PursuitRiskBranchDataset,
    flatten_branch_batch,
    move_batch_to_device,
    split_branch_indices_by_source,
)
from .model import RiskNet, risk_scalar_from_outputs, risk_scalar_from_targets


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--branch-dataset", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cache-items", type=int, default=16)
    parser.add_argument("--ranking-epsilon", type=float, default=0.02)
    return parser.parse_args()


def binary_auc(target: np.ndarray, score: np.ndarray) -> float | None:
    target = (target >= 0.5).astype(np.int64)
    positive = target == 1
    negative = target == 0
    if int(positive.sum()) == 0 or int(negative.sum()) == 0:
        return None
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    return float(
        (ranks[positive].sum() - positive.sum() * (positive.sum() + 1) / 2.0)
        / (positive.sum() * negative.sum())
    )


def average_precision(target: np.ndarray, score: np.ndarray) -> float | None:
    target = (target >= 0.5).astype(np.int64)
    positives = int(target.sum())
    if positives == 0:
        return None
    order = np.argsort(-score)
    sorted_target = target[order]
    true_positives = np.cumsum(sorted_target)
    precision = true_positives / (np.arange(len(sorted_target)) + 1)
    return float((precision * sorted_target).sum() / positives)


def binary_confusion(target: np.ndarray, score: np.ndarray, threshold: float):
    target = (target >= 0.5).astype(bool)
    predicted = score >= threshold
    tp = int((predicted & target).sum())
    tn = int((~predicted & ~target).sum())
    fp = int((predicted & ~target).sum())
    fn = int((~predicted & target).sum())
    return tp, tn, fp, fn


def binary_metrics_at(target: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    tp, tn, fp, fn = binary_confusion(target, score, threshold)
    total = tp + tn + fp + fn
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "threshold": float(threshold),
        "accuracy": float((tp + tn) / total),
        "balanced_accuracy": float(0.5 * (recall + specificity)),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def threshold_sweep(target: np.ndarray, score: np.ndarray) -> tuple[dict, dict]:
    thresholds = np.unique(np.concatenate(([0.0, 0.5, 1.0], score)))
    best_accuracy = None
    best_f1 = None
    for threshold in thresholds:
        metrics = binary_metrics_at(target, score, threshold)
        if best_accuracy is None or (
            metrics["accuracy"],
            metrics["balanced_accuracy"],
        ) > (
            best_accuracy["accuracy"],
            best_accuracy["balanced_accuracy"],
        ):
            best_accuracy = metrics
        if best_f1 is None or (
            metrics["f1"],
            metrics["accuracy"],
        ) > (
            best_f1["f1"],
            best_f1["accuracy"],
        ):
            best_f1 = metrics
    return best_accuracy, best_f1


def calibration_error(target: np.ndarray, probability: np.ndarray, bins: int) -> tuple[float, float]:
    target = (target >= 0.5).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    mce = 0.0
    for idx in range(bins):
        if idx == bins - 1:
            mask = (probability >= edges[idx]) & (probability <= edges[idx + 1])
        else:
            mask = (probability >= edges[idx]) & (probability < edges[idx + 1])
        if not mask.any():
            continue
        confidence = float(probability[mask].mean())
        empirical = float(target[mask].mean())
        gap = abs(confidence - empirical)
        ece += float(mask.mean()) * gap
        mce = max(mce, gap)
    return float(ece), float(mce)


def top_fraction_metrics(target: np.ndarray, score: np.ndarray, fraction: float) -> dict:
    target = (target >= 0.5).astype(np.int64)
    count = max(1, int(round(len(target) * fraction)))
    order = np.argsort(-score)[:count]
    positives = int(target.sum())
    true_positives = int(target[order].sum())
    precision = true_positives / count
    return {
        "fraction": float(fraction),
        "count": count,
        "precision": float(precision),
        "recall": float(true_positives / max(positives, 1)),
        "lift": float(precision / max(float(target.mean()), 1e-12)),
    }


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict:
    error = prediction - target
    denominator = float(np.sum((target - target.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "r2": float(1.0 - np.sum(error * error) / denominator) if denominator > 0.0 else None,
        "pearson": float(np.corrcoef(target, prediction)[0, 1])
        if np.std(target) > 0.0 and np.std(prediction) > 0.0
        else None,
        "target_mean": float(target.mean()),
        "prediction_mean": float(prediction.mean()),
    }


def binary_report(target: np.ndarray, score: np.ndarray, bins: int) -> dict:
    target = target.astype(np.float64)
    score = score.astype(np.float64)
    best_accuracy, best_f1 = threshold_sweep(target, score)
    ece, mce = calibration_error(target, score, bins)
    binary_target = (target >= 0.5).astype(np.int64)
    clipped_score = np.clip(score, 1e-7, 1.0 - 1e-7)
    return {
        "n": int(len(target)),
        "positive_rate": float(binary_target.mean()),
        "target_mean": float(target.mean()),
        "prediction_mean": float(score.mean()),
        "auroc": binary_auc(target, score),
        "ap": average_precision(target, score),
        "brier": float(np.mean((score - target) ** 2)),
        "nll": float(np.mean(-(target * np.log(clipped_score) + (1.0 - target) * np.log(1.0 - clipped_score)))),
        "ece": ece,
        "mce": mce,
        "at_threshold_0p5": binary_metrics_at(target, score, 0.5),
        "best_accuracy": best_accuracy,
        "best_f1": best_f1,
        "top_5pct": top_fraction_metrics(target, score, 0.05),
        "top_10pct": top_fraction_metrics(target, score, 0.10),
        "top_20pct": top_fraction_metrics(target, score, 0.20),
    }


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def spearman(predicted: np.ndarray, target: np.ndarray) -> float | None:
    if len(predicted) < 2 or np.std(predicted) == 0.0 or np.std(target) == 0.0:
        return None
    return float(np.corrcoef(rankdata(predicted), rankdata(target))[0, 1])


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    checkpoint_path = Path(args.checkpoint) if args.checkpoint is not None else run_dir / "checkpoint_best.pt"

    device = torch.device(args.device)

    dataset = PursuitRiskBranchDataset(
        args.branch_dataset,
        cache_items=args.cache_items,
    )
    train_indices, val_indices = split_branch_indices_by_source(
        dataset,
        config["val_fraction"],
        config["seed"],
    )
    indices = train_indices if args.split == "train" else val_indices
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    model = RiskNet(
        hidden_dim=config["hidden_dim"],
        resize_hw=tuple(config["lidar_resize"]),
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    anchors = []
    flat_pred_risk = []
    flat_true_risk = []
    flat_p50 = []
    flat_y50 = []
    flat_p150 = []
    flat_y150 = []
    flat_pred_severity50 = []
    flat_severity50 = []
    flat_pred_severity150 = []
    flat_severity150 = []
    flat_pred_recovery150 = []
    flat_recovery150 = []
    flat_recovery_valid150 = []
    pair_correct = 0
    pair_total = 0

    with torch.inference_mode():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            batch_size, candidates = batch["candidate_action"].shape[:2]
            lidar, s_red, action, targets, _group_ids = flatten_branch_batch(batch)
            outputs = model(lidar, s_red, action)
            predicted_risk = risk_scalar_from_outputs(outputs).cpu().numpy().reshape(batch_size, candidates)
            true_risk = risk_scalar_from_targets(
                {key: value.cpu() for key, value in targets.items()}
            ).numpy().reshape(batch_size, candidates)
            p50 = torch.sigmoid(outputs["p_loss_50_logit"]).cpu().numpy().reshape(batch_size, candidates)
            p150 = torch.sigmoid(outputs["p_loss_150_logit"]).cpu().numpy().reshape(batch_size, candidates)
            y50 = targets["loss_prob_50"].cpu().numpy().reshape(batch_size, candidates)
            y150 = targets["loss_prob_150"].cpu().numpy().reshape(batch_size, candidates)
            pred_severity50 = outputs["severity_50"].cpu().numpy().reshape(batch_size, candidates)
            pred_severity150 = outputs["severity_150"].cpu().numpy().reshape(batch_size, candidates)
            severity50 = targets["loss_severity_50"].cpu().numpy().reshape(batch_size, candidates)
            severity150 = targets["loss_severity_150"].cpu().numpy().reshape(batch_size, candidates)
            pred_recovery150 = torch.sigmoid(outputs["p_recover_150_logit"]).cpu().numpy().reshape(batch_size, candidates)
            recovery150 = targets["recovery_150"].cpu().numpy().reshape(batch_size, candidates)
            recovery_valid150 = targets["recovery_valid_150"].cpu().numpy().reshape(batch_size, candidates)

            flat_pred_risk.append(predicted_risk.reshape(-1))
            flat_true_risk.append(true_risk.reshape(-1))
            flat_p50.append(p50.reshape(-1))
            flat_y50.append(y50.reshape(-1))
            flat_p150.append(p150.reshape(-1))
            flat_y150.append(y150.reshape(-1))
            flat_pred_severity50.append(pred_severity50.reshape(-1))
            flat_severity50.append(severity50.reshape(-1))
            flat_pred_severity150.append(pred_severity150.reshape(-1))
            flat_severity150.append(severity150.reshape(-1))
            flat_pred_recovery150.append(pred_recovery150.reshape(-1))
            flat_recovery150.append(recovery150.reshape(-1))
            flat_recovery_valid150.append(recovery_valid150.reshape(-1))

            for anchor_idx in range(batch_size):
                pred = predicted_risk[anchor_idx]
                true = true_risk[anchor_idx]
                pred_best = int(np.argmin(pred))
                true_order = np.argsort(true)
                oracle_best = int(true_order[0])
                top3 = set(int(idx) for idx in true_order[: min(3, candidates)])
                true_range = float(np.max(true) - np.min(true))
                regret = float(true[pred_best] - true[oracle_best])
                anchor_pair_correct = 0
                anchor_pair_total = 0
                for left in range(candidates):
                    for right in range(left + 1, candidates):
                        true_diff = float(true[left] - true[right])
                        if abs(true_diff) <= args.ranking_epsilon:
                            continue
                        pred_diff = float(pred[left] - pred[right])
                        anchor_pair_total += 1
                        anchor_pair_correct += int(
                            (true_diff > 0.0 and pred_diff > 0.0)
                            or (true_diff < 0.0 and pred_diff < 0.0)
                        )
                pair_correct += anchor_pair_correct
                pair_total += anchor_pair_total
                anchors.append(
                    {
                        "hit_top1": int(pred_best == oracle_best),
                        "hit_top3": int(pred_best in top3),
                        "oracle_true_risk": float(true[oracle_best]),
                        "pred_selected_true_risk": float(true[pred_best]),
                        "random_true_risk_mean": float(np.mean(true)),
                        "regret": regret,
                        "normalized_regret": regret / max(true_range, 1e-8),
                        "spearman": spearman(pred, true),
                        "true_range": true_range,
                    }
                )

    flat_pred_risk = np.concatenate(flat_pred_risk)
    flat_true_risk = np.concatenate(flat_true_risk)
    flat_p50 = np.concatenate(flat_p50)
    flat_y50 = np.concatenate(flat_y50)
    flat_p150 = np.concatenate(flat_p150)
    flat_y150 = np.concatenate(flat_y150)
    flat_pred_severity50 = np.concatenate(flat_pred_severity50)
    flat_severity50 = np.concatenate(flat_severity50)
    flat_pred_severity150 = np.concatenate(flat_pred_severity150)
    flat_severity150 = np.concatenate(flat_severity150)
    flat_pred_recovery150 = np.concatenate(flat_pred_recovery150)
    flat_recovery150 = np.concatenate(flat_recovery150)
    flat_recovery_valid150 = np.concatenate(flat_recovery_valid150)

    spearman_values = [row["spearman"] for row in anchors if row["spearman"] is not None]
    pred_selected = np.asarray([row["pred_selected_true_risk"] for row in anchors], dtype=np.float64)
    oracle = np.asarray([row["oracle_true_risk"] for row in anchors], dtype=np.float64)
    random_mean = np.asarray([row["random_true_risk_mean"] for row in anchors], dtype=np.float64)
    regrets = np.asarray([row["regret"] for row in anchors], dtype=np.float64)
    normalized_regrets = np.asarray([row["normalized_regret"] for row in anchors], dtype=np.float64)
    true_ranges = np.asarray([row["true_range"] for row in anchors], dtype=np.float64)
    risk_report = regression_metrics(flat_true_risk, flat_pred_risk)
    h50_report = binary_report(flat_y50, flat_p50, 15)
    h150_report = binary_report(flat_y150, flat_p150, 15)

    report = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "branch_total_valid_pair_anchors": len(dataset),
        "anchors": len(subset),
        "candidate_count": len(flat_pred_risk) // max(len(subset), 1),
        "pairwise_accuracy": float(pair_correct / max(pair_total, 1)),
        "pairwise_pairs": int(pair_total),
        "spearman_mean": float(np.mean(spearman_values)) if spearman_values else None,
        "spearman_median": float(np.median(spearman_values)) if spearman_values else None,
        "top1_hit_rate": float(np.mean([row["hit_top1"] for row in anchors])),
        "top3_hit_rate": float(np.mean([row["hit_top3"] for row in anchors])),
        "mean_true_range": float(np.mean(true_ranges)),
        "mean_regret": float(np.mean(regrets)),
        "median_regret": float(np.median(regrets)),
        "mean_normalized_regret": float(np.mean(normalized_regrets)),
        "median_normalized_regret": float(np.median(normalized_regrets)),
        "pred_selected_true_risk_mean": float(np.mean(pred_selected)),
        "oracle_true_risk_mean": float(np.mean(oracle)),
        "random_candidate_true_risk_mean": float(np.mean(random_mean)),
        "improvement_vs_random": float(np.mean(random_mean) - np.mean(pred_selected)),
        "oracle_gap": float(np.mean(pred_selected) - np.mean(oracle)),
        "h50": h50_report,
        "h150": h150_report,
        "mAP_h50_h150": float(np.mean([h50_report["ap"], h150_report["ap"]])),
        "mean_auroc_h50_h150": float(np.mean([h50_report["auroc"], h150_report["auroc"]])),
        "severity_50": regression_metrics(flat_severity50, flat_pred_severity50),
        "severity_150": regression_metrics(flat_severity150, flat_pred_severity150),
        "risk_scalar": risk_report,
        "recovery_150": regression_metrics(
            flat_recovery150[flat_recovery_valid150],
            flat_pred_recovery150[flat_recovery_valid150],
        )
        if flat_recovery_valid150.any()
        else None,
    }

    output_path = Path(args.output) if args.output is not None else run_dir / f"branch_action_metrics_{args.split}.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
