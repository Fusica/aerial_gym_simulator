"""Evaluate action-conditioned branch ranking quality for a pursuit risk model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import (
    PursuitRiskBranchCacheDataset,
    flatten_branch_batch,
    move_batch_to_device,
)
from .model import RiskNet, risk_scalar_from_outputs, risk_scalar_from_targets


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--branch-cache", required=True)
    parser.add_argument("--checkpoint", default="checkpoint_best.pt")
    parser.add_argument("--output", default=None)
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--ranking-epsilon", type=float, default=0.02)
    return parser.parse_args()


def binary_auc(target: np.ndarray, score: np.ndarray) -> float:
    positive = target >= 0.5
    negative = ~positive
    order = np.argsort(score)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(score) + 1)
    return float(
        (ranks[positive].sum() - positive.sum() * (positive.sum() + 1) / 2.0)
        / (positive.sum() * negative.sum())
    )


def average_precision(target: np.ndarray, score: np.ndarray) -> float:
    target = target >= 0.5
    positives = target.sum()
    order = np.argsort(-score)
    sorted_target = target[order]
    true_positives = np.cumsum(sorted_target)
    precision = true_positives / (np.arange(len(sorted_target)) + 1)
    return float((precision * sorted_target).sum() / positives)


def binary_metrics_at(target: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    target = target >= 0.5
    predicted = score >= threshold
    tp = int((predicted & target).sum())
    tn = int((~predicted & ~target).sum())
    fp = int((predicted & ~target).sum())
    fn = int((~predicted & target).sum())
    total = tp + tn + fp + fn
    precision = np.divide(tp, tp + fp)
    recall = np.divide(tp, tp + fn)
    specificity = np.divide(tn, tn + fp)
    f1 = np.divide(2.0 * precision * recall, precision + recall)
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


def calibration_error(target: np.ndarray, probability: np.ndarray, bins: int):
    target = target >= 0.5
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
    target = target >= 0.5
    count = int(np.ceil(len(target) * fraction))
    order = np.argsort(-score)[:count]
    positives = target.sum()
    true_positives = target[order].sum()
    precision = true_positives / count
    return {
        "fraction": float(fraction),
        "count": count,
        "precision": float(precision),
        "recall": float(true_positives / positives),
        "lift": float(precision / target.mean()),
    }


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict:
    error = prediction - target
    denominator = np.sum((target - target.mean()) ** 2)
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "r2": float(1.0 - np.sum(error * error) / denominator),
        "pearson": float(np.corrcoef(target, prediction)[0, 1]),
        "target_mean": float(target.mean()),
        "prediction_mean": float(prediction.mean()),
    }


def binary_report(target: np.ndarray, score: np.ndarray, bins: int) -> dict:
    threshold_metrics = [binary_metrics_at(target, score, threshold) for threshold in np.unique(score)]
    ece, mce = calibration_error(target, score, bins)
    binary_target = target >= 0.5
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
        "best_accuracy": max(
            threshold_metrics,
            key=lambda item: (item["accuracy"], item["balanced_accuracy"]),
        ),
        "best_f1": max(threshold_metrics, key=lambda item: (item["f1"], item["accuracy"])),
        "top_5pct": top_fraction_metrics(target, score, 0.05),
        "top_10pct": top_fraction_metrics(target, score, 0.10),
        "top_20pct": top_fraction_metrics(target, score, 0.20),
    }


def spearman(predicted: np.ndarray, target: np.ndarray) -> float:
    pred_ranks = np.empty(len(predicted))
    target_ranks = np.empty(len(target))
    pred_ranks[np.argsort(predicted)] = np.arange(len(predicted))
    target_ranks[np.argsort(target)] = np.arange(len(target))
    return float(np.corrcoef(pred_ranks, target_ranks)[0, 1])


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    checkpoint_path = run_dir / args.checkpoint

    device = torch.device(args.device)

    dataset = PursuitRiskBranchCacheDataset(args.branch_cache, split=args.split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = RiskNet(
        hidden_dim=config["hidden_dim"],
        resize_hw=tuple(config["lidar_resize"]),
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    flat = {
        key: []
        for key in (
            "pred_risk",
            "true_risk",
            "p50",
            "y50",
            "p150",
            "y150",
            "pred_severity50",
            "severity50",
            "pred_severity150",
            "severity150",
            "pred_recovery150",
            "recovery150",
            "recovery_valid150",
        )
    }
    hit_top1 = []
    hit_top3 = []
    pred_selected = []
    oracle = []
    random_mean = []
    regrets = []
    normalized_regrets = []
    spearman_values = []
    true_ranges = []
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

            for key, value in (
                ("pred_risk", predicted_risk),
                ("true_risk", true_risk),
                ("p50", p50),
                ("y50", y50),
                ("p150", p150),
                ("y150", y150),
                ("pred_severity50", pred_severity50),
                ("severity50", severity50),
                ("pred_severity150", pred_severity150),
                ("severity150", severity150),
                ("pred_recovery150", pred_recovery150),
                ("recovery150", recovery150),
                ("recovery_valid150", recovery_valid150),
            ):
                flat[key].append(value.reshape(-1))

            for anchor_idx in range(batch_size):
                pred = predicted_risk[anchor_idx]
                true = true_risk[anchor_idx]
                pred_best = np.argmin(pred)
                true_order = np.argsort(true)
                oracle_best = true_order[0]
                top3 = set(true_order[: min(3, candidates)])
                true_range = np.max(true) - np.min(true)
                regret = true[pred_best] - true[oracle_best]
                anchor_pair_correct = 0
                anchor_pair_total = 0
                for left in range(candidates):
                    for right in range(left + 1, candidates):
                        true_diff = true[left] - true[right]
                        if abs(true_diff) <= args.ranking_epsilon:
                            continue
                        pred_diff = pred[left] - pred[right]
                        anchor_pair_total += 1
                        anchor_pair_correct += int(
                            (true_diff > 0.0 and pred_diff > 0.0)
                            or (true_diff < 0.0 and pred_diff < 0.0)
                        )
                pair_correct += anchor_pair_correct
                pair_total += anchor_pair_total
                hit_top1.append(pred_best == oracle_best)
                hit_top3.append(pred_best in top3)
                oracle.append(true[oracle_best])
                pred_selected.append(true[pred_best])
                random_mean.append(np.mean(true))
                regrets.append(regret)
                normalized_regrets.append(regret / true_range)
                spearman_values.append(spearman(pred, true))
                true_ranges.append(true_range)

    flat = {key: np.concatenate(values) for key, values in flat.items()}

    pred_selected = np.asarray(pred_selected)
    oracle = np.asarray(oracle)
    random_mean = np.asarray(random_mean)
    regrets = np.asarray(regrets)
    normalized_regrets = np.asarray(normalized_regrets)
    true_ranges = np.asarray(true_ranges)
    risk_report = regression_metrics(flat["true_risk"], flat["pred_risk"])
    h50_report = binary_report(flat["y50"], flat["p50"], 15)
    h150_report = binary_report(flat["y150"], flat["p150"], 15)

    report = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "branch_cache": args.branch_cache,
        "split": args.split,
        "anchors": len(dataset),
        "candidate_count": len(flat["pred_risk"]) // len(dataset),
        "pairwise_accuracy": float(pair_correct / pair_total),
        "pairwise_pairs": int(pair_total),
        "spearman_mean": float(np.mean(spearman_values)),
        "spearman_median": float(np.median(spearman_values)),
        "top1_hit_rate": float(np.mean(hit_top1)),
        "top3_hit_rate": float(np.mean(hit_top3)),
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
        "severity_50": regression_metrics(flat["severity50"], flat["pred_severity50"]),
        "severity_150": regression_metrics(flat["severity150"], flat["pred_severity150"]),
        "risk_scalar": risk_report,
        "recovery_150": regression_metrics(
            flat["recovery150"][flat["recovery_valid150"]],
            flat["pred_recovery150"][flat["recovery_valid150"]],
        ),
    }

    output_path = Path(args.output) if args.output is not None else run_dir / f"branch_action_metrics_{args.split}.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
