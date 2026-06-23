"""Train pursuit LiDAR observability risk model.

Two stages are supported with one model structure:

1. behavior_pretrain: cached schema7 behavior data, using behavior_action as u.
2. branch_finetune: same-anchor candidate-action data, with pairwise ranking.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .dataset import (
    PursuitRiskBehaviorDataset,
    PursuitRiskBranchDataset,
    flatten_branch_batch,
    move_batch_to_device,
    split_branch_indices_by_source,
)
from .model import (
    RiskNet,
    risk_loss,
    risk_scalar_from_outputs,
    risk_scalar_from_targets,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("behavior_pretrain", "branch_finetune"), required=True)
    parser.add_argument("--behavior-cache", default=None)
    parser.add_argument("--branch-dataset", default="runs/risk_dataset/D0_branch_full_v1_10ckpt_500ep_h150_k3_m16")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--pin-memory", dest="pin_memory", action="store_true")
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.add_argument("--persistent-workers", dest="persistent_workers", action="store_true")
    parser.add_argument("--no-persistent-workers", dest="persistent_workers", action="store_false")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--cache-items", type=int, default=16)
    parser.add_argument("--no-train-shuffle", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr-mult", type=float, default=0.25)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--lidar-resize", type=int, nargs=2, default=(128, 384), metavar=("H", "W"))
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=0)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--max-val-steps", type=int, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--ranking-epsilon", type=float, default=0.02)

    parser.add_argument("--loss-weight-p-loss-50", dest="p_loss_50", type=float, default=1.0)
    parser.add_argument("--loss-weight-p-loss-150", dest="p_loss_150", type=float, default=0.7)
    parser.add_argument("--loss-weight-severity-50", dest="severity_50", type=float, default=1.0)
    parser.add_argument("--loss-weight-severity-150", dest="severity_150", type=float, default=0.7)
    parser.add_argument("--loss-weight-p-recover-150", dest="p_recover_150", type=float, default=0.3)
    parser.add_argument("--loss-weight-ranking", dest="ranking", type=float, default=0.5)

    parser.add_argument("--log-interval", type=int, default=20)
    parser.set_defaults(pin_memory=True, persistent_workers=True)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(model: RiskNet, args) -> torch.optim.Optimizer:
    backbone_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("lidar_backbone."):
            backbone_params.append(param)
        else:
            other_params.append(param)
    param_groups = []
    if other_params:
        param_groups.append({"params": other_params, "lr": args.lr})
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": args.lr * args.backbone_lr_mult})
    if not param_groups:
        raise RuntimeError("optimizer has no trainable parameters")
    return torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)


def build_datasets(args):
    if args.stage == "behavior_pretrain":
        if args.behavior_cache is None:
            raise ValueError(
                "--stage behavior_pretrain requires --behavior-cache. "
                "Run prepare_cache.py first to build the cache."
        )
        train_dataset = PursuitRiskBehaviorDataset(args.behavior_cache, split="train")
        val_dataset = PursuitRiskBehaviorDataset(args.behavior_cache, split="val")
        return train_dataset, val_dataset
    else:
        dataset = PursuitRiskBranchDataset(
            args.branch_dataset,
            cache_items=args.cache_items,
        )
        train_indices, val_indices = split_branch_indices_by_source(
            dataset,
            args.val_fraction,
            args.seed,
        )
    if args.no_train_shuffle:
        train_indices = sorted(train_indices)
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def prepare_batch(batch: Dict, stage: str, device: torch.device):
    batch = move_batch_to_device(batch, device)
    if stage == "branch_finetune":
        return flatten_branch_batch(batch)
    return batch["lidar"], batch["s_red"], batch["action"], batch["targets"], None


def _binary_auc(target: np.ndarray, score: np.ndarray) -> Optional[float]:
    target = target.astype(np.int64)
    pos = target == 1
    neg = target == 0
    if not pos.any() or not neg.any():
        return None
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    pos_ranks = ranks[pos].sum()
    auc = (pos_ranks - pos.sum() * (pos.sum() + 1) / 2.0) / (pos.sum() * neg.sum())
    return float(auc)


def update_metric_sum(total: Dict[str, float], count: Dict[str, int], metrics: Dict[str, float]):
    for key, value in metrics.items():
        total[key] = total.get(key, 0.0) + value
        count[key] = count.get(key, 0) + 1


def average_metrics(total: Dict[str, float], count: Dict[str, int], prefix: str) -> Dict[str, float]:
    return {f"{prefix}/{key}": total[key] / max(count[key], 1) for key in sorted(total)}


def pairwise_accuracy(
    pred_rho: torch.Tensor,
    target_rho: torch.Tensor,
    group_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    epsilon: float,
) -> Tuple[int, int]:
    correct = 0
    total = 0
    for group in torch.unique(group_ids):
        mask = (group_ids == group) & valid_mask
        if mask.sum().item() < 2:
            continue
        pred = pred_rho[mask]
        target = target_rho[mask]
        for i in range(len(pred)):
            for j in range(i + 1, len(pred)):
                diff = float(target[i] - target[j])
                if abs(diff) <= epsilon:
                    continue
                total += 1
                pred_diff = float(pred[i] - pred[j])
                correct += (diff > 0.0 and pred_diff > 0.0) or (diff < 0.0 and pred_diff < 0.0)
    return correct, total


def train_one_epoch(model, loader, optimizer, scaler, args, device, epoch: int) -> Dict[str, float]:
    model.train()
    totals: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    use_amp = args.amp and device.type == "cuda"
    print(f"epoch {epoch}: training {args.stage}; metrics show supervised risk and ranking fit")
    data_time_total = 0.0
    step_time_total = 0.0
    interval_count = 0
    loop_end = time.perf_counter()
    for step, batch in enumerate(loader):
        data_time = time.perf_counter() - loop_end
        step_start = time.perf_counter()
        if args.max_train_steps is not None and step >= args.max_train_steps:
            break
        lidar, s_red, action, targets, group_ids = prepare_batch(batch, args.stage, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(lidar, s_red, action)
            loss, metrics = risk_loss(
                outputs,
                targets,
                weights=args,
                group_ids=group_ids,
                ranking_epsilon=args.ranking_epsilon,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        update_metric_sum(totals, counts, metrics)
        step_time = time.perf_counter() - step_start
        data_time_total += data_time
        step_time_total += step_time
        interval_count += 1
        if step % args.log_interval == 0:
            denom = max(interval_count, 1)
            print(
                f"epoch={epoch} step={step} loss={metrics['loss_total']:.4f} "
                f"data_time={data_time_total / denom:.3f}s "
                f"step_time={step_time_total / denom:.3f}s"
            )
            data_time_total = 0.0
            step_time_total = 0.0
            interval_count = 0
        loop_end = time.perf_counter()
    return average_metrics(totals, counts, "train")


def evaluate(model, loader, args, device: torch.device, max_steps: Optional[int]) -> Dict[str, float]:
    model.eval()
    totals: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    p_targets = {50: [], 150: []}
    p_scores = {50: [], 150: []}
    severity_errors = {50: [], 150: []}
    ranking_correct = 0
    ranking_total = 0
    with torch.inference_mode():
        for step, batch in enumerate(loader):
            if max_steps is not None and step >= max_steps:
                break
            lidar, s_red, action, targets, group_ids = prepare_batch(batch, args.stage, device)
            outputs = model(lidar, s_red, action)
            _loss, metrics = risk_loss(
                outputs,
                targets,
                weights=args,
                group_ids=group_ids,
                ranking_epsilon=args.ranking_epsilon,
            )
            update_metric_sum(totals, counts, metrics)
            for horizon in (50, 150):
                valid = targets[f"valid_{horizon}"].cpu().bool()
                if not valid.any():
                    continue
                p_targets[horizon].append(targets[f"loss_prob_{horizon}"].cpu()[valid].numpy())
                p_scores[horizon].append(
                    torch.sigmoid(outputs[f"p_loss_{horizon}_logit"]).cpu()[valid].numpy()
                )
                severity_errors[horizon].append(
                    torch.abs(
                        outputs[f"severity_{horizon}"].cpu()[valid]
                        - targets[f"loss_severity_{horizon}"].cpu()[valid]
                    ).numpy()
                )
            if group_ids is not None:
                pred_rho = risk_scalar_from_outputs(outputs).cpu()
                target_rho = risk_scalar_from_targets({k: v.cpu() for k, v in targets.items()})
                valid = (targets["valid_50"] & targets["valid_150"]).cpu().bool()
                groups = group_ids.cpu()
                correct, total = pairwise_accuracy(pred_rho, target_rho, groups, valid, args.ranking_epsilon)
                ranking_correct += correct
                ranking_total += total
    metrics = average_metrics(totals, counts, "val")
    for horizon in (50, 150):
        if p_targets[horizon]:
            target = np.concatenate(p_targets[horizon]).astype(np.float64)
            score = np.concatenate(p_scores[horizon]).astype(np.float64)
            prefix = f"val/p_loss_{horizon}"
            metrics[f"{prefix}_rate"] = float(target.mean())
            metrics[f"{prefix}_acc"] = float(np.mean((score >= 0.5) == (target >= 0.5)))
            metrics[f"{prefix}_brier"] = float(np.mean((score - target) ** 2))
            auc = _binary_auc(target, score)
            if auc is not None:
                metrics[f"{prefix}_auroc"] = auc
        if severity_errors[horizon]:
            metrics[f"val/severity_{horizon}_mae"] = float(np.concatenate(severity_errors[horizon]).mean())
    if ranking_total > 0:
        metrics["val/ranking_pair_accuracy"] = float(ranking_correct / ranking_total)
        metrics["val/ranking_pairs"] = float(ranking_total)
    return metrics


def save_checkpoint(path: Path, model: RiskNet, optimizer, args, epoch: int, metrics: Dict[str, float]):
    args_payload = vars(args).copy()
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "args": args_payload,
        "model_type": "RiskNet_CNN_GRU_v2",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, sort_keys=True)

    device = torch.device(args.device)

    train_dataset, val_dataset = build_datasets(args)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=not args.no_train_shuffle,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor
    )

    model = RiskNet(hidden_dim=args.hidden_dim, resize_hw=tuple(args.lidar_resize)).to(device)
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint)
        print(f"loaded checkpoint {args.resume} epoch={checkpoint.get('epoch') if isinstance(checkpoint, dict) else None}")
    freeze_backbone = args.stage == "branch_finetune" and args.freeze_backbone_epochs > 0
    for param in model.lidar_backbone.parameters():
        param.requires_grad = not freeze_backbone
    optimizer = build_optimizer(model, args)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    best_score = math.inf
    history_path = output_dir / "metrics_history.jsonl"
    if history_path.exists():
        history_path.unlink()
    for epoch in range(args.epochs):
        if freeze_backbone and epoch == args.freeze_backbone_epochs:
            print("unfreezing LiDAR backbone for local end-to-end fine-tuning")
            for param in model.lidar_backbone.parameters():
                param.requires_grad = True
            optimizer = build_optimizer(model, args)
            freeze_backbone = False
        train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, args, device, epoch)
        val_metrics = evaluate(model, val_loader, args, device, args.max_val_steps)
        metrics = {"epoch": epoch, **train_metrics, **val_metrics}
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, sort_keys=True) + "\n")
        save_checkpoint(output_dir / "checkpoint_last.pt", model, optimizer, args, epoch, metrics)
        val_loss = metrics["val/loss_total"]
        if val_loss < best_score:
            best_score = float(val_loss)
            save_checkpoint(output_dir / "checkpoint_best.pt", model, optimizer, args, epoch, metrics)
        print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
