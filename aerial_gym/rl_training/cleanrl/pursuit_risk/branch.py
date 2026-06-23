"""Branch-dataset pure helpers for pursuit observability-risk export."""

from __future__ import annotations

import os
from typing import Dict, Sequence

import numpy as np
import torch

from .contract import label_key


EMPTY_BBOX_XYXY = (-1, -1, -1, -1)


def _candidate_is_diverse(
    candidate: torch.Tensor,
    candidates: Sequence[torch.Tensor],
    min_l2: float,
) -> bool:
    if min_l2 <= 0.0 or not candidates:
        return True
    return all(
        torch.linalg.norm(candidate - existing).detach().cpu().item() >= min_l2
        for existing in candidates
    )


def candidate_actions_from_policy(
    agent,
    obs: torch.Tensor,
    num_candidates: int,
    args,
) -> torch.Tensor:
    min_l2 = args.branch_min_candidate_action_l2
    with torch.no_grad():
        candidates = []
        while len(candidates) < num_candidates:
            sample, _, _, _ = agent.get_action_and_value(obs)
            if _candidate_is_diverse(sample[0], candidates, min_l2):
                candidates.append(sample[0].detach().clone())
    return torch.stack(candidates, dim=0)


def branch_anchor_selection_stats(
    semantic_stats: dict,
    lidar_height: int,
    lidar_width: int,
    args,
) -> dict:
    pixel_count = semantic_stats["pixel_count"]
    bbox = tuple(semantic_stats["bbox_xyxy"])
    if pixel_count <= 0 or bbox == EMPTY_BBOX_XYXY:
        return {
            "score": 0.0,
            "edge_distance_px": -1,
            "edge_score": 0.0,
            "low_pixel_score": 0.0,
            "reason": "not_semantic_visible",
        }

    x0, y0, x1, y1 = bbox
    edge_distance = min(x0, y0, lidar_width - 1 - x1, lidar_height - 1 - y1)
    edge_margin = max(args.branch_edge_margin_px, 1)
    low_pixel_max = max(args.branch_low_pixel_max, args.label_min_visible_pixels)
    edge_score = max(0.0, (edge_margin - edge_distance) / edge_margin)
    low_pixel_score = max(0.0, (low_pixel_max - pixel_count) / low_pixel_max)
    score = edge_score + low_pixel_score
    if edge_score > 0.0 and low_pixel_score > 0.0:
        reason = "edge_and_low_pixel"
    elif edge_score > 0.0:
        reason = "edge"
    elif low_pixel_score > 0.0:
        reason = "low_pixel"
    else:
        reason = "visible_center"
    return {
        "score": score,
        "edge_distance_px": edge_distance,
        "edge_score": edge_score,
        "low_pixel_score": low_pixel_score,
        "reason": reason,
    }


def branch_loss_label(visible_values: Sequence[bool], persist_steps: int) -> tuple:
    visible_np = np.asarray(visible_values, dtype=np.bool_)
    invisible = ~visible_np
    if len(invisible) < persist_steps:
        return False, -1, 0.0
    for start in range(0, len(invisible) - persist_steps + 1):
        if np.all(invisible[start : start + persist_steps]):
            return True, start, np.mean(invisible)
    return False, -1, np.mean(invisible)


def branch_recovery_label(visible_values: Sequence[bool], persist_steps: int) -> tuple:
    visible_np = np.asarray(visible_values, dtype=np.bool_)
    lost, offset, _ = branch_loss_label(visible_np, persist_steps)
    if not lost:
        return False, False
    recovery_start = offset + persist_steps
    recovered = recovery_start < len(visible_np) and np.any(visible_np[recovery_start:])
    return recovered, True


def branch_horizon_label_payload(
    visible_history: Sequence[Sequence[bool]],
    horizons: Sequence[int],
    persist_steps: int,
) -> Dict[str, np.ndarray]:
    payload = {}
    for horizon_steps in horizons:
        losses = []
        severities = []
        valids = []
        recoveries = []
        recovery_valids = []
        for values in visible_history:
            prefix = list(values[:horizon_steps])
            valid = len(prefix) >= horizon_steps
            if valid:
                label, _, severity = branch_loss_label(prefix, persist_steps)
                recovered, recovery_valid = branch_recovery_label(prefix, persist_steps)
            else:
                label, severity = False, 0.0
                recovered, recovery_valid = False, False
            losses.append(label)
            severities.append(severity)
            valids.append(valid)
            recoveries.append(recovered)
            recovery_valids.append(recovery_valid and valid)
        payload[label_key("label_loss_prob", horizon_steps)] = np.asarray(losses, dtype=np.float32)
        payload[label_key("label_loss_severity", horizon_steps)] = np.asarray(severities, dtype=np.float32)
        payload[label_key("label_valid", horizon_steps)] = np.asarray(valids, dtype=np.bool_)
        payload[label_key("label_recovery", horizon_steps)] = np.asarray(recoveries, dtype=np.float32)
        payload[label_key("label_recovery_valid", horizon_steps)] = np.asarray(recovery_valids, dtype=np.bool_)
    return payload


def lidar_stack_from_history(history: Sequence[np.ndarray], stack_frames: int) -> np.ndarray:
    if not history:
        raise ValueError("LiDAR history is empty")
    frames = list(history[-stack_frames:])
    while len(frames) < stack_frames:
        frames.insert(0, frames[0])
    return np.stack(frames, axis=0).astype(np.float32, copy=False)


def write_branch_chunk(path: str, lidar_frames: np.ndarray, schema_version: int):
    np.savez_compressed(
        path,
        schema_version=np.array(schema_version, dtype=np.int32),
        lidar_range_norm=lidar_frames.astype(np.float16, copy=True),
        num_frames=np.array(lidar_frames.shape[0], dtype=np.int32),
        anchor_frame_index=np.array(lidar_frames.shape[0] - 1, dtype=np.int32),
    )
    return os.path.getsize(path)


def build_branch_metadata_payload(
    *,
    schema_version: int,
    anchor_lidar_relpath: str,
    anchor_ego_obs_np: np.ndarray,
    candidate_actions: torch.Tensor,
    horizon_label_payload: dict,
) -> dict:
    return dict(
        schema_version=np.array(schema_version, dtype=np.int32),
        anchor_lidar_relative_path=np.array(anchor_lidar_relpath, dtype="U256"),
        ego_obs=anchor_ego_obs_np,
        candidate_action=candidate_actions.detach().cpu().numpy().astype(np.float32),
        **horizon_label_payload,
    )


def build_branch_index_artifact(
    *,
    schema_version: int,
    transition_schema: str,
    args,
    item,
    dataset_anchor_id: int,
    episode_idx: int,
    step: int,
    num_candidates: int,
    loss_range: float,
    severity_range: float,
    label_has_separation: bool,
    anchor_lidar_bytes: int,
    anchor_lidar_stack: np.ndarray,
    metadata_relpath: str,
    semantic_stats: dict,
    anchor_detectable: bool,
    anchor_visible: bool,
    selection_stats: dict,
) -> dict:
    return {
        "schema_version": schema_version,
        "dataset_kind": "pursuit_lidar_branch_risk_dataset",
        "transition_schema": transition_schema,
        "dataset_anchor_id": dataset_anchor_id,
        "update": item.update,
        "source_episode_idx": episode_idx,
        "source_step": step,
        "branch_candidates": num_candidates,
        "branch_horizon": args.branch_horizon,
        "candidate_loss_range": loss_range,
        "candidate_severity_range": severity_range,
        "candidate_label_has_separation": label_has_separation,
        "label_source": "semantic_target_pixel_count",
        "label_min_visible_pixels": args.label_min_visible_pixels,
        "lidar_bytes": anchor_lidar_bytes,
        "lidar_frames": anchor_lidar_stack.shape[0],
        "metadata_relative_path": metadata_relpath,
        "target_pixel_count": semantic_stats["pixel_count"],
        "anchor_detectable": anchor_detectable,
        "anchor_visible": anchor_visible,
        "anchor_selection_score": selection_stats["score"],
    }
