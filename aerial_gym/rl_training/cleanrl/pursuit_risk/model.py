"""CNN+GRU action-conditioned observability risk model.

This module is intentionally independent from Isaac Gym runtime code.  It is
used for offline risk pretraining/fine-tuning and later loaded as a frozen
module by PPO.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .contract import ACTION_DIM, RISK_SCALAR_WEIGHTS, STATE_DIM


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2),
            nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LidarCnnGruBackbone(nn.Module):
    """Encode a K-frame LiDAR range-image stack into a 64D latent."""

    def __init__(
        self,
        latent_dim: int = 64,
        frame_feature_dim: int = 128,
        gru_hidden_dim: int = 128,
        resize_hw: Tuple[int, int] = (128, 384),
    ):
        super().__init__()
        self.resize_hw = resize_hw
        self.frame_encoder = nn.Sequential(
            ConvBlock(1, 16, stride=2),
            ConvBlock(16, 32, stride=2),
            ConvBlock(32, 64, stride=2),
            ConvBlock(64, 96, stride=2),
            ConvBlock(96, frame_feature_dim, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.temporal = nn.GRU(
            input_size=frame_feature_dim,
            hidden_size=gru_hidden_dim,
            batch_first=True,
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(gru_hidden_dim),
            nn.Linear(gru_hidden_dim, latent_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, lidar_stack: torch.Tensor) -> torch.Tensor:
        batch, frames, height, width = lidar_stack.shape
        x = lidar_stack.reshape(batch * frames, 1, height, width).float()
        if (height, width) != self.resize_hw:
            x = F.interpolate(x, size=self.resize_hw, mode="bilinear", align_corners=False)
        feat = self.frame_encoder(x).reshape(batch, frames, -1)
        gru_out, _ = self.temporal(feat)
        return self.proj(gru_out[:, -1])


class RiskNet(nn.Module):
    """Single end-to-end risk network.

    `s_red` is not encoded by the LiDAR CNN.  It is fused after the LiDAR
    backbone together with the candidate action and simple action transforms.
    """

    def __init__(
        self,
        lidar_latent_dim: int = 64,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        hidden_dim: int = 256,
        resize_hw: Tuple[int, int] = (128, 384),
    ):
        super().__init__()
        self.action_dim = action_dim
        self.lidar_backbone = LidarCnnGruBackbone(
            latent_dim=lidar_latent_dim,
            resize_hw=resize_hw,
        )
        action_feature_dim = action_dim * 4
        fusion_dim = lidar_latent_dim + state_dim + action_feature_dim
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.p_loss_50 = nn.Linear(hidden_dim, 1)
        self.p_loss_150 = nn.Linear(hidden_dim, 1)
        self.severity_50 = nn.Linear(hidden_dim, 1)
        self.severity_150 = nn.Linear(hidden_dim, 1)
        self.p_recover_150 = nn.Linear(hidden_dim, 1)

    def action_features(self, s_red: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        prev_action = s_red[:, -self.action_dim :]
        return torch.cat(
            (
                action,
                action - prev_action,
                torch.abs(action),
                action * action,
            ),
            dim=-1,
        )

    def forward(self, lidar_stack: torch.Tensor, s_red: torch.Tensor, action: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_lidar = self.lidar_backbone(lidar_stack)
        action_feat = self.action_features(s_red.float(), action.float())
        fused = self.fusion(torch.cat((z_lidar, s_red.float(), action_feat), dim=-1))
        return {
            "p_loss_50_logit": self.p_loss_50(fused).squeeze(-1),
            "p_loss_150_logit": self.p_loss_150(fused).squeeze(-1),
            "severity_50": torch.sigmoid(self.severity_50(fused).squeeze(-1)),
            "severity_150": torch.sigmoid(self.severity_150(fused).squeeze(-1)),
            "p_recover_150_logit": self.p_recover_150(fused).squeeze(-1),
        }


def risk_scalar_from_outputs(outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    return (
        RISK_SCALAR_WEIGHTS["loss_prob_50"] * torch.sigmoid(outputs["p_loss_50_logit"])
        + RISK_SCALAR_WEIGHTS["loss_prob_150"] * torch.sigmoid(outputs["p_loss_150_logit"])
        + RISK_SCALAR_WEIGHTS["loss_severity_50"] * outputs["severity_50"]
        + RISK_SCALAR_WEIGHTS["loss_severity_150"] * outputs["severity_150"]
    )


def risk_scalar_from_targets(targets: Dict[str, torch.Tensor]) -> torch.Tensor:
    return (
        RISK_SCALAR_WEIGHTS["loss_prob_50"] * targets["loss_prob_50"]
        + RISK_SCALAR_WEIGHTS["loss_prob_150"] * targets["loss_prob_150"]
        + RISK_SCALAR_WEIGHTS["loss_severity_50"] * targets["loss_severity_50"]
        + RISK_SCALAR_WEIGHTS["loss_severity_150"] * targets["loss_severity_150"]
    )


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not torch.any(mask):
        return values.new_tensor(0.0)
    return values[mask].mean()


def _masked_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    values = F.binary_cross_entropy_with_logits(logits, target.float(), reduction="none")
    return _masked_mean(values, mask)


def _masked_huber(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    delta: float = 0.1,
) -> torch.Tensor:
    values = F.huber_loss(pred, target.float(), reduction="none", delta=delta)
    return _masked_mean(values, mask)


def pairwise_ranking_loss(
    pred_rho: torch.Tensor,
    target_rho: torch.Tensor,
    group_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    epsilon: float = 0.02,
) -> torch.Tensor:
    losses = []
    for group_id in torch.unique(group_ids.detach()):
        mask = (group_ids == group_id) & valid_mask.bool()
        if mask.sum().item() < 2:
            continue
        pred = pred_rho[mask]
        target = target_rho[mask]
        diff_target = target[:, None] - target[None, :]
        pair_mask = torch.abs(diff_target) > epsilon
        upper = torch.triu(torch.ones_like(pair_mask, dtype=torch.bool), diagonal=1)
        pair_mask &= upper
        if not torch.any(pair_mask):
            continue
        diff_pred = pred[:, None] - pred[None, :]
        sign = torch.sign(diff_target[pair_mask])
        losses.append(F.softplus(-sign * diff_pred[pair_mask]).mean())
    if not losses:
        return pred_rho.new_tensor(0.0)
    return torch.stack(losses).mean()


def risk_loss(
    outputs: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    weights,
    group_ids: Optional[torch.Tensor] = None,
    ranking_epsilon: float = 0.02,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    valid_50 = targets["valid_50"].bool()
    valid_150 = targets["valid_150"].bool()
    recovery_valid_150 = targets["recovery_valid_150"].bool() & valid_150

    terms = {
        "p_loss_50": weights.p_loss_50
        * _masked_bce_with_logits(outputs["p_loss_50_logit"], targets["loss_prob_50"], valid_50),
        "p_loss_150": weights.p_loss_150
        * _masked_bce_with_logits(outputs["p_loss_150_logit"], targets["loss_prob_150"], valid_150),
        "severity_50": weights.severity_50
        * _masked_huber(outputs["severity_50"], targets["loss_severity_50"], valid_50),
        "severity_150": weights.severity_150
        * _masked_huber(outputs["severity_150"], targets["loss_severity_150"], valid_150),
        "p_recover_150": weights.p_recover_150
        * _masked_bce_with_logits(
            outputs["p_recover_150_logit"],
            targets["recovery_150"],
            recovery_valid_150,
        ),
    }
    if group_ids is not None and weights.ranking > 0.0:
        ranking = pairwise_ranking_loss(
            risk_scalar_from_outputs(outputs),
            risk_scalar_from_targets(targets),
            group_ids=group_ids,
            valid_mask=valid_50 & valid_150,
            epsilon=ranking_epsilon,
        )
        terms["ranking"] = weights.ranking * ranking
    else:
        terms["ranking"] = outputs["p_loss_50_logit"].new_tensor(0.0)

    total = torch.stack([value for value in terms.values()]).sum()
    metrics = {name: value.detach().cpu().item() for name, value in terms.items()}
    metrics["loss_total"] = total.detach().cpu().item()
    metrics["valid_50_frac"] = valid_50.float().mean().detach().cpu().item()
    metrics["valid_150_frac"] = valid_150.float().mean().detach().cpu().item()
    return total, metrics
