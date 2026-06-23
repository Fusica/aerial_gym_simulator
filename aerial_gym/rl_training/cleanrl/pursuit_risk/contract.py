"""Shared contract for pursuit LiDAR observability-risk data and models."""

from __future__ import annotations

import numpy as np


STATE_DIM = 19
ACTION_DIM = 4
DEFAULT_LIDAR_STACK_FRAMES = 3
PRIMARY_RISK_HORIZONS = (50, 150)
DEFAULT_RISK_HORIZONS = (50, 150, 300)

RISK_SCALAR_WEIGHTS = {
    "loss_prob_50": 0.40,
    "loss_prob_150": 0.15,
    "loss_severity_50": 0.35,
    "loss_severity_150": 0.10,
}


TARGET_SPECS = (
    ("loss_prob_50", "label_loss_prob_h050", np.float32),
    ("loss_prob_150", "label_loss_prob_h150", np.float32),
    ("loss_severity_50", "label_loss_severity_h050", np.float32),
    ("loss_severity_150", "label_loss_severity_h150", np.float32),
    ("valid_50", "label_valid_h050", np.bool_),
    ("valid_150", "label_valid_h150", np.bool_),
    ("recovery_150", "label_recovery_h150", np.float32),
    ("recovery_valid_150", "label_recovery_valid_h150", np.bool_),
)

def label_key(prefix: str, horizon_steps: int) -> str:
    return f"{prefix}_h{horizon_steps:03d}"
