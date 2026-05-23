from dataclasses import dataclass
import math
from typing import Optional, Sequence, Tuple

import torch


@dataclass(frozen=True)
class DetectionFrustum:
    range_m: float
    horizontal_fov_rad: float
    vertical_fov_rad: float
    min_forward_m: float = 1e-6
    horizontal_fov_min_rad: Optional[float] = None
    horizontal_fov_max_rad: Optional[float] = None
    vertical_fov_min_rad: Optional[float] = None
    vertical_fov_max_rad: Optional[float] = None
    sensor_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    sensor_quaternion: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)


def _quat_from_euler_xyz_tensor(euler_xyz_tensor: torch.Tensor) -> torch.Tensor:
    roll = euler_xyz_tensor[..., 0]
    pitch = euler_xyz_tensor[..., 1]
    yaw = euler_xyz_tensor[..., 2]
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp
    return torch.stack([qx, qy, qz, qw], dim=-1)


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.shape != b.shape:
        raise ValueError("quaternion operands must have the same shape")
    x1, y1, z1, w1 = torch.unbind(a, dim=-1)
    x2, y2, z2, w2 = torch.unbind(b, dim=-1)
    return torch.stack(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ),
        dim=-1,
    )


def _quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(q.shape[0], 1, 3), v.view(q.shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c


def relative_position_body(
    pursuer_position: torch.Tensor,
    pursuer_quaternion: torch.Tensor,
    target_position: torch.Tensor,
) -> torch.Tensor:
    """Return target position in the pursuer body frame."""
    if pursuer_position.shape[-1] != 3 or target_position.shape[-1] != 3:
        raise ValueError("pursuer_position and target_position must end with dimension 3")
    if pursuer_quaternion.shape[-1] != 4:
        raise ValueError("pursuer_quaternion must end with dimension 4")
    if pursuer_position.shape[:-1] != target_position.shape[:-1]:
        raise ValueError("pursuer_position and target_position leading dimensions must match")
    if pursuer_position.shape[:-1] != pursuer_quaternion.shape[:-1]:
        raise ValueError("pursuer_position and pursuer_quaternion leading dimensions must match")

    leading_shape = pursuer_position.shape[:-1]
    rel_world = target_position - pursuer_position
    flat_rel = rel_world.reshape(-1, 3)
    flat_quat = pursuer_quaternion.reshape(-1, 4)
    return _quat_rotate_inverse(flat_quat, flat_rel).reshape(*leading_shape, 3)


def relative_position_sensor(
    relative_body_position: torch.Tensor,
    frustum: DetectionFrustum,
) -> torch.Tensor:
    """Return body-frame relative target positions in the configured sensor frame."""
    if relative_body_position.shape[-1] != 3:
        raise ValueError("relative_body_position must end with dimension 3")

    leading_shape = relative_body_position.shape[:-1]
    flat_rel = relative_body_position.reshape(-1, 3)
    sensor_position = torch.tensor(
        frustum.sensor_position,
        dtype=flat_rel.dtype,
        device=flat_rel.device,
    ).view(1, 3)
    sensor_quat = torch.tensor(
        frustum.sensor_quaternion,
        dtype=flat_rel.dtype,
        device=flat_rel.device,
    ).view(1, 4).expand(flat_rel.shape[0], -1)
    flat_sensor = _quat_rotate_inverse(sensor_quat, flat_rel - sensor_position)
    return flat_sensor.reshape(*leading_shape, 3)


def detection_frustum_from_lidar_config(lidar_cfg, range_m: float = 150.0) -> DetectionFrustum:
    """Build the shared reward/export detection frustum from a LiDAR config class."""
    horizontal_min = math.radians(float(lidar_cfg.horizontal_fov_deg_min))
    horizontal_max = math.radians(float(lidar_cfg.horizontal_fov_deg_max))
    vertical_min = math.radians(float(lidar_cfg.vertical_fov_deg_min))
    vertical_max = math.radians(float(lidar_cfg.vertical_fov_deg_max))
    min_forward = max(float(getattr(lidar_cfg, "min_range", 1e-6)), 1e-6)

    nominal_position = getattr(lidar_cfg, "nominal_position", [0.0, 0.0, 0.0])
    min_translation = torch.tensor(
        getattr(lidar_cfg, "min_translation", nominal_position),
        dtype=torch.float32,
    )
    max_translation = torch.tensor(
        getattr(lidar_cfg, "max_translation", nominal_position),
        dtype=torch.float32,
    )
    sensor_position = 0.5 * (min_translation + max_translation)

    min_rotation = torch.deg2rad(
        torch.tensor(
            getattr(lidar_cfg, "min_euler_rotation_deg", [0.0, 0.0, 0.0]),
            dtype=torch.float32,
        )
    )
    max_rotation = torch.deg2rad(
        torch.tensor(
            getattr(lidar_cfg, "max_euler_rotation_deg", [0.0, 0.0, 0.0]),
            dtype=torch.float32,
        )
    )
    sensor_local_quat = _quat_from_euler_xyz_tensor(0.5 * (min_rotation + max_rotation))
    sensor_frame_quat = _quat_from_euler_xyz_tensor(
        torch.deg2rad(
            torch.tensor(
                getattr(lidar_cfg, "euler_frame_rot_deg", [0.0, 0.0, 0.0]),
                dtype=torch.float32,
            )
        )
    )
    sensor_quat = _quat_mul(sensor_local_quat.view(1, 4), sensor_frame_quat.view(1, 4))[0]
    return DetectionFrustum(
        range_m=float(range_m),
        horizontal_fov_rad=horizontal_max - horizontal_min,
        vertical_fov_rad=vertical_max - vertical_min,
        min_forward_m=min_forward,
        horizontal_fov_min_rad=horizontal_min,
        horizontal_fov_max_rad=horizontal_max,
        vertical_fov_min_rad=vertical_min,
        vertical_fov_max_rad=vertical_max,
        sensor_position=tuple(float(value) for value in sensor_position.tolist()),
        sensor_quaternion=tuple(float(value) for value in sensor_quat.tolist()),
    )


def _frustum_bounds(frustum: DetectionFrustum):
    horizontal_min = (
        -0.5 * float(frustum.horizontal_fov_rad)
        if frustum.horizontal_fov_min_rad is None
        else float(frustum.horizontal_fov_min_rad)
    )
    horizontal_max = (
        0.5 * float(frustum.horizontal_fov_rad)
        if frustum.horizontal_fov_max_rad is None
        else float(frustum.horizontal_fov_max_rad)
    )
    vertical_min = (
        -0.5 * float(frustum.vertical_fov_rad)
        if frustum.vertical_fov_min_rad is None
        else float(frustum.vertical_fov_min_rad)
    )
    vertical_max = (
        0.5 * float(frustum.vertical_fov_rad)
        if frustum.vertical_fov_max_rad is None
        else float(frustum.vertical_fov_max_rad)
    )
    return horizontal_min, horizontal_max, vertical_min, vertical_max


def target_in_detection_frustum(
    relative_body_position: torch.Tensor,
    frustum: DetectionFrustum,
) -> torch.Tensor:
    """Check if a body-frame target position is detectable by the configured sensor."""
    if relative_body_position.shape[-1] != 3:
        raise ValueError("relative_body_position must end with dimension 3")
    if frustum.range_m <= 0.0:
        raise ValueError("frustum.range_m must be positive")
    if frustum.horizontal_fov_rad <= 0.0 or frustum.vertical_fov_rad <= 0.0:
        raise ValueError("frustum FOV values must be positive")
    horizontal_min, horizontal_max, vertical_min, vertical_max = _frustum_bounds(frustum)
    if horizontal_min >= horizontal_max or vertical_min >= vertical_max:
        raise ValueError("frustum min FOV bounds must be smaller than max bounds")

    rel = relative_position_sensor(relative_body_position, frustum)
    forward = rel[..., 0]
    lateral = rel[..., 1]
    vertical = rel[..., 2]
    distance = torch.linalg.norm(rel, dim=-1)
    horizontal_angle = torch.atan2(lateral, torch.clamp(forward, min=frustum.min_forward_m))
    vertical_angle = torch.atan2(
        vertical,
        torch.clamp(torch.linalg.norm(rel[..., 0:2], dim=-1), min=frustum.min_forward_m),
    )

    return (
        (forward > frustum.min_forward_m)
        & (distance <= frustum.range_m)
        & (horizontal_angle >= horizontal_min)
        & (horizontal_angle <= horizontal_max)
        & (vertical_angle >= vertical_min)
        & (vertical_angle <= vertical_max)
    )

def target_in_detection_frustum_from_poses(
    pursuer_position: torch.Tensor,
    pursuer_quaternion: torch.Tensor,
    target_position: torch.Tensor,
    frustum: DetectionFrustum,
) -> torch.Tensor:
    rel_body = relative_position_body(pursuer_position, pursuer_quaternion, target_position)
    return target_in_detection_frustum(rel_body, frustum)


def persistent_loss_starts(detectable: torch.Tensor, persist_steps: int) -> torch.Tensor:
    """Return time/env positions where loss starts and persists for persist_steps."""
    if detectable.dtype != torch.bool:
        raise ValueError("detectable must be a bool tensor")
    if detectable.ndim != 2:
        raise ValueError("detectable must have shape [time, num_envs]")
    if persist_steps <= 0:
        raise ValueError("persist_steps must be positive")

    time_steps, num_envs = detectable.shape
    starts = torch.zeros((time_steps, num_envs), dtype=torch.bool, device=detectable.device)
    if time_steps < persist_steps:
        return starts

    lost = (~detectable).to(torch.int32)
    cumsum = torch.zeros((time_steps + 1, num_envs), dtype=torch.int32, device=detectable.device)
    cumsum[1:] = torch.cumsum(lost, dim=0)
    window_lost_count = cumsum[persist_steps:] - cumsum[:-persist_steps]
    starts[: time_steps - persist_steps + 1] = window_lost_count == persist_steps
    return starts


def observability_loss_labels(
    detectable: torch.Tensor,
    horizon_steps: int = 150,
    persist_steps: int = 10,
) -> torch.Tensor:
    """Label each time/env pair if persistent target loss starts within the horizon."""
    event_step = first_observability_loss_offset(detectable, horizon_steps, persist_steps)
    return event_step >= 0


def first_observability_loss_offset(
    detectable: torch.Tensor,
    horizon_steps: int = 150,
    persist_steps: int = 10,
) -> torch.Tensor:
    """Return first persistent-loss offset, or -1 when no event is in the horizon."""
    if horizon_steps <= 0:
        raise ValueError("horizon_steps must be positive")

    starts = persistent_loss_starts(detectable, persist_steps)
    time_steps, num_envs = starts.shape
    offsets = torch.full((time_steps, num_envs), -1, dtype=torch.long, device=starts.device)
    sentinel = torch.full((num_envs,), horizon_steps + 1, dtype=torch.long, device=starts.device)

    for t in range(time_steps):
        end = min(time_steps, t + horizon_steps)
        window = starts[t:end]
        if not torch.any(window):
            continue
        relative_indices = torch.arange(
            end - t, dtype=torch.long, device=starts.device
        ).unsqueeze(1)
        first = torch.where(window, relative_indices.expand(-1, num_envs), sentinel).min(dim=0).values
        has_event = first <= horizon_steps
        offsets[t, has_event] = first[has_event]

    return offsets


def observability_loss_labels_from_poses(
    pursuer_position: torch.Tensor,
    pursuer_quaternion: torch.Tensor,
    target_position: torch.Tensor,
    frustum: DetectionFrustum,
    horizon_steps: int = 150,
    persist_steps: int = 10,
) -> torch.Tensor:
    detectable = target_in_detection_frustum_from_poses(
        pursuer_position, pursuer_quaternion, target_position, frustum
    )
    return observability_loss_labels(detectable, horizon_steps, persist_steps)


def normalize_visibility_window(persist_steps: int, horizon_steps: int) -> Tuple[int, int]:
    persist_steps = max(int(persist_steps), 1)
    horizon_steps = max(int(horizon_steps), persist_steps + 1)
    return persist_steps, horizon_steps


def visibility_loss_penalty_from_steps(
    loss_steps: torch.Tensor,
    invisible: torch.Tensor,
    persist_steps: int,
    horizon_steps: int,
) -> torch.Tensor:
    persist_steps, horizon_steps = normalize_visibility_window(persist_steps, horizon_steps)
    horizon_span = max(float(horizon_steps - persist_steps), 1.0)
    loss_steps_f = loss_steps.float()
    persistent_loss = torch.clamp(
        (loss_steps_f - float(persist_steps)) / horizon_span,
        min=0.0,
        max=1.0,
    )
    over_horizon_loss = torch.clamp(
        (loss_steps_f - float(horizon_steps)) / float(horizon_steps),
        min=0.0,
        max=1.0,
    )
    return torch.where(
        invisible,
        1.0 + 2.0 * persistent_loss + 4.0 * over_horizon_loss,
        torch.zeros_like(loss_steps_f),
    )


def visibility_recovery_stats(
    detectable: Sequence[bool],
    persist_steps: int,
    horizon_steps: int,
) -> dict:
    """Summarize the same invisible-segment definition used by visibility reward logs."""
    if isinstance(detectable, torch.Tensor):
        detectable_values = detectable.detach().cpu().to(torch.bool).flatten().tolist()
    else:
        detectable_values = [bool(value) for value in detectable]
    if not detectable_values:
        return {
            "final_detectable": None,
            "max_consecutive_invisible_steps": None,
            "num_invisible_segments": 0,
            "mean_invisible_segment_steps": None,
            "recovered_invisible_segments": 0,
            "recovery_rate_within_persist_steps": None,
            "recovery_rate_within_horizon_steps": None,
            "mean_recovery_steps": None,
            "max_recovery_steps": None,
            "long_invisible_segments": 0,
            "over_horizon_invisible_segments": 0,
            "detectable_rate_first_half": None,
            "detectable_rate_second_half": None,
        }

    segment_lengths = []
    recovered_lengths = []
    current = 0
    for is_detectable in detectable_values:
        if not is_detectable:
            current += 1
        elif current > 0:
            segment_lengths.append(current)
            recovered_lengths.append(current)
            current = 0
    if current > 0:
        segment_lengths.append(current)

    persist_steps, horizon_steps = normalize_visibility_window(persist_steps, horizon_steps)
    num_segments = len(segment_lengths)
    if num_segments:
        recovered_within_persist = sum(1 for length in recovered_lengths if length <= persist_steps)
        recovered_within_horizon = sum(1 for length in recovered_lengths if length <= horizon_steps)
        mean_segment_steps = sum(segment_lengths) / float(num_segments)
        max_segment_steps = max(segment_lengths)
        long_segments = sum(1 for length in segment_lengths if length >= persist_steps)
        over_horizon_segments = sum(1 for length in segment_lengths if length > horizon_steps)
        recovery_rate = recovered_within_persist / float(num_segments)
        horizon_recovery_rate = recovered_within_horizon / float(num_segments)
    else:
        mean_segment_steps = 0.0
        max_segment_steps = 0
        long_segments = 0
        over_horizon_segments = 0
        recovery_rate = 1.0
        horizon_recovery_rate = 1.0

    midpoint = int(math.ceil(len(detectable_values) / 2.0))
    first_half = detectable_values[:midpoint]
    second_half = detectable_values[midpoint:]
    return {
        "final_detectable": bool(detectable_values[-1]),
        "max_consecutive_invisible_steps": int(max_segment_steps),
        "num_invisible_segments": int(num_segments),
        "mean_invisible_segment_steps": float(mean_segment_steps),
        "recovered_invisible_segments": int(len(recovered_lengths)),
        "recovery_rate_within_persist_steps": float(recovery_rate),
        "recovery_rate_within_horizon_steps": float(horizon_recovery_rate),
        "mean_recovery_steps": (
            sum(recovered_lengths) / float(len(recovered_lengths))
            if recovered_lengths
            else None
        ),
        "max_recovery_steps": int(max(recovered_lengths)) if recovered_lengths else None,
        "long_invisible_segments": int(long_segments),
        "over_horizon_invisible_segments": int(over_horizon_segments),
        "detectable_rate_first_half": (
            sum(first_half) / float(len(first_half)) if first_half else None
        ),
        "detectable_rate_second_half": (
            sum(second_half) / float(len(second_half)) if second_half else None
        ),
    }
