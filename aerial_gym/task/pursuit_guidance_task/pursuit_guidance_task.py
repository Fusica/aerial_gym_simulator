import torch
import numpy as np
import gymnasium as gym
from gym.spaces import Box, Dict
import pytorch3d.transforms as p3d_transforms

from aerial_gym.sim.sim_builder import SimBuilder
from aerial_gym.task.base_task import BaseTask
from aerial_gym.utils.math import (
    compute_vee_map,
    quat_axis,
    quat_from_euler_xyz_tensor,
    quat_rotate_inverse,
    quat_to_rotation_matrix,
)


def _as_tensor(value, device):
    return torch.tensor(value, dtype=torch.float32, device=device)


class PursuitGuidanceTask(BaseTask):
    def __init__(
        self, task_config, seed=None, num_envs=None, headless=None, device=None, use_warp=None
    ):
        if seed is not None:
            task_config.seed = seed
        if num_envs is not None:
            task_config.num_envs = num_envs
        if headless is not None:
            task_config.headless = headless
        if device is not None:
            task_config.device = device
        if use_warp is not None:
            task_config.use_warp = use_warp

        super().__init__(task_config)
        self.device = self.task_config.device
        self.sim_env = SimBuilder().build_env(
            sim_name=self.task_config.sim_name,
            env_name=self.task_config.env_name,
            robot_name=self.task_config.robot_name,
            controller_name=self.task_config.controller_name,
            args=self.task_config.args,
            device=self.device,
            num_envs=self.task_config.num_envs,
            use_warp=self.task_config.use_warp,
            headless=self.task_config.headless,
        )

        self.num_envs = self.sim_env.num_envs
        self.obs_dict = self.sim_env.get_obs()
        self.robot_state = self.obs_dict["robot_state_tensor"]
        self.target_state = self.obs_dict["env_asset_state_tensor"][:, 0, :]
        self.target_force_tensor = self.obs_dict["obstacle_force_tensor"]
        self.target_torque_tensor = self.obs_dict["obstacle_torque_tensor"]
        self.dt = float(self.obs_dict["dt"])

        self.actions = torch.zeros(
            (self.num_envs, self.task_config.action_space_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.prev_actions = torch.zeros_like(self.actions)
        self.last_actor1_attitude_command = torch.zeros_like(self.actions)
        self.last_actor1_output_thrust_normalized = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.last_actor1_output_torques = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device
        )
        self.last_actor1_force_z = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.rewards = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.terminations = self.obs_dict["crashes"]
        self.truncations = self.obs_dict["truncations"]
        self.trajectory_time = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.prev_relative_dist = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )

        self.target_command = torch.zeros((self.num_envs, 4), device=self.device)
        self.control2_kP = _as_tensor(self.task_config.control2.kP, self.device)
        self.control2_kV = _as_tensor(self.task_config.control2.kV, self.device)
        self.control2_kR = _as_tensor(self.task_config.control2.kR, self.device)
        self.control2_kOmega = _as_tensor(self.task_config.control2.kOmega, self.device)
        self.control2_scale = _as_tensor(self.task_config.control2.scale_input, self.device)

        self.motion_name_to_id = {
            "simple_linear": 0,
            "sine_linear": 1,
            "elliptical": 2,
            "apf_escape": 3,
        }
        self.target_motion_type = str(self.task_config.target_motion.type).lower()
        valid_motion_types = set(self.motion_name_to_id.keys()) | {"random"}
        if self.target_motion_type not in valid_motion_types:
            raise ValueError(
                f"Unsupported target motion type: {self.target_motion_type}. "
                f"Valid: {sorted(valid_motion_types)}"
            )
        random_choices = list(self.task_config.target_motion.random_choices)
        if len(random_choices) == 0:
            raise ValueError("target_motion.random_choices must contain at least one motion type")
        invalid_random_choices = [
            motion_name for motion_name in random_choices if motion_name not in self.motion_name_to_id
        ]
        if invalid_random_choices:
            raise ValueError(
                f"Unsupported motion(s) in target_motion.random_choices: {invalid_random_choices}"
            )
        self.random_motion_candidate_ids = torch.tensor(
            [self.motion_name_to_id[motion_name] for motion_name in random_choices],
            dtype=torch.long,
            device=self.device,
        )
        self.target_motion_wall_margin = float(self.task_config.target_motion.wall_margin)
        self.target_motion_obstacle_margin = float(self.task_config.target_motion.obstacle_margin)
        self.target_motion_validation_samples = max(
            2, int(self.task_config.target_motion.validation_samples)
        )
        self.target_motion_max_sample_attempts = max(
            1, int(self.task_config.target_motion.max_sample_attempts)
        )
        self.target_motion_horizon = max(
            self.dt, float(self.task_config.episode_len_steps) * self.dt
        )
        self.target_motion_validation_time = torch.linspace(
            0.0,
            self.target_motion_horizon,
            steps=self.target_motion_validation_samples,
            device=self.device,
        )
        self.target_motion_type_ids = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

        self.room_size = _as_tensor(self.task_config.room.size, self.device)
        self.actor1_reset_pos = _as_tensor(self.task_config.reset.actor1_position, self.device)
        self.actor1_reset_vel = _as_tensor(
            self.task_config.reset.actor1_linear_velocity, self.device
        )
        self.actor1_reset_angvel = _as_tensor(
            self.task_config.reset.actor1_angular_velocity, self.device
        )
        self.actor1_reset_quat = _as_tensor(self.task_config.reset.actor1_quaternion, self.device)
        self.actor2_reset_pos = _as_tensor(self.task_config.reset.actor2_position, self.device)
        self.actor2_reset_vel = _as_tensor(
            self.task_config.reset.actor2_linear_velocity, self.device
        )
        self.actor2_reset_angvel = _as_tensor(
            self.task_config.reset.actor2_angular_velocity, self.device
        )
        self.actor2_reset_quat = _as_tensor(self.task_config.reset.actor2_quaternion, self.device)

        self.static_obstacle_centers = _as_tensor(
            self.task_config.static_obstacles.positions, self.device
        )
        self.static_obstacle_radius = float(self.task_config.static_obstacles.radius)
        self.apf_velocity_state = torch.zeros((self.num_envs, 3), device=self.device)

        self.simple_start_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.simple_direction = torch.zeros((self.num_envs, 3), device=self.device)
        self.simple_speed = torch.zeros(self.num_envs, device=self.device)
        self.sine_start_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.sine_direction = torch.zeros((self.num_envs, 3), device=self.device)
        self.sine_lateral_axis = torch.zeros((self.num_envs, 3), device=self.device)
        self.sine_speed = torch.zeros(self.num_envs, device=self.device)
        self.sine_amp_y = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.sine_omega_y = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.sine_amp_z = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.sine_omega_z = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.sine_phase_y = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.sine_phase_z = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.ellipse_focus1 = torch.zeros((self.num_envs, 2), device=self.device)
        self.ellipse_focus2 = torch.zeros((self.num_envs, 2), device=self.device)
        self.ellipse_semi_major_axis = torch.zeros(self.num_envs, device=self.device)
        self.ellipse_angular_rate = torch.zeros(self.num_envs, device=self.device)
        self.ellipse_height = torch.zeros(self.num_envs, device=self.device)
        self.ellipse_initial_phase = torch.zeros(self.num_envs, device=self.device)
        self.apf_start_pos = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.apf_baseline_direction = torch.zeros((self.num_envs, 3), device=self.device)
        self.apf_baseline_speed = torch.zeros(self.num_envs, device=self.device)

        self.task_obs = {
            "observations": torch.zeros(
                (self.num_envs, self.task_config.observation_space_dim),
                dtype=torch.float32,
                device=self.device,
            ),
            "priviliged_obs": torch.zeros(
                (self.num_envs, self.task_config.privileged_observation_space_dim),
                dtype=torch.float32,
                device=self.device,
            ),
            "rewards": self.rewards.view(self.num_envs, 1),
            "terminations": self.terminations,
            "truncations": self.truncations,
        }
        self.infos = {}

        self.observation_space = Dict(
            {
                "observations": Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.task_config.observation_space_dim,),
                    dtype=np.float32,
                )
            }
        )
        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(self.task_config.action_space_dim,),
            dtype=np.float32,
        )

    def close(self):
        if hasattr(self.sim_env, "delete_env"):
            self.sim_env.delete_env()
        else:
            del self.sim_env
            torch.cuda.empty_cache()

    def render(self, mode="human"):
        self.sim_env.render(render_components="viewer")
        return None

    def reset(self):
        self.sim_env.reset()
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._reset_task_state(env_ids)
        self.rewards.zero_()
        self.terminations.zero_()
        self.truncations.zero_()
        self.infos = {}
        self.process_obs_for_task()
        return self.get_return_tuple()

    def reset_idx(self, env_ids):
        self.sim_env.reset_idx(env_ids)
        self._reset_task_state(env_ids)

    def step(self, actions):
        self.prev_actions[:] = self.actions
        self.actions[:] = torch.clamp(actions.to(self.device), -1.0, 1.0)

        self.trajectory_time += self.dt
        self._update_target_before_step()
        self._apply_target_controller()
        self.sim_env.step(actions=self.actions, env_actions=None)
        self._update_actor1_action_logs()

        reward_info, reset_mask = self._compute_reward_and_dones()
        self.rewards[:] = reward_info["total"]

        done_collision = self._actor1_collision_mask() & (~reward_info["success_mask"])
        done_timeout = self.sim_env.sim_steps >= self.task_config.episode_len_steps
        done_success = reward_info["success_mask"]
        done_far = reward_info["relative_dist"] > self.task_config.reward.far_terminate_distance
        self.truncations[:] = done_timeout
        self.terminations[:] = (
            done_success
            | done_far
            | (
                done_collision
                & bool(self.task_config.reward.terminate_on_collision)
            )
            | reset_mask.bool()
        )

        self.infos = {
            "time_outs": done_timeout.clone(),
            "done_reason_timeout": done_timeout.clone(),
            "done_reason_collision": done_collision.clone(),
            "done_reason_success": done_success.clone(),
            "done_reason_far": done_far.clone(),
        }
        for key, value in reward_info.items():
            self.infos[key] = value.clone() if torch.is_tensor(value) else value
        done_to_return = (self.terminations | self.truncations).clone()
        self.infos["terminal_valid"] = done_to_return
        self.infos["terminal_actor1_pos"] = torch.where(
            done_to_return.unsqueeze(1),
            self.robot_state[:, 0:3],
            torch.zeros_like(self.robot_state[:, 0:3]),
        )
        self.infos["terminal_actor2_pos"] = torch.where(
            done_to_return.unsqueeze(1),
            self.target_state[:, 0:3],
            torch.zeros_like(self.target_state[:, 0:3]),
        )
        self.infos["terminal_target_cmd"] = torch.where(
            done_to_return.unsqueeze(1),
            self.target_command[:, 0:3],
            torch.zeros_like(self.target_command[:, 0:3]),
        )

        reset_envs = self.sim_env.post_reward_calculation_step()
        if len(reset_envs) > 0:
            self._reset_task_state(reset_envs)

        self.process_obs_for_task()
        self.prev_relative_dist[:] = self._relative_distance()
        return self.get_return_tuple()

    def _update_actor1_action_logs(self):
        robot = self.sim_env.robot_manager.robot
        controller_last_scaled = getattr(robot.controller, "last_scaled_input", None)
        if hasattr(robot, "last_scaled_input"):
            self.last_actor1_attitude_command[:] = robot.last_scaled_input
        elif controller_last_scaled is not None:
            self.last_actor1_attitude_command[:] = controller_last_scaled
        else:
            self.last_actor1_attitude_command[:] = robot.action_tensor[:, : self.actions.shape[1]]

        self.last_actor1_force_z[:] = getattr(
            robot, "last_force_z", robot.robot_force_tensors[:, 0, 2]
        )
        self.last_actor1_output_torques[:] = getattr(
            robot, "last_output_torques", robot.robot_torque_tensors[:, 0, 0:3]
        )
        self.last_actor1_output_thrust_normalized[:] = getattr(
            robot,
            "last_output_thrust_normalized",
            self.last_actor1_force_z
            / torch.clamp(
                self.obs_dict["robot_mass"] * torch.norm(self.obs_dict["gravity"], dim=1),
                min=1e-6,
            ),
        )

    def _actor1_collision_mask(self):
        threshold = float(self.task_config.reward.collision_force_threshold)
        robot_body_count = self.sim_env.robot_manager.robot.robot_force_tensors.shape[1]
        contact_norm = torch.norm(
            self.obs_dict["global_contact_force_tensor"][:, :robot_body_count, :], dim=2
        )
        return torch.any(contact_norm > threshold, dim=1)

    def get_return_tuple(self):
        return (
            self.task_obs,
            self.rewards,
            self.terminations,
            self.truncations,
            self.infos,
        )

    def _reset_task_state(self, env_ids):
        if len(env_ids) == 0:
            return
        self.robot_state[env_ids, 0:3] = self.actor1_reset_pos
        self.robot_state[env_ids, 3:7] = self.actor1_reset_quat
        self.robot_state[env_ids, 7:10] = self.actor1_reset_vel
        self.robot_state[env_ids, 10:13] = self.actor1_reset_angvel
        self.target_state[env_ids, 0:3] = self.actor2_reset_pos
        self.target_state[env_ids, 3:7] = self.actor2_reset_quat
        self.target_state[env_ids, 7:10] = self.actor2_reset_vel
        self.target_state[env_ids, 10:13] = self.actor2_reset_angvel
        self.trajectory_time[env_ids] = -self.dt
        self.apf_velocity_state[env_ids] = self.actor2_reset_vel
        self._assign_target_motion_states(env_ids)
        self._set_target_state_at_reset(env_ids)
        self._sample_actor1_reset_positions(env_ids)
        self.sim_env.IGE_env.write_to_sim()
        self.sim_env.robot_manager.robot.update_states()
        self.prev_relative_dist[env_ids] = self._relative_distance()[env_ids]
        self.prev_actions[env_ids] = 0.0

    def _assign_target_motion_states(self, env_ids):
        if len(env_ids) == 0:
            return

        sampled_motion_ids = self._sample_target_motion_type_ids(env_ids)
        self.target_motion_type_ids[env_ids] = sampled_motion_ids

        simple_env_ids = env_ids[sampled_motion_ids == self.motion_name_to_id["simple_linear"]]
        sine_env_ids = env_ids[sampled_motion_ids == self.motion_name_to_id["sine_linear"]]
        elliptical_env_ids = env_ids[sampled_motion_ids == self.motion_name_to_id["elliptical"]]
        apf_env_ids = env_ids[sampled_motion_ids == self.motion_name_to_id["apf_escape"]]

        self._assign_simple_linear_states(simple_env_ids)
        self._assign_sine_linear_states(sine_env_ids)
        self._assign_elliptical_states(elliptical_env_ids)
        self._assign_apf_escape_states(apf_env_ids)

    def _sample_target_motion_type_ids(self, env_ids):
        num_envs = len(env_ids)
        if self.target_motion_type == "random":
            choice_indices = torch.randint(
                low=0,
                high=len(self.random_motion_candidate_ids),
                size=(num_envs,),
                device=self.device,
            )
            return self.random_motion_candidate_ids[choice_indices]
        return torch.full(
            (num_envs,),
            self.motion_name_to_id[self.target_motion_type],
            dtype=torch.long,
            device=self.device,
        )

    def _sample_centered(self, center_value, delta_value, shape):
        center_tensor = torch.as_tensor(center_value, dtype=torch.float32, device=self.device)
        delta_tensor = torch.as_tensor(delta_value, dtype=torch.float32, device=self.device)
        return center_tensor + (
            2.0 * torch.rand(shape, dtype=torch.float32, device=self.device) - 1.0
        ) * delta_tensor

    def _normalize_direction_batch(self, direction):
        direction_norm = torch.norm(direction, dim=1, keepdim=True)
        valid_mask = direction_norm.squeeze(-1) > 1e-6
        direction = direction / torch.clamp(direction_norm, min=1e-6)
        return direction, valid_mask

    def _build_lateral_axis(self, direction):
        lateral_xy = torch.stack(
            (-direction[:, 1], direction[:, 0], torch.zeros_like(direction[:, 0])),
            dim=1,
        )
        lateral_norm = torch.norm(lateral_xy[:, :2], dim=1, keepdim=True)
        valid_mask = lateral_norm.squeeze(-1) > 1e-6
        lateral_xy = lateral_xy / torch.clamp(lateral_norm, min=1e-6)
        if torch.any(~valid_mask):
            lateral_xy[~valid_mask] = torch.tensor(
                [1.0, 0.0, 0.0], dtype=torch.float32, device=self.device
            )
        return lateral_xy

    def _project_positions_to_target_motion_constraints(self, positions):
        projected = positions.clone()
        lower = torch.full(
            (3,), self.target_motion_wall_margin, dtype=torch.float32, device=self.device
        )
        upper = self.room_size - lower
        projected = torch.max(projected, lower.unsqueeze(0))
        projected = torch.min(projected, upper.unsqueeze(0))

        effective_radius = self.static_obstacle_radius + self.target_motion_obstacle_margin + 1e-3
        for obstacle_center in self.static_obstacle_centers:
            rel_xy = projected[:, :2] - obstacle_center[:2].unsqueeze(0)
            dist_xy = torch.norm(rel_xy, dim=1)
            inside_mask = dist_xy < effective_radius
            if not torch.any(inside_mask):
                continue

            safe_dir = rel_xy[inside_mask] / torch.clamp(
                dist_xy[inside_mask].unsqueeze(-1), min=1e-6
            )
            near_center_mask = dist_xy[inside_mask] <= 1e-6
            if torch.any(near_center_mask):
                safe_dir[near_center_mask] = torch.tensor(
                    [1.0, 0.0], dtype=torch.float32, device=self.device
                )
            projected[inside_mask, :2] = obstacle_center[:2].unsqueeze(0) + effective_radius * safe_dir

        return projected

    def _positions_respect_target_motion_constraints(self, positions):
        num_paths = positions.shape[0]
        valid = torch.isfinite(positions).reshape(num_paths, -1).all(dim=1)
        lower = torch.full(
            (3,), self.target_motion_wall_margin, dtype=torch.float32, device=self.device
        )
        upper = self.room_size - lower
        in_room = (positions >= lower.view(1, 1, 3)) & (positions <= upper.view(1, 1, 3))
        valid &= in_room.reshape(num_paths, -1).all(dim=1)

        effective_radius = self.static_obstacle_radius + self.target_motion_obstacle_margin
        rel_xy = positions[:, :, None, :2] - self.static_obstacle_centers[None, None, :, :2]
        dist_xy = torch.norm(rel_xy, dim=-1)
        obstacle_clear = dist_xy >= effective_radius
        valid &= obstacle_clear.reshape(num_paths, -1).all(dim=1)
        return valid

    def _positions_respect_actor1_reset_constraints(self, positions, target_positions):
        valid = torch.isfinite(positions).all(dim=1)
        lower = torch.full(
            (3,),
            float(self.task_config.safety.wall_hard_margin),
            dtype=torch.float32,
            device=self.device,
        )
        upper = self.room_size - lower
        in_room = (positions >= lower.unsqueeze(0)) & (positions <= upper.unsqueeze(0))
        valid &= in_room.all(dim=1)

        effective_radius = self.static_obstacle_radius + float(
            self.task_config.safety.obstacle_hard_margin
        )
        rel_xy = positions[:, None, :2] - self.static_obstacle_centers[None, :, :2]
        dist_xy = torch.norm(rel_xy, dim=-1)
        valid &= (dist_xy >= effective_radius).all(dim=1)

        relative_dist = torch.norm(positions - target_positions, dim=1)
        valid &= relative_dist >= 50.0
        valid &= relative_dist <= 100.0
        return valid

    def _build_simple_linear_candidate_positions(self, start_pos, direction, speed):
        time = self.target_motion_validation_time.view(1, -1, 1)
        return start_pos[:, None, :] + speed[:, None, None] * time * direction[:, None, :]

    def _build_sine_linear_candidate_positions(
        self,
        start_pos,
        direction,
        speed,
        lateral_axis,
        amp_y,
        omega_y,
        amp_z,
        omega_z,
        phase_y,
        phase_z,
    ):
        time = self.target_motion_validation_time.view(1, -1)
        positions = start_pos[:, None, :] + (
            speed[:, None, None] * time[:, :, None] * direction[:, None, :]
        )

        phase_y0 = phase_y[:, None]
        phase_z0 = phase_z[:, None]
        sine_y = torch.sin(omega_y[:, None] * time + phase_y0) - torch.sin(phase_y0)
        sine_z = torch.sin(omega_z[:, None] * time + phase_z0) - torch.sin(phase_z0)

        positions += amp_y[:, None, None] * sine_y[:, :, None] * lateral_axis[:, None, :]
        positions[:, :, 2] += amp_z[:, None] * sine_z
        return positions

    def _build_elliptical_candidate_positions(
        self,
        focus1,
        focus2,
        semi_major_axis,
        angular_rate,
        height,
        initial_phase,
    ):
        focal_vec = focus2 - focus1
        focal_dist = torch.norm(focal_vec, dim=1)
        major_axis_dir = focal_vec / torch.clamp(focal_dist.unsqueeze(-1), min=1e-6)
        minor_axis_dir = torch.stack((-major_axis_dir[:, 1], major_axis_dir[:, 0]), dim=1)
        center_xy = 0.5 * (focus1 + focus2)
        focal_half_distance = 0.5 * focal_dist
        semi_minor_axis = torch.sqrt(
            torch.clamp(semi_major_axis**2 - focal_half_distance**2, min=1e-6)
        )

        theta = initial_phase[:, None] + angular_rate[:, None] * self.target_motion_validation_time.view(1, -1)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        ellipse_xy = (
            center_xy[:, None, :]
            + semi_major_axis[:, None, None] * cos_theta[:, :, None] * major_axis_dir[:, None, :]
            + semi_minor_axis[:, None, None] * sin_theta[:, :, None] * minor_axis_dir[:, None, :]
        )

        positions = torch.zeros(
            (focus1.shape[0], self.target_motion_validation_samples, 3),
            dtype=torch.float32,
            device=self.device,
        )
        positions[:, :, :2] = ellipse_xy
        positions[:, :, 2] = height[:, None]
        return positions

    def _assign_simple_linear_states(self, env_ids):
        if len(env_ids) == 0:
            return

        motion_cfg = self.task_config.target_motion.simple_linear
        rand_cfg = motion_cfg.randomization
        pending_env_ids = env_ids.clone()
        pending_count = len(pending_env_ids)

        for _ in range(self.target_motion_max_sample_attempts):
            if pending_count == 0:
                break

            start_pos = self._sample_centered(
                self.task_config.reset.actor2_position,
                rand_cfg.start_position_delta,
                (pending_count, 3),
            )
            direction, valid_mask = self._normalize_direction_batch(
                self._sample_centered(motion_cfg.direction, rand_cfg.direction_delta, (pending_count, 3))
            )
            speed = self._sample_centered(float(motion_cfg.speed), float(rand_cfg.speed_delta), (pending_count,))
            valid_mask &= speed > 1e-3
            valid_mask &= self._positions_respect_target_motion_constraints(
                self._build_simple_linear_candidate_positions(start_pos, direction, speed)
            )

            if torch.any(valid_mask):
                accepted_env_ids = pending_env_ids[valid_mask]
                self.simple_start_pos[accepted_env_ids] = start_pos[valid_mask]
                self.simple_direction[accepted_env_ids] = direction[valid_mask]
                self.simple_speed[accepted_env_ids] = speed[valid_mask]

            pending_env_ids = pending_env_ids[~valid_mask]
            pending_count = len(pending_env_ids)

        if pending_count > 0:
            raise RuntimeError(
                "Failed to sample feasible simple_linear perturbations within max_sample_attempts."
            )

    def _assign_sine_linear_states(self, env_ids):
        if len(env_ids) == 0:
            return

        motion_cfg = self.task_config.target_motion.sine_linear
        rand_cfg = motion_cfg.randomization
        pending_env_ids = env_ids.clone()
        pending_count = len(pending_env_ids)

        for _ in range(self.target_motion_max_sample_attempts):
            if pending_count == 0:
                break

            start_pos = self._sample_centered(
                self.task_config.reset.actor2_position,
                rand_cfg.start_position_delta,
                (pending_count, 3),
            )
            direction, valid_mask = self._normalize_direction_batch(
                self._sample_centered(motion_cfg.direction, rand_cfg.direction_delta, (pending_count, 3))
            )
            lateral_axis = self._build_lateral_axis(direction)
            speed = self._sample_centered(float(motion_cfg.speed), float(rand_cfg.speed_delta), (pending_count,))
            amp_y = self._sample_centered(float(motion_cfg.amp_y), float(rand_cfg.amp_y_delta), (pending_count,))
            omega_y = self._sample_centered(float(motion_cfg.omega_y), float(rand_cfg.omega_y_delta), (pending_count,))
            amp_z = self._sample_centered(float(motion_cfg.amp_z), float(rand_cfg.amp_z_delta), (pending_count,))
            omega_z = self._sample_centered(float(motion_cfg.omega_z), float(rand_cfg.omega_z_delta), (pending_count,))
            phase_y = self._sample_centered(0.0, float(rand_cfg.phase_y_delta), (pending_count,))
            phase_z = self._sample_centered(0.0, float(rand_cfg.phase_z_delta), (pending_count,))

            valid_mask &= speed > 1e-3
            valid_mask &= amp_y >= 0.0
            valid_mask &= omega_y > 1e-3
            valid_mask &= amp_z >= 0.0
            valid_mask &= omega_z > 1e-3
            valid_mask &= self._positions_respect_target_motion_constraints(
                self._build_sine_linear_candidate_positions(
                    start_pos,
                    direction,
                    speed,
                    lateral_axis,
                    amp_y,
                    omega_y,
                    amp_z,
                    omega_z,
                    phase_y,
                    phase_z,
                )
            )

            if torch.any(valid_mask):
                accepted_env_ids = pending_env_ids[valid_mask]
                self.sine_start_pos[accepted_env_ids] = start_pos[valid_mask]
                self.sine_direction[accepted_env_ids] = direction[valid_mask]
                self.sine_speed[accepted_env_ids] = speed[valid_mask]
                self.sine_lateral_axis[accepted_env_ids] = lateral_axis[valid_mask]
                self.sine_amp_y[accepted_env_ids] = amp_y[valid_mask]
                self.sine_omega_y[accepted_env_ids] = omega_y[valid_mask]
                self.sine_amp_z[accepted_env_ids] = amp_z[valid_mask]
                self.sine_omega_z[accepted_env_ids] = omega_z[valid_mask]
                self.sine_phase_y[accepted_env_ids] = phase_y[valid_mask]
                self.sine_phase_z[accepted_env_ids] = phase_z[valid_mask]

            pending_env_ids = pending_env_ids[~valid_mask]
            pending_count = len(pending_env_ids)

        if pending_count > 0:
            raise RuntimeError(
                "Failed to sample feasible sine_linear perturbations within max_sample_attempts."
            )

    def _assign_elliptical_states(self, env_ids):
        if len(env_ids) == 0:
            return

        motion_cfg = self.task_config.target_motion.elliptical
        rand_cfg = motion_cfg.randomization
        pending_env_ids = env_ids.clone()
        pending_count = len(pending_env_ids)

        for _ in range(self.target_motion_max_sample_attempts):
            if pending_count == 0:
                break

            focus1 = self._sample_centered(motion_cfg.focus1, rand_cfg.focus1_delta, (pending_count, 2))
            focus2 = self._sample_centered(motion_cfg.focus2, rand_cfg.focus2_delta, (pending_count, 2))
            semi_major_axis = self._sample_centered(
                float(motion_cfg.semi_major_axis),
                float(rand_cfg.semi_major_axis_delta),
                (pending_count,),
            )
            angular_rate = self._sample_centered(
                float(motion_cfg.angular_rate),
                float(rand_cfg.angular_rate_delta),
                (pending_count,),
            )
            if bool(rand_cfg.randomize_rotation_direction):
                angular_rate = torch.abs(angular_rate) * torch.where(
                    torch.rand(pending_count, device=self.device) < 0.5,
                    -torch.ones(pending_count, dtype=torch.float32, device=self.device),
                    torch.ones(pending_count, dtype=torch.float32, device=self.device),
                )
            height = self._sample_centered(float(motion_cfg.height), float(rand_cfg.height_delta), (pending_count,))
            initial_phase = self._sample_centered(
                float(motion_cfg.initial_phase),
                float(rand_cfg.initial_phase_delta),
                (pending_count,),
            )

            focal_dist = torch.norm(focus2 - focus1, dim=1)
            valid_mask = focal_dist > 1e-6
            valid_mask &= semi_major_axis > 0.5 * focal_dist + 1e-3
            valid_mask &= torch.abs(angular_rate) > 1e-3
            valid_mask &= self._positions_respect_target_motion_constraints(
                self._build_elliptical_candidate_positions(
                    focus1,
                    focus2,
                    semi_major_axis,
                    angular_rate,
                    height,
                    initial_phase,
                )
            )

            if torch.any(valid_mask):
                accepted_env_ids = pending_env_ids[valid_mask]
                self.ellipse_focus1[accepted_env_ids] = focus1[valid_mask]
                self.ellipse_focus2[accepted_env_ids] = focus2[valid_mask]
                self.ellipse_semi_major_axis[accepted_env_ids] = semi_major_axis[valid_mask]
                self.ellipse_angular_rate[accepted_env_ids] = angular_rate[valid_mask]
                self.ellipse_height[accepted_env_ids] = height[valid_mask]
                self.ellipse_initial_phase[accepted_env_ids] = initial_phase[valid_mask]

            pending_env_ids = pending_env_ids[~valid_mask]
            pending_count = len(pending_env_ids)

        if pending_count > 0:
            raise RuntimeError(
                "Failed to sample feasible elliptical perturbations within max_sample_attempts."
            )

    def _assign_apf_escape_states(self, env_ids):
        if len(env_ids) == 0:
            return

        motion_cfg = self.task_config.target_motion.apf_escape
        rand_cfg = motion_cfg.randomization
        pending_env_ids = env_ids.clone()
        pending_count = len(pending_env_ids)

        for _ in range(self.target_motion_max_sample_attempts):
            if pending_count == 0:
                break

            start_pos = self._sample_centered(
                self.task_config.reset.actor2_position,
                rand_cfg.start_position_delta,
                (pending_count, 3),
            )
            baseline_direction, valid_mask = self._normalize_direction_batch(
                self._sample_centered(
                    motion_cfg.baseline_direction,
                    rand_cfg.baseline_direction_delta,
                    (pending_count, 3),
                )
            )
            baseline_speed = self._sample_centered(
                float(motion_cfg.baseline_speed),
                float(rand_cfg.baseline_speed_delta),
                (pending_count,),
            )

            valid_mask &= baseline_speed > 1e-3
            valid_mask &= self._positions_respect_target_motion_constraints(
                self._build_simple_linear_candidate_positions(
                    start_pos,
                    baseline_direction,
                    baseline_speed,
                )
            )

            if torch.any(valid_mask):
                accepted_env_ids = pending_env_ids[valid_mask]
                self.apf_start_pos[accepted_env_ids] = start_pos[valid_mask]
                self.apf_baseline_direction[accepted_env_ids] = baseline_direction[valid_mask]
                self.apf_baseline_speed[accepted_env_ids] = baseline_speed[valid_mask]
                self.apf_velocity_state[accepted_env_ids] = (
                    baseline_speed[valid_mask].unsqueeze(-1) * baseline_direction[valid_mask]
                )

            pending_env_ids = pending_env_ids[~valid_mask]
            pending_count = len(pending_env_ids)

        if pending_count > 0:
            raise RuntimeError(
                "Failed to sample feasible apf_escape perturbations within max_sample_attempts."
            )

    def _update_target_before_step(self):
        ids = torch.arange(self.num_envs, device=self.device)
        target_command = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)

        simple_env_ids = ids[
            self.target_motion_type_ids == self.motion_name_to_id["simple_linear"]
        ]
        if len(simple_env_ids) > 0:
            target_command[simple_env_ids] = self.generate_trajectory_simple_linear(simple_env_ids)

        sine_env_ids = ids[
            self.target_motion_type_ids == self.motion_name_to_id["sine_linear"]
        ]
        if len(sine_env_ids) > 0:
            target_command[sine_env_ids] = self.generate_trajectory_sine_linear(sine_env_ids)

        elliptical_env_ids = ids[
            self.target_motion_type_ids == self.motion_name_to_id["elliptical"]
        ]
        if len(elliptical_env_ids) > 0:
            target_command[elliptical_env_ids] = self.generate_trajectory_elliptical(
                elliptical_env_ids
            )

        apf_env_ids = ids[
            self.target_motion_type_ids == self.motion_name_to_id["apf_escape"]
        ]
        if len(apf_env_ids) > 0:
            target_command[apf_env_ids] = self.generate_trajectory_apf_escape(apf_env_ids)

        self.target_command[:] = target_command

    def _apply_target_controller(self):
        thrust_normalized, torques = self._old_lee_position_controller(
            self.target_state, self.target_command
        )
        self.target_force_tensor[:] = 0.0
        self.target_torque_tensor[:] = 0.0
        gravity_norm = torch.norm(self.obs_dict["gravity"], dim=1)
        target_weight = self.obs_dict["robot_mass"] * gravity_norm
        self.target_force_tensor[:, 0, 2] = torch.clamp(
            target_weight * thrust_normalized, min=0.0
        )
        self.target_torque_tensor[:, 0, 0:3] = torques

    def _old_lee_position_controller(self, robot_state, command_actions):
        command_actions = command_actions * self.control2_scale
        rotation_matrices = p3d_transforms.quaternion_to_matrix(robot_state[:, [6, 3, 4, 5]])
        rotation_matrix_transpose = torch.transpose(rotation_matrices, 1, 2)
        euler_angles = p3d_transforms.matrix_to_euler_angles(
            rotation_matrices, "ZYX"
        )[:, [2, 1, 0]]
        vehicle_position = robot_state[:, 0:3]
        desired_vehicle_position = command_actions[:, :3]
        pos_error = desired_vehicle_position - vehicle_position
        accel_command = self.control2_kP * pos_error - self.control2_kV * robot_state[:, 7:10]
        accel_command[:, 2] += 1.0
        thrust_command = torch.sum(accel_command * rotation_matrices[:, :, 2], dim=1)
        b3_c = accel_command / torch.clamp(torch.norm(accel_command, dim=1).unsqueeze(1), min=1e-6)
        temp_dir = torch.zeros_like(euler_angles)
        temp_dir[:, 0] = torch.cos(euler_angles[:, 2])
        temp_dir[:, 1] = torch.sin(euler_angles[:, 2])
        b2_c = torch.cross(b3_c, temp_dir, dim=1)
        b2_c = b2_c / torch.clamp(torch.norm(b2_c, dim=1).unsqueeze(1), min=1e-6)
        b1_c = torch.cross(b2_c, b3_c, dim=1)
        rotation_matrix_desired = torch.zeros_like(rotation_matrices)
        rotation_matrix_desired[:, :, 0] = b1_c
        rotation_matrix_desired[:, :, 1] = b2_c
        rotation_matrix_desired[:, :, 2] = b3_c
        rotation_matrix_desired_transpose = torch.transpose(rotation_matrix_desired, 1, 2)
        rot_err_mat = torch.bmm(rotation_matrix_desired_transpose, rotation_matrices) - torch.bmm(
            rotation_matrix_transpose, rotation_matrix_desired
        )
        rot_err = 0.5 * compute_vee_map(rot_err_mat)
        rotmat_euler_to_body_rates = torch.zeros_like(rotation_matrices)
        s_pitch = torch.sin(euler_angles[:, 1])
        c_pitch = torch.cos(euler_angles[:, 1])
        s_roll = torch.sin(euler_angles[:, 0])
        c_roll = torch.cos(euler_angles[:, 0])
        rotmat_euler_to_body_rates[:, 0, 0] = 1.0
        rotmat_euler_to_body_rates[:, 1, 1] = c_roll
        rotmat_euler_to_body_rates[:, 0, 2] = -s_pitch
        rotmat_euler_to_body_rates[:, 2, 1] = -s_roll
        rotmat_euler_to_body_rates[:, 1, 2] = s_roll * c_pitch
        rotmat_euler_to_body_rates[:, 2, 2] = c_roll * c_pitch
        euler_angle_rates = torch.zeros_like(euler_angles)
        yaw_setpoint = euler_angles[:, 2]
        euler_angle_rates[:, 2] = torch.remainder(
            command_actions[:, 3] - yaw_setpoint,
            np.pi * 2.0,
        )
        euler_angle_rates[:, 2] = torch.where(
            euler_angle_rates[:, 2] > np.pi,
            euler_angle_rates[:, 2] - np.pi * 2.0,
            euler_angle_rates[:, 2],
        )
        omega_desired_body = torch.bmm(
            rotmat_euler_to_body_rates, euler_angle_rates.unsqueeze(2)
        ).squeeze(2)
        desired_angvel_err = torch.bmm(
            rotation_matrix_transpose,
            torch.bmm(rotation_matrix_desired, omega_desired_body.unsqueeze(2)),
        ).squeeze(2)
        actual_angvel_err = torch.bmm(
            rotation_matrix_transpose, robot_state[:, 10:13].unsqueeze(2)
        ).squeeze(2)
        angvel_err = actual_angvel_err - desired_angvel_err
        torque = (
            -self.control2_kR * rot_err
            - self.control2_kOmega * angvel_err
            + torch.cross(robot_state[:, 10:13], robot_state[:, 10:13], dim=1)
        )
        return thrust_command, torque

    def _set_target_state_at_reset(self, env_ids):
        if len(env_ids) == 0:
            return
        saved_time = self.trajectory_time[env_ids].clone()
        position = self.target_state[env_ids, 0:3].clone()
        velocity = self.target_state[env_ids, 7:10].clone()
        yaw = torch.atan2(velocity[:, 1], velocity[:, 0])
        simple = self.target_motion_type_ids[env_ids] == self.motion_name_to_id["simple_linear"]
        sine = self.target_motion_type_ids[env_ids] == self.motion_name_to_id["sine_linear"]
        ellipse = self.target_motion_type_ids[env_ids] == self.motion_name_to_id["elliptical"]
        apf = self.target_motion_type_ids[env_ids] == self.motion_name_to_id["apf_escape"]
        reset_time = torch.zeros(len(env_ids), dtype=torch.float32, device=self.device)
        if torch.any(simple):
            pos, vel, cmd_yaw = self._evaluate_simple_linear_reference(
                reset_time[simple],
                env_ids[simple],
            )
            position[simple], velocity[simple], yaw[simple] = pos, vel, cmd_yaw
        if torch.any(sine):
            pos, vel, cmd_yaw = self._evaluate_sine_linear_reference(
                reset_time[sine],
                env_ids[sine],
            )
            position[sine], velocity[sine], yaw[sine] = pos, vel, cmd_yaw
        if torch.any(ellipse):
            pos, vel, cmd_yaw = self._evaluate_elliptical_reference(
                reset_time[ellipse],
                env_ids[ellipse],
            )
            position[ellipse], velocity[ellipse], yaw[ellipse] = pos, vel, cmd_yaw
        if torch.any(apf):
            pos, vel, cmd_yaw = self._evaluate_apf_baseline_reference(
                reset_time[apf],
                env_ids[apf],
            )
            position[apf], velocity[apf], yaw[apf] = pos, vel, cmd_yaw
            self.apf_velocity_state[env_ids[apf]] = vel
        self.target_state[env_ids, 0:3] = position
        self.target_state[env_ids, 3:7] = quat_from_euler_xyz_tensor(
            torch.stack((torch.zeros_like(yaw), torch.zeros_like(yaw), yaw), dim=1)
        )
        self.target_state[env_ids, 7:10] = velocity
        self.target_state[env_ids, 10:13] = 0.0
        self.target_command[env_ids, 0:3] = position
        self.target_command[env_ids, 3] = yaw
        self.trajectory_time[env_ids] = saved_time

    def _evaluate_simple_linear_reference(self, time, env_ids):
        start_pos = self.simple_start_pos[env_ids]
        direction = self.simple_direction[env_ids]
        speed = self.simple_speed[env_ids]
        position = start_pos + speed.unsqueeze(-1) * time.unsqueeze(-1) * direction
        velocity = speed.unsqueeze(-1) * direction
        yaw = torch.atan2(direction[:, 1], direction[:, 0])
        return position, velocity, yaw

    def _evaluate_sine_linear_reference(self, time, env_ids):
        start_pos = self.sine_start_pos[env_ids]
        direction = self.sine_direction[env_ids]
        speed = self.sine_speed[env_ids]
        lateral_axis = self.sine_lateral_axis[env_ids]
        amp_y = self.sine_amp_y[env_ids]
        omega_y = self.sine_omega_y[env_ids]
        amp_z = self.sine_amp_z[env_ids]
        omega_z = self.sine_omega_z[env_ids]
        phase_y = self.sine_phase_y[env_ids]
        phase_z = self.sine_phase_z[env_ids]

        arg_y = omega_y * time + phase_y
        arg_z = omega_z * time + phase_z
        sine_y = torch.sin(arg_y) - torch.sin(phase_y)
        sine_z = torch.sin(arg_z) - torch.sin(phase_z)

        position = start_pos.clone()
        position += speed.unsqueeze(-1) * time.unsqueeze(-1) * direction
        position += amp_y.unsqueeze(-1) * sine_y.unsqueeze(-1) * lateral_axis
        position[:, 2] += amp_z * sine_z

        velocity = speed.unsqueeze(-1) * direction
        velocity += (
            amp_y.unsqueeze(-1)
            * omega_y.unsqueeze(-1)
            * torch.cos(arg_y).unsqueeze(-1)
            * lateral_axis
        )
        velocity[:, 2] += amp_z * omega_z * torch.cos(arg_z)
        yaw = torch.atan2(velocity[:, 1], velocity[:, 0])
        return position, velocity, yaw

    def _evaluate_apf_baseline_reference(self, time, env_ids):
        start_pos = self.apf_start_pos[env_ids]
        direction = self.apf_baseline_direction[env_ids]
        speed = self.apf_baseline_speed[env_ids]
        position = start_pos + speed.unsqueeze(-1) * time.unsqueeze(-1) * direction
        velocity = speed.unsqueeze(-1) * direction
        yaw = torch.atan2(direction[:, 1], direction[:, 0])
        return position, velocity, yaw

    def _evaluate_elliptical_reference(self, time, env_ids):
        focus1 = self.ellipse_focus1[env_ids]
        focus2 = self.ellipse_focus2[env_ids]
        focal_vec = focus2 - focus1
        focal_dist = torch.norm(focal_vec, dim=1)
        semi_major_axis = self.ellipse_semi_major_axis[env_ids]
        focal_half_distance = 0.5 * focal_dist
        if torch.any(semi_major_axis <= focal_half_distance):
            raise ValueError(
                "Sampled elliptical trajectory is invalid: semi_major_axis must be greater than half the focal distance"
            )

        semi_minor_axis = torch.sqrt(
            torch.clamp(semi_major_axis**2 - focal_half_distance**2, min=1e-6)
        )
        angular_rate = self.ellipse_angular_rate[env_ids]
        height = self.ellipse_height[env_ids]
        initial_phase = self.ellipse_initial_phase[env_ids]

        major_axis_dir = focal_vec / torch.clamp(focal_dist.unsqueeze(-1), min=1e-6)
        minor_axis_dir = torch.stack((-major_axis_dir[:, 1], major_axis_dir[:, 0]), dim=1)
        center_xy = 0.5 * (focus1 + focus2)

        theta = initial_phase + angular_rate * time
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        ellipse_xy = (
            center_xy
            + semi_major_axis.unsqueeze(-1) * cos_theta.unsqueeze(-1) * major_axis_dir
            + semi_minor_axis.unsqueeze(-1) * sin_theta.unsqueeze(-1) * minor_axis_dir
        )

        position = torch.zeros((time.shape[0], 3), dtype=torch.float32, device=self.device)
        position[:, :2] = ellipse_xy
        position[:, 2] = height

        vel_xy = (
            -semi_major_axis.unsqueeze(-1)
            * angular_rate.unsqueeze(-1)
            * sin_theta.unsqueeze(-1)
            * major_axis_dir
            + semi_minor_axis.unsqueeze(-1)
            * angular_rate.unsqueeze(-1)
            * cos_theta.unsqueeze(-1)
            * minor_axis_dir
        )
        velocity = torch.zeros((time.shape[0], 3), dtype=torch.float32, device=self.device)
        velocity[:, :2] = vel_xy
        yaw = torch.atan2(velocity[:, 1], velocity[:, 0])
        return position, velocity, yaw

    def generate_trajectory_simple_linear(self, env_ids):
        time = self.trajectory_time[env_ids]
        reference_pos, _reference_vel, reference_yaw = self._evaluate_simple_linear_reference(
            time,
            env_ids,
        )
        traj2 = torch.zeros(len(env_ids), 4, dtype=torch.float32, device=self.device)
        traj2[:, :3] = reference_pos
        traj2[:, 3] = reference_yaw
        return traj2

    def generate_trajectory_sine_linear(self, env_ids):
        time = self.trajectory_time[env_ids]
        reference_pos, _reference_vel, reference_yaw = self._evaluate_sine_linear_reference(
            time,
            env_ids,
        )
        traj2 = torch.zeros(len(env_ids), 4, dtype=torch.float32, device=self.device)
        traj2[:, :3] = reference_pos
        traj2[:, 3] = reference_yaw
        return traj2

    def generate_trajectory_elliptical(self, env_ids):
        time = self.trajectory_time[env_ids]
        ellipse_pos, _ellipse_vel, ellipse_yaw = self._evaluate_elliptical_reference(
            time,
            env_ids,
        )
        traj2 = torch.zeros(len(env_ids), 4, dtype=torch.float32, device=self.device)
        traj2[:, :3] = ellipse_pos
        traj2[:, 3] = ellipse_yaw
        return traj2

    def generate_trajectory_apf_escape(self, env_ids):
        motion_cfg = self.task_config.target_motion.apf_escape
        time = self.trajectory_time[env_ids]
        current_pos = self.target_state[env_ids, 0:3]
        desired_velocity = self._apf_velocity(env_ids, current_pos, time)
        self.apf_velocity_state[env_ids] = desired_velocity

        command_pos = current_pos + float(motion_cfg.command_lookahead) * desired_velocity
        command_pos = self._project_positions_to_target_motion_constraints(command_pos)
        command_yaw = torch.atan2(desired_velocity[:, 1], desired_velocity[:, 0])

        fallback_yaw = torch.atan2(
            self.apf_baseline_direction[env_ids, 1],
            self.apf_baseline_direction[env_ids, 0],
        )
        use_fallback = torch.norm(desired_velocity[:, :2], dim=1) < 1e-3
        command_yaw[use_fallback] = fallback_yaw[use_fallback]

        traj2 = torch.zeros(len(env_ids), 4, dtype=torch.float32, device=self.device)
        traj2[:, :3] = command_pos
        traj2[:, 3] = command_yaw
        return traj2

    def _sample_actor1_reset_positions(self, env_ids):
        if len(env_ids) == 0:
            return
        pending_env_ids = env_ids.clone()
        min_target_distance = 50.0
        max_target_distance = 100.0
        for _ in range(self.target_motion_max_sample_attempts):
            if len(pending_env_ids) == 0:
                break
            target_positions = self.target_state[pending_env_ids, 0:3]
            direction = torch.randn((len(pending_env_ids), 3), device=self.device)
            direction = direction / torch.clamp(torch.norm(direction, dim=1, keepdim=True), min=1e-6)
            radius = min_target_distance + (
                max_target_distance - min_target_distance
            ) * torch.rand(len(pending_env_ids), device=self.device)
            candidates = target_positions + radius.unsqueeze(1) * direction
            valid = self._positions_respect_actor1_reset_constraints(
                candidates,
                target_positions,
            )
            if torch.any(valid):
                accepted = pending_env_ids[valid]
                self.robot_state[accepted, 0:3] = candidates[valid]
            pending_env_ids = pending_env_ids[~valid]
        if len(pending_env_ids) > 0:
            raise RuntimeError(
                "Failed to sample feasible actor1 reset positions within max_sample_attempts."
            )

    def _actor1_reset_positions_valid(self, positions, target_positions):
        return self._positions_respect_actor1_reset_constraints(positions, target_positions)

    def _apf_velocity(self, env_ids, positions, time):
        cfg = self.task_config.target_motion.apf_escape
        _baseline_pos, baseline_velocity, _baseline_yaw = self._evaluate_apf_baseline_reference(
            time,
            env_ids,
        )
        attacker_relative = positions - self.robot_state[env_ids, 0:3]
        distance = torch.norm(attacker_relative, dim=1)
        attacker_dir = attacker_relative / torch.clamp(distance.unsqueeze(1), min=1e-6)
        response = 1.0 / (
            1.0
            + torch.exp(
                torch.clamp(
                    (distance - float(cfg.response_distance)) / float(cfg.response_width),
                    -60.0,
                    60.0,
                )
            )
        )
        response = torch.clamp(response, float(cfg.min_escape_ratio), float(cfg.max_escape_ratio))
        speed_scale = self.apf_baseline_speed[env_ids].unsqueeze(1)
        pursuer_velocity = float(cfg.pursuer_repulsion_weight) * speed_scale * attacker_dir
        attacker_dist_xy = torch.norm(attacker_relative[:, :2], dim=1)
        tangent_xy = torch.zeros((len(env_ids), 2), dtype=torch.float32, device=self.device)
        valid_tangent = attacker_dist_xy > 1e-6
        if torch.any(valid_tangent):
            tangent_xy[valid_tangent, 0] = (
                -attacker_relative[valid_tangent, 1] / attacker_dist_xy[valid_tangent]
            )
            tangent_xy[valid_tangent, 1] = (
                attacker_relative[valid_tangent, 0] / attacker_dist_xy[valid_tangent]
            )
        tangent_velocity = torch.zeros_like(positions)
        tangent_velocity[:, :2] = float(cfg.pursuer_tangent_weight) * speed_scale * tangent_xy
        escape = (
            pursuer_velocity
            + tangent_velocity
            + self._apf_wall_velocity(env_ids, positions)
            + self._apf_obstacle_velocity(env_ids, positions)
        )
        baseline_weight = (
            float(cfg.baseline_weight)
            - response
            * (float(cfg.baseline_weight) - float(cfg.evasive_baseline_weight))
        )
        desired = baseline_weight.unsqueeze(1) * baseline_velocity + response.unsqueeze(1) * escape
        previous = self.apf_velocity_state[env_ids]
        smoothing = float(np.clip(float(cfg.velocity_smoothing), 0.0, 0.999))
        velocity = smoothing * previous + (1.0 - smoothing) * desired
        max_delta = max(float(cfg.max_accel), 0.0) * self.dt
        if max_delta > 0.0:
            delta = velocity - previous
            delta_norm = torch.norm(delta, dim=1)
            delta_scale = torch.clamp(max_delta / torch.clamp(delta_norm, min=1e-6), max=1.0)
            velocity = previous + delta * delta_scale.unsqueeze(1)
        speed = torch.norm(velocity, dim=1)
        speed_scale = torch.clamp(float(cfg.max_speed) / torch.clamp(speed, min=1e-6), max=1.0)
        return velocity * speed_scale.unsqueeze(1)

    def _apf_wall_velocity(self, env_ids, positions):
        cfg = self.task_config.target_motion.apf_escape
        influence = max(float(cfg.wall_influence_distance), 1e-6)
        margin = float(self.task_config.target_motion.wall_margin)
        lower = torch.full((3,), margin, dtype=torch.float32, device=self.device)
        upper = self.room_size - lower
        lower_clearance = positions - lower
        upper_clearance = upper - positions
        lower_intrusion = torch.relu(influence - lower_clearance) / influence
        upper_intrusion = torch.relu(influence - upper_clearance) / influence
        speed_scale = self.apf_baseline_speed[env_ids].unsqueeze(1)
        return float(cfg.wall_repulsion_weight) * speed_scale * (
            lower_intrusion - upper_intrusion
        )

    def _apf_obstacle_velocity(self, env_ids, positions):
        cfg = self.task_config.target_motion.apf_escape
        influence = max(float(cfg.obstacle_influence_distance), 1e-6)
        effective_radius = self.static_obstacle_radius + float(
            self.task_config.target_motion.obstacle_margin
        )
        rel_xy = positions[:, None, :2] - self.static_obstacle_centers[None, :, :2]
        dist_xy = torch.norm(rel_xy, dim=-1)
        clearance = dist_xy - effective_radius
        intrusion = torch.relu(influence - clearance) / influence
        direction_xy = rel_xy / torch.clamp(dist_xy.unsqueeze(-1), min=1e-6)
        velocity_xy = (
            float(cfg.obstacle_repulsion_weight)
            * self.apf_baseline_speed[env_ids].view(-1, 1, 1)
            * intrusion.unsqueeze(-1)
            * direction_xy
        ).sum(dim=1)
        velocity = torch.zeros_like(positions)
        velocity[:, :2] = velocity_xy
        return velocity

    def _project_to_room(self, positions):
        margin = float(self.task_config.target_motion.wall_margin)
        lower = torch.full((3,), margin, dtype=torch.float32, device=self.device)
        upper = self.room_size - lower
        return torch.minimum(torch.maximum(positions, lower), upper)

    def process_obs_for_task(self):
        obs = self.task_obs["observations"]
        robot_pos = self.obs_dict["robot_position"]
        robot_quat = self.obs_dict["robot_orientation"]
        robot_linvel = self.obs_dict["robot_linvel"]
        robot_body_linvel = self.obs_dict["robot_body_linvel"]
        robot_body_angvel = self.obs_dict["robot_body_angvel"]
        target_pos = self.target_state[:, 0:3]
        target_vel = self.target_state[:, 7:10]
        rel_pos_world = target_pos - robot_pos
        rel_vel_world = target_vel - robot_linvel
        rel_pos_body = quat_rotate_inverse(robot_quat, rel_pos_world)
        rel_vel_body = quat_rotate_inverse(robot_quat, rel_vel_world)
        rel_dist = torch.norm(rel_pos_body, dim=1, keepdim=True)
        rel_dir = rel_pos_body / torch.clamp(rel_dist, min=1e-6)
        closing_speed = -torch.sum(rel_dir * rel_vel_body, dim=1, keepdim=True)
        rot = quat_to_rotation_matrix(robot_quat).reshape(self.num_envs, 9)
        obs[:, 0:3] = robot_body_linvel
        obs[:, 3:6] = robot_body_angvel
        obs[:, 6:15] = rot
        obs[:, 15:18] = rel_pos_body
        obs[:, 18:21] = rel_vel_body
        obs[:, 21:22] = closing_speed
        obs[:, 22:28] = self._wall_clearances(
            robot_pos, margin=float(self.task_config.safety.wall_hard_margin)
        )
        obs[:, 28:32] = self._obstacle_relative_body_xy(robot_quat, robot_pos)
        self.task_obs["rewards"] = self.rewards.view(self.num_envs, 1)
        self.task_obs["terminations"] = self.terminations
        self.task_obs["truncations"] = self.truncations

    def _wall_clearances(self, positions, margin=0.0):
        clearances = torch.zeros((self.num_envs, 6), device=self.device)
        clearances[:, 0] = positions[:, 0] - margin
        clearances[:, 1] = self.room_size[0] - positions[:, 0] - margin
        clearances[:, 2] = positions[:, 1] - margin
        clearances[:, 3] = self.room_size[1] - positions[:, 1] - margin
        clearances[:, 4] = positions[:, 2] - margin
        clearances[:, 5] = self.room_size[2] - positions[:, 2] - margin
        return clearances

    def _obstacle_relative_body_xy(self, robot_quat, robot_pos):
        rel_world = self.static_obstacle_centers[None, :, :3] - robot_pos[:, None, :3]
        flat_rel = rel_world.reshape(-1, 3)
        quat = robot_quat[:, None, :].expand(-1, self.static_obstacle_centers.shape[0], -1)
        flat_quat = quat.reshape(-1, 4)
        rel_body = quat_rotate_inverse(flat_quat, flat_rel).reshape(
            self.num_envs, self.static_obstacle_centers.shape[0], 3
        )
        return rel_body[:, :, :2].reshape(self.num_envs, -1)

    def _nearest_hazard_clearance(self):
        robot_pos = self.obs_dict["robot_position"]
        wall = self._wall_clearances(robot_pos).min(dim=1).values
        rel_xy = robot_pos[:, None, :2] - self.static_obstacle_centers[None, :, :2]
        obstacle = torch.norm(rel_xy, dim=2).min(dim=1).values - self.static_obstacle_radius
        return torch.minimum(wall, obstacle)

    def _relative_distance(self):
        return torch.norm(self.target_state[:, 0:3] - self.robot_state[:, 0:3], dim=1)

    def _compute_reward_and_dones(self):
        cfg = self.task_config.reward
        relative_pos = self.target_state[:, 0:3] - self.robot_state[:, 0:3]
        relative_dist = torch.norm(relative_pos, dim=1)
        relative_dir = relative_pos / torch.clamp(relative_dist.unsqueeze(1), min=float(cfg.eps))
        rel_vel = self.target_state[:, 7:10] - self.robot_state[:, 7:10]
        closing_speed = -torch.sum(relative_dir * rel_vel, dim=1)
        forward_axis = quat_axis(self.robot_state[:, 3:7], 0)
        forward_alignment = torch.clamp(torch.sum(forward_axis * relative_dir, dim=1), -1.0, 1.0)
        success_mask = (relative_dist <= float(cfg.success_threshold)) & (
            forward_alignment >= float(cfg.success_forward_alignment_cos)
        )

        progress_reward = self.prev_relative_dist - relative_dist
        remaining_frac = 1.0 - (
            self.sim_env.sim_steps.float() / float(self.task_config.episode_len_steps)
        )
        remaining_frac = torch.clamp(remaining_frac, 0.0, 1.0)
        success_bonus = torch.zeros_like(relative_dist)
        success_bonus[success_mask] = float(cfg.base_success_bonus) + float(
            cfg.early_bonus_scale
        ) * remaining_frac[success_mask]
        time_penalty = -torch.full_like(relative_dist, float(cfg.time_penalty_per_step))
        timeout_mask = (self.sim_env.sim_steps >= self.task_config.episode_len_steps) & (
            ~success_mask
        )
        timeout_penalty = -float(cfg.timeout_penalty) * timeout_mask.float()
        action_delta = self.actions - self.prev_actions
        smooth_penalty = float(cfg.smooth_penalty_coef) * torch.mean(action_delta**2, dim=1)
        amplitude_penalty = float(cfg.amplitude_penalty_coef) * torch.mean(self.actions**2, dim=1)
        effort = smooth_penalty + amplitude_penalty
        nearest_clearance = self._nearest_hazard_clearance()
        avoid_intrusion = torch.relu(float(self.task_config.safety.avoid_margin) - nearest_clearance)
        avoid_penalty = -float(cfg.weight_avoid) * (
            avoid_intrusion / max(float(self.task_config.safety.avoid_margin), float(cfg.eps))
        )
        contrib_progress = float(cfg.weight_progress) * progress_reward
        contrib_success_bonus = float(cfg.weight_success_bonus) * success_bonus
        contrib_time_penalty = time_penalty
        contrib_timeout_penalty = timeout_penalty
        contrib_effort = float(cfg.weight_effort) * effort
        contrib_avoid_penalty = avoid_penalty
        contrib_alignment = float(cfg.weight_alignment) * forward_alignment
        total = (
            contrib_progress
            + contrib_success_bonus
            + contrib_time_penalty
            + contrib_timeout_penalty
            + contrib_effort
            + contrib_avoid_penalty
            + contrib_alignment
        )
        collision_mask = self._actor1_collision_mask() & (~success_mask)
        collision_penalty = -collision_mask.float() * float(cfg.collision_penalty_actor1)
        total = total + collision_penalty
        reset = (
            timeout_mask
            | success_mask
            | (relative_dist > float(cfg.far_terminate_distance))
            | (collision_mask & bool(cfg.terminate_on_collision))
        )
        return {
            "total": total,
            "progress": progress_reward,
            "success_bonus": success_bonus,
            "time_penalty": time_penalty,
            "timeout_penalty": timeout_penalty,
            "effort": effort,
            "collision_penalty": collision_penalty,
            "avoid_penalty": avoid_penalty,
            "contrib_progress": contrib_progress,
            "contrib_success_bonus": contrib_success_bonus,
            "contrib_time_penalty": contrib_time_penalty,
            "contrib_timeout_penalty": contrib_timeout_penalty,
            "contrib_effort": contrib_effort,
            "contrib_collision_penalty": collision_penalty,
            "contrib_avoid_penalty": contrib_avoid_penalty,
            "contrib_alignment": contrib_alignment,
            "min_hazard_clearance": nearest_clearance,
            "relative_dist": relative_dist,
            "closing_speed": closing_speed,
            "forward_alignment": forward_alignment,
            "success_mask": success_mask,
        }, reset
