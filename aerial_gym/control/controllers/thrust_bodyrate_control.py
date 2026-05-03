import torch

from aerial_gym.control.controllers.base_controller import BaseController


class ThrustBodyRateController(BaseController):
    def __init__(self, config, num_envs, device):
        super().__init__(config, num_envs, device)
        self.rate_error_integral = None
        self.prev_rate_error = None
        self.prev_error_valid = None

    def init_tensors(self, global_tensor_dict=None):
        super().init_tensors(global_tensor_dict)
        self.dt = float(global_tensor_dict["dt"])
        self.scale_input = torch.tensor(
            self.cfg.scale_input, dtype=torch.float32, device=self.device
        )
        self.kOmega = torch.tensor(
            self.cfg.kOmega, dtype=torch.float32, device=self.device
        )
        self.kOmegaI = torch.tensor(
            self.cfg.kOmegaI, dtype=torch.float32, device=self.device
        )
        self.kOmegaD = torch.tensor(
            self.cfg.kOmegaD, dtype=torch.float32, device=self.device
        )
        self.omega_integral_limit = torch.tensor(
            self.cfg.omegaIntegralLimit, dtype=torch.float32, device=self.device
        )
        self.torque_limit = torch.tensor(
            self.cfg.torqueLimit, dtype=torch.float32, device=self.device
        )
        self.wrench_command = torch.zeros((self.num_envs, 6), device=self.device)
        self.last_scaled_input = torch.zeros((self.num_envs, 4), device=self.device)
        self.rate_error_integral = torch.zeros((self.num_envs, 3), device=self.device)
        self.prev_rate_error = torch.zeros((self.num_envs, 3), device=self.device)
        self.prev_error_valid = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    def reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.rate_error_integral[env_ids] = 0.0
        self.prev_rate_error[env_ids] = 0.0
        self.prev_error_valid[env_ids] = False

    def reset(self):
        self.reset_idx(None)

    def randomize_params(self, env_ids):
        return

    def __call__(self, command_actions):
        return self.update(command_actions)

    def update(self, command_actions):
        self.wrench_command[:] = 0.0
        scaled_input = command_actions * self.scale_input
        self.last_scaled_input[:] = scaled_input

        omega_cmd = scaled_input[:, 1:4]
        rate_error = omega_cmd - self.robot_body_angvel

        dt = max(self.dt, 1e-6)
        self.rate_error_integral += rate_error * dt
        integral_limit = self.omega_integral_limit.unsqueeze(0).expand_as(rate_error)
        self.rate_error_integral = torch.clamp(
            self.rate_error_integral, min=-integral_limit, max=integral_limit
        )

        rate_error_derivative = torch.zeros_like(rate_error)
        valid_mask = self.prev_error_valid
        if torch.any(valid_mask):
            rate_error_derivative[valid_mask] = (
                rate_error[valid_mask] - self.prev_rate_error[valid_mask]
            ) / dt

        self.prev_rate_error[:] = rate_error
        self.prev_error_valid[:] = True

        torques = (
            self.kOmega * rate_error
            + self.kOmegaI * self.rate_error_integral
            + self.kOmegaD * rate_error_derivative
        )

        torque_limit = self.torque_limit.unsqueeze(0).expand_as(torques)
        finite_limit_mask = torch.isfinite(torque_limit)
        if torch.any(finite_limit_mask):
            torques = torch.where(
                finite_limit_mask,
                torch.clamp(torques, min=-torque_limit, max=torque_limit),
                torques,
            )

        thrust_normalized = scaled_input[:, 0] + 1.0
        gravity_norm = torch.norm(self.gravity, dim=1)
        self.wrench_command[:, 2] = thrust_normalized * self.mass.squeeze(1) * gravity_norm
        self.wrench_command[:, 3:6] = torques
        return self.wrench_command
