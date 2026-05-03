import torch

from aerial_gym.robots.base_multirotor import BaseMultirotor


class PursuitQuadDirect(BaseMultirotor):
    """
    Multirotor path that matches the old aerial_0803 force application semantics.

    The old pursuit environment applies the controller output directly to the root
    body as local-space force/torque. The default Aerial Gym v2 multirotor routes
    controller output through allocation and motor dynamics, which changes the
    low-level behavior. This robot keeps the v2 tensor interface but bypasses
    allocation for this migrated task.
    """

    def step(self, action_tensor):
        self.update_states()
        if action_tensor.shape[0] != self.num_envs:
            raise ValueError("Action tensor does not have the correct number of environments")

        self.action_tensor[:] = torch.clamp(action_tensor, -1.0, 1.0)
        command_wrench = self.controller(self.action_tensor)

        self.robot_force_tensors[:] = 0.0
        self.robot_torque_tensors[:] = 0.0
        self.robot_force_tensors[:, 0, 2] = torch.clamp(command_wrench[:, 2], min=0.0)
        self.robot_torque_tensors[:, 0, 0:3] = command_wrench[:, 3:6]

        self.last_scaled_input = self.controller.last_scaled_input
        gravity_norm = torch.norm(self.gravity, dim=1)
        mass = self.controller.mass.squeeze(1)
        denom = torch.clamp(mass * gravity_norm, min=1e-6)
        self.last_output_thrust_normalized = self.robot_force_tensors[:, 0, 2] / denom
        self.last_force_z = self.robot_force_tensors[:, 0, 2]
        self.last_output_torques = self.robot_torque_tensors[:, 0, 0:3]
